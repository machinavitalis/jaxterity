# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""MCP tools exposing the layered S1-S4 API + sysid recipes.

Each public S1/S2/S3/S4 entry point gets a typed MCP tool. The handles
are stateful (a Robot + S4/S3 instance), so the tool surface uses the
ADR-008 stable-handle pattern: callers create a handle, then issue tool
calls referencing it by id. Handles live in a per-server registry.

This module exports the registry + the tool functions; the MCP server
wires them via ``mcp.tool(fn)`` in :func:`build_server`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any

import numpy as np

from .. import zoo
from ..policies import S3, S4
from ..policies.s1 import S1Result, pick, reach, walk_to
from ..policies.s2 import move_joint, track_trajectory
from ..wbc import PostureTask


@dataclasses.dataclass
class _Handle:
    """A registered S4/S3 pair for one robot, keyed by a content hash."""

    s4: S4
    s3: S3


# Module-level registry — one process per MCP server, so a simple dict is enough.
_REGISTRY: dict[str, _Handle] = {}


def _new_handle_id(robot_name: str, q0: list[float] | None) -> str:
    """SHA-256 of (robot_name, q0) → 16-hex-char handle id (stable for replay)."""
    payload = json.dumps({"robot": robot_name, "q0": q0 or []}).encode()
    return "@" + hashlib.sha256(payload).hexdigest()[:16]


# --- handle lifecycle -----------------------------------------------------


def create_robot_handle(robot_name: str, q0: list[float] | None = None) -> str:
    """Instantiate a controllable Robot and return its stable handle id.

    Args
    ----
    robot_name : str
        One of :func:`jaxterity.zoo.available`.
    q0 : list[float] or None
        Initial joint positions, defaults to zeros.

    Returns
    -------
    str
        Handle id of the form ``"@<16-hex>"``. Pass to all subsequent tools.
    """
    robot = zoo.load(robot_name)
    s4 = S4(robot, q0=q0)
    s3 = S3(s4)
    handle = _new_handle_id(robot_name, q0)
    _REGISTRY[handle] = _Handle(s4=s4, s3=s3)
    return handle


def list_handles() -> list[str]:
    """Return all live handle ids."""
    return list(_REGISTRY)


def release_handle(handle: str) -> None:
    """Drop a handle from the registry. No-op if absent."""
    _REGISTRY.pop(handle, None)


def clear_registry() -> None:
    """Drop every live handle.

    Useful in test fixtures that don't want state leaking across tests and
    for MCP-server shutdown hooks. Production servers normally call
    ``release_handle`` per closed session instead.
    """
    _REGISTRY.clear()


def _get(handle: str) -> _Handle:
    if handle not in _REGISTRY:
        raise KeyError(
            f"unknown handle {handle!r}. Use create_robot_handle first; "
            f"live handles: {list_handles()}."
        )
    return _REGISTRY[handle]


# --- S4: raw actuator + encoder ------------------------------------------


def s4_read_encoder(handle: str, joint_name: str) -> float:
    """S4: read a joint's current position by name (rad)."""
    return _get(handle).s4.read_encoder(joint_name)


def s4_read_velocity(handle: str, joint_name: str) -> float:
    """S4: read a joint's current velocity by name (rad/s)."""
    return _get(handle).s4.read_velocity(joint_name)


def s4_apply_torque(handle: str, joint_name: str, tau: float) -> str:
    """S4: set the commanded torque (N·m) on a joint."""
    _get(handle).s4.apply_torque(joint_name, tau)
    return f"applied tau={tau} N·m to {joint_name}"


def s4_joint_names(handle: str) -> list[str]:
    """S4: list the robot's joint names in canonical (q, qd) order."""
    return list(_get(handle).s4.joint_names)


# --- S3: task-level -------------------------------------------------------


def s3_add_posture_task(
    handle: str,
    name: str,
    q_desired: list[float],
    *,
    kp: float = 50.0,
    kd: float = 10.0,
    weight: float = 1.0,
) -> str:
    """S3: register a posture task in the WBC stack."""
    h = _get(handle)
    import jax.numpy as jnp

    h.s3.add_task(
        PostureTask(
            name=name, q_desired=jnp.asarray(q_desired), kp=kp, kd=kd, weight=weight
        )
    )
    return f"added posture task {name!r} (weight={weight})"


def s3_set_task_priority(handle: str, name: str, weight: float) -> str:
    """S3: update an existing task's soft-priority weight."""
    _get(handle).s3.set_task_priority(name, weight)
    return f"set {name!r}.weight = {weight}"


def s3_step_wbc(handle: str) -> list[float]:
    """S3: solve the WBC QP at the current state and return ``qdd``."""
    return list(np.asarray(_get(handle).s3.step_wbc()).tolist())


# --- S2: skills -----------------------------------------------------------


def s2_move_joint(
    handle: str,
    joint_name: str,
    q_des: float,
    *,
    max_steps: int = 200,
    dt: float = 0.01,
) -> dict[str, Any]:
    """S2: drive a single joint to ``q_des`` via posture-task WBC."""
    h = _get(handle)
    qs, _ = move_joint(h.s3, joint_name, q_des, max_steps=max_steps, dt=dt)
    return {
        "n_steps": int(qs.shape[0]),
        "final_q": qs[-1].tolist(),
        "joint_final": float(qs[-1][h.s4.dof_index(joint_name)]),
    }


def s2_track_trajectory(
    handle: str,
    times: list[float],
    q_desired: list[list[float]],
    *,
    dt: float = 0.01,
) -> dict[str, Any]:
    """S2: follow a joint-trajectory ``(t, q_des(t))`` sequence."""
    h = _get(handle)
    qs, _ = track_trajectory(h.s3, np.asarray(times), np.asarray(q_desired), dt=dt)
    return {"n_steps": int(qs.shape[0]), "final_q": qs[-1].tolist()}


# --- S1: high-level -------------------------------------------------------


def _result_to_dict(r: S1Result) -> dict[str, Any]:
    return {
        "success": r.success,
        "summary": r.summary,
        "trajectory_shape": list(r.trajectory.shape)
        if r.trajectory is not None
        else None,
    }


def s1_reach(
    handle: str, q_target: list[float], *, max_steps: int = 500
) -> dict[str, Any]:
    """S1: reach a target joint configuration."""
    return _result_to_dict(
        reach(_get(handle).s3, np.asarray(q_target), max_steps=max_steps)
    )


def s1_pick(handle: str, object_handle: str, pose: list[float]) -> dict[str, Any]:
    """S1: pick object at pose (returns not-implemented summary)."""
    return _result_to_dict(pick(_get(handle).s3, object_handle, np.asarray(pose)))


def s1_walk_to(handle: str, target_xy: list[float]) -> dict[str, Any]:
    """S1: walk to 2D target (returns not-implemented summary)."""
    return _result_to_dict(walk_to(_get(handle).s3, np.asarray(target_xy)))


#: Source-of-truth registry the MCP server adds to its tool table.
LAYERED_TOOLS = (
    # lifecycle
    create_robot_handle,
    list_handles,
    release_handle,
    clear_registry,
    # S4
    s4_read_encoder,
    s4_read_velocity,
    s4_apply_torque,
    s4_joint_names,
    # S3
    s3_add_posture_task,
    s3_set_task_priority,
    s3_step_wbc,
    # S2
    s2_move_joint,
    s2_track_trajectory,
    # S1
    s1_reach,
    s1_pick,
    s1_walk_to,
)


__all__ = [
    "LAYERED_TOOLS",
    "create_robot_handle",
    "list_handles",
    "release_handle",
    "clear_registry",
    "s4_read_encoder",
    "s4_read_velocity",
    "s4_apply_torque",
    "s4_joint_names",
    "s3_add_posture_task",
    "s3_set_task_priority",
    "s3_step_wbc",
    "s2_move_joint",
    "s2_track_trajectory",
    "s1_reach",
    "s1_pick",
    "s1_walk_to",
]
