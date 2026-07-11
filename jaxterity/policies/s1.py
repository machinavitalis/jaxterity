# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""High-level interface — ``policies.S1``.

S1 is the agent's primary surface: ``pick(handle, pose)``, ``walk_to(target)``,
``reach(point)``. Each call assembles the right S2 skills + S3 tasks for the
behavior and returns a status code (success / failure / in-progress) the
agent can act on.

This first cut ships interfaces and minimal-viable implementations that
exercise the layered stack end-to-end on the existing zoo robots:

* :func:`reach` — call :func:`s2.move_joint` for every joint to drive the
  arm toward a Cartesian target via a posture-task setpoint. (Real Cartesian
  reach needs IK; here we treat the call as a posture command for arm
  classes — manipulator-specific kinematics live in S2/S3 follow-ups.)
* :func:`pick` and :func:`walk_to` — skeleton implementations that return
  ``NotImplementedError`` with clear pointers to the missing kinematics /
  contact-scheduler / grasp-planner integration. They're enumerated here
  so the surface is visible to the MCP layer and the agent eval harness.

The status code is the contract the agent acts on; richer typed results
(intermediate plans, parity reports against expected motion) are a
follow-up.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from . import s2
from .s3 import S3


@dataclasses.dataclass(frozen=True)
class S1Result:
    """Status of an S1 call.

    Skills that integrate a trajectory (today: :func:`reach`) populate
    ``trajectory``. Skills that are not implemented yet (today: :func:`pick`,
    :func:`walk_to`) return with ``trajectory=None`` and a ``summary`` that
    names the missing primitives. Use ``trajectory is None`` to branch on
    whether the skill executed a controller.

    Fields
    ------
    success : bool
        True iff the skill reached its termination condition cleanly.
    summary : str
        Human / agent readable one-liner. For not-implemented skills, names
        the missing primitives (grasp planner, gait scheduler, ...).
    trajectory : numpy.ndarray, shape ``(n_steps, n_dof)`` or None
        The captured joint trajectory, if the skill integrated one. ``None``
        for not-implemented skills.
    extras : dict[str, object]
        Skill-specific diagnostic payload (final error, intermediate plan,
        parity-report dict, ...). Empty by default. Populated by the skill
        when there's structured data the agent should be able to inspect.
    """

    success: bool
    summary: str
    trajectory: np.ndarray | None = None
    extras: dict = dataclasses.field(default_factory=dict)


def reach(s3_ctrl: S3, q_target: np.ndarray, *, max_steps: int = 500) -> S1Result:
    """Reach a target configuration via per-joint posture commands.

    First-cut implementation: posture-task setpoint for every joint, WBC +
    Euler integration via S2's :func:`~jaxterity.policies.s2.track_trajectory`
    on a single setpoint repeated for ``max_steps`` ticks. Proper Cartesian
    reach (compute IK from a 3D target → ``q_target``) is the next add when
    the manipulator-kinematics S2 follow-up lands.
    """
    q_target = np.asarray(q_target, dtype=np.float64)
    n_steps = max_steps
    q_des = np.broadcast_to(q_target, (n_steps, q_target.shape[0]))
    times = np.arange(n_steps, dtype=np.float64) * 0.01
    qs, _ = s2.track_trajectory(s3_ctrl, times, q_des)
    q_final = qs[-1]
    err = float(np.linalg.norm(q_final - q_target))
    return S1Result(
        success=err < 1e-2,
        summary=f"reach: final error {err:.3e} rad (target n_dof={q_target.shape[0]})",
        trajectory=qs,
        extras={"final_error_rad": err, "n_steps": int(n_steps)},
    )


def pick(s3_ctrl: S3, object_handle: str, pose: np.ndarray) -> S1Result:
    """Pick ``object_handle`` at ``pose``.

    Not implemented in this first cut. Requires (a) a grasp planner that
    maps ``pose`` to an end-effector goal, (b) gripper actuation primitives,
    (c) collision-aware motion planning. Each is its own follow-up; the
    interface is exposed so the agent eval harness can probe the
    failure mode.
    """
    return S1Result(
        success=False,
        summary=(
            f"pick({object_handle!r}, pose={pose.tolist()}): not implemented. "
            "Needs grasp planner + gripper primitives + collision planning."
        ),
    )


def walk_to(s3_ctrl: S3, target_xy: np.ndarray) -> S1Result:
    """Walk to a 2D ground target.

    Not implemented in this first cut. Requires (a) a gait scheduler
    (the scheduler module lays the groundwork), (b) contact-aware WBC tasks
    (the ContactTask follow-up), (c) a planner that maps ``target_xy``
    to a sequence of footstep targets.
    """
    return S1Result(
        success=False,
        summary=(
            f"walk_to({target_xy.tolist()}): not implemented. "
            "Needs gait scheduler integration + ContactTask "
            "+ footstep planner."
        ),
    )


__all__ = ["S1Result", "reach", "pick", "walk_to"]
