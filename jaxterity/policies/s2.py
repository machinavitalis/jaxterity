# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Skill-level interface — ``policies.S2``.

S2 is the "primitive skills" layer: ``move_joint(joint, q_des)`` and
``track_trajectory(traj)``. Internally each call materializes the right
S3 tasks and lets S3 / WBC handle the per-tick QP. The agent's mental
model: "I want this joint to reach this angle" — no controllers, no
tasks.

Two skills shipped first cut:

* :func:`move_joint` — set a posture-task setpoint for one joint and step
  the WBC until convergence (or a step budget).
* :func:`track_trajectory` — feed a ``(t, q_des)`` sequence into the WBC
  one tick at a time.

The functions return ``(q, qd)`` trajectories so the caller can inspect
or replay. They are not JIT-compiled (mostly Python orchestration of S3),
but they call into the JAX-traceable S3/WBC kernels.

API hygiene: the public surfaces used here are
:meth:`S4.dof_index` and :meth:`S3.update_task_field` — these functions
do not reach into S3/S4 private attributes.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..wbc import PostureTask
from .s3 import S3


def move_joint(
    s3: S3,
    joint_name: str,
    q_des: float,
    *,
    max_steps: int = 200,
    dt: float = 0.01,
    tolerance: float = 1e-2,
) -> tuple[np.ndarray, np.ndarray]:
    """Drive a single joint to ``q_des`` via a posture task + WBC.

    Adds a temporary :class:`PostureTask` targeting only this joint, steps
    the WBC + Euler integration until the joint angle is within
    ``tolerance`` or ``max_steps`` exhausted, then removes the task.

    Returns the captured ``(q_traj, qd_traj)`` arrays — shape
    ``(n_steps, n_dof)``.
    """
    q0, _ = s3.s4.state()
    target = np.asarray(q0).copy()
    idx = s3.s4.dof_index(joint_name)
    target[idx] = q_des
    task = PostureTask(
        name=f"_move_{joint_name}",
        q_desired=jnp.asarray(target),
        kp=50.0,
        kd=10.0,
        weight=10.0,
    )
    s3.add_task(task)
    qs, qds = [], []
    for _ in range(max_steps):
        qdd = s3.step_wbc()
        # Realize the WBC qdd through real MJX dynamics (inverse dynamics
        # → torque → MJX step), not Euler-on-qdd + set_state.
        s3.s4.step_with_acceleration(qdd, dt)
        q, qd = s3.s4.state()
        qs.append(np.asarray(q))
        qds.append(np.asarray(qd))
        if abs(float(q[idx]) - q_des) < tolerance:
            break
    s3.remove_task(task.name)
    return np.asarray(qs), np.asarray(qds)


def track_trajectory(
    s3: S3,
    times: np.ndarray,
    q_desired: np.ndarray,
    *,
    dt: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Track a joint-trajectory ``(t, q_des(t))`` over time.

    ``q_desired`` is shape ``(n_steps, n_dof)`` aligned with ``times``. For
    each tick: update the posture task's setpoint via the public
    :meth:`S3.update_task_field`, step WBC + Euler, record. Returns the
    captured ``(q_traj, qd_traj)``.
    """
    assert q_desired.shape[0] == len(times), "q_desired and times must align"
    q, _ = s3.s4.state()
    n = q.shape[0]
    task_name = "_track_trajectory"
    s3.add_task(
        PostureTask(
            name=task_name, q_desired=jnp.zeros(n), kp=50.0, kd=10.0, weight=10.0
        )
    )
    qs, qds = [], []
    for i in range(len(times)):
        s3.update_task_field(task_name, q_desired=jnp.asarray(q_desired[i]))
        qdd = s3.step_wbc()
        # Integrate the WBC qdd through real MJX dynamics, not Euler.
        s3.s4.step_with_acceleration(qdd, dt)
        q, qd = s3.s4.state()
        qs.append(np.asarray(q))
        qds.append(np.asarray(qd))
    s3.remove_task(task_name)
    return np.asarray(qs), np.asarray(qds)


__all__ = ["move_joint", "track_trajectory"]
