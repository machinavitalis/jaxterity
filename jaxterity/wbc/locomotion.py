# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Wire a gait :class:`ContactSchedule` into the whole-body-control QP.

The scheduler (``scheduler.py``) says *which* feet are on the ground at time
``t``; the centroidal primitives (``centroidal.py``) say *how* a foot couples to
the generalized accelerations. This module joins them: it turns a schedule + the
foot bodies into the per-foot :class:`ContactTask` list the WBC consumes, with
each task's ``active`` gate set from the contact mask (swing foot → ``0`` → the
task drops out of the QP). Combine that contact set with posture / CoM / momentum
tasks and the same task list serves every phase of a gait.

* :func:`foot_body_ids` — resolve foot link names to MJX body ids once.
* :func:`contact_tasks_from_schedule` — per-foot :class:`ContactTask`s gated by
  ``schedule.contact(t)``.
* :func:`stance_swing_masks` — the boolean stance/swing split at ``t``.
"""

from __future__ import annotations

import mujoco

from .centroidal import contact_jacobian, contact_jacobian_dot
from .scheduler import ContactSchedule
from .tasks import ContactTask


def foot_body_ids(robot, names: tuple[str, ...]) -> tuple[int, ...]:
    """Resolve foot/link names to MJX body ids for ``robot``.

    Build the MuJoCo model once and look each name up; the ids are stable for the
    life of the model and can be passed to :func:`contact_tasks_from_schedule`.
    """
    mj = mujoco.MjModel.from_xml_string(robot.to_mjcf())
    return tuple(int(mj.body(n).id) for n in names)


def _contact_closures(model, body_id: int, with_bias: bool):
    """Build the (jacobian_fn, bias_fn) closures for one foot body."""

    def jac_fn(q):
        return contact_jacobian(model, q, body_id)

    bias_fn = None
    if with_bias:

        def bias_fn(q, qd):  # noqa: F811  (only defined in the with_bias branch)
            return contact_jacobian_dot(model, q, qd, body_id) @ qd

    return jac_fn, bias_fn


def contact_tasks_from_schedule(
    model,
    foot_body_ids: tuple[int, ...],
    schedule: ContactSchedule,
    t: float,
    *,
    weight: float = 1000.0,
    with_bias: bool = False,
) -> tuple[ContactTask, ...]:
    """Build one :class:`ContactTask` per foot, gated by the schedule at time ``t``.

    A stance foot (``schedule.contact(t)[i]`` true) gets ``active = 1`` and
    enters the QP as a near-hard zero-acceleration constraint; a swing foot gets
    ``active = 0`` and contributes nothing. Each task's Jacobian closure is
    :func:`contact_jacobian` bound to that foot's body id (and, when
    ``with_bias``, the exact convective term from :func:`contact_jacobian_dot`).

    Args
    ----
    model : mjx.Model
        From ``robot.build_system()._mjx_model``.
    foot_body_ids : tuple[int, ...]
        Foot body ids, in the same order as ``schedule.foot_names``.
    schedule : ContactSchedule
    t : float
        Time at which to sample the contact mask (concrete; for a jitted control
        loop, gate via the mask directly rather than rebuilding tasks).
    weight : float
        Soft-priority weight for stance contacts (high ≈ hard constraint).
    with_bias : bool
        If ``True``, pass the exact ``J̇_c qd`` bias for a true zero-acceleration
        contact; otherwise use the zero-acceleration simplification.

    Returns
    -------
    tuple[ContactTask, ...]
        One task per foot, ordered like ``foot_body_ids``.
    """
    mask = schedule.contact(t)
    tasks = []
    for i, bid in enumerate(foot_body_ids):
        active = float(mask[i])
        jac_fn, bias_fn = _contact_closures(model, bid, with_bias)
        name = (
            f"{schedule.foot_names[i]}_contact"
            if i < len(schedule.foot_names)
            else f"foot{i}_contact"
        )
        tasks.append(
            ContactTask(
                name=name,
                jacobian_fn=jac_fn,
                bias_fn=bias_fn,
                active=active,
                weight=weight,
            )
        )
    return tuple(tasks)


def stance_swing_masks(schedule: ContactSchedule, t: float):
    """Return ``(stance, swing)`` boolean arrays at time ``t`` (``swing = ~stance``)."""
    stance = schedule.contact(t)
    return stance, ~stance.astype(bool)


__all__ = [
    "foot_body_ids",
    "contact_tasks_from_schedule",
    "stance_swing_masks",
]
