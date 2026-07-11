# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Gait-schedule → WBC contact-task wiring."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaxterity import zoo
from jaxterity.wbc import MomentumTask, WBCSolver, periodic_gait
from jaxterity.wbc.centroidal import centroidal_momentum_matrix
from jaxterity.wbc.locomotion import (
    contact_tasks_from_schedule,
    foot_body_ids,
    stance_swing_masks,
)


def _trot():
    return periodic_gait(
        foot_names=("FL", "FR", "RL", "RR"),
        period=1.0,
        duty=0.5,
        phases=(0.0, 0.5, 0.5, 0.0),
    )


def test_contact_tasks_gated_by_schedule():
    """Stance feet get active=1, swing feet active=0; the mask flips mid-cycle."""
    sched = _trot()
    tasks = contact_tasks_from_schedule(None, (0, 1, 2, 3), sched, 0.0)
    assert [t.active for t in tasks] == [1.0, 0.0, 0.0, 1.0]
    assert tasks[0].name == "FL_contact" and tasks[3].name == "RR_contact"
    later = contact_tasks_from_schedule(None, (0, 1, 2, 3), sched, 0.6)
    assert [t.active for t in later] == [0.0, 1.0, 1.0, 0.0]


def test_contact_tasks_bias_toggle():
    sched = _trot()
    no_bias = contact_tasks_from_schedule(
        None, (0, 1, 2, 3), sched, 0.0, with_bias=False
    )
    with_bias = contact_tasks_from_schedule(
        None, (0, 1, 2, 3), sched, 0.0, with_bias=True
    )
    assert all(t.bias_fn is None for t in no_bias)
    assert all(t.bias_fn is not None for t in with_bias)


def test_contact_tasks_weight_and_count():
    tasks = contact_tasks_from_schedule(None, (0, 1), _trot(), 0.0, weight=2500.0)
    assert len(tasks) == 2
    assert all(t.weight == 2500.0 for t in tasks)


def test_stance_swing_masks_complementary():
    stance, swing = stance_swing_masks(_trot(), 0.0)
    np.testing.assert_array_equal(np.asarray(stance).astype(bool), ~np.asarray(swing))


@pytest.mark.slow
def test_locomotion_contacts_hold_feet_on_g1():
    """Integration: double-support contact tasks on the real G1 → the QP solves
    and the (high-weight) stance-foot contact accelerations come out ~0."""
    robot = zoo.unitree_g1.load(floating_base=True)
    model = robot.build_system()._mjx_model
    ids = foot_body_ids(robot, ("left_ankle_roll_link", "right_ankle_roll_link"))
    both_down = periodic_gait(
        foot_names=("L", "R"), period=1.0, duty=1.0, phases=(0.0, 0.0)
    )
    contacts = contact_tasks_from_schedule(model, ids, both_down, 0.0, weight=1e5)
    assert [t.active for t in contacts] == [1.0, 1.0]

    q = jnp.asarray(np.array(model.qpos0), float)
    qd = jnp.zeros(model.nv)
    # ask for forward momentum via a MomentumTask so the QP is non-trivial;
    # the contacts must still pin the feet.
    mom = MomentumTask(
        name="mom",
        momentum_jac_fn=lambda qq: centroidal_momentum_matrix(model, qq),
        h_desired=jnp.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0]),
        kp=1.0,
        weight=1.0,
    )
    qdd = WBCSolver(tasks=(*contacts, mom)).solve(q, qd, n_dof=int(model.nv))
    assert qdd.shape == (model.nv,)
    assert bool(jnp.all(jnp.isfinite(qdd)))
    # stance-foot contact-point accelerations (J_c qdd) are driven to ~0.
    for t in contacts:
        Jc = t.jacobian_fn(q)
        np.testing.assert_allclose(np.asarray(Jc @ qdd), np.zeros(3), atol=1e-2)
