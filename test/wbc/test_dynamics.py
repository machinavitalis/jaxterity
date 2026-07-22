# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Dynamics-consistent WBC: torques, contact forces, unactuated-row equality."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mujoco import mjx

from jaxterity import zoo
from jaxterity.wbc import (
    PostureTask,
    WBCSolver,
    actuated_mask,
    inverse_dynamics,
    solve_whole_body,
)
from jaxterity.wbc.dynamics import _bias_forces, _forward
from jaxterity.wbc.locomotion import contact_tasks_from_schedule, foot_body_ids
from jaxterity.wbc.scheduler import periodic_gait


@pytest.fixture(scope="module")
def so100_model():
    return zoo.load("so100").build_system()._mjx_model


# --- inverse dynamics (fixed base, no contact) --------------------------------


def test_inverse_dynamics_round_trips_through_mjx(so100_model):
    """tau = ID(qdd) applied as qfrc_applied reproduces qdd in mjx.forward.

    Property test on an interior (limit-inactive) pose: the equation of
    motion the module writes down must be the one MJX integrates.
    """
    model = so100_model
    q = jnp.zeros(model.nq)
    qd = jnp.full(model.nv, 0.1)
    qdd_desired = jnp.linspace(-1.0, 1.0, int(model.nv))

    tau = inverse_dynamics(model, q, qd, qdd_desired)
    data = mjx.make_data(model).replace(qpos=q, qvel=qd, qfrc_applied=tau)
    qacc = mjx.forward(model, data).qacc
    np.testing.assert_allclose(np.asarray(qacc), np.asarray(qdd_desired), atol=1e-6)


def test_fixed_base_solve_matches_kinematic_solver_plus_id(so100_model):
    """No contacts + fully actuated → qdd equals WBCSolver's, tau equals ID(qdd)."""
    model = so100_model
    nv = int(model.nv)
    q = jnp.zeros(model.nq)
    qd = jnp.zeros(nv)
    task = PostureTask(name="posture", q_desired=jnp.full(model.nq, 0.2))

    sol = solve_whole_body(model, q, qd, (task,))
    qdd_kin = WBCSolver(tasks=(task,)).solve(q, qd, n_dof=nv)

    np.testing.assert_allclose(np.asarray(sol.qdd), np.asarray(qdd_kin), atol=1e-8)
    np.testing.assert_allclose(
        np.asarray(sol.tau),
        np.asarray(inverse_dynamics(model, q, qd, qdd_kin)),
        atol=1e-6,
    )
    assert sol.contact_forces == ()


def test_solve_is_jittable(so100_model):
    model = so100_model
    task = PostureTask(name="posture", q_desired=jnp.zeros(model.nq))

    @jax.jit
    def run(q, qd):
        return solve_whole_body(model, q, qd, (task,)).tau

    tau = run(jnp.zeros(model.nq), jnp.zeros(model.nv))
    assert tau.shape == (model.nv,)
    assert bool(jnp.all(jnp.isfinite(tau)))


# --- floating base (G1) -------------------------------------------------------


@pytest.fixture(scope="module")
def g1():
    robot = zoo.unitree_g1.load(floating_base=True)
    model = robot.build_system()._mjx_model
    ids = foot_body_ids(robot, ("left_ankle_roll_link", "right_ankle_roll_link"))
    return model, ids


def test_actuated_mask_marks_free_joint(g1):
    model, _ = g1
    mask = actuated_mask(model)
    assert mask.shape == (int(model.nv),)
    assert not mask[:6].any()  # floating base
    assert mask[6:].all()


@pytest.mark.slow
def test_double_support_supports_the_weight(g1):
    """Standing double support: the QP finds forces carrying the robot's weight.

    Physics checks on the solution:
    * unactuated (base) rows of the equations of motion are satisfied,
    * normal forces are non-negative (unilateral),
    * total vertical contact force ≈ m·g when the posture task holds still,
    * tau is exactly zero on the base rows.
    """
    model, ids = g1
    nv = int(model.nv)
    q = jnp.asarray(np.array(model.qpos0), float)
    qd = jnp.zeros(nv)

    both_down = periodic_gait(
        foot_names=("L", "R"), period=1.0, duty=1.0, phases=(0.0, 0.0)
    )
    contacts = contact_tasks_from_schedule(model, ids, both_down, 0.0, weight=1e5)

    # PostureTask spans only the actuated joints; embed it over nv via a
    # floating-base-aware wrapper (J = [0 | I]).
    class _JointPosture(PostureTask):
        def residual(self, q_, qd_):
            return self.kp * (self.q_desired - q_[7:]) - self.kd * qd_[6:]

        def jacobian(self, q_, qd_):
            return jnp.concatenate([jnp.zeros((nv - 6, 6)), jnp.eye(nv - 6)], axis=1)

    posture = _JointPosture(name="hold", q_desired=q[7:], kp=50.0, kd=10.0)

    sol = solve_whole_body(model, q, qd, (posture,), contacts=contacts)

    assert sol.qdd.shape == (nv,)
    assert len(sol.contact_forces) == 2
    assert bool(jnp.all(jnp.isfinite(sol.qdd)))

    # (1) unactuated rows of M qdd + h - sum Jc^T lam = 0
    data = _forward(model, q, qd)
    M = mjx.full_m(model, data)
    h = _bias_forces(model, data)
    generalized = M @ sol.qdd + h
    for c, lam in zip(contacts, sol.contact_forces):
        generalized = generalized - c.jacobian_fn(q).T @ lam
    base_residual = np.asarray(generalized[:6])
    scale = float(np.abs(np.asarray(h[:6])).max()) + 1.0
    np.testing.assert_allclose(base_residual, np.zeros(6), atol=2e-2 * scale)

    # (2) unilateral normals
    for lam in sol.contact_forces:
        assert float(lam[2]) >= -1e-6

    # (3) total vertical force ≈ weight (posture holds still → qdd small)
    total_mass = float(jnp.sum(model.body_mass))
    fz = sum(float(lam[2]) for lam in sol.contact_forces)
    assert fz == pytest.approx(total_mass * 9.81, rel=0.05), (
        f"fz={fz} weight={total_mass * 9.81}"
    )

    # (4) tau masked on the base
    np.testing.assert_array_equal(np.asarray(sol.tau[:6]), np.zeros(6))
    assert bool(jnp.any(sol.tau[6:] != 0.0))


@pytest.mark.slow
def test_swing_contact_transmits_no_force(g1):
    """active=0 on a contact zeroes its force block (gated bounds)."""
    model, ids = g1
    nv = int(model.nv)
    q = jnp.asarray(np.array(model.qpos0), float)
    qd = jnp.zeros(nv)

    left_only = periodic_gait(
        foot_names=("L", "R"), period=1.0, duty=0.5, phases=(0.0, 0.5)
    )
    contacts = contact_tasks_from_schedule(model, ids, left_only, 0.0, weight=1e5)
    assert [t.active for t in contacts] == [1.0, 0.0]

    sol = solve_whole_body(model, q, qd, (), contacts=contacts)
    stance, swing = sol.contact_forces
    np.testing.assert_allclose(np.asarray(swing), np.zeros(3), atol=1e-6)
    assert float(stance[2]) > 0.0  # the stance foot carries load
