# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""WBC task DSL + contact scheduler."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaxterity.wbc import (
    CoMTask,
    ContactSchedule,
    ContactTask,
    EndEffectorTask,
    JointAccelerationLimits,
    MomentumTask,
    PostureTask,
    Task,
    WBCSolver,
    periodic_gait,
)

# --- task DSL --------------------------------------------------------


def test_posture_task_pd_residual():
    """r = kp (q_des - q) - kd qd; J = I."""
    task = PostureTask(
        name="posture",
        q_desired=jnp.array([1.0, 0.0]),
        kp=10.0,
        kd=2.0,
    )
    r = task.residual(jnp.array([0.0, 0.0]), jnp.array([0.5, 0.0]))
    np.testing.assert_allclose(np.asarray(r), [10.0 - 1.0, 0.0], atol=1e-9)
    J = task.jacobian(jnp.zeros(2), jnp.zeros(2))
    np.testing.assert_array_equal(np.asarray(J), np.eye(2))


def test_wbc_solver_single_posture_task_recovers_pd_acceleration():
    """A single posture task → qdd equals the PD residual (unconstrained QP)."""
    q_des = jnp.array([0.5, -0.3, 0.1])
    posture = PostureTask(name="p", q_desired=q_des, kp=20.0, kd=4.0)
    solver = WBCSolver(tasks=(posture,))
    q = jnp.zeros(3)
    qd = jnp.zeros(3)
    qdd = solver.solve(q, qd, n_dof=3)
    # H = I (since J=I, weight=1), g = -r → qdd = -H^-1 g = r.
    expected = 20.0 * np.asarray(q_des)
    np.testing.assert_allclose(np.asarray(qdd), expected, atol=1e-4)


def test_wbc_solver_weighted_priority():
    """Higher-weight task dominates when two tasks conflict."""
    primary = PostureTask(
        name="primary", q_desired=jnp.array([1.0]), kp=10.0, weight=100.0
    )
    secondary = PostureTask(
        name="secondary",
        q_desired=jnp.array([0.0]),
        kp=10.0,
        weight=1.0,
    )
    solver = WBCSolver(tasks=(primary, secondary))
    qdd = solver.solve(jnp.zeros(1), jnp.zeros(1), n_dof=1)
    # Weighted-least-squares: qdd ≈ (100·10 + 1·0)/(100 + 1) = 9.9... per unit error.
    expected = (100.0 * 10.0 + 1.0 * 0.0) / (100.0 + 1.0)
    assert float(qdd[0]) == pytest.approx(expected, rel=1e-3)


def test_wbc_solver_with_acceleration_limits_clips_output():
    """JointAccelerationLimits clip the QP solution to the box."""
    posture = PostureTask(name="p", q_desired=jnp.array([10.0]), kp=10.0)
    solver = WBCSolver(tasks=(posture,))
    limits = JointAccelerationLimits(
        lb=jnp.array([-1.0]),
        ub=jnp.array([1.0]),
    )
    qdd = solver.solve(jnp.zeros(1), jnp.zeros(1), n_dof=1, limits=limits)
    # Unconstrained → 100; clipped to 1.
    assert float(qdd[0]) == pytest.approx(1.0, abs=5e-3)


def test_end_effector_task_uses_user_supplied_kinematics():
    """The task delegates pose / Jacobian closures to the caller."""
    J_const = jnp.array([[1.0, 0.5]])
    task = EndEffectorTask(
        name="ee",
        pose_fn=lambda q: jnp.array([q[0] + 0.5 * q[1]]),
        jacobian_fn=lambda q: J_const,
        x_desired=jnp.array([0.5]),
        kp=10.0,
        kd=2.0,
    )
    r = task.residual(jnp.zeros(2), jnp.zeros(2))
    np.testing.assert_allclose(np.asarray(r), [5.0], atol=1e-9)
    J = task.jacobian(jnp.zeros(2), jnp.zeros(2))
    np.testing.assert_array_equal(np.asarray(J), np.asarray(J_const))


def test_com_task_drives_com_toward_setpoint():
    com_fn = lambda q: jnp.array([q[0], q[1], 0.0])  # noqa: E731
    J_fn = lambda q: jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])  # noqa: E731
    task = CoMTask(
        name="com",
        com_fn=com_fn,
        jacobian_fn=J_fn,
        com_desired=jnp.array([0.1, -0.2, 0.0]),
        kp=10.0,
    )
    r = task.residual(jnp.zeros(2), jnp.zeros(2))
    np.testing.assert_allclose(np.asarray(r), [1.0, -2.0, 0.0], atol=1e-9)


# --- required closures (no silent default) ------------------------


def test_end_effector_task_requires_pose_and_jacobian_closures():
    """EndEffectorTask refuses to construct without pose_fn / jacobian_fn.

    Prior versions had `jacobian_fn=lambda q: jnp.zeros((3, 0))` as a silent
    default that crashed at residual() time with a dot-product shape mismatch.
    The fix: closures are required kw_only args; dataclass raises TypeError.
    """
    with pytest.raises(TypeError, match="pose_fn"):
        EndEffectorTask(name="ee")


def test_com_task_requires_com_and_jacobian_closures():
    """CoMTask refuses to construct without com_fn / jacobian_fn."""
    with pytest.raises(TypeError, match="com_fn"):
        CoMTask(name="com")


def test_posture_task_requires_q_desired():
    """PostureTask refuses to construct without q_desired."""
    with pytest.raises(TypeError, match="q_desired"):
        PostureTask(name="p")


# --- mixing tasks with different Jacobian shapes ------------------


def test_wbc_solver_mixes_posture_and_end_effector_jacobians():
    """A 1D end-effector task + a 2D posture task on a 2-DOF robot compose.

    fix: prior tests only mixed two posture tasks. Here the stacked
    Jacobian becomes shape (3, 2): rows 0-1 from posture (I₂), row 2 from
    EE (1×2). The QP must solve a well-posed 2-DOF problem.
    """
    posture = PostureTask(
        name="posture",
        q_desired=jnp.array([0.5, 0.0]),
        kp=10.0,
        weight=1.0,
    )
    ee = EndEffectorTask(
        name="ee",
        pose_fn=lambda q: jnp.array([q[0] + 0.5 * q[1]]),  # 1-D position
        jacobian_fn=lambda q: jnp.array([[1.0, 0.5]]),  # (1, 2)
        x_desired=jnp.array([0.3]),
        kp=10.0,
        weight=1.0,
    )
    solver = WBCSolver(tasks=(posture, ee))
    qdd = solver.solve(jnp.zeros(2), jnp.zeros(2), n_dof=2)
    # assert the *closed-form* WBC minimiser, not just a sign. The QP
    # solves ``(JᵀWJ + λI) qdd = JᵀW r``; here W = I, J stacks posture I₂ over
    # the EE row ``[1, 0.5]``, and ``r = kp·(desired − current) = [5, 0, 3]``.
    # A sign-only check passes for any J with the right sign — this pins the
    # actual least-squares solution (≈ [4.111, −0.444]).
    J = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.5]])
    r = np.array([5.0, 0.0, 3.0])
    qdd_expected = np.linalg.solve(J.T @ J + solver.regularizer * np.eye(2), J.T @ r)
    assert qdd.shape == (2,)
    np.testing.assert_allclose(np.asarray(qdd), qdd_expected, rtol=1e-5, atol=1e-6)


def test_task_base_is_abstract():
    """Base Task raises NotImplementedError when called directly."""
    t = Task(name="base")
    with pytest.raises(NotImplementedError):
        t.residual(jnp.zeros(1), jnp.zeros(1))


# --- contact scheduler ----------------------------------------------


def test_periodic_gait_trot_alternates_diagonal_pairs():
    """Trot: feet 0,3 in contact during first half-period; 1,2 during second."""
    schedule = periodic_gait(
        foot_names=("FL", "FR", "RL", "RR"),
        period=1.0,
        duty=0.5,
        phases=(0.0, 0.5, 0.5, 0.0),
    )
    np.testing.assert_array_equal(np.asarray(schedule.contact(0.0)), [1, 0, 0, 1])
    np.testing.assert_array_equal(np.asarray(schedule.contact(0.25)), [1, 0, 0, 1])
    np.testing.assert_array_equal(np.asarray(schedule.contact(0.6)), [0, 1, 1, 0])


def test_periodic_gait_walk_is_4_beat():
    """Walk: 75% stance per foot, 25% phase offsets → at any time ≥3 feet down."""
    schedule = periodic_gait(
        foot_names=("FL", "FR", "RL", "RR"),
        period=1.0,
        duty=0.75,
        phases=(0.0, 0.25, 0.5, 0.75),
    )
    for t in [0.0, 0.1, 0.5, 0.9]:
        n_stance = int(jnp.sum(schedule.contact(t)))
        assert n_stance >= 3, f"walk at t={t} has only {n_stance} stance feet"


def test_contact_schedule_period_field_stored():
    schedule = periodic_gait(
        foot_names=("a", "b"),
        period=0.4,
        duty=0.5,
        phases=(0.0, 0.5),
    )
    assert schedule.period == 0.4
    assert isinstance(schedule, ContactSchedule)


# --- ContactTask + MomentumTask -------------------------------------------

# A 6-DOF toy system; the "contact" is the first 3 DOFs (J_c = [I3 | 0]).
_JC = jnp.concatenate([jnp.eye(3), jnp.zeros((3, 3))], axis=1)  # (3, 6)


def test_contact_task_holds_point_stationary():
    """High-weight ContactTask drives the contact-point acceleration to zero."""
    posture = PostureTask(name="posture", q_desired=jnp.ones(6), kp=10.0, weight=1.0)
    contact = ContactTask(name="foot", jacobian_fn=lambda q: _JC, weight=1e5)
    qdd = WBCSolver(tasks=(posture, contact)).solve(jnp.zeros(6), jnp.zeros(6), n_dof=6)
    # contact coords (first 3) ≈ 0; free coords (last 3) track the posture PD (=10).
    np.testing.assert_allclose(np.asarray(_JC @ qdd), np.zeros(3), atol=1e-3)
    np.testing.assert_allclose(np.asarray(qdd[3:]), 10.0 * np.ones(3), atol=1e-2)


def test_contact_task_active_zero_disables_it():
    """active=0 removes the task — qdd equals the posture-only solution."""
    posture = PostureTask(name="posture", q_desired=jnp.ones(6), kp=10.0, weight=1.0)
    contact = ContactTask(
        name="foot", jacobian_fn=lambda q: _JC, weight=1e5, active=0.0
    )
    with_swing = WBCSolver(tasks=(posture, contact)).solve(
        jnp.zeros(6), jnp.zeros(6), n_dof=6
    )
    posture_only = WBCSolver(tasks=(posture,)).solve(
        jnp.zeros(6), jnp.zeros(6), n_dof=6
    )
    np.testing.assert_allclose(
        np.asarray(with_swing), np.asarray(posture_only), atol=1e-6
    )


def test_contact_task_bias_zeros_full_acceleration():
    """A convective bias drives J_c@qdd toward -bias (so contact accel stays ~0)."""
    bias = jnp.array([0.1, -0.2, 0.3])
    contact = ContactTask(
        name="foot",
        jacobian_fn=lambda q: _JC,
        bias_fn=lambda q, qd: bias,
        weight=1e5,
    )
    posture = PostureTask(name="posture", q_desired=jnp.zeros(6), kp=1.0, weight=1.0)
    qdd = WBCSolver(tasks=(contact, posture)).solve(jnp.zeros(6), jnp.zeros(6), n_dof=6)
    contact_accel = np.asarray(_JC @ qdd) + np.asarray(bias)
    np.testing.assert_allclose(contact_accel, np.zeros(3), atol=1e-4)


def test_contact_task_requires_jacobian_fn():
    """No silent default Jacobian."""
    with pytest.raises(TypeError, match="jacobian_fn"):
        ContactTask(name="foot")


def test_momentum_task_residual_and_jacobian():
    """r = kp (h_des − A qd) − bias ; J = A."""
    rng = np.random.default_rng(0)
    A = jnp.asarray(rng.normal(size=(6, 8)))
    qd = jnp.asarray(rng.normal(size=8))
    task = MomentumTask(name="mom", momentum_jac_fn=lambda q: A, kp=3.0)
    r = task.residual(jnp.zeros(8), qd)
    np.testing.assert_allclose(
        np.asarray(r), 3.0 * (-(np.asarray(A) @ np.asarray(qd))), atol=1e-6
    )
    np.testing.assert_array_equal(
        np.asarray(task.jacobian(jnp.zeros(8), qd)), np.asarray(A)
    )


def test_momentum_task_drives_momentum_rate():
    """Solving with the momentum task hits its centroidal-momentum-rate target."""
    rng = np.random.default_rng(1)
    A = jnp.asarray(rng.normal(size=(6, 8)))
    qd = jnp.asarray(rng.normal(size=8))
    task = MomentumTask(name="mom", momentum_jac_fn=lambda q: A, kp=2.0, weight=1.0)
    qdd = WBCSolver(tasks=(task,)).solve(jnp.zeros(8), qd, n_dof=8)
    target = 2.0 * (-(np.asarray(A) @ np.asarray(qd)))  # h_desired = 0
    np.testing.assert_allclose(np.asarray(A @ qdd), target, atol=1e-3)


def test_momentum_task_requires_momentum_jac_fn():
    """No silent default centroidal-momentum matrix."""
    with pytest.raises(TypeError, match="momentum_jac_fn"):
        MomentumTask(name="mom")
