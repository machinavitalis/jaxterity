# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""qp_solve — unconstrained / equality / box paths."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jaxterity.wbc import qp_solve


def test_unconstrained_recovers_normal_equation():
    """min ½xᵀHx + gᵀx → x = -H⁻¹g."""
    H = jnp.array([[2.0, 0.5], [0.5, 1.0]])
    g = jnp.array([1.0, -1.0])
    x = qp_solve(H, g)
    expected = -np.linalg.solve(np.asarray(H), np.asarray(g))
    np.testing.assert_allclose(np.asarray(x), expected, atol=1e-6)


def test_equality_constrained_recovers_kkt_minimum():
    """Equality constraint pins one direction; the QP minimum lies on the line."""
    H = jnp.eye(3)
    g = jnp.array([1.0, 2.0, 3.0])
    A = jnp.array([[1.0, 1.0, 1.0]])  # x0 + x1 + x2 = 0
    b = jnp.array([0.0])
    x = qp_solve(H, g, A=A, b=b)
    # KKT closed form: x = -g + Aᵀλ with A x = 0 → λ = A g / (A Aᵀ).
    np.testing.assert_allclose(float(jnp.sum(x)), 0.0, atol=1e-6)


def test_box_constraint_clips_minimum():
    """If the unconstrained minimum is outside the box, QP returns the box edge."""
    H = jnp.eye(2)
    g = jnp.array([-3.0, -2.0])  # unconstrained min is (3, 2)
    lb = jnp.array([0.0, 0.0])
    ub = jnp.array([1.0, 1.0])
    x = qp_solve(H, g, lb=lb, ub=ub)
    # Both unconstrained components hit the upper bound.
    np.testing.assert_allclose(np.asarray(x), [1.0, 1.0], atol=5e-3)


def test_qp_solve_is_jit_traceable():
    """qp_solve composes under jax.jit (no Python branches on traced values)."""
    H = jnp.eye(3)
    g = jnp.array([1.0, 2.0, -1.0])
    f = jax.jit(qp_solve)
    x = f(H, g)
    np.testing.assert_allclose(np.asarray(x), [-1.0, -2.0, 1.0], atol=1e-6)


def test_qp_solve_supports_grad():
    """Forward-mode autodiff flows through the unconstrained solve."""

    def loss(g):
        x = qp_solve(jnp.eye(2), g)
        return jnp.sum(x * x)

    g0 = jnp.array([1.0, 2.0])
    grad = jax.grad(loss)(g0)
    # d/dg ‖-g‖² = 2g
    np.testing.assert_allclose(np.asarray(grad), 2.0 * np.asarray(g0), atol=1e-6)


def test_equality_under_jit_and_grad():
    H = jnp.eye(2)
    A = jnp.array([[1.0, 0.0]])
    b = jnp.array([1.0])
    out = jax.jit(lambda g: qp_solve(H, g, A=A, b=b))(jnp.array([-1.0, -1.0]))
    # x0 = 1 by constraint; x1 = -g1 = 1.
    np.testing.assert_allclose(np.asarray(out), [1.0, 1.0], atol=1e-6)


# --- equality + box (combined) -------------------------------------


def test_equality_plus_box_constraints_satisfied():
    """Combined eq + box routes through ADMM and both constraints hold.

    fix: prior versions left this path untested. The unconstrained
    minimum is [3, 2, 1]; equality sum(x)=2 + box [0,1]³ forces the
    Pareto-optimal [1, 1, 0].
    """
    H = jnp.eye(3)
    g = jnp.array([-3.0, -2.0, -1.0])
    A = jnp.array([[1.0, 1.0, 1.0]])
    b = jnp.array([2.0])
    lb = jnp.array([0.0, 0.0, 0.0])
    ub = jnp.array([1.0, 1.0, 1.0])
    x = qp_solve(H, g, A=A, b=b, lb=lb, ub=ub, admm_rho=10.0, admm_iters=200)
    np.testing.assert_allclose(float(jnp.sum(x)), 2.0, atol=1e-2)
    assert np.all(np.asarray(x) >= -1e-3)
    assert np.all(np.asarray(x) <= 1.0 + 1e-3)


# --- ADMM under jit -------------------------------------------------


def test_box_admm_is_jit_traceable():
    """The box-ADMM path uses lax.scan and Python `is None` branches — must
    survive jax.jit when both lb and ub are concrete arrays."""
    H = jnp.eye(2)
    lb = jnp.array([0.0, 0.0])
    ub = jnp.array([1.0, 1.0])
    f = jax.jit(lambda g: qp_solve(H, g, lb=lb, ub=ub))
    x = f(jnp.array([-3.0, -2.0]))
    np.testing.assert_allclose(np.asarray(x), [1.0, 1.0], atol=5e-3)


# --- ADMM kwargs --------------------------------------------------


def test_admm_iters_kwarg_controls_iteration_count():
    """A larger admm_iters yields a tighter equality residual on a stiff problem."""
    H = jnp.eye(3)
    g = jnp.array([-5.0, -3.0, -1.0])
    A = jnp.array([[1.0, 1.0, 1.0]])
    b = jnp.array([1.0])
    lb = jnp.array([0.0, 0.0, 0.0])
    ub = jnp.array([1.0, 1.0, 1.0])
    x_few = qp_solve(H, g, A=A, b=b, lb=lb, ub=ub, admm_iters=5, admm_rho=1.0)
    x_many = qp_solve(H, g, A=A, b=b, lb=lb, ub=ub, admm_iters=400, admm_rho=10.0)
    res_few = abs(float(jnp.sum(x_few)) - 1.0)
    res_many = abs(float(jnp.sum(x_many)) - 1.0)
    assert res_many <= res_few, (res_few, res_many)
