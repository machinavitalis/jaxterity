# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""JAX-traceable QP solver wrapper.

``qp_solve`` solves the standard quadratic program

    min ½ xᵀ H x + gᵀ x   s.t.   A x = b,   lb ≤ x ≤ ub

with H symmetric positive semi-definite. The wrapper presents a single
``qp_solve(H, g, *, A=None, b=None, lb=None, ub=None)`` entry point and
dispatches to the appropriate algorithm:

* **Unconstrained** (no A/lb/ub): closed-form via ``jnp.linalg.solve`` with a
  small Tikhonov regularizer on the diagonal for robustness.
* **Equality-constrained** (A only): KKT system solved by stacking
  ``[[H, Aᵀ], [A, 0]]`` and a single linear solve.
* **Box-constrained** (lb/ub): ADMM-style projected-gradient iterations
  (Heun integrator-style updates) — a simple JAX-traceable loop with a
  fixed iteration budget. Good for low-dimensional WBC where box bounds
  encode joint torque limits or contact normals.

For the centroidal-dynamics-style QPs Pink and Mink solve, the
equality-constrained path is the working horse; box constraints are
secondary and ADMM is sufficient.

All paths are ``jax.jit`` / ``jax.grad`` / ``jax.vmap`` clean — fixed
iteration counts and no Python branches on traced values.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

#: Tikhonov regularization on the QP Hessian — keeps the unconstrained linear
#: solve well-conditioned even when ``H`` is numerically singular (e.g. an
#: underdetermined task set).
_RIDGE = 1e-9
#: Default ADMM penalty parameter ``ρ`` for the box-constrained path.
_RHO_DEFAULT = 1.0
#: Default iteration budget for the ADMM loop. Fixed for ``jit`` compatibility.
_ADMM_ITERS_DEFAULT = 50


def qp_solve(
    H,
    g,
    *,
    A=None,
    b=None,
    lb=None,
    ub=None,
    admm_rho: float = _RHO_DEFAULT,
    admm_iters: int = _ADMM_ITERS_DEFAULT,
):
    """Solve ``min ½ xᵀ H x + gᵀ x`` subject to equality / box constraints.

    Args
    ----
    H : jax.Array, shape ``(n, n)``
        Symmetric positive semi-definite Hessian.
    g : jax.Array, shape ``(n,)``
        Linear term.
    A : jax.Array, shape ``(m, n)`` or None
        Equality-constraint matrix. ``None`` skips equality constraints.
    b : jax.Array, shape ``(m,)`` or None
        Equality-constraint RHS. Must be provided when ``A`` is.
    lb, ub : jax.Array, shape ``(n,)`` or None
        Box bounds. Either may be ``None`` to leave that side unbounded.
    admm_rho : float
        ADMM penalty parameter ρ (box-constrained path only). Larger ρ
        tightens the constraint at the cost of slower progress on the
        objective. Default 1.0 — fine for box-only WBC; equality+box on
        tightly-coupled problems may want 10..100.
    admm_iters : int
        ADMM iteration budget. Default 50. Must be a static int (no traced
        values) so ``lax.scan`` can fix the trip count under ``jit``.

    Returns
    -------
    jax.Array, shape ``(n,)``
        The QP minimizer.

    Notes
    -----
    Combining equality and box constraints (both provided) routes through the
    ADMM loop, projecting onto the box at every iteration and re-solving the
    equality-constrained KKT sub-system.
    """
    n = H.shape[0]
    H_reg = H + _RIDGE * jnp.eye(n)
    if A is None and lb is None and ub is None:
        # Unconstrained: x = -H⁻¹ g.
        return jnp.linalg.solve(H_reg, -g)
    if (lb is None and ub is None) and A is not None:
        # Equality-only: KKT [[H, Aᵀ], [A, 0]] @ [x, λ] = [-g, b].
        return _solve_kkt(H_reg, g, A, b)
    # Box (and optional equality): ADMM.
    return _admm_box(H_reg, g, A=A, b=b, lb=lb, ub=ub, rho=admm_rho, iters=admm_iters)


def _solve_kkt(H, g, A, b):
    """Stacked KKT solve for the equality-constrained QP."""
    n, m = H.shape[0], A.shape[0]
    K = jnp.block([[H, A.T], [A, jnp.zeros((m, m))]])
    rhs = jnp.concatenate([-g, b])
    return jnp.linalg.solve(K, rhs)[:n]


def _admm_box(H, g, *, A, b, lb, ub, rho, iters):
    """ADMM for ``min ½ xᵀ H x + gᵀ x`` s.t. (A x = b)? and lb ≤ z ≤ ub, x = z."""
    n = H.shape[0]
    # Pre-factor the inner KKT system once (constant across iterations).
    K_top = jnp.concatenate(
        [H + rho * jnp.eye(n), A.T if A is not None else jnp.zeros((n, 0))], axis=1
    )
    if A is not None:
        K_bot = jnp.concatenate([A, jnp.zeros((A.shape[0], A.shape[0]))], axis=1)
        K = jnp.concatenate([K_top, K_bot], axis=0)
    else:
        K = K_top  # square already, since the right block is (n, 0)
    z = jnp.zeros(n)
    u = jnp.zeros(n)

    def step(carry, _):
        z, u = carry
        # x-update: argmin_x ½ xᵀHx + gᵀx + ρ/2 ‖x - z + u‖² + (eq constraints)
        rhs_x = -g + rho * (z - u)
        if A is not None:
            rhs = jnp.concatenate([rhs_x, b])
            sol = jnp.linalg.solve(K, rhs)
            x = sol[:n]
        else:
            x = jnp.linalg.solve(K, rhs_x)
        # z-update: project x + u onto the box.
        z_new = x + u
        if lb is not None:
            z_new = jnp.maximum(z_new, lb)
        if ub is not None:
            z_new = jnp.minimum(z_new, ub)
        # u-update.
        u_new = u + (x - z_new)
        return (z_new, u_new), None

    (z_final, _), _ = jax.lax.scan(step, (z, u), None, length=iters)
    return z_final


__all__ = ["qp_solve"]
