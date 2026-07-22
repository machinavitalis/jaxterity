# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Reverse-mode differentiable MJX constraint solve (implicit function theorem).

MuJoCo's constraint solver is a ``lax.while_loop`` with a data-dependent trip
count, so ``jax.grad`` through ``mjx.forward`` raises — only forward-mode
(:func:`jaxterity.sysid.parameter_jacobian`) survives, at one pass *per
parameter*. This module wraps the constraint solve with
:func:`jaxonomy.optimization.implicit_solver`: the forward pass runs MJX's
solver untaped, and the backward pass solves the adjoint of the solver's own
optimality (KKT) condition, reconstructed row-by-row from the constraint
Jacobian. One backward pass then yields gradients with respect to *all*
parameters and the state at once.

The optimality condition of MuJoCo's convex acceleration-space problem is::

    M (qacc - qacc_smooth) - efc_J.T @ f(jar) = 0,   jar = efc_J @ qacc - efc_aref

with the per-row constraint force ``f`` determined by the row type:
equality rows ``-D.jar`` (always active), joint/tendon friction rows
``clip(-D.jar, -frictionloss, +frictionloss)``, and inequality
(limit / frictionless- or pyramidal-contact) rows ``-D.min(jar, 0)`` (active
only when violated). Everything in the residual is reconstructed from MJX's
*smooth* pipeline stages (``fwd_position`` … ``fwd_acceleration``), which are
reverse-differentiable — only the iterative solve itself is bypassed.

Scope (first cut): **joint limits, equality, and dof/tendon friction rows** —
elliptic contact cones are not supported, and models that can generate
contacts are rejected at construction (`scope: T-218`; the contact
complementarity residual is tracked follow-up work). The active set is frozen
at the solution, so the gradient is the exact IFT gradient almost everywhere
(the boundary where a constraint switches active/inactive has measure zero).

Accuracy note: the gradient is exact for the KKT *equation*; if MJX's solver
returns a loose approximate solution the gradient corresponds to the exact
root near it (tighten ``model.opt.iterations`` / ``tolerance`` if primal and
gradient must agree to machine precision).

Solver-pairing note: the ``custom_vjp`` wrap forecloses forward-mode (jvp)
through the dynamics, which rules out the ``bdf`` integrator entirely — BDF's
Newton iteration takes a jvp of the RHS even on the primal pass. Pair
``implicit_vjp=True`` with an explicit integrator (``rk4``); keep the default
raw path for ``bdf`` / forward-mode workflows.
"""

from __future__ import annotations

from typing import Any, Callable

import jax.numpy as jnp
from jaxonomy.optimization import implicit_solver

from ..errors import JaxterityError

__all__ = ["implicit_qacc", "constraint_residual"]


def _smooth_data(
    mjx: Any, model: Any, q: jnp.ndarray, qd: jnp.ndarray, tau: jnp.ndarray
):
    """Run MJX's smooth (solver-free) pipeline stages and return the data.

    ``fwd_position`` builds kinematics, the joint-space inertia, and the
    constraint rows (``efc_J``, ``efc_pos``); ``fwd_velocity`` the bias forces
    and the reference acceleration ``efc_aref``; ``fwd_actuation`` /
    ``fwd_acceleration`` the smooth force and ``qacc_smooth``. None of these
    stages contains a data-dependent ``while_loop``, so the result is
    reverse-differentiable.
    """
    data = mjx.make_data(model).replace(qpos=q, qvel=qd, qfrc_applied=tau)
    data = mjx.fwd_position(model, data)
    data = mjx.fwd_velocity(model, data)
    data = mjx.fwd_actuation(model, data)
    data = mjx.fwd_acceleration(model, data)
    return data


def constraint_residual(
    mjx: Any,
    qacc: jnp.ndarray,
    model: Any,
    q: jnp.ndarray,
    qd: jnp.ndarray,
    tau: jnp.ndarray,
) -> jnp.ndarray:
    """KKT residual of MJX's acceleration-space constraint problem.

    Evaluates ``M (qacc - qacc_smooth) - efc_J.T @ f(jar)`` where ``f`` is the
    per-row constraint force law (see module docstring). At the solver's
    solution the residual is zero to solver tolerance; its Jacobians supply
    both sides of the implicit function theorem.

    Args
    ----
    mjx : module
        The ``mujoco.mjx`` module (passed in so this file stays import-light).
    qacc : Float[Array, "nv"]
        Candidate joint acceleration, rad/s^2 (or m/s^2 for slide joints).
    model : mjx.Model
        The (possibly parameter-threaded) MJX model.
    q, qd : Float[Array, "nq"], Float[Array, "nv"]
        Joint positions (rad / m) and velocities (rad/s / m/s).
    tau : Float[Array, "nv"]
        Applied generalized force (``qfrc_applied``), N.m / N.

    Returns
    -------
    Float[Array, "nv"]
        The residual, in generalized-force units (N.m / N).

    Units
    -----
    SI throughout; the residual has the units of a generalized force.
    """
    import mujoco

    data = _smooth_data(mjx, model, q, qd, tau)
    impl = data._impl  # MJX >= 3.10 keeps constraint arrays on Data._impl
    M = mjx.full_m(model, data)
    jar = impl.efc_J @ qacc - impl.efc_aref
    row_type = impl.efc_type
    f_equality = -impl.efc_D * jar
    f_friction = jnp.clip(
        -impl.efc_D * jar, -impl.efc_frictionloss, impl.efc_frictionloss
    )
    f_inequality = -impl.efc_D * jnp.minimum(jar, 0.0)
    is_equality = row_type == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
    is_friction = (row_type == int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF)) | (
        row_type == int(mujoco.mjtConstraint.mjCNSTR_FRICTION_TENDON)
    )
    f = jnp.where(
        is_equality, f_equality, jnp.where(is_friction, f_friction, f_inequality)
    )
    return M @ (qacc - data.qacc_smooth) - impl.efc_J.T @ f


def implicit_qacc(mjx: Any, template_model: Any) -> Callable[..., jnp.ndarray]:
    """Build a reverse-mode differentiable ``qacc`` solve for an MJX model.

    Returns ``qacc_fn(model, q, qd, tau) -> qacc`` that is byte-equivalent to
    ``mjx.forward(model, make_data(model).replace(qpos=q, qvel=qd,
    qfrc_applied=tau)).qacc`` in the forward pass, and reverse-mode
    differentiable with respect to every argument (the threaded ``model``
    arrays, the state, and the applied force) via the implicit function
    theorem. Forward-mode (``jvp`` / ``jacfwd``) is foreclosed by the
    ``custom_vjp`` — use the raw MJX path (the default) for forward-mode
    workflows.

    Args
    ----
    mjx : module
        The ``mujoco.mjx`` module.
    template_model : mjx.Model
        A representative compiled model, used once at construction to verify
        the constraint scope (contacts rejected). The runtime ``model`` passed
        to the returned function may differ in threaded array *values* but
        must share this structure.

    Returns
    -------
    Callable[[Model, Array, Array, Array], Array]
        ``qacc_fn(model, q, qd, tau) -> qacc`` (shape ``(nv,)``).

    Raises
    ------
    JaxterityError
        If the model can generate contact rows — the first-cut scope is
        joint-limit / equality / friction constraints only.

    Examples
    --------
    >>> qacc_fn = implicit_qacc(mjx, mjx_model)  # doctest: +SKIP
    >>> jax.grad(lambda d: qacc_fn(
    ...     mjx_model.replace(dof_damping=d),
    ...     q, qd, tau).sum())(damping)  # doctest: +SKIP
    """
    n_contact_rows = int(mjx.make_data(template_model)._impl.contact.dist.shape[0])
    if n_contact_rows > 0:
        raise JaxterityError(
            "implicit_qacc: the compiled model can generate contact constraint "
            f"rows ({n_contact_rows} padded contacts), but the reverse-mode "
            "implicit constraint solve currently supports joint-limit, "
            "equality, and friction rows only (scope: T-218 first cut). "
            "Fix: disable collisions (geom contype/conaffinity 0) or use "
            "forward-mode `parameter_jacobian` for contact-capable models."
        )

    def solver(theta):
        model, q, qd, tau = theta
        data = mjx.make_data(model).replace(qpos=q, qvel=qd, qfrc_applied=tau)
        return mjx.forward(model, data).qacc

    def residual(qacc, theta):
        model, q, qd, tau = theta
        return constraint_residual(mjx, qacc, model, q, qd, tau)

    wrapped = implicit_solver(solver, residual)

    def qacc_fn(model, q, qd, tau):
        return wrapped((model, q, qd, tau))

    return qacc_fn
