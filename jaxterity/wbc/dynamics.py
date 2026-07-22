# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Dynamics-consistent whole-body control: torques and contact forces.

:class:`~jaxterity.wbc.WBCSolver` answers the *kinematic* question — which
``qdd`` best satisfies the task stack. This module answers the *dynamic* one:
which ``qdd``, contact forces ``lambda``, and actuator torques ``tau`` are
simultaneously consistent with the equations of motion

    M(q) qdd + h(q, qd) = S_a^T tau + sum_i J_ci^T lambda_i

where ``h = qfrc_bias - qfrc_passive`` (gravity + Coriolis minus passive
joint forces) and ``S_a`` selects the actuated DOFs. On a floating base the
six base DOFs carry no actuator, so the base rows

    S_u (M qdd + h - sum_i J_ci^T lambda_i) = 0

become a hard equality constraint: the base can only be accelerated through
contact. This is the constraint that makes WBC output *physically realizable*
torques on an under-actuated system (KNOWN_GAPS "dynamics-consistency"
entry).

:func:`solve_whole_body` builds one QP over the stacked unknown
``x = [qdd, lambda_1, ..., lambda_nc]``:

* task costs from the same :class:`~jaxterity.wbc.Task` DSL (soft weights),
* the unactuated-row equality above (skipped for a fully actuated robot),
* box bounds on each contact force — unilateral normal (``lambda_z >= 0``,
  z-up convention) and a magnitude cap, both scaled by the contact's
  ``active`` gate so a swing foot transmits no force,

then recovers ``tau = M qdd + h - sum J_c^T lambda`` on the actuated rows.
The friction cone is **not** enforced (named limitation): tangential
components are bounded only by the magnitude cap. A pyramid/cone constraint
needs general inequality rows, which :func:`~jaxterity.wbc.qp_solve` does not
yet expose.

:func:`inverse_dynamics` is the no-contact, fully-actuated special case —
the ``qdd -> tau`` map that turns a kinematic WBC solution into joint
torques on a fixed-base arm.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
from mujoco import mjx

from .qp import qp_solve
from .tasks import ContactTask, JointAccelerationLimits, Task

#: Ridge on the qdd block (matches WBCSolver's default).
_REG = 1e-6
#: Ridge on the contact-force block — prefers the smallest force distribution
#: among the (generally redundant) solutions in multi-contact stances.
_FORCE_REG = 1e-4
#: Default per-component contact-force magnitude cap, N. Generous — it exists
#: to keep the ADMM iterates bounded, not to model actuator or surface limits.
_FORCE_LIMIT = 1e4
#: ADMM iteration budget for the coupled equality+box problem (stiffer than
#: the box-only WBC default of 50).
_ADMM_ITERS = 200


def _bias_forces(model, data) -> jnp.ndarray:
    """``h = qfrc_bias - qfrc_passive`` so that ``M qdd + h = qfrc_applied + J^T f``."""
    return data.qfrc_bias - data.qfrc_passive


def _forward(model, q, qd):
    data = mjx.make_data(model).replace(qpos=q, qvel=qd)
    return mjx.forward(model, data)


def actuated_mask(model) -> np.ndarray:
    """Default actuation mask: every DOF except free-joint (floating-base) DOFs.

    Returns a **static** boolean array of shape ``(nv,)`` — True where a DOF
    is actuated. Hinge/slide DOFs are assumed actuated (the zoo robots carry
    no ``<actuator>`` elements; torque enters as ``qfrc_applied``); the six
    DOFs of a free joint are never actuatable. Pass an explicit mask to
    :func:`solve_whole_body` for robots with passive joints.
    """
    mask = np.ones(int(model.nv), dtype=bool)
    jnt_type = np.asarray(model.jnt_type)
    dofadr = np.asarray(model.jnt_dofadr)
    for jid in range(int(model.njnt)):
        if int(jnt_type[jid]) == 0:  # mjJNT_FREE
            mask[int(dofadr[jid]) : int(dofadr[jid]) + 6] = False
    return mask


@dataclasses.dataclass(frozen=True)
class WholeBodySolution:
    """Result of :func:`solve_whole_body`.

    Fields
    ------
    qdd : jax.Array, shape ``(nv,)``
        Generalized acceleration, rad/s^2 (m/s^2 for slide/translation DOFs).
    contact_forces : tuple[jax.Array, ...]
        One world-frame force per contact, each shape ``(k,)`` matching that
        contact's Jacobian rows (``k = 3``: fx, fy, fz in N; ``k = 6`` adds
        torque rows in N.m). Empty tuple when no contacts were passed.
    tau : jax.Array, shape ``(nv,)``
        Actuator torques (N.m; N for slide DOFs), **zero on unactuated rows**
        — the equality constraint makes the recovered generalized force
        vanish there up to solver tolerance, and the returned value is masked
        exactly.
    """

    qdd: jnp.ndarray
    contact_forces: tuple[jnp.ndarray, ...]
    tau: jnp.ndarray


def inverse_dynamics(model, q, qd, qdd) -> jnp.ndarray:
    """Joint torques realizing ``qdd`` on a fully actuated, contact-free robot.

    ``tau = M(q) qdd + qfrc_bias - qfrc_passive`` — the map from a kinematic
    WBC solution (:meth:`WBCSolver.solve`) to the torque command a fixed-base
    arm needs. For floating bases or contacts use :func:`solve_whole_body`.

    Args
    ----
    model : mjx.Model
        The compiled model (``robot.build_system()._mjx_model``).
    q, qd : jax.Array, shapes ``(nq,)``, ``(nv,)``
        Joint positions (rad / m) and velocities (rad/s / m/s).
    qdd : jax.Array, shape ``(nv,)``
        Desired generalized acceleration.

    Returns
    -------
    jax.Array, shape ``(nv,)``
        Generalized force, N.m (N for slide DOFs).

    Units
    -----
    SI throughout (rad, m, s, N.m).
    """
    data = _forward(model, q, qd)
    M = mjx.full_m(model, data)
    return M @ qdd + _bias_forces(model, data)


def solve_whole_body(
    model,
    q,
    qd,
    tasks: tuple[Task, ...],
    *,
    contacts: tuple[ContactTask, ...] = (),
    actuated: np.ndarray | None = None,
    limits: JointAccelerationLimits | None = None,
    regularizer: float = _REG,
    force_regularizer: float = _FORCE_REG,
    contact_force_limit: float = _FORCE_LIMIT,
    admm_iters: int = _ADMM_ITERS,
    admm_rho: float = 10.0,
) -> WholeBodySolution:
    """Solve the dynamics-consistent whole-body QP for ``(qdd, lambda, tau)``.

    Minimizes the weighted task residuals (the same soft hierarchy as
    :class:`~jaxterity.wbc.WBCSolver`) over ``x = [qdd, lambda]`` subject to
    the unactuated-row equations of motion and unilateral, gated contact
    force bounds — then recovers actuated torques by substitution. See the
    module docstring for the formulation and its named limitations (no
    friction cone).

    Args
    ----
    model : mjx.Model
        The compiled model (``robot.build_system()._mjx_model``).
    q, qd : jax.Array, shapes ``(nq,)``, ``(nv,)``
        Current joint positions and velocities (rad / m, rad/s / m/s).
    tasks : tuple[Task, ...]
        The motion task stack (posture, CoM, momentum, end-effector, ...).
        Contact tasks belong in ``contacts``, not here.
    contacts : tuple[ContactTask, ...]
        Contacts that may transmit force. Each contributes (a) its gated
        no-acceleration motion task rows, exactly as in
        :meth:`WBCSolver.solve`, and (b) a force unknown ``lambda`` of the
        same row dimension as its Jacobian, with bounds scaled by its
        ``active`` gate (``active = 0`` forces ``lambda = 0``).
    actuated : np.ndarray | None
        Static boolean mask, shape ``(nv,)``, True on actuated DOFs.
        ``None`` (default) uses :func:`actuated_mask` — everything actuated
        except free-joint DOFs.
    limits : JointAccelerationLimits | None
        Optional box bounds on ``qdd``.
    regularizer, force_regularizer : float
        Ridge terms on the ``qdd`` and ``lambda`` blocks.
    contact_force_limit : float
        Per-component force magnitude cap, N (see module constants).
    admm_iters, admm_rho : int, float
        Budget / penalty for the coupled equality+box ADMM (only used when
        the problem actually has bounds or equality rows).

    Returns
    -------
    WholeBodySolution
        ``qdd``, per-contact world-frame forces, and masked actuated ``tau``.

    Frame
    -----
    Contact forces are world-frame, matching the world-frame contact
    Jacobians from :func:`~jaxterity.wbc.contact_jacobian`. The unilateral
    bound applies to the world-z (index 2) component — a flat, z-up ground
    convention.

    Units
    -----
    SI throughout (rad, m, s, N, N.m).

    Examples
    --------
    Fixed-base arm, torque output (no contacts)::

        sol = solve_whole_body(model, q, qd, (PostureTask(q_desired=q_ref),))
        tau_command = sol.tau

    Floating-base double support::

        feet = tuple(
            ContactTask(name=f"foot{i}", jacobian_fn=Jc, active=1.0)
            for i, Jc in enumerate(foot_jacobians)
        )
        sol = solve_whole_body(model, q, qd, tasks, contacts=feet)
    """
    nv = int(model.nv)
    data = _forward(model, q, qd)
    M = mjx.full_m(model, data)
    h = _bias_forces(model, data)

    if actuated is None:
        actuated = actuated_mask(model)
    actuated = np.asarray(actuated, dtype=bool)
    unactuated_rows = np.flatnonzero(~actuated)

    # Raw (ungated) contact Jacobians for the force coupling; the motion rows
    # below use the tasks' own sqrt(active) gating.
    Jcs = [c.jacobian_fn(q) for c in contacts]
    k_dims = [int(Jc.shape[0]) for Jc in Jcs]
    n = nv + sum(k_dims)

    # --- cost: task rows (over qdd) + ridge on both blocks -----------------
    stack: list[jnp.ndarray] = []
    res: list[jnp.ndarray] = []
    for t in (*tasks, *contacts):
        stack.append(t.jacobian(q, qd) * jnp.sqrt(t.weight))
        res.append(t.residual(q, qd) * jnp.sqrt(t.weight))
    J = jnp.concatenate(stack, axis=0)
    r = jnp.concatenate(res, axis=0)
    J_full = jnp.concatenate([J, jnp.zeros((J.shape[0], n - nv))], axis=1)
    ridge = jnp.concatenate(
        [jnp.full(nv, regularizer), jnp.full(n - nv, force_regularizer)]
    )
    H = J_full.T @ J_full + jnp.diag(ridge)
    g = -J_full.T @ r

    # --- equality: unactuated rows of the equations of motion --------------
    A = b = None
    if unactuated_rows.size:
        blocks = [M[unactuated_rows, :]]
        for Jc in Jcs:
            blocks.append(-Jc.T[unactuated_rows, :])
        A = jnp.concatenate(blocks, axis=1)
        b = -h[unactuated_rows]

    # --- box bounds: qdd limits + gated unilateral force caps --------------
    lb = ub = None
    if limits is not None or contacts:
        qdd_lb = limits.lb if limits is not None else jnp.full(nv, -jnp.inf)
        qdd_ub = limits.ub if limits is not None else jnp.full(nv, jnp.inf)
        lb_parts, ub_parts = [qdd_lb], [qdd_ub]
        for c, k in zip(contacts, k_dims):
            cap = contact_force_limit * jnp.asarray(c.active)
            lo = jnp.full(k, -1.0) * cap
            hi = jnp.full(k, 1.0) * cap
            lo = lo.at[2].set(0.0) if k >= 3 else lo  # unilateral normal (z-up)
            lb_parts.append(lo)
            ub_parts.append(hi)
        lb = jnp.concatenate(lb_parts)
        ub = jnp.concatenate(ub_parts)

    x = qp_solve(H, g, A=A, b=b, lb=lb, ub=ub, admm_rho=admm_rho, admm_iters=admm_iters)

    qdd = x[:nv]
    forces: list[jnp.ndarray] = []
    offset = nv
    for k in k_dims:
        forces.append(x[offset : offset + k])
        offset += k

    generalized = M @ qdd + h
    for Jc, lam in zip(Jcs, forces):
        generalized = generalized - Jc.T @ lam
    tau = jnp.where(jnp.asarray(actuated), generalized, 0.0)
    return WholeBodySolution(qdd=qdd, contact_forces=tuple(forces), tau=tau)


__all__ = [
    "WholeBodySolution",
    "actuated_mask",
    "inverse_dynamics",
    "solve_whole_body",
]
