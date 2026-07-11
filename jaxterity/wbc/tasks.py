# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Whole-body-control task DSL and hierarchical QP composition.

A :class:`Task` declares a residual function ``r(q, qd)`` and a Jacobian
``J(q, qd)`` such that the task is satisfied when ``J · qdd ≈ r``. The
:class:`WBCSolver` composes a list of weighted tasks into a single QP minimizing

    min ½ ‖W^½ (J qdd − r)‖² + ε ‖qdd‖²
    qdd

solved at every control tick. Tasks are weighted (Pink/Mink-style soft
hierarchy), not strictly prioritized — soft hierarchy is straightforward to
JIT and yields the desired qualitative ordering (10× weight ≈ 10× tighter
residual) without the cascade of QPs a strict hierarchy needs.

Concrete tasks shipped here:

* :class:`PostureTask` — track a joint-position setpoint (J = I).
* :class:`EndEffectorTask` — track a Cartesian end-effector pose via a
  user-supplied Jacobian closure.
* :class:`CoMTask` — track a center-of-mass position.
* :class:`ContactTask` — hold a contact point (foot) stationary, with a
  per-foot ``active`` gate driven by a :class:`ContactSchedule` (swing → 0).
* :class:`MomentumTask` — regulate centroidal momentum (zero angular
  momentum is the standard balance objective).
* :class:`JointAccelerationLimits` — symmetric box bounds on ``qdd``.

Each task is a frozen dataclass so it can flow through ``jax.jit`` and
``jax.vmap`` cleanly. The Jacobian is a closure (callable) rather than a
stored array so it can be recomputed from the current state per tick.

``ContactTask`` and ``MomentumTask`` consume the centroidal-dynamics
primitives (``contact_jacobian`` / ``centroidal_momentum_matrix``, a separate
MJX-backed module) as their Jacobian closures, keeping this module
kinematics-agnostic like :class:`EndEffectorTask`.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import equinox as eqx
import jax.numpy as jnp

from .qp import qp_solve

#: Default regularizer on ``qdd`` so the QP stays well-conditioned even when
#: the task Jacobian is rank-deficient (an underdetermined task set).
_REG = 1e-6


@dataclasses.dataclass(frozen=True)
class Task:
    """Base task: residual + Jacobian + weight.

    Subclasses bind specific (residual, Jacobian) computations. Carrying the
    Jacobian as a callable lets it recompute from the current state each tick.

    Fields
    ------
    name : str
        Human-readable label, used in diagnostics and the WBC priority table.
    weight : float
        Soft-priority weight. A 10× weight gives a ~10× tighter residual.
        Must be ≥ 0.
    """

    name: str
    weight: float = 1.0

    def residual(self, q, qd) -> jnp.ndarray:
        """Desired ``J qdd`` value: r ∈ ℝ^k_task."""
        raise NotImplementedError

    def jacobian(self, q, qd) -> jnp.ndarray:
        """``∂r/∂qdd`` = J ∈ ℝ^(k_task × n_dof)."""
        raise NotImplementedError


@dataclasses.dataclass(frozen=True, kw_only=True)
class PostureTask(Task):
    """Drive ``q`` toward ``q_desired`` with PD-on-acceleration shaping.

    Residual: ``r = Kp (q_desired − q) − Kd qd``. With J = I this is the
    classic PD-on-acceleration controller — qdd tracks the reference.

    Fields
    ------
    q_desired : jax.Array, shape ``(n_dof,)``
        Setpoint. Required — passing a wrong-shape value here is the most
        common bug, so the dataclass refuses to construct without it.
    kp, kd : float
        Position / velocity gains.
    """

    q_desired: jnp.ndarray
    kp: float = 50.0
    kd: float = 10.0

    def residual(self, q, qd):
        return self.kp * (self.q_desired - q) - self.kd * qd

    def jacobian(self, q, qd):
        return jnp.eye(q.shape[0])


@dataclasses.dataclass(frozen=True, kw_only=True)
class EndEffectorTask(Task):
    """Drive an end-effector Cartesian pose toward a setpoint.

    The user supplies ``J_fn(q) -> (k, n_dof)`` and ``pose_fn(q) -> (k,)``;
    typically computed from MJX via ``mjx.jac_site_translation`` + the
    site's current Cartesian position. We keep them as callables so the
    user owns the kinematics integration (no MJX-specific code in this
    module).

    Residual: ``r = kp · (x_des − pose_fn(q)) − kd · J_fn(q) qd``.

    Fields
    ------
    pose_fn, jacobian_fn : Callable
        Required — silent zero-Jacobian defaults were a footgun (``J @ qd``
        crashed with a shape mismatch). The dataclass refuses to construct
        without both closures.
    x_desired : jax.Array, shape ``(k,)``
        Target Cartesian pose.
    kp, kd : float
        Cartesian PD gains.
    """

    pose_fn: Callable
    jacobian_fn: Callable
    x_desired: jnp.ndarray = dataclasses.field(default_factory=lambda: jnp.zeros(3))
    kp: float = 100.0
    kd: float = 20.0

    def residual(self, q, qd):
        x = self.pose_fn(q)
        J = self.jacobian_fn(q)
        return self.kp * (self.x_desired - x) - self.kd * J @ qd

    def jacobian(self, q, qd):
        return self.jacobian_fn(q)


@dataclasses.dataclass(frozen=True, kw_only=True)
class CoMTask(Task):
    """Drive the center of mass toward a 3D setpoint.

    Fields
    ------
    com_fn, jacobian_fn : Callable
        Required — same rationale as :class:`EndEffectorTask`'s closures.
    com_desired : jax.Array, shape ``(3,)``
        Target center-of-mass position.
    kp, kd : float
        CoM PD gains.
    """

    com_fn: Callable
    jacobian_fn: Callable
    com_desired: jnp.ndarray = dataclasses.field(default_factory=lambda: jnp.zeros(3))
    kp: float = 50.0
    kd: float = 10.0

    def residual(self, q, qd):
        c = self.com_fn(q)
        J = self.jacobian_fn(q)
        return self.kp * (self.com_desired - c) - self.kd * J @ qd

    def jacobian(self, q, qd):
        return self.jacobian_fn(q)


@dataclasses.dataclass(frozen=True, kw_only=True)
class ContactTask(Task):
    """Hold a contact point (foot) stationary — drive its acceleration to zero.

    A point in contact must not accelerate: ``J_c qdd + J̇_c qd = 0``. As a
    weighted task this is ``J_c qdd ≈ −J̇_c qd`` at high weight (contact is
    near-hard). The ``active`` gate in ``[0, 1]`` — driven by a
    :class:`ContactSchedule` — scales the task out of the QP for a swing foot
    (``active = 0`` contributes zero rows), so the same task list serves every
    phase of a gait.

    Fields
    ------
    jacobian_fn : Callable
        ``q -> J_c`` of shape ``(k, n_dof)`` (k = 3 for a point contact, 6 for a
        6-DOF contact). Required — no silent default.
    bias_fn : Callable | None
        ``(q, qd) -> J̇_c qd`` of shape ``(k,)``, the convective term. ``None``
        (default) uses the zero-acceleration simplification ``J_c qdd ≈ 0``.
    active : float
        Contact gate in ``[0, 1]``; ``0`` removes the task (swing foot). The
        gate folds into the effective weight by scaling both J and r by
        ``sqrt(active)`` (so ``active·weight`` is the stacked least-squares
        weight). Default ``1.0``.
    weight : float
        Soft-priority weight; high by default since contact ≈ a hard constraint.
    """

    jacobian_fn: Callable
    bias_fn: Callable | None = None
    active: float = 1.0
    weight: float = 1000.0

    def residual(self, q, qd):
        J = self.jacobian_fn(q)
        bias = (
            self.bias_fn(q, qd) if self.bias_fn is not None else jnp.zeros(J.shape[0])
        )
        return jnp.sqrt(self.active) * (-bias)

    def jacobian(self, q, qd):
        return jnp.sqrt(self.active) * self.jacobian_fn(q)


@dataclasses.dataclass(frozen=True, kw_only=True)
class MomentumTask(Task):
    """Regulate centroidal momentum toward a setpoint.

    Centroidal momentum ``h = A_G qd`` evolves as ``ḣ = A_G qdd + Ȧ_G qd``. This
    task drives ``ḣ`` so ``h`` approaches ``h_desired`` at rate ``kp``:

        ``A_G qdd ≈ kp (h_desired − A_G qd) − Ȧ_G qd``.

    With ``h_desired = 0`` it damps angular (and linear) momentum — the standard
    balance/locomotion objective (keep the body from spinning up).

    Fields
    ------
    momentum_jac_fn : Callable
        ``q -> A_G`` of shape ``(6, n_dof)`` (centroidal momentum matrix; rows
        are the angular-then-linear components by the producing primitive's
        convention). Required — no silent default.
    bias_fn : Callable | None
        ``(q, qd) -> Ȧ_G qd`` of shape ``(6,)``. ``None`` drops the convective
        term.
    h_desired : jax.Array, shape ``(6,)``
        Target centroidal momentum (default zero).
    kp : float
        Regulation rate (how fast ``h`` is driven to ``h_desired``).
    """

    momentum_jac_fn: Callable
    bias_fn: Callable | None = None
    h_desired: jnp.ndarray = dataclasses.field(default_factory=lambda: jnp.zeros(6))
    kp: float = 10.0

    def residual(self, q, qd):
        A = self.momentum_jac_fn(q)
        bias = (
            self.bias_fn(q, qd) if self.bias_fn is not None else jnp.zeros(A.shape[0])
        )
        return self.kp * (self.h_desired - A @ qd) - bias

    def jacobian(self, q, qd):
        return self.momentum_jac_fn(q)


@dataclasses.dataclass(frozen=True)
class JointAccelerationLimits:
    """Per-DOF symmetric box bounds on the QP solution ``qdd``.

    Encoded as ``lb / ub`` arrays applied directly in :func:`qp_solve`'s
    box-constraint path.
    """

    lb: jnp.ndarray  # shape (n_dof,)
    ub: jnp.ndarray  # shape (n_dof,)


class WBCSolver(eqx.Module):
    """Compose tasks into a hierarchical-soft-priority QP.

    Args
    ----
    tasks : tuple[Task, ...]
        Tasks in arbitrary order; weights set the soft priority.
    regularizer : float
        Ridge added to the Hessian so the QP stays well-conditioned even when
        the stacked Jacobian is rank-deficient.

    Methods
    -------
    solve(q, qd, n_dof, *, limits=None) -> jax.Array
        Stack tasks, build H/g, call ``qp_solve``, return ``qdd``.
    """

    tasks: tuple[Task, ...]
    regularizer: float = eqx.field(static=True, default=_REG)

    def solve(
        self,
        q,
        qd,
        n_dof: int,
        *,
        limits: JointAccelerationLimits | None = None,
    ):
        """Solve the WBC QP at the current ``(q, qd)``.

        Builds the stacked-Jacobian normal-equation H/g, optionally projects
        onto box bounds via :func:`qp_solve`'s ADMM path, returns the
        n_dof-length ``qdd`` minimizer.
        """
        Js = [t.jacobian(q, qd) * jnp.sqrt(t.weight) for t in self.tasks]
        rs = [t.residual(q, qd) * jnp.sqrt(t.weight) for t in self.tasks]
        J = jnp.concatenate(Js, axis=0)
        r = jnp.concatenate(rs, axis=0)
        H = J.T @ J + self.regularizer * jnp.eye(n_dof)
        g = -J.T @ r
        if limits is None:
            return qp_solve(H, g)
        return qp_solve(H, g, lb=limits.lb, ub=limits.ub)


__all__ = [
    "Task",
    "PostureTask",
    "EndEffectorTask",
    "CoMTask",
    "ContactTask",
    "MomentumTask",
    "JointAccelerationLimits",
    "WBCSolver",
]
