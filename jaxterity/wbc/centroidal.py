# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Centroidal-dynamics primitives for whole-body control (locomotion).

These are the MJX-backed quantities :class:`~jaxterity.wbc.ContactTask` and
:class:`~jaxterity.wbc.MomentumTask` consume as their Jacobian closures:

* :func:`contact_jacobian` — the ``(k, nv)`` Jacobian mapping generalized
  velocities to a contact point's spatial velocity (``k = 3`` translational,
  ``6`` with rotation). Built from ``mjx.jac``, which handles the floating-base
  velocity map correctly (an autodiff-through-the-quaternion-map shortcut does
  not).
* :func:`contact_jacobian_dot` — the convective term ``J̇_c`` (via
  ``mjx.jac_dot``); ``J̇_c @ qd`` is the contact-point acceleration bias.
* :func:`centroidal_momentum_matrix` — the ``(6, nv)`` matrix ``A_G`` with
  ``h = A_G qd`` the whole-body spatial momentum (angular about the CoM, then
  linear). Centroidal momentum is linear in ``qd``, so ``A_G`` is a single
  ``jax.jacobian`` of the momentum (assembled from ``mjx.subtree_vel``) — not a
  per-body construction.
* :func:`com_position` / :func:`centroidal_momentum` — the supporting CoM and
  ``h`` queries.

All functions take the MJX ``model`` (``robot.build_system()._mjx_model``) and a
configuration ``q``; they build and forward an ``mjx.Data`` internally so they
compose as ``q -> array`` closures. They are ``jax.jit`` / ``vmap`` clean.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from mujoco import mjx
from mujoco.mjx._src import (
    support as _mjx_support,
)  # jac_dot is not re-exported at mjx.*

#: Index of the floating-base root body. Body 0 is the MuJoCo world; body 1 is
#: the first real body, whose kinematic subtree is the whole robot — so its
#: ``subtree_linvel`` / ``subtree_angmom`` are the whole-body momentum.
_ROOT_BODY = 1


def _forward(model, q, qd=None):
    """Build an ``mjx.Data`` at ``(q, qd)`` and run forward kinematics+dynamics."""
    if qd is None:
        qd = jnp.zeros(model.nv)
    data = mjx.make_data(model).replace(qpos=q, qvel=qd)
    return mjx.forward(model, data)


def _subtree(data, name):
    """Read a (deprecated-on-Data) subtree field via the ``_impl`` accessor."""
    impl = getattr(data, "_impl", data)
    return getattr(impl, name)


def com_position(model, q):
    """Whole-body center-of-mass position in the world frame, shape ``(3,)``."""
    data = _forward(model, q)
    total_mass = jnp.sum(model.body_mass)
    return jnp.sum(model.body_mass[:, None] * data.xipos, axis=0) / total_mass


def contact_jacobian(model, q, body_id, *, point=None, rotational=False):
    """Spatial-velocity Jacobian of a contact point.

    Args
    ----
    model : mjx.Model
    q : jax.Array, shape ``(nq,)``
    body_id : int
        Body carrying the contact (e.g. a foot/ankle link).
    point : jax.Array, shape ``(3,)`` or None
        World contact point. ``None`` uses the body's CoM (``xipos[body_id]``).
    rotational : bool
        If ``True`` return a ``(6, nv)`` Jacobian (translational rows then
        rotational rows); otherwise ``(3, nv)`` translational only.

    Returns
    -------
    jax.Array, shape ``(3, nv)`` or ``(6, nv)``
        ``J_c`` such that the contact-point spatial velocity is ``J_c @ qd``.
    """
    data = _forward(model, q)
    pt = data.xipos[body_id] if point is None else point
    jacp, jacr = mjx.jac(model, data, pt, body_id)  # each (nv, 3)
    if rotational:
        return jnp.concatenate([jacp.T, jacr.T], axis=0)
    return jacp.T


def contact_jacobian_dot(model, q, qd, body_id, *, point=None, rotational=False):
    """Time derivative ``J̇_c`` of the contact Jacobian (the convective term).

    ``J̇_c @ qd`` is the contact-point acceleration bias, i.e. the acceleration
    a stationary contact would see at zero ``qdd`` — pass it as the ``bias_fn``
    of a :class:`~jaxterity.wbc.ContactTask` for an exact zero-acceleration
    contact constraint. Shapes mirror :func:`contact_jacobian`.
    """
    data = _forward(model, q, qd)
    pt = data.xipos[body_id] if point is None else point
    jacp_dot, jacr_dot = _mjx_support.jac_dot(model, data, pt, body_id)  # each (nv, 3)
    if rotational:
        return jnp.concatenate([jacp_dot.T, jacr_dot.T], axis=0)
    return jacp_dot.T


def centroidal_momentum(model, q, qd):
    """Whole-body spatial momentum ``h = [L_ang; p_lin]`` about the CoM, shape ``(6,)``.

    ``L_ang`` is the angular momentum about the whole-body CoM and ``p_lin`` the
    linear momentum, both in the world frame. Assembled from ``mjx.subtree_vel``
    (``subtree_angmom`` / ``subtree_linvel`` of the root subtree).
    """
    data = _forward(model, q, qd)
    data = mjx.subtree_vel(model, data)
    total_mass = jnp.sum(model.body_mass)
    p_lin = total_mass * _subtree(data, "subtree_linvel")[_ROOT_BODY]
    L_ang = _subtree(data, "subtree_angmom")[_ROOT_BODY]
    return jnp.concatenate([L_ang, p_lin])


def centroidal_momentum_matrix(model, q):
    """Centroidal momentum matrix ``A_G``, shape ``(6, nv)``, with ``h = A_G qd``.

    Computed as the Jacobian of :func:`centroidal_momentum` with respect to
    ``qd`` (momentum is linear in ``qd``, so the Jacobian is exact and
    independent of the evaluation point).
    """
    nv = int(model.nv)
    return jax.jacobian(lambda qd: centroidal_momentum(model, q, qd))(jnp.zeros(nv))


__all__ = [
    "com_position",
    "contact_jacobian",
    "contact_jacobian_dot",
    "centroidal_momentum",
    "centroidal_momentum_matrix",
]
