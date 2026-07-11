# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Reduced-order centroidal models for locomotion planning, balance, and deploy.

The whole-body MJX dynamics cannot be lowered to embedded C (the MJX constraint
solver emits ``lax.while_loop``; see ``jaxility`` KNOWN_GAPS). Locomotion control
therefore plans on a *reduced* model — closed-form, smooth, deployable — and the
whole-body controller (the WBC of :mod:`jaxterity.wbc`) tracks its plan. Two
classic reductions live here:

* :func:`srbd_dynamics` — the single-rigid-body / centroidal model. State is
  ``[c, ċ, L]`` (CoM position, CoM velocity, angular momentum about the CoM);
  the controls are the contact forces at the (parameterised) contact points.
  The dynamics are linear in the forces — exactly the structure a convex
  centroidal MPC wants — and use only smooth ops, so :mod:`jaxility` can lower
  them to C.
* :func:`lip_dynamics` + :func:`capture_point` — the linear inverted pendulum
  and its capture point, the principled basis for push recovery: holding the
  centre of pressure at the instantaneous capture point puts the CoM exactly on
  the LIP's stable manifold, so it coasts to rest (see
  :func:`capture_point_recovery`).

:func:`reduced_params` extracts a specific robot's mass and nominal CoM height so
the reduced model is grounded in the real platform (e.g. the Unitree G1).
"""

from __future__ import annotations

import jax.numpy as jnp
from mujoco import mjx

#: Standard gravity vector (world frame, z up).
GRAVITY = jnp.array([0.0, 0.0, -9.81])


def srbd_dynamics(state, forces, points, *, mass, gravity=GRAVITY):
    """Single-rigid-body centroidal dynamics ``ẋ = f(x, forces)``.

    Args
    ----
    state : jax.Array, shape ``(9,)``
        ``[c (3), ċ (3), L (3)]`` — CoM position, CoM velocity, angular momentum
        about the CoM (world frame).
    forces : jax.Array, shape ``(n_c, 3)``
        Contact forces (world frame), one per contact point.
    points : jax.Array, shape ``(n_c, 3)``
        Contact-point positions (world frame).
    mass : float
        Total robot mass.
    gravity : jax.Array, shape ``(3,)``

    Returns
    -------
    jax.Array, shape ``(9,)``
        ``[ċ, c̈, L̇]`` with ``c̈ = (Σ fᵢ)/m + g`` and
        ``L̇ = Σ (pᵢ − c) × fᵢ``. Linear in ``forces``.
    """
    c, cdot = state[0:3], state[3:6]
    f_tot = jnp.sum(forces, axis=0)
    cddot = f_tot / mass + gravity
    Ldot = jnp.sum(jnp.cross(points - c, forces), axis=0)
    return jnp.concatenate([cdot, cddot, Ldot])


def lip_omega(com_height, g=9.81):
    """LIP natural frequency ``ω = √(g / h)``."""
    return jnp.sqrt(g / com_height)


def capture_point(c_xy, cdot_xy, omega):
    """Instantaneous capture point ``ξ = c + ċ/ω`` (horizontal plane).

    Stepping the centre of pressure to ``ξ`` brings the LIP to rest (it is the
    point that zeroes the pendulum's orbital energy).
    """
    return c_xy + cdot_xy / omega


def lip_dynamics(state, p_cop, omega):
    """Linear inverted pendulum ``c̈ = ω² (c − p_cop)`` (horizontal plane).

    Args
    ----
    state : jax.Array, shape ``(4,)``
        ``[c_xy (2), ċ_xy (2)]``.
    p_cop : jax.Array, shape ``(2,)``
        Centre of pressure (the control).
    omega : float
        LIP frequency from :func:`lip_omega`.
    """
    c, cdot = state[0:2], state[2:4]
    cddot = omega**2 * (c - p_cop)
    return jnp.concatenate([cdot, cddot])


def capture_point_recovery(c0, cdot0, omega, *, dt=0.005, n_steps=600):
    """Simulate LIP push recovery: hold the CoP at the initial capture point.

    Returns the time series ``(t, c, cdot)``. Because the CoP is fixed at
    ``ξ₀ = c₀ + ċ₀/ω``, the initial state lies on the LIP's stable manifold, so
    the CoM coasts to rest at ``ξ₀`` — the textbook capture-point result.
    """
    p_cop = capture_point(c0, cdot0, omega)
    state = jnp.concatenate([c0, cdot0])

    def step(s, _):
        s = s + dt * lip_dynamics(s, p_cop, omega)
        return s, s

    import jax

    _, traj = jax.lax.scan(step, state, None, length=n_steps)
    t = jnp.arange(1, n_steps + 1) * dt
    return t, traj[:, 0:2], traj[:, 2:4]


def reduced_params(robot):
    """Extract a robot's reduced-model parameters: total mass and nominal CoM height.

    Returns
    -------
    dict
        ``{"mass": float, "com_height": float}`` — ``mass`` is the sum of body
        masses; ``com_height`` is the whole-body CoM height at the home posture.
    """
    model = robot.build_system()._mjx_model
    mass = float(jnp.sum(model.body_mass))
    q0 = jnp.asarray(model.qpos0, float)
    data = mjx.forward(model, mjx.make_data(model).replace(qpos=q0))
    com = jnp.sum(model.body_mass[:, None] * data.xipos, axis=0) / mass
    # CoM height above the lowest body (a ground proxy) — robust to where the
    # model's free-joint origin sits, unlike the absolute CoM z.
    ground_z = jnp.min(data.xipos[1:, 2])  # skip body 0 (world)
    com_height = float(com[2] - ground_z)
    return {"mass": mass, "com_height": com_height}


__all__ = [
    "GRAVITY",
    "srbd_dynamics",
    "lip_omega",
    "capture_point",
    "lip_dynamics",
    "capture_point_recovery",
    "reduced_params",
]
