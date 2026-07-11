# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Reduced-order centroidal models (SRBD + LIP capture-point push recovery)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jaxterity import zoo
from jaxterity.locomotion.reduced import (
    GRAVITY,
    capture_point,
    capture_point_recovery,
    lip_omega,
    reduced_params,
    srbd_dynamics,
)


def test_srbd_hover_equilibrium():
    """Forces summing to m·g through the CoM give zero CoM and angular accel."""
    mass = 40.0
    c = jnp.array([0.0, 0.0, 0.9])
    state = jnp.concatenate([c, jnp.zeros(3), jnp.zeros(3)])
    # two feet, each carrying half the weight, directly below the CoM.
    points = jnp.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
    fz = mass * 9.81 / 2
    forces = jnp.array([[0.0, 0.0, fz], [0.0, 0.0, fz]])
    dstate = srbd_dynamics(state, forces, points, mass=mass)
    cddot, Ldot = dstate[3:6], dstate[6:9]
    np.testing.assert_allclose(np.asarray(cddot), np.zeros(3), atol=1e-9)
    # symmetric vertical forces about the CoM x → no net moment.
    np.testing.assert_allclose(np.asarray(Ldot), np.zeros(3), atol=1e-9)


def test_srbd_linear_in_forces():
    """Centroidal dynamics are linear in the contact forces (convex MPC structure)."""
    mass = 30.0
    state = jnp.array([0.0, 0.0, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
    points = jnp.array([[0.1, 0.05, 0.0], [-0.1, -0.05, 0.0]])

    def dyn(forces):
        return srbd_dynamics(state, forces, points, mass=mass)

    f0 = jnp.zeros((2, 3))
    J = jax.jacobian(dyn)(f0)  # (9, 2, 3)
    # linearity: dyn(f) == dyn(0) + J·f for an arbitrary f
    rng = np.random.default_rng(0)
    f = jnp.asarray(rng.normal(size=(2, 3)))
    lin = dyn(f0) + jnp.tensordot(J, f, axes=([1, 2], [0, 1]))
    np.testing.assert_allclose(np.asarray(dyn(f)), np.asarray(lin), atol=1e-9)


def test_srbd_angular_momentum_from_offset_force():
    """A horizontal force offset from the CoM produces the expected moment."""
    mass = 20.0
    c = jnp.array([0.0, 0.0, 1.0])
    state = jnp.concatenate([c, jnp.zeros(3), jnp.zeros(3)])
    points = jnp.array([[0.0, 0.0, 0.0]])  # contact 1 m below the CoM
    forces = jnp.array([[10.0, 0.0, 0.0]])  # +x force at the foot
    dstate = srbd_dynamics(state, forces, points, mass=mass)
    # L̇ = (p − c) × f = (0,0,−1) × (10,0,0) = (0,−10,0)
    np.testing.assert_allclose(np.asarray(dstate[6:9]), [0.0, -10.0, 0.0], atol=1e-9)


def test_capture_point_recovery_brings_com_to_rest():
    """Holding the CoP at the initial capture point recovers a pushed LIP:
    the CoM coasts to rest at the capture point (stable-manifold result)."""
    omega = lip_omega(0.9)
    c0 = jnp.array([0.0, 0.0])
    cdot0 = jnp.array([0.4, -0.2])  # a sideways/forward push
    xi0 = capture_point(c0, cdot0, omega)
    t, c, cdot = capture_point_recovery(c0, cdot0, omega, dt=0.004, n_steps=900)
    # velocity decays to ~0 and the CoM settles at the capture point.
    assert float(jnp.linalg.norm(cdot[-1])) < 1e-2
    np.testing.assert_allclose(np.asarray(c[-1]), np.asarray(xi0), atol=1e-2)
    # monotone-ish energy decay: final speed well below the push.
    assert float(jnp.linalg.norm(cdot[-1])) < 0.05 * float(jnp.linalg.norm(cdot0))


def test_reduced_params_g1():
    """G1 reduced params: physically plausible humanoid mass + standing CoM height."""
    robot = zoo.unitree_g1.load(floating_base=True)
    p = reduced_params(robot)
    assert 20.0 < p["mass"] < 80.0, p["mass"]  # G1-class humanoid mass
    assert 0.3 < p["com_height"] < 1.2, p["com_height"]  # standing CoM height
    assert GRAVITY[2] < 0  # sanity: gravity points down
