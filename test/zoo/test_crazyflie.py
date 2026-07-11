# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Crazyflie nano-quadrotor — floating base + thrust/moment wrench.

The shared zoo harness (``test_zoo.py``) covers the free-fall golden. This file
covers what makes it a *quadrotor*: the body thrust+moment wrench model. The
control input is the physical wrench ``u = [thrust, Mx, My, Mz]``; the headline
checks are that hover thrust cancels gravity and that tilting the body tilts the
thrust vector (the attitude → world-force coupling), with the quaternion config
map (ADR-018) staying consistent on a free body.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxterity import zoo
from jaxterity.robot import JointType

pytest.importorskip("mujoco")

# state = [x y z, quat(w,x,y,z), v(3), ω(3)]; level hover at z = 0.1, at rest.
_LEVEL = jnp.array([0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_topology_is_a_single_floating_body():
    """One free body: a FLOATING base over the single cf2 link (nq 7, nv 6)."""
    robot = zoo.load("crazyflie")
    assert len(robot.links) == 1
    assert [j.joint_type for j in robot.joints] == [JointType.FLOATING]
    _, sysm = robot.to_diagram(
        q0=[0, 0, 0, 1, 0, 0, 0], qd0=[0] * 6, return_system=True
    )
    assert (sysm._nq, sysm._nv) == (7, 6)


def test_hover_thrust_cancels_gravity():
    """At ``u = [m·g, 0, 0, 0]`` the vehicle's acceleration is ~zero."""
    f = zoo.crazyflie.thrust_dynamics()
    dstate = f(_LEVEL, zoo.crazyflie.hover_control())
    accel = np.asarray(dstate[7:13])  # [a_world(3), α_body(3)]
    assert np.allclose(accel, 0.0, atol=1e-5), accel


def test_zero_thrust_is_free_fall():
    """No thrust ⇒ pure ½·g·t² drop, no spurious lateral/angular accel."""
    f = zoo.crazyflie.thrust_dynamics()
    accel = np.asarray(f(_LEVEL, jnp.zeros(4))[7:13])
    assert accel[2] == pytest.approx(-9.81, rel=1e-3)
    assert np.allclose(np.delete(accel, 2), 0.0, atol=1e-6)


def test_thrust_tilts_with_attitude():
    """Thrust acts along body +z: pitching the body produces lateral force.

    With the body pitched about +y by θ, hover-magnitude collective thrust has a
    world-x component ``m·g·sin θ`` and a reduced world-z component ``m·g·cos θ``
    — the coupling a position controller relies on. Verifies the quaternion →
    world-frame rotation in ``control_to_generalized_force``.
    """
    theta = 0.3  # rad, pitch about +y
    quat = jnp.array([jnp.cos(theta / 2), 0.0, jnp.sin(theta / 2), 0.0])
    state = _LEVEL.at[3:7].set(quat)
    f = zoo.crazyflie.thrust_dynamics()
    accel = np.asarray(f(state, zoo.crazyflie.hover_control())[7:13])
    # a = (thrust/m)·ẑ_body − g·ẑ_world; thrust/m = g, so a = g·(ẑ_body − ẑ_world).
    g = 9.81
    assert accel[0] == pytest.approx(g * np.sin(theta), rel=2e-3)  # forward
    assert accel[2] == pytest.approx(g * (np.cos(theta) - 1.0), rel=2e-3)  # sinks


def test_thrust_dynamics_is_jit_and_vmap_safe():
    """The control env is a pure function: jit + vmap over a batch of states."""
    f = zoo.crazyflie.thrust_dynamics()
    states = jnp.broadcast_to(_LEVEL, (8, 13))
    us = jnp.broadcast_to(zoo.crazyflie.hover_control(), (8, 4))
    out = jax.jit(jax.vmap(f))(states, us)
    assert out.shape == (8, 13)
    assert np.all(np.isfinite(np.asarray(out)))


def test_hover_holds_altitude_in_closed_loop():
    """Integrated over 0.5 s, hover thrust keeps the vehicle within ~1 mm of z0.

    End-to-end check that the wrench model + quaternion integration produce a
    genuine equilibrium, not just an instantaneous zero acceleration.
    """
    f = zoo.crazyflie.thrust_dynamics()
    u = zoo.crazyflie.hover_control()
    dt, n = 0.005, 100
    state = _LEVEL

    def rk4(s):
        k1 = f(s, u)
        k2 = f(s + 0.5 * dt * k1, u)
        k3 = f(s + 0.5 * dt * k2, u)
        k4 = f(s + dt * k3, u)
        return s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    step = jax.jit(rk4)
    for _ in range(n):
        state = step(state)
    z = float(state[2])
    assert abs(z - 0.1) < 1e-3, z
    assert abs(float(jnp.linalg.norm(state[3:7])) - 1.0) < 1e-6
