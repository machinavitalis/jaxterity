# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Crazyflie 2.0 reference robot — a 27 g nano-quadrotor (the zoo's first flyer).

Source: google-deepmind/mujoco_menagerie (``bitcraze_crazyflie_2/cf2.xml``).
Upstream model is (c) 2024 whoenig under the **MIT** license — see the vendored
``LICENSE`` in this directory and ``AGENTS/LIBRARIES.md``.

The vendored ``cf2.xml`` is a **mesh-free reduction**: a single rigid body with
its explicit ``<inertial>`` (mass + diagonal inertia); all geometry, sites,
actuators, and the upstream ``<freejoint>`` are stripped (Jaxterity drives
dynamics from inertials — same policy as the other zoo entries).

A quadrotor is a **single free body** driven by its rotors, so — unlike the
articulated arms/humanoid — it has no meaningful *fixed-base* form (a welded
body has zero DOF). :func:`load` therefore returns the **floating-base** body (a
``FLOATING`` joint, ADR-018; ``nq == 7``, ``nv == 6``).

Rotors enter as a **body wrench** — total thrust along body ``+z`` plus three
body moments — exactly the lumped control model the upstream MJCF uses (a
``body_thrust`` + ``x/y/z`` moment actuator at a site), in SI units. The wrench
is mapped to the free base's generalized force by :func:`control_to_generalized_force`
(thrust is rotated into the world frame by the current attitude; the moments are
already body-frame, which is the free joint's rotational coordinate). Individual
four-rotor control allocation (motor speeds → wrench, needing per-rotor thrust /
drag coefficients) is the documented next layer on top of this.
"""

from __future__ import annotations

import dataclasses
import pathlib

import jax.numpy as jnp

from ...robot import (
    Frame,
    FramedQuantity,
    Joint,
    JointLimits,
    JointType,
    Position,
    Robot,
    Transform,
    from_mjcf,
)

#: Vendored mesh-free MJCF. ``URDF_PATH`` aliases it so the generic zoo
#: machinery (golden-path lookup, which addresses every entry via ``URDF_PATH``)
#: works unchanged for this MJCF-sourced robot.
MJCF_PATH = pathlib.Path(__file__).with_name("cf2.xml")
URDF_PATH = MJCF_PATH
SOURCE = "https://github.com/google-deepmind/mujoco_menagerie (bitcraze_crazyflie_2)"
LICENSE = "MIT"

#: The single rigid body of the model (the welded root after import).
_ROOT_LINK = "cf2"

#: Total vehicle mass [kg] (from the vendored ``<inertial>``).
MASS = 0.027
#: Standard gravity [m/s²] used by the default MJX model.
_G = 9.81


def _with_floating_base(robot: Robot) -> Robot:
    """Attach a 6-DoF :class:`JointType.FLOATING` base (world → cf2).

    ``from_mjcf`` welds the upstream ``<freejoint>`` (ADR-018: importers don't
    yet round-trip free joints), so the free base is reconstructed here as the
    single supported ``FLOATING`` joint.
    """
    free = Joint(
        name="floating_base",
        joint_type=JointType.FLOATING,
        parent_link="world",
        child_link=_ROOT_LINK,
        axis=FramedQuantity(Position(jnp.array([0.0, 0.0, 1.0])), Frame("world")),
        limits=JointLimits(lower=-1.0, upper=1.0, effort=1.0, velocity=1.0),
        origin=Transform.identity(Frame("world")),
        damping=jnp.asarray(0.0),
        friction=jnp.asarray(0.0),
    )
    return dataclasses.replace(robot, joints=(free,) + robot.joints)


def load() -> Robot:
    """Load the (uncalibrated) Crazyflie 2.0 nano-quadrotor (floating base).

    Returns
    -------
    Robot
        A single rigid body on a ``FLOATING`` base (``nq == 7`` — position +
        unit quaternion; ``nv == 6``), built from the vendored mesh-free MJCF.

    Examples
    --------
    >>> from jaxterity import zoo
    >>> drone = zoo.load("crazyflie")                  # floating-base body
    >>> f = zoo.crazyflie.thrust_dynamics()            # f(state, u) -> dstate
    >>> u_hover = zoo.crazyflie.hover_control()         # [m·g, 0, 0, 0]
    >>> # f(state, u_hover) has ~zero vertical acceleration — see
    >>> # test/zoo/test_crazyflie.py::test_hover_thrust_cancels_gravity.
    """
    return _with_floating_base(from_mjcf(MJCF_PATH))


def hover_thrust() -> float:
    """Collective thrust [N] that exactly cancels gravity: ``MASS · g``."""
    return MASS * _G


def hover_control() -> jnp.ndarray:
    """The control wrench ``[thrust, Mx, My, Mz]`` for a level hover.

    Returns ``[MASS·g, 0, 0, 0]`` — collective thrust at weight, no moments.
    """
    return jnp.array([hover_thrust(), 0.0, 0.0, 0.0])


def control_to_generalized_force(state: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
    """Map a body wrench ``u = [thrust, Mx, My, Mz]`` to the base generalized force.

    The free joint's six velocity coordinates are ``[v_world(3), ω_body(3)]``, so
    its generalized force is ``[F_world(3), M_body(3)]``. Collective ``thrust``
    acts along body ``+z``; rotating it by the current attitude gives the world
    force ``F_world = thrust · ẑ_body``. The moments are already body-frame.

    Args
    ----
    state : Array
        ``concatenate([q, qd])``; ``q[3:7]`` is the body quaternion ``(w,x,y,z)``.
    u : Array
        ``[thrust (N), Mx, My, Mz (N·m)]`` in the body frame.

    Returns
    -------
    Array
        Length-6 generalized force for the floating base (``qfrc_applied`` rows
        of the free joint), ``[Fx, Fy, Fz, Mx, My, Mz]``.
    """
    w, x, y, z = state[3], state[4], state[5], state[6]
    # Third column of R(quat): the body +z axis expressed in the world frame.
    z_body_world = jnp.array(
        [
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        ]
    )
    thrust, moments = u[0], u[1:4]
    return jnp.concatenate([thrust * z_body_world, moments])


def thrust_dynamics():
    """Return a pure, ``jit``/``vmap``-able quadrotor dynamics ``f(state, u)``.

    Composes :meth:`Robot.functional_dynamics` (the fast MJX continuous-time
    dynamics) with :func:`control_to_generalized_force`, so the control input is
    the physical body wrench ``u = [thrust, Mx, My, Mz]`` rather than a raw
    generalized force. ``f(state, u) -> dstate`` with ``state =
    concatenate([q, qd])`` (lengths 7 + 6) and ``dstate = concatenate([q̇, q̈])``.

    Build once and reuse (the MJX model is compiled at call time)::

        f = zoo.crazyflie.thrust_dynamics()
        dstate = f(state, u)
    """
    robot = load()
    base = robot.functional_dynamics()

    def f(state: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return base(state, control_to_generalized_force(state, u))

    return f


__all__ = [
    "load",
    "thrust_dynamics",
    "control_to_generalized_force",
    "hover_control",
    "hover_thrust",
    "MASS",
    "MJCF_PATH",
    "URDF_PATH",
    "SOURCE",
    "LICENSE",
]
