# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Collision-geometry lowering.

Covers the new :class:`~jaxterity.robot.CollisionGeom` structure, its MJCF
emission, the ``Robot.ground_plane`` flag, and the end-to-end physics oracle:
a free body with a collision sphere dropped on the floor comes to rest at its
radius instead of falling forever (the geometry-free historical behaviour).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxterity.robot import (
    CollisionGeom,
    Frame,
    FramedQuantity,
    Joint,
    JointLimits,
    JointType,
    Link,
    Position,
    Robot,
    Transform,
)

pytest.importorskip("mujoco")


def _free_ball(radius: float = 0.1, *, ground_plane: bool = True) -> Robot:
    """A single free-floating sphere link above a floor plane."""
    ball = Link(
        name="ball",
        body_frame=Frame("ball"),
        mass=jnp.asarray(1.0),
        com=jnp.zeros(3),
        inertia=jnp.eye(3) * (2.0 / 5.0) * radius**2,
        collision=(CollisionGeom(kind="sphere", size=(radius,), name="ball_geom"),),
    )
    free = Joint(
        name="free_base",
        joint_type=JointType.FLOATING,
        parent_link="world",
        child_link="ball",
        axis=FramedQuantity(Position(jnp.array([0.0, 0.0, 1.0])), Frame("world")),
        limits=JointLimits(lower=-1.0, upper=1.0, effort=1.0, velocity=1.0),
        origin=Transform.identity(Frame("world")),
        damping=jnp.asarray(0.0),
        friction=jnp.asarray(0.0),
    )
    return Robot(
        links=(ball,), joints=(free,), name="free_ball", ground_plane=ground_plane
    )


def test_collision_geom_validates_kind_and_size():
    """Unknown kinds and wrong size arity raise with the expected wording."""
    with pytest.raises(ValueError, match="not supported"):
        CollisionGeom(kind="mesh", size=(0.1,))
    with pytest.raises(ValueError, match="size entr"):
        CollisionGeom(kind="box", size=(0.1, 0.2))
    # The documented arities build cleanly.
    CollisionGeom(kind="sphere", size=(0.1,))
    CollisionGeom(kind="capsule", size=(0.05, 0.2))
    CollisionGeom(kind="box", size=(0.1, 0.2, 0.3))


def test_lowering_emits_geoms_and_floor():
    """The MJCF carries the floor plane and one geom per CollisionGeom."""
    import mujoco

    robot = _free_ball()
    mjcf = robot.to_mjcf()
    m = mujoco.MjModel.from_xml_string(mjcf)
    names = {m.geom(i).name for i in range(m.ngeom)}
    assert names == {"floor", "ball_geom"}

    # Without the flag the model is geometry-free except the link geom.
    no_floor = _free_ball(ground_plane=False)
    m2 = mujoco.MjModel.from_xml_string(no_floor.to_mjcf())
    assert {m2.geom(i).name for i in range(m2.ngeom)} == {"ball_geom"}


def test_opaque_urdf_collision_handles_stay_unlowered():
    """A string collision handle (URDF mesh filename) emits no geom."""
    import dataclasses

    import mujoco

    robot = _free_ball(ground_plane=False)
    link = dataclasses.replace(robot.links[0], collision="mesh.stl")
    robot = dataclasses.replace(robot, links=(link,))
    m = mujoco.MjModel.from_xml_string(robot.to_mjcf())
    assert m.ngeom == 0


def test_ground_plane_changes_attestation_handle():
    """Adding geometry is a model change — the handle must move (ADR-008)."""
    assert (
        _free_ball(ground_plane=True).attestation_handle
        != _free_ball(ground_plane=False).attestation_handle
    )


def test_dropped_sphere_rests_at_its_radius():
    """End-to-end contact oracle: the sphere lands and rests at ``z = radius``.

    Free fall from 0.3 m onto the plane, integrated with fixed-step RK4 over
    ``functional_dynamics`` (the MJX fast path). Without contact the analytic
    ballistic answer at 0.6 s is z ≈ -1.47 m; with contact the centre must
    finish within a centimetre of the radius.
    """
    radius = 0.1
    robot = _free_ball(radius)
    f = robot.functional_dynamics()
    dt, n_steps = 0.002, 300
    tau = jnp.zeros(6)
    q0 = jnp.zeros(7).at[2].set(0.3).at[3].set(1.0)
    state0 = jnp.concatenate([q0, jnp.zeros(6)])

    def rk4(state, _):
        k1 = f(state, tau)
        k2 = f(state + 0.5 * dt * k1, tau)
        k3 = f(state + 0.5 * dt * k2, tau)
        k4 = f(state + dt * k3, tau)
        s = state + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return s, s[2]

    _, zs = jax.jit(lambda s: jax.lax.scan(rk4, s, None, length=n_steps))(state0)
    zs = np.asarray(zs)
    assert np.all(np.isfinite(zs))
    # Settled on the plane (soft-contact band, not ballistic).
    assert zs[-1] == pytest.approx(radius, abs=0.01), zs[-1]
    # Never tunnelled through the floor.
    assert zs.min() > radius * 0.5, zs.min()
