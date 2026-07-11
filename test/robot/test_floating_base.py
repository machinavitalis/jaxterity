# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Floating-base (free joint) support: MJCF emission, dynamics, validation.

A single ``FLOATING`` joint (parent ``"world"``, child = the root link) makes the
model floating-base: the root body gets a ``<freejoint>`` (nq = nv + 1, the extra
component being the unit quaternion). The free part of the configuration
derivative is the quaternion rate ½·quat⊗ω, not the angular velocity.
"""

from __future__ import annotations

import jax.numpy as jnp
import jaxonomy
import numpy as np
import pytest
from jaxonomy import SimulatorOptions

from jaxterity.errors import JaxterityError
from jaxterity.robot import (
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

_RK4 = SimulatorOptions(
    math_backend="jax", ode_solver_method="rk4", max_minor_step_size=0.002
)


def _free_joint(child: str, name: str = "root") -> Joint:
    return Joint(
        name=name,
        joint_type=JointType.FLOATING,
        parent_link="world",
        child_link=child,
        axis=FramedQuantity(Position(jnp.array([0.0, 0.0, 1.0])), Frame("world")),
        limits=JointLimits(lower=-1.0, upper=1.0, effort=1.0, velocity=1.0),
        origin=Transform.identity(Frame("world")),
        damping=jnp.asarray(0.0),
        friction=jnp.asarray(0.0),
    )


def _body(name: str, inertia=(0.1, 0.2, 0.3)) -> Link:
    return Link(
        name=name,
        body_frame=Frame(name),
        mass=jnp.asarray(1.0),
        com=jnp.zeros(3),
        inertia=jnp.diag(jnp.asarray(inertia)),
    )


def _free_body() -> Robot:
    return Robot(
        links=(_body("body"),), joints=(_free_joint("body"),), name="free_body"
    )


def test_floating_base_emits_freejoint_and_has_quaternion_state():
    """A FLOATING joint becomes a ``<freejoint>``; the model gains a quaternion."""
    import mujoco

    robot = _free_body()
    mjcf = robot.to_mjcf()
    assert "<freejoint" in mjcf
    model = mujoco.MjModel.from_xml_string(mjcf)
    assert model.nq == 7 and model.nv == 6  # 3 + quaternion(4); 3 + 3


def test_floating_joint_has_no_identifiable_parameters():
    """A free base exposes no joint dynamics (only the body's inertials)."""
    params = _free_body().parameters()
    assert set(params) == {"body.mass", "body.com", "body.inertia"}
    assert not any(k.startswith("root.") for k in params)


def test_free_body_falls_under_gravity():
    """With gravity, a free body accelerates downward by ≈ ½gt²."""
    robot = Robot(
        links=(_body("body"),), joints=(_free_joint("body"),), name="fb", gravity=9.81
    )
    diagram, system = robot.to_diagram(return_system=True)  # q0 defaults to qpos0
    res = jaxonomy.simulate(diagram, diagram.create_context(), (0.0, 1.0), options=_RK4)
    x = np.asarray(res.context[system.system_id].continuous_state)
    assert np.all(np.isfinite(x))
    assert float(x[2]) == pytest.approx(-0.5 * 9.81 * 1.0**2, rel=2e-3)  # z(1s)
    assert float(np.linalg.norm(x[3:7])) == pytest.approx(1.0, abs=1e-6)  # unit quat


def test_floating_base_with_articulation():
    """A free base plus an articulated hinge (mixed nq≠nv map) simulates."""
    base = _body("base")
    arm = Link(
        name="arm",
        body_frame=Frame("arm"),
        mass=jnp.asarray(0.5),
        com=jnp.array([0.0, 0.0, -0.3]),
        inertia=jnp.eye(3) * 0.01,
    )
    elbow = Joint(
        name="elbow",
        joint_type=JointType.CONTINUOUS,
        parent_link="base",
        child_link="arm",
        axis=FramedQuantity(Position(jnp.array([0.0, 1.0, 0.0])), Frame("base")),
        limits=JointLimits(lower=-3.0, upper=3.0, effort=10.0, velocity=10.0),
        origin=Transform(
            jnp.eye(3),
            jnp.array([0.0, 0.0, -0.2]),
            source=Frame("base"),
            target=Frame("arm"),
        ),
        damping=jnp.asarray(0.0),
        friction=jnp.asarray(0.0),
    )
    robot = Robot(
        links=(base, arm), joints=(_free_joint("base"), elbow), name="floatarm"
    )

    import mujoco

    model = mujoco.MjModel.from_xml_string(robot.to_mjcf())
    assert model.nq == 8 and model.nv == 7  # free(7/6) + hinge(1/1)
    # only the hinge is identifiable; the free base contributes nothing
    assert "elbow.damping" in robot.parameters()
    assert not any(k.startswith("root.") for k in robot.parameters())

    diagram, system = robot.to_diagram(
        qd0=[0, 0, 0, 0.5, 0, 0, 0.3], return_system=True
    )
    res = jaxonomy.simulate(diagram, diagram.create_context(), (0.0, 0.5), options=_RK4)
    x = np.asarray(res.context[system.system_id].continuous_state)
    assert np.all(np.isfinite(x))
    assert float(np.linalg.norm(x[3:7])) == pytest.approx(1.0, abs=1e-5)  # unit quat
    assert abs(float(x[7])) > 1e-3  # the elbow moved


def test_multiple_floating_bases_rejected():
    """At most one floating base is supported."""
    robot = Robot(
        links=(_body("a"), _body("b")),
        joints=(_free_joint("a", "ra"), _free_joint("b", "rb")),
        name="two_free",
    )
    with pytest.raises(JaxterityError, match="one floating base"):
        robot.to_mjcf()
