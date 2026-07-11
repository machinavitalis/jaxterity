# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Joint and Link tests: PyTree round-trip, vmap, grad, parameter_spec."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jaxterity.robot import (
    Frame,
    FramedQuantity,
    Joint,
    JointLimits,
    JointType,
    Link,
    Position,
    Transform,
)
from jaxterity.testing import tolerances as tol

_ATOL = tol.atol("joint_position", "dopri5", "float32")


def _link(mass: float = 1.5) -> Link:
    return Link(
        name="link1",
        body_frame=Frame("link1"),
        mass=jnp.asarray(mass),
        com=jnp.array([0.0, 0.0, 0.1]),
        inertia=jnp.eye(3) * 0.01,
    )


def _joint(damping: float = 0.2, friction: float = 0.05) -> Joint:
    return Joint(
        name="j1",
        joint_type=JointType.REVOLUTE,
        parent_link="base",
        child_link="link1",
        axis=FramedQuantity(Position(jnp.array([0.0, 0.0, 1.0])), Frame("base")),
        limits=JointLimits(lower=-3.14, upper=3.14, effort=10.0, velocity=5.0),
        origin=Transform.identity(Frame("base")),
        damping=jnp.asarray(damping),
        friction=jnp.asarray(friction),
    )


# --- parameter_spec --------------------------------------------------------


def test_parameter_specs():
    assert tuple(_link().parameter_spec()) == ("mass", "com", "inertia")
    assert tuple(_joint().parameter_spec()) == ("damping", "friction", "armature")
    assert "mass" in _link().parameter_spec()
    assert "name" not in _link().parameter_spec()


# --- PyTree round-trip -----------------------------------------------------


def test_link_pytree_roundtrip():
    link = _link()
    leaves, treedef = jax.tree_util.tree_flatten(link)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt.name == link.name
    assert rebuilt.body_frame == link.body_frame
    np.testing.assert_array_equal(np.asarray(rebuilt.mass), np.asarray(link.mass))
    np.testing.assert_array_equal(np.asarray(rebuilt.inertia), np.asarray(link.inertia))


def test_joint_pytree_roundtrip_and_static_fields():
    joint = _joint()
    leaves, treedef = jax.tree_util.tree_flatten(joint)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    # static metadata survives
    assert rebuilt.joint_type is JointType.REVOLUTE
    assert rebuilt.parent_link == "base" and rebuilt.child_link == "link1"
    assert rebuilt.limits.effort == 10.0
    # only the array leaves are dynamic (no strings/enums among leaves)
    assert all(hasattr(leaf, "dtype") for leaf in leaves)


# --- vmap ------------------------------------------------------------------


def test_vmap_over_parameter_values():
    """vmap a derived quantity over a batch of joint damping values."""

    def stiffness_proxy(damping):
        j = _joint(damping=damping)
        return j.damping**2 + j.friction**2

    dampings = jnp.linspace(0.1, 1.0, 6)
    out = jax.vmap(stiffness_proxy)(dampings)
    assert out.shape == (6,)
    np.testing.assert_allclose(
        np.asarray(out), np.asarray(dampings) ** 2 + 0.05**2, atol=_ATOL
    )


# --- grad ------------------------------------------------------------------


def test_grad_through_link_mass():
    """jax.grad flows through a Link's identifiable mass to a scalar."""

    def cost(mass):
        link = _link(mass=mass)
        # a downstream scalar that depends on the inertial parameter
        return link.mass * jnp.sum(link.com**2)

    g = jax.grad(cost)(jnp.asarray(2.0))
    np.testing.assert_allclose(np.asarray(g), 0.1**2, atol=_ATOL)


def test_grad_through_joint_damping():
    def cost(damping):
        return _joint(damping=damping).damping ** 2

    g = jax.grad(cost)(jnp.asarray(0.3))
    np.testing.assert_allclose(np.asarray(g), 2 * 0.3, atol=_ATOL)
