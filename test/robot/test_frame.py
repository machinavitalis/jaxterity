# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Frame, FramedQuantity, and Transform tests.

Covers the load-bearing frame-safety invariant (mismatched-frame arithmetic
raises), the rigid-transform algebra (``T_AB @ T_BA == I``), and vmap-safety
(batched composition is bit-identical to the Python loop).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxterity.errors import FrameError, JaxterityError
from jaxterity.robot import (
    Frame,
    FramedQuantity,
    Position,
    Transform,
    Twist,
    Wrench,
)
from jaxterity.testing import tolerances as tol

# JAX defaults to float32 (x64 disabled); match the tolerance to that dtype.
# Float64-precision geometry is exercised under the dtype matrix.
_ATOL = tol.atol("joint_position", "dopri5", "float32")  # geometric tolerance


def _rz(theta: float) -> jnp.ndarray:
    """Rotation about +z by ``theta`` radians."""
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# --- Frame -----------------------------------------------------------------


def test_frame_equality_by_name():
    assert Frame("world") == Frame("world")
    assert Frame("world") != Frame("body")
    assert hash(Frame("world")) == hash(Frame("world"))


# --- FramedQuantity arithmetic --------------------------------------------


def test_same_frame_addition():
    a = FramedQuantity(Position(jnp.zeros(3)), Frame("world"))
    b = FramedQuantity(Position(jnp.ones(3)), Frame("world"))
    out = a + b
    assert out.frame == Frame("world")
    np.testing.assert_allclose(np.asarray(out.value.xyz), np.ones(3), atol=_ATOL)


def test_mismatched_frame_addition_raises_with_names():
    a = FramedQuantity(Position(jnp.zeros(3)), Frame("world"))
    b = FramedQuantity(Position(jnp.ones(3)), Frame("body"))
    with pytest.raises(FrameError, match="'world'.*'body'|'body'.*'world'"):
        _ = a + b


def test_mismatched_value_type_raises():
    a = FramedQuantity(Position(jnp.zeros(3)), Frame("world"))
    b = FramedQuantity(Twist(jnp.zeros(3), jnp.zeros(3)), Frame("world"))
    with pytest.raises(JaxterityError, match="value types must match"):
        _ = a + b


def test_wrench_subtraction_same_frame():
    a = FramedQuantity(Wrench(jnp.ones(3), 2 * jnp.ones(3)), Frame("foot"))
    b = FramedQuantity(Wrench(jnp.ones(3), jnp.ones(3)), Frame("foot"))
    out = a - b
    np.testing.assert_allclose(np.asarray(out.value.force), np.zeros(3), atol=_ATOL)
    np.testing.assert_allclose(np.asarray(out.value.torque), np.ones(3), atol=_ATOL)


# --- Transform -------------------------------------------------------------


def test_roundtrip_inverse_is_identity():
    """T_AB @ T_BA == identity to the geometric tolerance."""
    a, b = Frame("a"), Frame("b")
    t_ab = Transform(_rz(0.7), jnp.array([1.0, -2.0, 3.0]), source=b, target=a)
    rt = t_ab @ t_ab.inverse()
    np.testing.assert_allclose(np.asarray(rt.rotation), np.eye(3), atol=_ATOL)
    np.testing.assert_allclose(np.asarray(rt.translation), np.zeros(3), atol=_ATOL)
    assert rt.source == a and rt.target == a


def test_apply_moves_frame_and_transforms_point():
    a, b = Frame("a"), Frame("b")
    # pure +x translation by 1, no rotation: a point at origin of b lands at x=1 in a.
    t_ab = Transform(jnp.eye(3), jnp.array([1.0, 0.0, 0.0]), source=b, target=a)
    p_b = FramedQuantity(Position(jnp.zeros(3)), b)
    p_a = t_ab.apply(p_b)
    assert p_a.frame == a
    np.testing.assert_allclose(
        np.asarray(p_a.value.xyz), np.array([1.0, 0.0, 0.0]), atol=_ATOL
    )


def test_apply_wrong_frame_raises():
    a, b, c = Frame("a"), Frame("b"), Frame("c")
    t_ab = Transform(jnp.eye(3), jnp.zeros(3), source=b, target=a)
    with pytest.raises(FrameError, match="'b'"):
        t_ab.apply(FramedQuantity(Position(jnp.zeros(3)), c))


def test_compose_inner_frame_mismatch_raises():
    a, b, c, d = Frame("a"), Frame("b"), Frame("c"), Frame("d")
    t_ab = Transform(jnp.eye(3), jnp.zeros(3), source=b, target=a)
    t_cd = Transform(jnp.eye(3), jnp.zeros(3), source=d, target=c)
    with pytest.raises(FrameError, match="must match"):
        _ = t_ab @ t_cd


# --- vmap safety -----------------------------------------------------------


def test_vmap_compose_matches_loop_bit_exact():
    """Batched compose-with-inverse under vmap is byte-identical to the loop."""
    a, b = Frame("a"), Frame("b")

    def compose_inv(rotation, translation):
        t = Transform(rotation, translation, source=b, target=a)
        out = t @ t.inverse()
        return out.rotation, out.translation

    thetas = jnp.linspace(0.0, 1.5, 8)
    rots = jax.vmap(_rz)(thetas)
    trans = jnp.stack([jnp.array([i, -i, 2.0 * i]) for i in range(8)])

    v_rot, v_tr = jax.vmap(compose_inv)(rots, trans)
    loop = [compose_inv(rots[i], trans[i]) for i in range(8)]
    l_rot = jnp.stack([r for r, _ in loop])
    l_tr = jnp.stack([t for _, t in loop])

    assert np.asarray(v_rot).tobytes() == np.asarray(l_rot).tobytes()
    assert np.asarray(v_tr).tobytes() == np.asarray(l_tr).tobytes()


def test_grad_flows_through_framed_value():
    """jax.grad flows through a FramedQuantity's value (frame is static)."""

    def cost(xyz):
        p = FramedQuantity(Position(xyz), Frame("world"))
        q = FramedQuantity(Position(jnp.ones(3)), Frame("world"))
        diff = (p - q).value.xyz
        return jnp.sum(diff**2)

    g = jax.grad(cost)(jnp.zeros(3))
    np.testing.assert_allclose(np.asarray(g), -2.0 * np.ones(3), atol=_ATOL)
