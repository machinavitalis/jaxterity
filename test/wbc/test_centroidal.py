# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Centroidal-dynamics primitives (contact Jacobian, centroidal momentum matrix)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from mujoco import mjx

from jaxterity import zoo
from jaxterity.wbc.centroidal import (
    centroidal_momentum,
    centroidal_momentum_matrix,
    com_position,
    contact_jacobian,
    contact_jacobian_dot,
)

_FOOT = "left_ankle_roll_link"


def _g1(floating_base):
    robot = zoo.unitree_g1.load(floating_base=floating_base)
    model = robot.build_system()._mjx_model
    foot_id = int(mujoco.MjModel.from_xml_string(robot.to_mjcf()).body(_FOOT).id)
    return model, foot_id


def test_contact_jacobian_shapes():
    model, foot_id = _g1(True)
    q = jnp.asarray(np.array(model.qpos0), float)
    J = contact_jacobian(model, q, foot_id)
    J6 = contact_jacobian(model, q, foot_id, rotational=True)
    assert J.shape == (3, model.nv)
    assert J6.shape == (6, model.nv)


def test_contact_jacobian_matches_autodiff_fixed_base():
    """On the fixed-base robot (no quaternion map, q̇ = qvel), mjx.jac equals
    the autodiff position Jacobian — validating our use of ``mjx.jac``."""
    model, foot_id = _g1(False)
    rng = np.random.default_rng(0)
    q = jnp.asarray(np.array(model.qpos0) + 0.1 * rng.normal(size=model.nq), float)

    def foot_pos(qq):
        d = mjx.forward(model, mjx.make_data(model).replace(qpos=qq))
        return d.xipos[foot_id]

    J_ad = jax.jacobian(foot_pos)(q)
    J_cj = contact_jacobian(model, q, foot_id)
    np.testing.assert_allclose(np.asarray(J_cj), np.asarray(J_ad), atol=1e-5)


def test_contact_jacobian_dot_shape():
    model, foot_id = _g1(True)
    q = jnp.asarray(np.array(model.qpos0), float)
    qd = jnp.zeros(model.nv)
    Jdot = contact_jacobian_dot(model, q, qd, foot_id)
    assert Jdot.shape == (3, model.nv)


def test_centroidal_momentum_matrix_shape_and_consistency():
    """A_G is (6, nv) and reproduces h = A_G qd exactly (momentum is linear in qd)."""
    model, _ = _g1(True)
    rng = np.random.default_rng(1)
    q = jnp.asarray(np.array(model.qpos0), float)
    qd = jnp.asarray(0.3 * rng.normal(size=model.nv), float)
    A_G = centroidal_momentum_matrix(model, q)
    assert A_G.shape == (6, model.nv)
    h = centroidal_momentum(model, q, qd)
    np.testing.assert_allclose(np.asarray(A_G @ qd), np.asarray(h), atol=1e-9)


def test_centroidal_linear_part_is_total_mass_times_com_velocity():
    """A_G's linear rows = total_mass · (CoM Jacobian) on the fixed-base robot."""
    model, _ = _g1(False)
    rng = np.random.default_rng(2)
    q = jnp.asarray(np.array(model.qpos0) + 0.05 * rng.normal(size=model.nq), float)
    qd = jnp.asarray(0.4 * rng.normal(size=model.nv), float)
    A_G = centroidal_momentum_matrix(model, q)
    total_mass = float(jnp.sum(model.body_mass))
    J_com = jax.jacobian(lambda qq: com_position(model, qq))(q)  # (3, nv)
    np.testing.assert_allclose(
        np.asarray(A_G[3:] @ qd), total_mass * np.asarray(J_com @ qd), atol=1e-5
    )


def test_com_position_matches_mass_weighted_bodies():
    model, _ = _g1(True)
    q = jnp.asarray(np.array(model.qpos0), float)
    c = com_position(model, q)
    d = mjx.forward(model, mjx.make_data(model).replace(qpos=q))
    ref = jnp.sum(model.body_mass[:, None] * d.xipos, axis=0) / jnp.sum(model.body_mass)
    np.testing.assert_allclose(np.asarray(c), np.asarray(ref), atol=1e-9)
    assert c.shape == (3,)


@pytest.mark.slow
def test_centroidal_momentum_conserved_in_free_fall():
    """Gold-standard physical check: with gravity off and no contacts/actuation,
    the whole-body spatial momentum h = A_G(q) qd is conserved along an MJX
    rollout. This validates the *full* A_G (angular momentum included), not just
    its self-consistency."""
    model, _ = _g1(True)
    # zero gravity so linear momentum is conserved; lift the base so no contacts.
    model = model.tree_replace({"opt.gravity": jnp.zeros(3)})
    q0 = np.array(model.qpos0, dtype=float)
    q0[2] += 5.0  # lift the floating base well clear of any ground plane
    rng = np.random.default_rng(3)
    qd0 = jnp.asarray(0.5 * rng.normal(size=model.nv), float)
    data = mjx.make_data(model).replace(qpos=jnp.asarray(q0), qvel=qd0)
    data = mjx.forward(model, data)

    step = jax.jit(lambda d: mjx.step(model, d))
    h0 = centroidal_momentum_matrix(model, data.qpos) @ data.qvel
    max_drift = 0.0
    for _ in range(25):
        data = step(data)
        h = centroidal_momentum_matrix(model, data.qpos) @ data.qvel
        max_drift = max(max_drift, float(jnp.max(jnp.abs(h - h0))))
    # small integrator drift over 25 steps; conservation holds to a tight band.
    rel = max_drift / float(jnp.max(jnp.abs(h0)))
    assert rel < 5e-2, f"centroidal momentum drifted {rel:.3f} (not conserved)"
