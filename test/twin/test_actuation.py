# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Twin actuation scale (``K_t_scale``) + the G1 unit twin.

The actuation leaf models motor-constant drift: delivered torque =
``K_t_scale · commanded``. Evidence layers: validation, exact equivalence with
pre-scaling the commanded torque (body *and* thermal), differentiability
(it is a fit target), attestation sensitivity/stability, and the G1 factory
satisfying the declared drift-parameter schema.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxterity.twin import Twin, g1_drift_parameters, unitree_g1_twin

pytest.importorskip("mujoco")

from jaxterity import zoo  # noqa: E402  (after the mujoco skip-guard)


@pytest.fixture(scope="module")
def cartpole_robot():
    return zoo.load("cartpole")


def _twin(robot, K_t_scale=None) -> Twin:
    n = 2  # cartpole: slider + pole
    return Twin(
        robot,
        serial="TEST-001",
        C_th=jnp.full(n, 15.0),
        R_th=jnp.full(n, 4.0),
        k_pwr=jnp.full(n, 6.0),
        K_t_scale=K_t_scale,
    )


def test_k_t_scale_shape_is_validated(cartpole_robot):
    with pytest.raises(ValueError, match="K_t_scale"):
        _twin(cartpole_robot, K_t_scale=jnp.ones(3))


def test_scaled_twin_equals_prescaled_command(cartpole_robot):
    """``K_t_scale = s`` ≡ commanding ``s · tau`` on a nominal twin.

    Exact (bit-level path identity): body trajectory, temperature, all of it —
    the delivered torque drives every physics channel.
    """
    n_steps, dt = 50, 0.005
    tau = jnp.tile(jnp.array([0.8, -0.3]), (n_steps, 1))
    scale = jnp.array([0.5, 1.25])
    q0, qd0, T0 = jnp.zeros(2), jnp.zeros(2), jnp.full(2, 298.15)

    scaled = _twin(cartpole_robot, K_t_scale=scale)
    nominal = _twin(cartpole_robot)
    out_scaled = scaled.rollout(q0, qd0, T0, tau, dt)
    out_prescaled = nominal.rollout(q0, qd0, T0, tau * scale[None, :], dt)

    np.testing.assert_array_equal(np.asarray(out_scaled.q), np.asarray(out_prescaled.q))
    np.testing.assert_array_equal(
        np.asarray(out_scaled.temperature), np.asarray(out_prescaled.temperature)
    )


def test_weaker_actuation_heats_less(cartpole_robot):
    """Half the torque constant ⇒ a quarter of the i²R heating."""
    n_steps, dt = 100, 0.005
    tau = jnp.tile(jnp.array([1.0, 1.0]), (n_steps, 1))
    q0, qd0, T0 = jnp.zeros(2), jnp.zeros(2), jnp.full(2, 298.15)

    hot = _twin(cartpole_robot).rollout(q0, qd0, T0, tau, dt)
    cool = _twin(cartpole_robot, K_t_scale=jnp.full(2, 0.5)).rollout(
        q0, qd0, T0, tau, dt
    )
    rise_hot = float(hot.temperature[-1, 0] - 298.15)
    rise_cool = float(cool.temperature[-1, 0] - 298.15)
    assert rise_hot > 0.0
    assert rise_cool == pytest.approx(0.25 * rise_hot, rel=1e-6)


def test_k_t_scale_is_differentiable(cartpole_robot):
    """The actuation leaf is a fit target: gradients flow (thermal path)."""
    n_steps, dt = 20, 0.005
    tau = jnp.tile(jnp.array([1.0, 0.5]), (n_steps, 1))
    q0, qd0, T0 = jnp.zeros(2), jnp.zeros(2), jnp.full(2, 298.15)
    base = _twin(cartpole_robot)

    def final_heat(scale):
        out = base.with_actuation(scale).rollout(q0, qd0, T0, tau, dt)
        return jnp.sum(out.temperature[-1])

    grad = jax.grad(final_heat)(jnp.ones(2))
    assert np.all(np.isfinite(np.asarray(grad)))
    assert np.all(np.asarray(grad) > 0.0)  # more torque constant ⇒ more heat


def test_actuation_moves_the_handle_and_appends_provenance(cartpole_robot):
    """Attestation: absent ≠ present; a re-fit is a new unit version."""
    nominal = _twin(cartpole_robot)
    unit_scaled = _twin(cartpole_robot, K_t_scale=jnp.ones(2))
    # Even an all-ones scale is a *declared* leaf — present ≠ absent.
    assert nominal.attestation_handle != unit_scaled.attestation_handle
    # Same declared parameters ⇒ same handle (determinism).
    assert (
        unit_scaled.attestation_handle
        == _twin(cartpole_robot, K_t_scale=jnp.ones(2)).attestation_handle
    )

    refit = unit_scaled.with_actuation(
        jnp.array([0.95, 1.05]),
        provenance_record=("sysid.actuation", "1.0", "deadbeef"),
    )
    assert refit.attestation_handle != unit_scaled.attestation_handle
    assert refit.provenance[-1] == ("sysid.actuation", "1.0", "deadbeef")
    assert "offbody.K_t_scale" in refit.identifiable_params()


# --- the G1 unit twin + declared drift schema -----------------------------


def test_unitree_g1_twin_satisfies_drift_schema():
    """Every declared drift parameter resolves into the G1 twin's leaves."""
    twin = unitree_g1_twin("G1-0047")
    assert twin.n_servos == 29
    assert not twin.has_battery
    # Servos are addressable by joint name, in DOF order.
    assert twin.servo_names[0] == "left_hip_pitch_joint"
    assert "right_knee_joint" in twin.servo_names

    params = twin.identifiable_params()
    schema = g1_drift_parameters()
    assert len(schema) == 14  # 12 leg frictions + K_t_scale + R_th
    for entry in schema:
        assert entry.path in params, entry.path
        assert entry.lower < entry.upper

    # Thermal defaults follow the actuator classes: a knee (139 N·m class)
    # carries more thermal mass than a wrist (5 N·m class).
    names = twin.servo_names
    knee = names.index("left_knee_joint")
    wrist = names.index("left_wrist_yaw_joint")
    assert float(twin.C_th[knee]) > float(twin.C_th[wrist])
    # Nominal actuation scale: ones, declared (present in the handle).
    np.testing.assert_array_equal(np.asarray(twin.K_t_scale), np.ones(29))


def test_unitree_g1_twin_distinct_serials_distinct_handles():
    a = unitree_g1_twin("G1-0001").attestation_handle
    b = unitree_g1_twin("G1-0002").attestation_handle
    assert a != b
