# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Predictive safety monitor — OK/derate/veto, monotonicity, constraints.

Most checks feed the monitor synthetic `TwinRollout` predictions so the decision
logic is exercised precisely and fast (the monitor's compute is pure NumPy); one
integration test wires the co-processor → monitor end to end on the real twin.
"""

from __future__ import annotations

import time

import jax.numpy as jnp
import pytest

from jaxterity.selfcheck import (
    CoProcessor,
    SafetyAction,
    SafetyEnvelope,
    SafetyMonitor,
)
from jaxterity.twin import Twin, TwinRollout, so101_twin

pytest.importorskip("mujoco")


@pytest.fixture(scope="module")
def twin() -> Twin:
    return so101_twin("SO101-0001")


@pytest.fixture(scope="module")
def dims(twin):
    return twin.robot.build_system()._nq, twin.n_servos


def _flat_prediction(twin, dims, *, temp_k=None, horizon=10, bus_v=None):
    """A synthetic prediction: flat temperature, at-rest joints, optional bus."""
    nq, nv = dims
    temp_k = twin.T_amb if temp_k is None else temp_k
    bus = None if bus_v is None else jnp.full(horizon, bus_v)
    return TwinRollout(
        q=jnp.zeros((horizon + 1, nq)),
        qd=jnp.zeros((horizon + 1, nv)),
        temperature=jnp.full((horizon + 1, nv), temp_k),
        bus_voltage=bus,
    )


# --- OK / Veto on the sustained-hold thermal check -------------------------


def test_safe_command_is_ok(twin, dims):
    _, nv = dims
    mon = SafetyMonitor(twin, SafetyEnvelope(temp_ceiling_k=twin.T_amb + 35.0))
    pred = _flat_prediction(twin, dims)
    d = mon.evaluate(pred, action=jnp.full(nv, 0.3))  # low hold → cool steady state
    assert d.action is SafetyAction.OK
    assert d.factor == 1.0
    assert d.time_to_breach_s is None


def test_sustained_hold_vetoes_with_time_to_breach(twin, dims):
    _, nv = dims
    mon = SafetyMonitor(twin, SafetyEnvelope(temp_ceiling_k=twin.T_amb + 35.0))
    pred = _flat_prediction(twin, dims)
    d = mon.evaluate(pred, action=jnp.full(nv, 2.0))  # heavy hold → overheats
    assert d.action is SafetyAction.VETO
    assert d.binding_constraint == "servo_temperature"
    assert d.factor == 0.0
    assert d.time_to_breach_s is not None and d.time_to_breach_s > 0.0


def test_derate_between_ok_and_veto(twin, dims):
    _, nv = dims
    mon = SafetyMonitor(twin, SafetyEnvelope(temp_ceiling_k=twin.T_amb + 35.0))
    pred = _flat_prediction(twin, dims)
    # tuned so steady state lands inside the derate band (just below the ceiling).
    d = mon.evaluate(pred, action=jnp.full(nv, 1.2))
    assert d.action is SafetyAction.DERATE
    assert 0.0 < d.factor < 1.0


# --- monotonicity + boundedness -------------------------------------------


def test_factor_is_monotone_and_bounded(twin, dims):
    _, nv = dims
    mon = SafetyMonitor(twin, SafetyEnvelope(temp_ceiling_k=twin.T_amb + 35.0))
    pred = _flat_prediction(twin, dims)
    factors = [
        mon.evaluate(pred, action=jnp.full(nv, t)).factor
        for t in (0.3, 0.8, 1.2, 1.6, 2.0)
    ]
    assert all(0.0 <= f <= 1.0 for f in factors)
    assert all(factors[i] >= factors[i + 1] - 1e-9 for i in range(len(factors) - 1))


# --- bus-voltage floor (LeKiwi) -------------------------------------------


def test_bus_voltage_floor_binds(twin, dims):
    _, nv = dims
    lekiwi = Twin(
        twin.robot,
        serial="LK-0001",
        C_th=twin.C_th,
        R_th=twin.R_th,
        k_pwr=twin.k_pwr,
        V_oc=12.0,
        R_int=0.4,
        k_i=tuple([1.5] * nv),
    )
    env = SafetyEnvelope(
        temp_ceiling_k=lekiwi.T_amb + 200.0,  # thermal never binds here
        bus_floor_v=10.0,
        bus_band_v=1.0,
    )
    mon = SafetyMonitor(lekiwi, env)
    pred = _flat_prediction(lekiwi, dims, bus_v=9.5)  # below the floor
    d = mon.evaluate(pred, action=jnp.zeros(nv))
    assert d.binding_constraint == "bus_voltage"
    assert d.action is SafetyAction.VETO


# --- joint-limit check is opt-in ------------------------------------------


def test_joint_limit_is_opt_in(twin, dims):
    _, nv = dims
    pred = _flat_prediction(twin, dims)
    off = SafetyMonitor(twin, SafetyEnvelope(temp_ceiling_k=twin.T_amb + 200.0))
    assert "joint_limit" not in off.evaluate(pred, action=jnp.zeros(nv)).factors
    on = SafetyMonitor(
        twin, SafetyEnvelope(temp_ceiling_k=twin.T_amb + 200.0, joint_margin_rad=0.1)
    )
    assert "joint_limit" in on.evaluate(pred, action=jnp.zeros(nv)).factors


# --- decision fits one co-processor tick budget ----------------------------


def test_decision_is_cheap(twin, dims):
    _, nv = dims
    mon = SafetyMonitor(twin, SafetyEnvelope(temp_ceiling_k=twin.T_amb + 35.0))
    pred = _flat_prediction(twin, dims)
    action = jnp.full(nv, 1.0)
    mon.evaluate(pred, action=action)  # warm any laziness
    t0 = time.perf_counter()
    for _ in range(20):
        mon.evaluate(pred, action=action)
    per_call = (time.perf_counter() - t0) / 20
    # pure-NumPy decision — comfortably inside even a 200 Hz (5 ms) tick budget.
    assert per_call < 5e-3, f"decision took {per_call * 1e3:.2f} ms/call"


# --- integration: co-processor → monitor on the real twin ------------------


def test_coprocessor_to_monitor_end_to_end(twin, dims):
    nq, nv = dims
    cp = CoProcessor(twin, rate_hz=50.0, horizon_steps=10, dt=0.02)
    mon = SafetyMonitor(twin, SafetyEnvelope(temp_ceiling_k=twin.T_amb + 35.0))
    q, qd, T = jnp.zeros(nq), jnp.zeros(nv), jnp.full(nv, twin.T_amb)

    safe = mon.evaluate(
        cp.predict(q, qd, T, jnp.full(nv, 0.3)), action=jnp.full(nv, 0.3)
    )
    hot = mon.evaluate(
        cp.predict(q, qd, T, jnp.full(nv, 2.0)), action=jnp.full(nv, 2.0)
    )
    assert safe.action is SafetyAction.OK
    assert hot.action is SafetyAction.VETO
    assert hot.binding_constraint == "servo_temperature"
