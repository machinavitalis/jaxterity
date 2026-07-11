# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Reduced on-device co-processor: NumPy↔JAX parity + on-Pi timing.

The Raspberry Pi that owns the Feetech bus has NumPy only — no JAX/MJX — so the
full MJX-body :class:`jaxterity.selfcheck.CoProcessor` cannot run there. The
honest lowered path (`scripts/lekiwi_reduced_coprocessor.py`) forward-integrates
only the safety-relevant off-body models (per-servo thermal ODE + battery sag) in
pure NumPy. These tests check (1) that NumPy forecast is faithful to the JAX
`jaxterity.physics` models, and (2) that the co-processor met its deadlines on the
real Pi (committed timing reports).
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import jax.numpy as jnp
import numpy as np
import pytest

from jaxterity.physics.battery import bus_voltage as jax_bus_voltage
from jaxterity.physics.thermal import thermal_response

_SCRIPT = (
    pathlib.Path(__file__).parent.parent.parent
    / "scripts"
    / "lekiwi_reduced_coprocessor.py"
)
_DATA = pathlib.Path(__file__).with_name("data")


def _load_script():
    spec = importlib.util.spec_from_file_location("lekiwi_reduced_coprocessor", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cp():
    return _load_script()


def test_thermal_forecast_matches_closed_form(cp):
    # The NumPy RK4 thermal forecast tracks the JAX closed-form solution.
    C_th, R_th, k_pwr, T_amb = 15.0, 4.0, 6.0, 298.15
    tau = np.array([0.8, 0.5, 1.2, 0.3, 0.6, 0.4])
    T0 = np.full(6, T_amb + 5.0)
    dt, horizon = 0.05, 40
    T_traj, _ = cp.forecast(
        T0, tau, horizon=horizon, dt=dt, C_th=C_th, R_th=R_th, k_pwr=k_pwr, T_amb=T_amb
    )
    for h in (1, 10, 39):
        ref = np.asarray(
            thermal_response(
                (h + 1) * dt, tau, T0=T0, T_amb=T_amb, C_th=C_th, R_th=R_th, k_pwr=k_pwr
            )
        )
        np.testing.assert_allclose(T_traj[h], ref, rtol=1e-4, atol=1e-4)


def test_bus_voltage_matches_jax_model(cp):
    tau = np.array([0.8, 0.5, 1.2, 0.3, 0.6, 0.4])
    V_oc, R_int, k_i = 12.166, 0.0653, 1.5
    mine = cp.bus_voltage(tau, V_oc=V_oc, R_int=R_int, k_i=k_i)
    ref = float(
        jax_bus_voltage(jnp.asarray(tau), V_oc=V_oc, R_int=R_int, k_i=jnp.full(6, k_i))
    )
    assert mine == pytest.approx(ref, abs=1e-3)


def test_forecast_is_cheap_enough_for_rate():
    # The forecast itself (the co-processor's per-tick compute) is sub-millisecond
    # for a 10-step horizon — comfortably inside a 100+ Hz budget (validated on the
    # real Pi below; this is a host smoke-check of the compute cost).
    cp = _load_script()
    tau = np.full(6, 0.8)
    T0 = np.full(6, 303.0)
    import time

    cp.forecast(T0, tau, horizon=10, dt=0.05)  # warm up
    t0 = time.perf_counter()
    for _ in range(200):
        cp.forecast(T0, tau, horizon=10, dt=0.05)
    per_tick_ms = (time.perf_counter() - t0) / 200 * 1e3
    assert per_tick_ms < 5.0


@pytest.mark.parametrize(
    "report,max_tick_ms",
    [("pi_coproc_offline_500hz.json", 2.0), ("pi_coproc_live_50hz.json", 20.0)],
)
def test_on_pi_zero_deadline_misses(report, max_tick_ms):
    # Committed reports from the real Raspberry Pi: the reduced co-processor met
    # every deadline. Offline (compute only) holds 500 Hz; the live read+forecast
    # loop holds 50 Hz (bus-read-bound, not compute-bound).
    rep = json.loads((_DATA / report).read_text())
    assert rep["deadline_misses"] == 0
    assert rep["n_ticks"] > 100
    assert rep["tick_ms_max"] < max_tick_ms
