# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""LeKiwi brownout veto from a REAL aggregate-load battery fit.

The Thévenin battery parameters are fit from a *real* multi-servo load trace
captured on the physical LeKiwi (committed under ``test/calibrate/data/lekiwi/``:
three arm servos driven by a synchronized multisine while the shared bus voltage
was recorded). The fit (``fit_battery_sag``) recovers ``V_oc``/``R_int``; the
trace shows a real ~0.7 V sag strongly anti-correlated with the aggregate load.

The catch: the battery-bearing ``lekiwi_twin`` predicts the bus voltage
and the ``SafetyMonitor`` vetoes a sustained load that would brown it out, naming
``bus_voltage`` as the binding constraint — while a body-only twin (``so101_twin``,
no battery) is structurally blind to the electrical limit and passes. The healthy
real-fit battery correctly does *not* false-alarm; an aged battery (``R_int`` grown
by wear — exactly the drift the twin's online re-fit tracks) browns out under the
same load.

Runs in CI (the trace is committed); only the capture itself needs hardware.
"""

from __future__ import annotations

import pathlib

import jax.numpy as jnp
import numpy as np
import pytest

from jaxterity.calibrate import fit_battery_sag, load_bus_load_trace
from jaxterity.selfcheck import CoProcessor, SafetyAction, SafetyEnvelope, SafetyMonitor
from jaxterity.twin import lekiwi_twin, so101_twin

_TRACE = (
    pathlib.Path(__file__).parent.parent
    / "calibrate"
    / "data"
    / "lekiwi"
    / ("aggregate_load_trace.json")
)


@pytest.fixture(scope="module")
def fit():
    return fit_battery_sag(load_bus_load_trace(_TRACE))


def _vmin_and_decision(twin, tau_val, floor):
    nv = twin.n_servos
    cp = CoProcessor(twin, rate_hz=20.0, horizon_steps=8, dt=0.02, warmup=False)
    q = jnp.zeros(twin.robot.build_system()._nq)
    qd = jnp.zeros(nv)
    temp = jnp.full(nv, twin.T_amb)
    tau = jnp.full(nv, tau_val)
    pred = cp.predict(q, qd, temp, tau)
    env = SafetyEnvelope(
        temp_ceiling_k=twin.T_amb + 500.0, bus_floor_v=floor, bus_band_v=0.5
    )
    decision = SafetyMonitor(twin, env).evaluate(pred, action=tau)
    return float(np.asarray(pred.bus_voltage).min()), decision


def test_battery_fit_from_real_trace_is_physical(fit):
    v_oc = float(fit.fitted["V_oc"])
    r_int = float(fit.fitted["R_int"])
    assert 11.5 < v_oc < 12.6  # healthy ~12 V pack
    assert 0.0 < r_int < 1.0  # plausible pack + wiring resistance
    assert fit.residuals["r_squared"] > 0.5  # the sag tracks the aggregate load
    assert fit.residuals["sag_v"] > 0.3  # a real, measurable sag


def test_healthy_battery_does_not_false_alarm(fit):
    # The real-fit healthy battery: a sustained load sags the bus but stays above
    # a 10 V floor — the monitor must NOT raise a (false) brownout veto.
    twin = lekiwi_twin(
        "LK-real", V_oc=float(fit.fitted["V_oc"]), R_int=float(fit.fitted["R_int"])
    )
    v_min, decision = _vmin_and_decision(twin, 1.5, 10.0)
    assert v_min > 10.0
    assert decision.binding_constraint != "bus_voltage"


def test_aged_battery_browns_out_and_is_vetoed(fit):
    # Aged pack: R_int grown ~5x by wear (the degradation the twin's re-fit tracks).
    # The same sustained load now browns out below the floor → veto on bus_voltage.
    twin = lekiwi_twin(
        "LK-aged",
        V_oc=float(fit.fitted["V_oc"]),
        R_int=5.0 * float(fit.fitted["R_int"]),
    )
    v_min, decision = _vmin_and_decision(twin, 1.5, 10.0)
    assert v_min < 10.0
    assert decision.action is SafetyAction.VETO
    assert decision.binding_constraint == "bus_voltage"


def test_body_only_twin_is_blind_to_brownout():
    # The contrast: a body-only twin has no battery state, so the bus-voltage
    # check is skipped — it passes a load the battery-bearing twin would veto.
    body = so101_twin("SO101-body")
    assert body.has_battery is False
    nv = body.n_servos
    cp = CoProcessor(body, rate_hz=20.0, horizon_steps=8, dt=0.02, warmup=False)
    q = jnp.zeros(body.robot.build_system()._nq)
    pred = cp.predict(q, jnp.zeros(nv), jnp.full(nv, body.T_amb), jnp.full(nv, 1.5))
    assert pred.bus_voltage is None
    env = SafetyEnvelope(temp_ceiling_k=body.T_amb + 500.0, bus_floor_v=10.0)
    decision = SafetyMonitor(body, env).evaluate(pred, action=jnp.full(nv, 1.5))
    assert decision.binding_constraint != "bus_voltage"
