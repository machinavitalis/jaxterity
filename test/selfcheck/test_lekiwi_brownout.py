# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""LeKiwi brownout check — the shared-battery veto path.

The bench SO-101 path needs only the thermal model; LeKiwi adds a shared battery.
This wires `lekiwi_twin` → CoProcessor → SafetyMonitor end to end and shows the
body-only-vs-twin contrast on the *electrical* axis: under aggregate servo load the
predicted bus voltage sags below the floor and the monitor vetoes (brownout),
while a light load passes.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from jaxterity.selfcheck import CoProcessor, SafetyAction, SafetyEnvelope, SafetyMonitor
from jaxterity.twin import _LEKIWI_R_INT, lekiwi_twin

pytest.importorskip("mujoco")


@pytest.fixture(scope="module")
def twin():
    # The default (fitted, healthy-pack) R_int barely sags under 2 N·m — the
    # honest physical outcome. To exercise the brownout veto path this test
    # uses an aged-pack scenario: R_int grown 5x by wear, the same factor the
    # real-trace test and examples/selfcheck_demo_lekiwi.py use.
    return lekiwi_twin("LK-0001", R_int=5.0 * _LEKIWI_R_INT)


@pytest.fixture(scope="module")
def dims(twin):
    return twin.robot.build_system()._nq, twin.n_servos


def test_lekiwi_twin_has_battery(twin):
    assert twin.has_battery
    assert twin.V_oc is not None and twin.R_int is not None


def _monitor(twin):
    # thermal ceiling set out of the way so the bus-voltage floor is the test.
    return SafetyMonitor(
        twin,
        SafetyEnvelope(
            temp_ceiling_k=twin.T_amb + 500.0, bus_floor_v=10.5, bus_band_v=0.5
        ),
    )


def test_heavy_load_browns_out_and_vetoes(twin, dims):
    nq, nv = dims
    cp = CoProcessor(twin, rate_hz=20.0, horizon_steps=8, dt=0.02)
    mon = _monitor(twin)
    heavy = jnp.full(nv, 2.0)
    pred = cp.predict(jnp.zeros(nq), jnp.zeros(nv), jnp.full(nv, twin.T_amb), heavy)
    decision = mon.evaluate(pred, action=heavy)
    assert decision.action is SafetyAction.VETO
    assert decision.binding_constraint == "bus_voltage"
    assert decision.worst_value < 10.5  # bus sagged below the floor


def test_light_load_is_ok(twin, dims):
    nq, nv = dims
    cp = CoProcessor(twin, rate_hz=20.0, horizon_steps=8, dt=0.02)
    mon = _monitor(twin)
    light = jnp.full(nv, 0.3)
    pred = cp.predict(jnp.zeros(nq), jnp.zeros(nv), jnp.full(nv, twin.T_amb), light)
    decision = mon.evaluate(pred, action=light)
    assert decision.action is SafetyAction.OK


def test_co_processor_publishes_bus_voltage(twin, dims):
    nq, nv = dims
    cp = CoProcessor(twin, rate_hz=20.0, horizon_steps=8, dt=0.02)
    pred = cp.predict(
        jnp.zeros(nq), jnp.zeros(nv), jnp.full(nv, twin.T_amb), jnp.full(nv, 1.0)
    )
    assert pred.bus_voltage is not None
    assert pred.bus_voltage.shape == (8,)
