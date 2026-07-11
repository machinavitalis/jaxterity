# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Self-check co-processor — publishing, determinism, skip-and-flag, timing.

The behavioural tests use an injected clock so the deadline logic is
deterministic (no wall-clock flakiness): the co-processor still does the real
rollout each tick, but lateness is judged by the scripted clock. The actual
"holds the target rate on the dev host" assertion is the ``pi``-marked
real-wall-clock test, skipped by ``scripts/check.sh``.
"""

from __future__ import annotations

import time

import jax.numpy as jnp
import numpy as np
import pytest

from jaxterity.selfcheck import CoProcessor
from jaxterity.twin import Twin, so101_twin

pytest.importorskip("mujoco")


@pytest.fixture(scope="module")
def twin() -> Twin:
    return so101_twin("SO101-0001")


@pytest.fixture(scope="module")
def dims(twin):
    return twin.robot.build_system()._nq, twin.n_servos


def _scripted_clock(elapseds):
    """A clock yielding start/end pairs so tick k's measured compute == elapseds[k]."""
    vals = []
    t = 0.0
    for e in elapseds:
        vals.append(t)  # tick start
        t += e
        vals.append(t)  # tick end
        t += 1.0  # idle gap between ticks (irrelevant to the deadline check)
    it = iter(vals)
    return lambda: next(it)


def _const_source(nq, nv, *, action=0.2):
    def source(_k):
        return (
            jnp.zeros(nq),
            jnp.full(nv, 0.05),
            jnp.full(nv, 298.15),
            jnp.full(nv, action),
        )

    return source


# --- construction ----------------------------------------------------------


def test_rate_must_be_positive(twin):
    with pytest.raises(ValueError, match="rate_hz"):
        CoProcessor(twin, rate_hz=0.0)


# --- publishing + prediction shape -----------------------------------------


def test_runs_all_ticks_and_publishes(twin, dims):
    nq, nv = dims
    n = 6
    # All ticks well under the deadline → every tick publishes, zero misses.
    cp = CoProcessor(
        twin,
        rate_hz=20.0,
        horizon_steps=10,
        dt=0.02,
        clock=_scripted_clock([0.001] * n),
    )
    pub = []
    report = cp.run(_const_source(nq, nv), n, publish=lambda k, p: pub.append(k))
    assert report.published == n
    assert report.deadline_misses == 0
    assert pub == list(range(n))


def test_prediction_shapes(twin, dims):
    nq, nv = dims
    cp = CoProcessor(twin, rate_hz=20.0, horizon_steps=10, dt=0.02)
    pred = cp.predict(
        jnp.zeros(nq), jnp.full(nv, 0.05), jnp.full(nv, 298.15), jnp.full(nv, 0.2)
    )
    assert pred.temperature.shape == (11, nv)
    assert pred.q.shape == (11, nq)
    assert pred.bus_voltage is None  # bench twin


# --- determinism -----------------------------------------------------------


def test_deterministic_predictions(twin, dims):
    nq, nv = dims
    cp = CoProcessor(twin, rate_hz=20.0, horizon_steps=10, dt=0.02)
    src = _const_source(nq, nv)
    a = cp.predict(*src(0))
    b = cp.predict(*src(0))
    np.testing.assert_array_equal(np.asarray(a.temperature), np.asarray(b.temperature))
    np.testing.assert_array_equal(np.asarray(a.q), np.asarray(b.q))


# --- skip-and-flag ---------------------------------------------------------


def test_injected_slow_tick_is_skipped_and_flagged(twin, dims):
    nq, nv = dims
    # deadline = 1/20 = 0.05 s; tick 2 "takes" 0.5 s, the rest 1 ms.
    elapseds = [0.001, 0.001, 0.5, 0.001, 0.001]
    cp = CoProcessor(
        twin,
        rate_hz=20.0,
        horizon_steps=10,
        dt=0.02,
        clock=_scripted_clock(elapseds),
    )
    pub = []
    report = cp.run(
        _const_source(nq, nv), len(elapseds), publish=lambda k, p: pub.append(k)
    )
    assert report.deadline_misses == 1
    assert report.published == len(elapseds) - 1
    late = [r.tick for r in report.reports if r.late]
    assert late == [2]
    assert 2 not in pub  # the late tick was skipped, not published
    assert report.reports[2].prediction is None


# --- battery (LeKiwi) twin predicts bus voltage ----------------------------


def test_battery_twin_predicts_bus_voltage(twin, dims):
    nq, nv = dims
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
    cp = CoProcessor(lekiwi, rate_hz=20.0, horizon_steps=8, dt=0.02)
    pred = cp.predict(
        jnp.zeros(nq), jnp.full(nv, 0.05), jnp.full(nv, 298.15), jnp.full(nv, 0.3)
    )
    assert pred.bus_voltage is not None
    assert pred.bus_voltage.shape == (8,)
    assert float(pred.bus_voltage.min()) < 12.0  # sags under load


# --- simulated controller + arm in the loop --------------------------------


def test_closed_loop_with_simulated_arm(twin, dims):
    """Run beside a simulated arm + controller; every tick publishes a forecast."""
    nq, nv = dims
    step = twin.compiled_rollout(0.02)  # ground-truth one-step advance
    state = {"q": jnp.zeros(nq), "qd": jnp.full(nv, 0.05), "T": jnp.full(nv, 298.15)}

    def source(_k):
        action = jnp.full(nv, 0.15)  # simple constant controller
        # advance the "true" arm one control step
        roll = step(state["q"], state["qd"], state["T"], action[None, :])
        state["q"], state["qd"], state["T"] = (
            roll.q[-1],
            roll.qd[-1],
            roll.temperature[-1],
        )
        return state["q"], state["qd"], state["T"], action

    cp = CoProcessor(
        twin,
        rate_hz=20.0,
        horizon_steps=10,
        dt=0.02,
        clock=_scripted_clock([0.001] * 8),
    )
    report = cp.run(source, 8)
    assert report.published == 8 and report.deadline_misses == 0


# --- pi: real-wall-clock timing (skipped by scripts/check.sh) --------------


@pytest.mark.pi
def test_holds_target_rate_on_host(twin, dims):  # pragma: no cover - timing
    """A warmed co-processor holds tens-of-Hz with zero deadline misses on host."""
    nq, nv = dims
    cp = CoProcessor(
        twin, rate_hz=20.0, horizon_steps=10, dt=0.02, clock=time.perf_counter
    )
    report = cp.run(_const_source(nq, nv), 40)
    assert report.deadline_misses == 0, (
        f"{report.deadline_misses}/40 ticks missed the 20 Hz deadline — escalate "
        "to a lowered co-processor."
    )
