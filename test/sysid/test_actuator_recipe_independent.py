# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""``sysid.actuator`` LM fitters — recovery from an *independent* integrator.

``test_actuator_recipe.py`` fits each LM device against truth telemetry it
generates with the *same* fixed-step RK4 the fitter uses. That proves the fit is
self-consistent, but a shared discretization or sign error in that RK4 would
pass silently — the fit would recover the bug, not the physics.

These tests close that hole at two levels:

* **Integrator independence** (Servo / BLDC / SEA): frozen traces produced by a
  *different* integrator — SciPy's implicit adaptive ``Radau`` — applied to a
  standalone transcription of each device ODE (``generate.py``). A
  discretization/sign bug in the fitter's rollout can no longer hide behind an
  identical bug in the data generator.
* **Model-form independence** (SEA): a frozen trace whose ground truth is *not*
  a transcription of ``_sea_rhs`` at all — it is jaxonomy's acausal DAE built
  from independently-tested rotational primitives (``generate_jaxonomy.py``). A
  sign or coupling error in jaxterity's own SEA equations would fail to recover
  here even if it were copied identically into the Radau transcription.

The traces are committed (``data/independent/*.npz``); SciPy / jaxonomy are only
needed to regenerate them, not to run this test.

What this still does NOT cover: a physics error shared across the whole jax*y
stack, or effects absent from every in-repo model. Catching that needs telemetry
whose ground truth is external to the stack entirely — a published third-party
reference log, or real actuator-bench data — tracked as a follow-up in
KNOWN_GAPS.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import jaxterity.sysid as sysid
from jaxterity.actuators import BLDC, SeriesElasticActuator, Servo
from jaxterity.data import (
    ActuatorChirpTelemetry,
    ActuatorStepTelemetry,
    SeriesElasticTelemetry,
)

_DATA = Path(__file__).resolve().parent / "data" / "independent"


def _load(name: str) -> tuple[dict[str, np.ndarray], dict]:
    """Load a frozen oracle trace and its metadata (truth + warm-start params)."""
    z = np.load(_DATA / f"{name}.npz")
    meta = json.loads(str(z["meta_json"]))
    arrays = {k: z[k] for k in z.files if k != "meta_json"}
    # Guard: the oracle must never come from the fitter's own explicit RK4 —
    # that would silently restore the circularity. Independent integrators
    # (SciPy Radau) and cross-model sources (jaxonomy acausal) are both fine.
    method = meta["integrator"]["method"]
    assert "rk4" not in method.lower(), (
        f"oracle trace must not come from the fitter's own RK4; got {method!r}. "
        "Regenerate under data/independent/."
    )
    return arrays, meta


@pytest.mark.nightly
def test_bldc_recovers_kt_over_j_from_independent_trace():
    """BLDC fit recovers the identifiable ratio Kt/J from a Radau-generated chirp.

    R/L/Ke/Kp_i are only weakly identifiable from a free-shaft current chirp
    (documented in the recipe), so the assertion is on Kt/J — the ratio the
    velocity dynamics pin — plus a small predictive residual. Crucially the
    telemetry comes from an integrator the fitter does not use.
    """
    a, m = _load("bldc_chirp")
    t = m["truth"]
    tel = ActuatorChirpTelemetry(
        time=a["time"],
        sample_rate_hz=m["sample_rate_hz"],
        actuator_name="m",
        setpoint=a["setpoint"],
        velocity=a["velocity"],
        current=a["current"],
    )
    device = BLDC(
        **{k: m["warm"][k] for k in ("R", "L", "Kt", "Ke", "Kp_i", "J", "b", "Fc")}
    )
    res = sysid.actuator(tel, device, hints={"max_steps": 200})

    kt_over_j = float(res.fitted["Kt"]) / float(res.fitted["J"])
    assert kt_over_j == pytest.approx(t["Kt"] / t["J"], rel=1e-3)
    assert res.residuals is not None
    # Small but not machine-zero: the fitter's RK4 and Radau are different
    # discretizations, so a residual well below sensor scale is the tell that
    # the fit generalizes across integrators rather than memorizing its own.
    assert res.residuals["velocity_rms"] < 1e-3
    assert res.residuals["current_rms"] < 1e-3
    assert not res.warnings  # a clean recovery must not raise the stall warning


@pytest.mark.nightly
def test_series_elastic_recovers_truth_from_independent_trace():
    """SEA fit recovers k, c, and both inertias from a Radau-generated torque chirp."""
    a, m = _load("sea_chirp")
    t = m["truth"]
    tel = SeriesElasticTelemetry(
        time=a["time"],
        sample_rate_hz=m["sample_rate_hz"],
        actuator_name="s",
        tau_input=a["tau_input"],
        q_motor=a["q_motor"],
        q_load=a["q_load"],
    )
    device = SeriesElasticActuator(
        k=m["warm"]["k"],
        c=m["warm"]["c"],
        J_motor=m["warm"]["J_m"],
        J_load=m["warm"]["J_l"],
        b_m=m["warm"]["b_m"],
        b_l=m["warm"]["b_l"],
    )
    res = sysid.actuator(tel, device, hints={"max_steps": 200})

    assert float(res.fitted["k"]) == pytest.approx(t["k"], rel=1e-2)
    assert float(res.fitted["c"]) == pytest.approx(t["c"], rel=5e-2)
    assert float(res.fitted["J_motor"]) == pytest.approx(t["J_m"], rel=2e-2)
    assert float(res.fitted["J_load"]) == pytest.approx(t["J_l"], rel=2e-2)
    assert not res.warnings


@pytest.mark.nightly
def test_sea_recovers_from_jaxonomy_acausal_oracle():
    """SEA fit recovers the truth from a *cross-model* oracle.

    Stronger than the Radau trace above: this ground truth is not a transcription
    of ``_sea_rhs`` at all — it is jaxonomy's acausal DAE assembled from
    independently-tested rotational primitives (Inertia / Spring / Damper /
    TorqueSource); see ``data/independent/generate_jaxonomy.py``. A sign or
    coupling error in jaxterity's own SEA rollout would fail to recover here even
    though it might survive a same-transcription check. Tolerances are looser
    than the Radau case because the oracle drives a continuous torque (vs the
    fitter's ZOH) and is resampled onto the telemetry grid — a genuine model gap,
    not self-consistency.
    """
    a, m = _load("sea_jaxonomy")
    t = m["truth"]
    assert m["integrator"]["method"] == "jaxonomy-acausal-BDF"  # the cross-model oracle
    tel = SeriesElasticTelemetry(
        time=a["time"],
        sample_rate_hz=m["sample_rate_hz"],
        actuator_name="s",
        tau_input=a["tau_input"],
        q_motor=a["q_motor"],
        q_load=a["q_load"],
    )
    device = SeriesElasticActuator(
        k=m["warm"]["k"],
        c=m["warm"]["c"],
        J_motor=m["warm"]["J_m"],
        J_load=m["warm"]["J_l"],
        b_m=m["warm"]["b_m"],
        b_l=m["warm"]["b_l"],
    )
    res = sysid.actuator(tel, device, hints={"max_steps": 200})

    assert float(res.fitted["k"]) == pytest.approx(t["k"], rel=3e-2)
    assert float(res.fitted["J_motor"]) == pytest.approx(t["J_m"], rel=8e-2)
    assert float(res.fitted["J_load"]) == pytest.approx(t["J_l"], rel=2e-2)
    # c (coupling damping) is the weakly-identifiable channel — a looser bound.
    assert float(res.fitted["c"]) == pytest.approx(t["c"], rel=1.5e-1)
    assert res.residuals is not None
    assert res.residuals["q_motor_rms"] < 1e-2
    assert res.residuals["q_load_rms"] < 1e-2
    assert not res.warnings


@pytest.mark.nightly
def test_servo_recovers_gains_from_independent_trace():
    """Servo fit recovers Kp/Kd/J from a Radau-generated step response.

    The step-response cost landscape is shallow: a single-start LM stalls at the
    warm start (and the recipe flags it). The multi-start variant recovers the
    gains from the independent trace — a stronger check than the sibling
    round-trip, since the data was not produced by the fitter's own integrator.
    """
    a, m = _load("servo_step")
    t = m["truth"]
    tel = ActuatorStepTelemetry(
        time=a["time"],
        sample_rate_hz=m["sample_rate_hz"],
        actuator_name="s",
        setpoint=a["setpoint"],
        position=a["position"],
    )
    device = Servo(
        Kp=m["warm"]["Kp"],
        Kd=m["warm"]["Kd"],
        T_max=m["warm"]["T_max"],
        deadband=m["warm"]["deadband"],
        J=m["warm"]["J"],
    )
    hints = {"max_steps": 128, "n_starts": 5, "perturbation": 0.2, "seed": 0}
    res = sysid.actuator(tel, device, hints=hints)

    assert float(res.fitted["Kp"]) == pytest.approx(t["Kp"], rel=5e-2)
    assert float(res.fitted["Kd"]) == pytest.approx(t["Kd"], rel=5e-2)
    assert float(res.fitted["J"]) == pytest.approx(t["J"], rel=8e-2)
    assert res.residuals is not None
    assert res.residuals["position_rms"] < 1e-3


@pytest.mark.nightly
def test_single_start_servo_stalls_on_independent_trace():
    """Guard the negative case: single-start LM cannot recover from the warm start.

    This is why the recovery test above uses multi-start. If a future change made
    single-start silently "recover" here, that would signal the stall detector or
    the cost landscape changed — worth a look, not a silent pass.
    """
    a, m = _load("servo_step")
    t = m["truth"]
    tel = ActuatorStepTelemetry(
        time=a["time"],
        sample_rate_hz=m["sample_rate_hz"],
        actuator_name="s",
        setpoint=a["setpoint"],
        position=a["position"],
    )
    device = Servo(
        Kp=m["warm"]["Kp"],
        Kd=m["warm"]["Kd"],
        T_max=m["warm"]["T_max"],
        deadband=m["warm"]["deadband"],
        J=m["warm"]["J"],
    )
    single = sysid.actuator(tel, device, hints={"max_steps": 128, "n_starts": 1})
    # The warm start is 25% off in Kp; a stalled single-start stays near it.
    assert abs(float(single.fitted["Kp"]) - t["Kp"]) > 1.0
