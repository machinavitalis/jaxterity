# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""The static-hold thermal capture loader (`scripts/lekiwi_thermal_capture.py`).

Unit-level checks on the boundary conversions that matter: whole-degree Celsius
from the register becomes kelvin (the SI-at-the-boundary rule), the duty proxy
becomes the same torque frame the closed-loop fit uses, the register's wild
single-sample readings are despiked, phases are selectable (a rise-only or
decay-only fit is how the heating gain and cooling constant get separated), and
the short commanded-excitation schema — which carries no usable thermal signal —
is refused by name rather than silently fitted.
"""

from __future__ import annotations

import numpy as np
import pytest

from jaxterity.calibrate import THERMAL_CAPTURE_SCHEMA, load_thermal_capture
from jaxterity.errors import CalibrationError


def _doc(n=30, temps=None, loads=None):
    """A minimal two-servo capture in the on-device schema."""
    phase = ["baseline"] * (n // 3) + ["hold"] * (n // 3)
    phase += ["cool"] * (n - len(phase))
    base = list(range(28, 28 + n)) if temps is None else list(temps)
    return {
        "schema": THERMAL_CAPTURE_SCHEMA,
        "servo_ids": [1, 2],
        "joints": {"1": "shoulder_pan", "2": "shoulder_lift"},
        "dry_run": False,
        "aborted": False,
        "excitation": {
            "kind": "static_hold",
            "baseline_s": 10.0,
            "hold_s": 10.0,
            "cool_s": 10.0,
            "target_rate_hz": 1.0,
        },
        "plan": {"schema": "jaxterity-checkup-plan-v2", "serial": "LK-TEST"},
        "pose_ticks": {"1": 2900, "2": 1500},
        "calibration": {},
        "achieved_rate_hz": 1.0,
        "t": [float(i) for i in range(n)],
        "phase": phase,
        "pos_ticks": {"1": [2900] * n, "2": [1500] * n},
        "load_raw": {
            "1": ([200] * n if loads is None else list(loads)),
            "2": [400] * n,
        },
        "volt_dV": {"1": [122] * n, "2": [122] * n},
        "temp_C": {"1": base, "2": base},
    }


def test_celsius_becomes_kelvin_at_the_boundary():
    telemetry, _tau, _meta = load_thermal_capture(_doc(n=30))
    assert telemetry.temperature.shape == (30, 2)
    # 28 degC is the first sample.
    assert float(telemetry.temperature[0, 0]) == pytest.approx(28 + 273.15)


def test_the_duty_proxy_becomes_the_closed_loop_torque_frame():
    _telemetry, tau, _meta = load_thermal_capture(_doc(n=9))
    # (load/1000) * volts: 200/1000 * 12.2 and 400/1000 * 12.2.
    assert float(tau[0, 0]) == pytest.approx(0.2 * 12.2)
    assert float(tau[0, 1]) == pytest.approx(0.4 * 12.2)


def test_a_wild_register_sample_is_despiked():
    temps = [30] * 20
    temps[10] = 95  # the Feetech register's occasional nonsense
    telemetry, _tau, _meta = load_thermal_capture(_doc(n=20, temps=temps))
    assert float(telemetry.temperature[10, 0]) == pytest.approx(30 + 273.15)


def test_a_real_step_is_not_despiked_away():
    # A genuine monotone rise must survive: despiking must not flatten signal.
    temps = list(range(28, 48))
    telemetry, _tau, _meta = load_thermal_capture(_doc(n=20, temps=temps))
    assert float(telemetry.temperature[-1, 0]) > float(telemetry.temperature[0, 0]) + 15


def test_phases_are_selectable_and_time_is_rebased():
    hold, _tau, _meta = load_thermal_capture(_doc(n=30), phases=("hold",))
    cool, _tau2, _meta2 = load_thermal_capture(_doc(n=30), phases=("cool",))
    assert hold.temperature.shape[0] == 10
    assert cool.temperature.shape[0] == 10
    # Each slice starts at t=0 so the fit's derivative is not offset.
    assert float(hold.time[0]) == 0.0 and float(cool.time[0]) == 0.0


def test_an_absent_phase_is_refused_with_what_was_logged():
    with pytest.raises(CalibrationError, match="no samples in phase"):
        load_thermal_capture(_doc(), phases=("soak",))


def test_the_short_excitation_schema_is_refused_by_name():
    doc = _doc()
    doc["schema"] = "lekiwi-commanded-capture-v1"
    with pytest.raises(CalibrationError, match="no usable thermal signal"):
        load_thermal_capture(doc)


def test_missing_samples_do_not_crash_the_loader():
    doc = _doc(n=12)
    doc["temp_C"]["1"][3] = None
    doc["load_raw"]["1"][4] = None
    telemetry, tau, _meta = load_thermal_capture(doc)
    assert telemetry.temperature.shape == (12, 2)
    assert float(tau[4, 0]) == 0.0  # a dropped load reads as no torque


def test_metadata_carries_the_provenance_a_fit_needs():
    _telemetry, _tau, meta = load_thermal_capture(_doc())
    assert meta["excitation"]["kind"] == "static_hold"
    assert meta["plan"]["schema"] == "jaxterity-checkup-plan-v2"
    assert meta["pose_ticks"]["1"] == 2900
    assert meta["dry_run"] is False


def test_the_loader_feeds_fit_servo_thermal():
    from jaxterity.calibrate import fit_servo_thermal

    # A physically-shaped rise so the regression is well posed.
    n = 200
    T_amb, R_th, k_pwr, C_th = 300.0, 3.0, 0.5, 40.0
    tau_proxy = 0.2 * 12.2
    temps, T = [], T_amb
    for _ in range(n):
        T += (k_pwr * tau_proxy**2 - (T - T_amb) / R_th) / C_th
        temps.append(int(round(T - 273.15)))
    doc = _doc(n=n, temps=temps)
    telemetry, tau, _meta = load_thermal_capture(doc)
    fit = fit_servo_thermal(telemetry, tau, C_th=C_th, T_amb=T_amb)
    assert set(fit) >= {"R_th", "k_pwr"}
    assert np.all(np.isfinite(fit["R_th"]))


# --- the real LeKiwi hold, and what it does and does not identify -----------

_REAL = (
    __import__("pathlib").Path(__file__).parent
    / "data"
    / "lekiwi"
    / "thermal_hold.json"
)


def test_the_real_hold_dissipates_on_only_the_loaded_joints():
    """LK-0001, 2026-07-29: the first thermally-excited trace on this unit.

    9 minutes at the vetted home pose (60 s baseline / 300 s hold / 180 s cool).
    It is a real improvement on the 16 s commanded captures — two joints carry
    genuine load and rise 2-3 degC — but the pose only loads *two* of six, and
    the other four read exactly zero duty. Their ~1 degC "rise" is one
    quantization code: ambient/neighbour warming, not self-heating, and nothing
    the single-node model can attribute.
    """
    telemetry, tau, meta = load_thermal_capture(_REAL)
    assert meta["excitation"]["kind"] == "static_hold"
    assert telemetry.temperature.shape[0] == 540  # 9 min at 1 Hz
    duty = {
        name: float(abs(tau[:, j]).max())
        for j, name in enumerate(telemetry.servo_names)
    }
    loaded = {n for n, d in duty.items() if d > 0}
    assert loaded == {"shoulder_lift", "elbow_flex", "gripper"}, duty
    # The two shoulder/elbow joints carry the real gravity load.
    assert duty["elbow_flex"] > duty["shoulder_lift"] > 0.5
    for name in ("shoulder_pan", "wrist_flex", "wrist_roll"):
        assert duty[name] == 0.0


def test_the_real_hold_does_not_yet_identify_the_thermal_model():
    """The honest verdict, pinned so a better capture can be recognized.

    Every servo fits non-physically: R_th comes out negative with a standard
    error far larger than the value. That is not a fitter bug — the trace has no
    cooling information (the loaded joints were still climbing when torque cut,
    and the 3-minute tail decayed by 0.0 degC), so the regression cannot see the
    ``-(T - T_amb)/R_th`` term at all.

    When a longer capture lands, this test should fail — that is the point. It
    is the bar: a capture that identifies the model gives positive R_th with a
    standard error below the value on at least the loaded joints.
    """
    from jaxterity.calibrate import fit_servo_thermal

    telemetry, tau, _meta = load_thermal_capture(_REAL)
    fit = fit_servo_thermal(
        telemetry, tau, C_th=40.0, T_amb=float(telemetry.temperature[0].min())
    )
    assert np.all(fit["R_th"] < 0.0), (
        "R_th came out positive — if a longer capture is now committed, this "
        "test has served its purpose: replace it with the real fitted values."
    )
    assert np.all(np.sqrt(fit["R_th_var"]) > np.abs(fit["R_th"]))


def test_the_loaded_joints_were_still_climbing_when_torque_cut():
    """Why the hold length, not the pose, is the binding constraint."""
    import json

    doc = json.loads(_REAL.read_text())
    phase = np.asarray(doc["phase"], dtype=object)
    telemetry, _tau, _meta = load_thermal_capture(_REAL)
    time = np.asarray(doc["t"], float)
    j = telemetry.servo_names.index("elbow_flex")
    hold = phase == "hold"
    late_slope = np.polyfit(time[hold][-60:], telemetry.temperature[hold, j][-60:], 1)[
        0
    ]
    assert late_slope * 60 > 0.5, "elbow_flex should still be warming at hold end"
    # And the cooling tail carries no decay to fit against.
    cool = telemetry.temperature[phase == "cool", j]
    assert abs(float(cool[-1] - cool[0])) <= 1.0  # within one quantization code
