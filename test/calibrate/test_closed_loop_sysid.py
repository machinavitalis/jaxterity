# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Closed-loop active sysid on REAL LeKiwi telemetry.

These tests load telemetry captured on a *physical* LeKiwi arm (committed under
``test/calibrate/data/lekiwi/``: a band-limited position multisine commanded
through each STS3215 servo's inner loop, torque held on — see
``scripts/lekiwi_commanded_capture.py``) and reproduce the inverse-dynamics
equation-error fit (:func:`jaxterity.calibrate.identify_closed_loop`).

The captures are the split-capture half of the supported route: the Pi records,
the workstation fits. No hardware is needed to run these — the data is committed —
so they guard the fit against regressions in CI. The geared STS3215 does *not*
free-coast, so this commanded/equation-error path supersedes the coast
``Excitation`` for these joints.

The gate the fit must clear: on a held-out split of real data, the per-unit fit
predicts joint torque measurably better than the incoming (URDF/datasheet,
frictionless) model.
"""

from __future__ import annotations

import pathlib

import pytest

from jaxterity import zoo
from jaxterity.calibrate import (
    CommandedCapture,
    identify_closed_loop,
    load_commanded_capture,
)

_DATA = pathlib.Path(__file__).with_name("data") / "lekiwi"
_CALIB = _DATA / "calibration_final.json"


@pytest.fixture(scope="module")
def robot():
    # The LeKiwi arm is the SO-101 (== zoo so100 6-DoF arm). The fit identifies
    # friction from the *uncalibrated* URDF — the frictionless datasheet baseline the
    # gate compares against (zoo.calibrated now carries the real fit).
    return zoo.load("so100")


# (capture file, expected damping, expected viscous, expected held-out reduction)
_CASES = [
    ("commanded_shoulder_pan.json", "shoulder_pan", 0.4150, 0.45),
    ("commanded_elbow_flex.json", "elbow_flex", 0.0674, 0.15),
]


@pytest.mark.parametrize("fname,joint,exp_damping,min_reduction", _CASES)
def test_real_capture_fit_beats_baseline(
    robot, fname, joint, exp_damping, min_reduction
):
    cap = load_commanded_capture(_DATA / fname, robot, calibration=_CALIB)
    assert isinstance(cap, CommandedCapture)
    assert cap.joint == joint
    assert cap.n_samples == 1280

    res = identify_closed_loop(robot, cap)

    # The fitted viscous damping is positive and matches the recorded value.
    assert f"{joint}.damping" in res.fitted
    assert float(res.fitted[f"{joint}.damping"]) == pytest.approx(exp_damping, abs=2e-3)

    # GATE: held-out torque prediction beats the frictionless baseline.
    r = res.residuals
    assert r["held_out_reduction"] >= min_reduction
    assert r["held_out_rms_fitted"] < r["held_out_rms_baseline"]

    # Applying the fit yields a robot whose joint carries the identified params.
    calibrated = res.apply_to(robot)
    j = calibrated.joint_map[joint]
    assert float(j.damping) == pytest.approx(exp_damping, abs=2e-3)
    assert float(j.damping) > 0.0  # baseline was frictionless


def test_fit_is_deterministic(robot):
    # Equation-error fit (FFT filter + mj_inverse + lstsq) is fully deterministic:
    # the same committed telemetry reproduces the same fit bit-for-bit.
    cap = load_commanded_capture(
        _DATA / "commanded_shoulder_pan.json", robot, calibration=_CALIB
    )
    a = identify_closed_loop(robot, cap)
    b = identify_closed_loop(robot, cap)
    assert a.telemetry_hash == b.telemetry_hash
    for k in a.fitted:
        assert float(a.fitted[k]) == float(b.fitted[k])
    for k in a.residuals:
        assert a.residuals[k] == b.residuals[k]


def test_uncertainty_intervals_are_reported(robot):
    # The fit carries a covariance, so interval() is available; viscous
    # damping is tightly bounded and strictly positive.
    cap = load_commanded_capture(
        _DATA / "commanded_shoulder_pan.json", robot, calibration=_CALIB
    )
    res = identify_closed_loop(robot, cap)
    intervals = res.interval(0.95)
    assert intervals is not None
    lo, hi = intervals["shoulder_pan.damping"]
    assert lo > 0.0 and hi > lo


def test_aborted_capture_is_rejected(robot):
    bad = {
        "schema": "lekiwi-commanded-capture-v1",
        "joint": "shoulder_pan",
        "servo_id": 1,
        "aborted": True,
        "calibration": {
            "sign": -1,
            "ticks_per_rad": 651.9,
            "offset_ticks": 3043.5,
            "range_min_ticks": 1792,
            "range_max_ticks": 4095,
        },
        "pose_ticks": {},
        "t": [0.0],
        "pos_ticks": [2000],
        "load_raw": [0],
        "volt_dV": [122],
    }
    with pytest.raises(Exception, match="aborted"):
        load_commanded_capture(bad, robot)
