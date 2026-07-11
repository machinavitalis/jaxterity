# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Real closed-loop recalibration of the SO-100 zoo robot.

The committed ``so100/calibrated.json`` is now a **real** fit (``synthetic:false``):
``zoo.recalibrate("so100", captures, recipe="closed_loop")`` runs the closed-loop
active-sysid fit (commanded-multisine telemetry captured on a physical LeKiwi) and
writes per-joint viscous/Coulomb friction. This verifies the recalibration is a
**data swap, not an API change** — ``zoo.calibrated("so100")`` reads the same file
shape and applies it the same way; only the contents changed from synthetic to real.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from jaxterity import zoo
from jaxterity.calibrate import load_commanded_capture

_DATA = pathlib.Path(__file__).parent.parent / "calibrate" / "data" / "lekiwi"
_CALIB = _DATA / "calibration_final.json"
_CAPTURES = (
    "commanded_shoulder_pan.json",
    "commanded_shoulder_lift.json",
    "commanded_elbow_flex.json",
    "commanded_wrist_flex.json",
    "commanded_wrist_roll.json",
    "commanded_gripper.json",
)


def test_committed_so100_calibration_is_real_not_synthetic():
    pkg = pathlib.Path(zoo.so100.__file__).parent
    payload = json.loads((pkg / "calibrated.json").read_text())
    assert payload["synthetic"] is False
    assert payload["recipe_name"] == "active.closed_loop"
    # Real per-joint friction/damping (not the old synthetic masses) — all 6 arm joints.
    damped = {k for k in payload["fitted"] if k.endswith(".damping")}
    for j in (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ):
        assert f"{j}.damping" in damped
    assert payload["fitted"]["shoulder_pan.damping"] > 0.0


def test_calibrated_so100_applies_real_damping():
    # zoo.calibrated reads the same artifact shape (data swap, not API change) and
    # the calibrated arm now carries nonzero joint damping (was frictionless).
    cal = zoo.calibrated("so100")
    assert float(cal.joint_map["shoulder_pan"].damping) > 0.0
    assert float(cal.joint_map["elbow_flex"].damping) > 0.0


def test_recalibrate_closed_loop_dry_run_reproduces_payload():
    robot = zoo.load("so100")
    caps = [
        load_commanded_capture(_DATA / f, robot, calibration=_CALIB) for f in _CAPTURES
    ]
    _, payload = zoo.recalibrate(
        "so100", caps, recipe="closed_loop", write=False, note="unit test"
    )
    assert payload["synthetic"] is False
    assert payload["recipe_name"] == "active.closed_loop"
    assert payload["note"] == "unit test"
    # One held-out reduction per excited joint, all positive (beats baseline).
    reductions = [v for k, v in payload["residuals"].items() if k.endswith("reduction")]
    assert len(reductions) == len(_CAPTURES)
    assert all(r > 0.0 for r in reductions)


def test_recalibrate_unknown_recipe_raises():
    robot = zoo.load("so100")
    cap = load_commanded_capture(_DATA / _CAPTURES[0], robot, calibration=_CALIB)
    with pytest.raises(Exception, match="unknown recipe"):
        zoo.recalibrate("so100", cap, recipe="nope", write=False)
