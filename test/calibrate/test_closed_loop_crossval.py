# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Cross-trajectory generalization of the closed-loop fit, real LeKiwi.

An earlier check showed the fit beats the frictionless baseline on a held-out
*tail* of the same excitation. This is the stronger gate check: the plant friction
identified on one real capture must predict the joint torque of a **separately
captured, different trajectory** better than the frictionless URDF model. Three
joints are exercised —
two with a paired train/validation excitation (different center/amplitude/duration),
plus shoulder_lift (the heavy gravity joint) on a held-out split.

All telemetry is committed under ``test/calibrate/data/lekiwi/`` (``synthetic:false``),
so these run in CI without hardware.
"""

from __future__ import annotations

import pathlib

import pytest

from jaxterity import zoo
from jaxterity.calibrate import (
    cross_trajectory_validation,
    identify_closed_loop,
    load_commanded_capture,
)

_DATA = pathlib.Path(__file__).with_name("data") / "lekiwi"
_CALIB = _DATA / "calibration_final.json"


@pytest.fixture(scope="module")
def robot():
    # Uncalibrated URDF = the frictionless datasheet baseline the fit improves on
    # (zoo.calibrated now carries the real fit).
    return zoo.load("so100")


def _load(name, robot):
    return load_commanded_capture(_DATA / name, robot, calibration=_CALIB)


# (train capture, validation capture, joint, min cross-trajectory reduction)
_PAIRS = [
    (
        "commanded_shoulder_pan.json",
        "validation_shoulder_pan.json",
        "shoulder_pan",
        0.20,
    ),
    ("commanded_elbow_flex.json", "validation_elbow_flex.json", "elbow_flex", 0.10),
]


@pytest.mark.parametrize("train,val,joint,min_reduction", _PAIRS)
def test_fit_generalizes_to_a_different_real_trajectory(
    robot, train, val, joint, min_reduction
):
    fit = identify_closed_loop(robot, _load(train, robot))
    report = cross_trajectory_validation(robot, fit, _load(val, robot))
    # The plant friction identified on one trajectory predicts a *different* real
    # trajectory's torque better than the frictionless model.
    assert report["reduction"] >= min_reduction
    assert report["rms_fitted"] < report["rms_baseline"]


def test_shoulder_lift_heavy_gravity_joint_identifies(robot):
    # The heaviest arm joint (lifts the whole arm) — strong dynamic signal.
    fit = identify_closed_loop(robot, _load("commanded_shoulder_lift.json", robot))
    assert fit.residuals["held_out_reduction"] >= 0.25
    assert float(fit.fitted["shoulder_lift.damping"]) > 0.0


def test_measurement_calibration_is_not_transferred(robot):
    # cross_trajectory_validation refits K_t/c0 per capture (measurement-side), so a
    # joint's own validation set is at least as good as the frictionless baseline —
    # the friction can only help once the duty-proxy calibration is fair.
    fit = identify_closed_loop(robot, _load("commanded_shoulder_pan.json", robot))
    same = cross_trajectory_validation(
        robot, fit, _load("commanded_shoulder_pan.json", robot)
    )
    assert same["reduction"] > 0.0
