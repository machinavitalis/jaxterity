# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""The 2026-07-30 LK-0001 bench session, pinned.

Two things were owed to hardware and are now measured on it: the per-joint lost
motion (T-238), and a **controlled** two-pose test of the claim that a joint's
fitted damping tracks the inertia it is swinging (T-235). The second was
previously supported only by an accidental pair of sessions whose recorded poses
turned out to be untrustworthy — see
:func:`test_the_two_poses_were_actually_different`.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from jaxterity import zoo
from jaxterity.calibrate import (
    axis_inertia_feature,
    identify_closed_loop,
    load_commanded_capture,
)
from jaxterity.calibrate.types import JointCalibration
from jaxterity.checkup import backlash_summary, reversal_lost_motion

_DATA = pathlib.Path(__file__).parent / "data"
_CAL = str(
    pathlib.Path(__file__).parents[1]
    / "calibrate"
    / "data"
    / "lekiwi"
    / "calibration_final.json"
)
_BACKLASH = _DATA / "lk0001_backlash.json"
_POSES = _DATA / "lk0001_pose"


@pytest.fixture(scope="module")
def robot():
    return zoo.load("so100")


@pytest.fixture(scope="module")
def calibrations():
    fields = JointCalibration.__dataclass_fields__
    rows = json.loads(pathlib.Path(_CAL).read_text())["calibrations"]
    return [
        JointCalibration(**{k: v for k, v in c.items() if k in fields}) for c in rows
    ]


@pytest.fixture(scope="module")
def pose_fits(robot, calibrations):
    """Fitted damping and axis inertia at each of the two deliberate poses."""
    feature = axis_inertia_feature(robot, calibrations, "shoulder_pan")
    out = {}
    for name in ("tucked", "extended"):
        doc = json.loads((_POSES / f"pan_{name}.json").read_text())
        capture = load_commanded_capture(doc, robot, calibration=_CAL)
        result = identify_closed_loop(robot, capture)
        pose = tuple(
            (int(s), int(t))
            for s, t in sorted(doc["pose_ticks"].items(), key=lambda kv: int(kv[0]))
        )
        out[name] = {
            "damping": float(result.fitted["shoulder_pan.damping"]),
            "inertia": feature(pose),
            "pose": dict(pose),
            "provenance": doc["plan_transit"],
        }
    return out


# --- T-235: the controlled two-pose test ----------------------------------


def test_the_two_poses_were_actually_different(pose_fits):
    """The precondition the earlier evidence could not establish.

    An on-device bug recorded ``pose_ticks`` *before* the plan transit, so a
    capture could report the pose it was parked in rather than the one it was
    excited at. Both fixtures here were taken after that fix and carry
    provenance saying the pose was read after the transit.
    """
    for name, fit in pose_fits.items():
        assert fit["provenance"]["used_plan"] is True, name
        assert fit["provenance"]["pose_read_after_transit"] is True, name
    tucked, extended = pose_fits["tucked"]["pose"], pose_fits["extended"]["pose"]
    # shoulder_pan itself is at the same centre; the *rest* of the arm moved.
    assert abs(tucked[1] - extended[1]) < 25
    assert abs(tucked[2] - extended[2]) > 1000
    assert abs(tucked[3] - extended[3]) > 1000


def test_the_inertia_lever_is_large(pose_fits):
    ratio = pose_fits["extended"]["inertia"] / pose_fits["tucked"]["inertia"]
    assert ratio == pytest.approx(3.57, rel=0.05)


def test_damping_moves_with_pose_far_outside_fit_noise(pose_fits):
    raw = pose_fits["extended"]["damping"] / pose_fits["tucked"]["damping"]
    assert raw > 3.0, f"raw damping ratio {raw:.2f}x"


def test_normalizing_by_inertia_removes_most_of_the_move(pose_fits):
    """T-235's acceptance criterion, on a controlled experiment.

    Two deliberately different poses, one unit, identical recipe, minutes apart:
    the raw damping differs >3x and ``damping/inertia`` agrees to ~10 %, which is
    inside the fit's own repeat scatter (measured at 1-12 % on this unit).
    """
    tucked, extended = pose_fits["tucked"], pose_fits["extended"]
    normalized = (extended["damping"] / extended["inertia"]) / (
        tucked["damping"] / tucked["inertia"]
    )
    assert normalized == pytest.approx(1.0, abs=0.15), (
        f"pose-normalized ratio {normalized:.3f}x — pose no longer explains the move"
    )
    raw = extended["damping"] / tucked["damping"]
    assert abs(normalized - 1.0) < abs(raw - 1.0) / 5.0


# --- T-238: the owed lost-motion number -----------------------------------


@pytest.fixture(scope="module")
def widths():
    return reversal_lost_motion(_BACKLASH, calibration=_CAL)


def test_all_five_arm_joints_were_measured(widths):
    assert [w.joint for w in widths] == [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    ]


def test_the_measured_lost_motion_is_a_few_encoder_ticks(widths):
    # 8-12 ticks, i.e. 12-18 mrad (0.7-1.05 deg), on a healthy unit. This is the
    # number T-238 owed; it is total lost motion, not gear play.
    ticks = {w.joint: w.lost_motion_ticks for w in widths}
    assert all(6.0 <= v <= 16.0 for v in ticks.values()), ticks
    assert ticks["elbow_flex"] == max(ticks.values())


def test_every_width_is_flagged_quantization_limited(widths):
    """The honest limit of this measurement on this encoder.

    The per-goal gaps are integers and the widths are single-digit ticks, so the
    bootstrap cannot see below the quantum — three joints originally reported a
    *zero-width* interval, which reads as certainty and is not. Flagged, and the
    interval widened to the encoder's real resolution of half a tick.
    """
    for width in widths:
        assert width.quantization_limited, width.joint
        lo, hi = width.ci95_rad
        assert lo < width.lost_motion_rad < hi
        span_ticks = (hi - lo) * width.lost_motion_ticks / width.lost_motion_rad
        assert span_ticks == pytest.approx(1.0, rel=1e-6)


def test_the_summary_reports_the_quantization_flag(widths):
    text = backlash_summary(widths)
    assert "quantized" in text.splitlines()[0]
    assert text.count("yes") == len(widths)


def test_the_compliant_mesh_model_is_refuted_on_this_hardware():
    """A negative result, and the reason the width is not called backlash.

    The fitted dead zone is essentially zero on every joint (|play| < 0.25x half
    the measured width) while the *linear* term fits at R^2 > 0.95. So the load
    tracks following error with no dead band: the hysteresis is the position
    loop's own deadband plus compliance, not a gear-mesh dead zone. One encoder
    cannot separate those, which is exactly why nothing here is named backlash.
    """
    from jaxterity.checkup import fit_reversal

    widths = {w.joint: w for w in reversal_lost_motion(_BACKLASH, calibration=_CAL)}
    for fit in fit_reversal(_BACKLASH, calibration=_CAL):
        half = widths[fit.joint].lost_motion_rad / 2.0
        assert fit.r_squared > 0.95, fit.joint
        assert abs(fit.play_rad) < 0.25 * half, (
            f"{fit.joint}: fitted play {fit.play_rad * 1e3:.2f} mrad is no longer "
            "negligible against the measured half-width — the mesh model may now fit"
        )


def test_the_load_register_sign_is_opposite_to_following_error():
    # Measured, and the reason fitted stiffnesses come out negative: the servo
    # reports load with the opposite sign to (goal - position).
    import numpy as np

    from jaxterity.checkup import fit_reversal

    for fit in fit_reversal(_BACKLASH, calibration=_CAL):
        assert fit.k_mesh_proxy < 0, fit.joint
    doc = json.loads(_BACKLASH.read_text())
    sid = np.asarray(doc["servo_id"])
    goal = np.asarray(doc["goal_ticks"], float)
    pos = np.asarray([np.nan if v is None else v for v in doc["pos_ticks"]], float)
    load = np.asarray([np.nan if v is None else v for v in doc["load_raw"]], float)
    mine = (sid == 1) & np.isfinite(pos) & np.isfinite(load)
    correlation = float(np.corrcoef(goal[mine] - pos[mine], load[mine])[0, 1])
    assert correlation < -0.9, correlation
