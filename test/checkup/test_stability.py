# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Tip-over gating of checkup capture plans (the T-228 flip regression).

On 2026-07-28 the v0 checkup's captures tipped a real LeKiwi: each capture
parked the previous joint at its band center, the arm accumulated into a
forward-extended pose, and the elbow/wrist excitations rocked the base off
its rear wheel. The recorded poses of that session are hardcoded here as the
regression fixture: the stability gate must REFUSE the captures during which
the hardware misbehaved, and must ACCEPT a full-amplitude-where-possible plan
built from the bring-up choreography's tucked home pose. Margins use the
representative ``LEKIWI_BASE`` profile — the assertions are about verdicts
and orderings, which are robust to moderate profile error, not about exact
millimetres.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from jaxterity import zoo
from jaxterity.checkup import (
    LEKIWI_BASE,
    PLAN_SCHEMA_V1,
    CapturePlan,
    build_capture_plan,
    validate_capture_motion,
)
from jaxterity.checkup.stability import (
    MULTISINE_KIND,
    MULTISINE_KIND_V1,
    multisine_peak_to_peak,
    polygon_margin,
    schroeder_multisine,
)
from jaxterity.cli import main as cli_main
from jaxterity.errors import CalibrationError

#: The unit's bring-up calibration. Use the **committed** copy under
#: ``test/calibrate/data/`` — the ``scripts/lekiwi_out/`` working directory is
#: gitignored, so pointing at it made this whole module fail on any fresh
#: checkout (including CI) with a bare FileNotFoundError.
_CALIB = (
    pathlib.Path(__file__).parents[1]
    / "calibrate"
    / "data"
    / "lekiwi"
    / "calibration_final.json"
)

#: The real 2026-07-28 session, as recorded in its capture files: the arm
#: accumulated into extension (servo 2 raised to ~1982, servo 3 extended to
#: ~1976) because each capture left the previous joint at its band center.
_T228_ELBOW = {
    "pose": {"1": 2943, "2": 1966, "3": 2999, "4": 1434, "5": 611, "6": 2597},
    "servo": 3,
    "center": 1976,
    "amp": 200,
}
_T228_WRIST = {
    "pose": {"1": 2943, "2": 1986, "3": 1965, "4": 1474, "5": 610, "6": 2598},
    "servo": 4,
    "center": 2096,
    "amp": 200,
}
_T228_FIRST = {  # the session's FIRST capture — arm still parked, was fine
    "pose": {"1": 3190, "2": 799, "3": 3078, "4": 1424, "5": 611, "6": 2598},
    "servo": 1,
    "center": 2943,
    "amp": 200,
}

#: The bring-up choreography's home pose (jaxterity model joint values, rad).
_HOME = {
    "shoulder_pan": 1.916,
    "shoulder_lift": -0.8311,
    "elbow_flex": -1.5417,
    "wrist_flex": 0.1449,
    "wrist_roll": -0.5704,
}


@pytest.fixture(scope="module")
def robot():
    return zoo.load("so100")


def _validate(robot, case):
    return validate_capture_motion(
        robot,
        str(_CALIB),
        base=LEKIWI_BASE,
        pose_ticks=case["pose"],
        excited_servo_id=case["servo"],
        center_ticks=case["center"],
        amp_ticks=case["amp"],
        # The 2026-07-28 session was recorded under the legacy sqrt(2n)
        # normalization; scoring it with today's true-peak waveform would
        # understate the swing it really commanded by 52% and could exonerate a
        # capture that physically rocked the base.
        excitation_kind=MULTISINE_KIND_V1,
    )


def test_flip_session_captures_are_refused(robot):
    # The two captures during which the base physically rocked/tipped must
    # fail the gate; the session's first capture (arm parked) may pass.
    elbow = _validate(robot, _T228_ELBOW)
    wrist = _validate(robot, _T228_WRIST)
    first = _validate(robot, _T228_FIRST)
    assert not elbow.ok and not wrist.ok
    assert first.ok
    # The extension is visible statically: by the wrist capture, just HOLDING
    # the accumulated pose is worse than holding the first capture's pose.
    assert wrist.static_margin_m < first.static_margin_m
    # And the excitation eats further into the margin.
    assert elbow.min_margin_m < elbow.static_margin_m


def test_home_pose_plan_is_feasible_and_derated(robot):
    plan = build_capture_plan(
        robot, str(_CALIB), serial="LK-TEST", base=LEKIWI_BASE, pose_rad=_HOME
    )
    assert plan.pose_static_margin_m >= LEKIWI_BASE.margin_min_m
    assert len(plan.entries) == 7  # 6 joints + the load trace
    assert all(e.feasible for e in plan.entries)
    assert all(e.min_margin_m >= LEKIWI_BASE.margin_min_m for e in plan.entries)
    # Band-edge joints get reduced amplitudes; no amplitude exceeds request.
    amps = {e.joints[0]: e.amp_ticks for e in plan.entries if e.kind == "joint"}
    assert all(0 < a <= 200 for a in amps.values())
    load = next(e for e in plan.entries if e.kind == "load")
    assert load.servo_ids == (1, 2, 3) and 0 < load.amp_ticks <= 180
    # Round-trips through JSON unchanged.
    again = CapturePlan.from_dict(json.loads(json.dumps(plan.to_dict())))
    assert again.to_dict() == plan.to_dict()


def test_true_peak_normalization_buys_back_amplitude(robot):
    """The point of 0.2: a tighter waveform is a *less derated* capture.

    The loose sqrt(2n) normalization spent 52% of its commanded swing on
    overshoot, and the ZMP gate — correctly — charged the plan for it. Under the
    true-peak waveform the same requested amplitude produces less excursion, so
    margins rise and the load trace stops needing a derate.
    """
    kwargs = dict(serial="LK-TEST", base=LEKIWI_BASE, pose_rad=_HOME)
    legacy = build_capture_plan(
        robot, str(_CALIB), excitation_kind=MULTISINE_KIND_V1, **kwargs
    )
    current = build_capture_plan(
        robot, str(_CALIB), excitation_kind=MULTISINE_KIND, **kwargs
    )
    assert current.excitation_kind == MULTISINE_KIND
    by_joint = {e.joints: e for e in legacy.entries}
    for entry in current.entries:
        was = by_joint[entry.joints]
        assert entry.min_margin_m >= was.min_margin_m - 1e-9, entry.joints
        assert entry.amp_ticks >= was.amp_ticks, entry.joints
    # Concretely: the in-phase load trace no longer needs derating.
    load_now = next(e for e in current.entries if e.kind == "load")
    load_was = next(e for e in legacy.entries if e.kind == "load")
    assert load_was.amp_ticks < load_now.amp_ticks == 180


def test_a_legacy_plan_reads_back_as_the_legacy_waveform(robot):
    # A v1 plan JSON carries no excitation_kind; reading it as v2 would
    # understate every swing it was vetted for, so the loader must not.
    plan = build_capture_plan(
        robot,
        str(_CALIB),
        serial="LK-TEST",
        base=LEKIWI_BASE,
        pose_rad=_HOME,
        excitation_kind=MULTISINE_KIND_V1,
    )
    legacy_doc = plan.to_dict()
    legacy_doc["schema"] = PLAN_SCHEMA_V1
    del legacy_doc["excitation_kind"]
    assert CapturePlan.from_dict(legacy_doc).excitation_kind == MULTISINE_KIND_V1


def test_statically_unstable_pose_is_rejected(robot):
    # The T-228 wrist-capture pose (arm accumulated into extension) is not a
    # legal start pose: the builder must refuse it outright.
    from jaxterity.calibrate.closed_loop import LEKIWI_ARM_JOINT_BY_ID
    from jaxterity.checkup.plan import _cal_rows, _ticks_to_rad

    rows = _cal_rows(str(_CALIB))
    pose_rad = {
        LEKIWI_ARM_JOINT_BY_ID[int(s)]: _ticks_to_rad(float(t), rows[int(s)])
        for s, t in _T228_WRIST["pose"].items()
    }
    with pytest.raises(CalibrationError, match="statically unstable"):
        build_capture_plan(
            robot, str(_CALIB), serial="LK-TEST", base=LEKIWI_BASE, pose_rad=pose_rad
        )


def test_smaller_amplitude_never_hurts_the_margin(robot):
    big = _validate(robot, _T228_ELBOW)
    small = _validate(robot, dict(_T228_ELBOW, amp=60))
    assert small.min_margin_m >= big.min_margin_m


def test_polygon_margin_signs():
    square = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], float)  # CCW
    m = polygon_margin(np.array([[0.0, 0.0], [0.9, 0.0], [2.0, 0.0]]), square)
    assert m[0] == pytest.approx(1.0)
    assert m[1] == pytest.approx(0.1)
    assert m[2] < 0.0


def test_multisine_is_peak_normalized_and_deterministic():
    t = np.linspace(0.0, 16.0, 1281)
    w = schroeder_multisine(t, 16.0)
    # v2 normalizes by the sum's measured peak: the commanded swing never
    # exceeds the requested amplitude.
    assert float(np.max(np.abs(w))) == pytest.approx(1.0, abs=1e-3)
    assert np.array_equal(w, schroeder_multisine(t, 16.0))


def test_the_legacy_waveform_is_still_reproducible():
    # The measured T-229 finding, preserved: sqrt(2n) is a crest *estimate*
    # this 5-tone set overshoots by ~1.52x (amp 200 commanded ~±303 ticks,
    # matching the ~46° travel recorded on hardware). Historical captures must
    # keep scoring against the waveform they actually commanded.
    # Sampled densely: a capture-rate grid lands just below the true peak.
    t = np.linspace(0.0, 16.0, 16001)
    legacy = schroeder_multisine(t, 16.0, kind=MULTISINE_KIND_V1)
    assert float(np.max(np.abs(legacy))) == pytest.approx(1.5163, rel=1e-3)


def test_peak_to_peak_is_not_twice_the_amplitude():
    # The Schroeder sum is asymmetric (+4.795/-3.696), so even peak-normalized
    # its swing is 1.77x the amplitude — the assumption the response-ratio gate
    # used to make (2x) was wrong for both waveform versions.
    assert multisine_peak_to_peak(16.0, MULTISINE_KIND) == pytest.approx(
        1.771, rel=1e-3
    )
    assert multisine_peak_to_peak(16.0, MULTISINE_KIND_V1) == pytest.approx(
        2.685, rel=1e-3
    )


def test_an_unknown_waveform_kind_is_refused():
    with pytest.raises(CalibrationError, match="Unknown multisine kind"):
        schroeder_multisine(np.linspace(0.0, 1.0, 10), 1.0, kind="vibes")


def test_cli_plan_mode_writes_a_vetted_plan(tmp_path, capsys):
    pose_file = tmp_path / "pose.json"
    pose_file.write_text(json.dumps(_HOME))
    out = tmp_path / "plan.json"
    rc = cli_main(
        [
            "checkup",
            "--plan-out",
            str(out),
            "--serial",
            "LK-CLI-PLAN",
            "--calibration",
            str(_CALIB),
            "--pose-file",
            str(pose_file),
        ]
    )
    text = capsys.readouterr().out
    assert rc == 0 and out.exists()
    plan = CapturePlan.from_dict(json.loads(out.read_text()))
    assert plan.serial == "LK-CLI-PLAN" and plan.feasible_entries
    assert "capture plan" in text
