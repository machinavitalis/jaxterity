# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Drive-wheel health from a free-spin capture.

Every expectation here is pinned to the real LK-0001 capture in
``data/lk0001_wheels.json`` (wheels on blocks, 2026-07-29), including its two
measurement defects — a temperature register that garbles under drive, and a
speed register that dithers by one count at rest. Both of those produced
confident wrong answers before they were handled, so both are asserted on.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from jaxterity.checkup import (
    WheelHealth,
    load_wheel_capture,
    wheel_gates,
    wheel_health,
    wheel_summary,
)
from jaxterity.errors import CalibrationError

CAPTURE = Path(__file__).parent / "data" / "lk0001_wheels.json"


@pytest.fixture(scope="module")
def health():
    return wheel_health(CAPTURE)


def test_loader_rejects_a_foreign_schema():
    with pytest.raises(CalibrationError, match="unrecognized schema"):
        load_wheel_capture({"schema": "something-else", "t": [0.0]})


def test_loader_rejects_an_empty_capture():
    doc = json.loads(CAPTURE.read_text())
    with pytest.raises(CalibrationError, match="no samples"):
        load_wheel_capture({**doc, "t": []})


def test_all_three_wheels_are_summarized(health):
    assert [h.servo_id for h in health] == [7, 8, 9]
    assert [h.joint for h in health] == ["wheel_left", "wheel_back", "wheel_right"]


def test_the_healthy_unit_tracks_every_step_and_raises_no_gate(health):
    # LK-0001 hit every commanded speed exactly, forward and reverse.
    for h in health:
        assert h.tracks_every_step
        assert h.worst_tracking == pytest.approx(1.0, abs=1e-9)
    assert wheel_gates(health) == []


def test_tracking_covers_both_directions(health):
    steps = sorted(health[0].tracking)
    assert steps == [-600, -400, -200, 200, 400, 600]


def test_a_wheel_that_cannot_reach_speed_fails_the_tracking_gate(health):
    lame = dataclasses.replace(health[0], tracking={**health[0].tracking, 600: 0.55})
    assert not lame.tracks_every_step
    gates = wheel_gates((lame, *health[1:]))
    assert [g.gate for g in gates] == ["wheel_tracking"]
    assert gates[0].level == "fail"
    assert gates[0].scope == "wheel_left"
    assert "600" in gates[0].detail and "55%" in gates[0].detail


def test_duty_symmetry_is_a_warning_not_a_failure(health):
    # The whole point of duty symmetry is that it needs no reference unit, but a
    # three-wheel median is thin evidence, so an outlier warns rather than fails.
    hog = dataclasses.replace(
        health[0], duty={k: v * 2.0 for k, v in health[0].duty.items()}
    )
    gates = wheel_gates((hog, *health[1:]))
    assert [g.gate for g in gates] == ["wheel_symmetry"]
    assert gates[0].level == "warn"
    assert gates[0].scope == "wheel_left"


def test_one_duty_outlier_warns_once_not_three_times(health):
    # Against a median-of-the-others the bad wheel drags both healthy wheels'
    # reference and all three warn. Against the median of all three it cannot.
    hog = dataclasses.replace(
        health[0], duty={k: v * 2.0 for k, v in health[0].duty.items()}
    )
    scopes = [g.scope for g in wheel_gates((hog, *health[1:]))]
    assert scopes == ["wheel_left"], f"one outlier, {len(scopes)} warnings: {scopes}"


def test_symmetry_is_skipped_rather_than_degraded_below_three_wheels(health):
    # A two-wheel median is just an average of the pair: it cannot tell which of
    # the two is the outlier, so the check is withheld instead of guessing.
    hog = dataclasses.replace(
        health[0], duty={k: v * 2.0 for k, v in health[0].duty.items()}
    )
    assert wheel_gates((hog, health[1])) == []
    assert wheel_gates((hog,)) == []


def test_healthy_duty_spread_is_well_inside_the_symmetry_tolerance(health):
    top = max(h for h in health[0].duty if h > 0)
    duties = np.array([h.duty[top] for h in health])
    spread = float(np.max(np.abs(duties - np.median(duties))) / np.median(duties))
    assert spread < 0.05, f"LK-0001 wheels matched within 2 %, got {spread:.1%}"


def test_coast_is_reported_but_never_gated(health):
    # Deliberate: 0.1 s at 50 Hz is five samples. Resolution too coarse to gate.
    frozen = dataclasses.replace(health[0], coast_s=0.0)
    runaway = dataclasses.replace(health[0], coast_s=60.0, coast_censored=True)
    assert wheel_gates((frozen, *health[1:])) == []
    assert wheel_gates((runaway, *health[1:])) == []


def test_at_rest_speed_dither_is_not_counted_as_coasting(health):
    # The capture's coast tail alternates 0 and 50 (one register count) on a
    # stopped wheel. Reading that as motion put two wheels at 2.98 s against a
    # 3.0 s window, inventing a 25x asymmetry on a symmetric robot.
    coasts = np.array([h.coast_s for h in health])
    assert np.all(coasts < 0.5), f"dither leaked into coast_s: {coasts}"
    assert np.ptp(coasts) < 0.1, f"wheels should agree, got {coasts}"
    assert not any(h.coast_censored for h in health)


def test_coast_censoring_is_flagged_when_the_window_cuts_the_spin_short():
    doc = json.loads(CAPTURE.read_text())
    speed = list(doc["speed_raw"])
    phase, sid = doc["phase"], doc["servo_id"]
    # Leave wheel 7 spinning fast right through the end of its coast window.
    for i, (p, s) in enumerate(zip(phase, sid)):
        if p == "coast" and s == 7:
            speed[i] = 400
    (only,) = [h for h in wheel_health({**doc, "speed_raw": speed}) if h.servo_id == 7]
    assert only.coast_censored
    assert only.coast_s > 2.5


def test_temperature_is_the_median_and_survives_the_garbled_reads(health):
    # The bus showed 32-33 C right after this capture. A max over the same
    # channel says 78 C, because ~1.5 % of driven-phase reads are garbage.
    for h in health:
        assert 30.0 <= h.median_temp_c <= 36.0
        # The fixture keeps every garbled read while downsampling the good ones
        # 3x, so this fraction reads ~3x the 1.5-1.7 % of the raw capture.
        assert 0.0 < h.garbled_temp_fraction < 0.08


def test_the_garbled_reads_really_are_in_the_channel_not_the_wheel():
    # Guards the reason median is used: if the max were real the wheels would
    # have been at 78 C, and a rolling median would have removed an isolated
    # spike. Neither is true — hence "report the median, report the fraction".
    doc = json.loads(CAPTURE.read_text())
    temp = np.array([np.nan if v is None else v for v in doc["temp_C"]], float)
    phase = np.asarray(doc["phase"], dtype=object)
    off = np.abs(temp - np.nanmedian(temp)) > 3.0
    assert np.nanmax(temp) > 50.0, "fixture must retain the garbled reads"
    assert off[phase == "coast"].sum() == 0, "garbling was driven-phase only"
    assert off[phase == "staircase"].sum() > 0


def test_a_wheel_with_no_temperature_channel_degrades_to_nan():
    doc = json.loads(CAPTURE.read_text())
    blank = wheel_health({**doc, "temp_C": [None] * len(doc["temp_C"])})
    assert all(np.isnan(h.median_temp_c) for h in blank)
    assert all(np.isnan(h.garbled_temp_fraction) for h in blank)
    assert wheel_gates(blank) == []  # a dead sensor is not a wheel fault


def test_summary_renders_every_wheel_and_marks_censoring(health):
    text = wheel_summary(health)
    assert text.count("\n") == len(health)  # header + one row each
    for h in health:
        assert h.joint in text
    censored = dataclasses.replace(health[0], coast_s=3.0, coast_censored=True)
    assert ">" in wheel_summary((censored,)).splitlines()[1]


def test_to_dict_is_json_round_trippable(health):
    d = health[0].to_dict()
    assert json.loads(json.dumps(d)) == d
    assert set(d["tracking"]) == {str(k) for k in health[0].tracking}


def test_wheel_health_is_frozen(health):
    with pytest.raises(dataclasses.FrozenInstanceError):
        health[0].coast_s = 1.0  # type: ignore[misc]
    assert isinstance(health[0], WheelHealth)
