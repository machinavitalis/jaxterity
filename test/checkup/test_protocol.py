# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Protocol pinning — the drift series' precondition, made checkable.

The claims under test are the ones a fleet study would rest on: two sessions
that commanded the same motion compare equal *despite* servos settling on
different ticks; anything that really changes the excitation (amplitude,
waveform, tones, duration, rate, center, held pose) compares unequal; the
difference is reported in words rather than as "the hash changed"; and a stored
series that spans a protocol boundary says so instead of presenting method
sensitivity as degradation.

Note the split the tests keep honest: the *recipe* (what the plan sets) is
hashed, positions are compared against a tolerance. A moved pose therefore has
the same hash and does not match — which is precisely the run-0001 failure a
hash-only pin would have missed.
"""

from __future__ import annotations

import copy
import json
import pathlib

import numpy as np
import pytest

from jaxterity.checkup import (
    POSE_TOLERANCE_TICKS,
    CaptureProtocol,
    CheckupStore,
    capture_protocol,
)
from jaxterity.errors import CalibrationError

_DATA = pathlib.Path(__file__).parents[1] / "calibrate" / "data" / "lekiwi"
_CAPTURES = sorted(_DATA.glob("commanded_*.json"))
_BATTERY = _DATA / "aggregate_load_trace.json"


def _write(tmp_path, doc, name="commanded_x.json"):
    path = tmp_path / name
    path.write_text(json.dumps(doc))
    return path


@pytest.fixture
def capture_doc():
    return json.loads((_DATA / "commanded_shoulder_pan.json").read_text())


# --- canonicalization and hashing ------------------------------------------


def test_protocol_reads_every_capture_and_sorts_by_joint():
    protocol = capture_protocol(_CAPTURES, battery_trace=str(_BATTERY))
    names = [j.joint for j in protocol.joints]
    assert names == sorted(names)
    assert len(names) == len(_CAPTURES)
    assert protocol.battery is not None
    assert protocol.battery.servo_ids == (1, 2, 3)


def test_hash_is_stable_and_round_trips():
    a = capture_protocol(_CAPTURES)
    b = capture_protocol(list(reversed(_CAPTURES)))  # file order must not matter
    assert a.hash == b.hash
    assert CaptureProtocol.from_dict(json.loads(json.dumps(a.to_dict()))) == a
    assert CaptureProtocol.from_dict(a.to_dict()).hash == a.hash


def test_settling_noise_is_still_the_same_experiment(tmp_path, capture_doc):
    """Why positions are compared by distance instead of hashed: a servo never
    settles on the same tick, and a pin that flips on that is worse than none."""
    baseline = capture_protocol([_write(tmp_path, capture_doc)])
    jittered = copy.deepcopy(capture_doc)
    jittered["pose_ticks"] = {
        sid: int(t) + 2 for sid, t in capture_doc["pose_ticks"].items()
    }
    other = capture_protocol([_write(tmp_path, jittered, "b.json")])
    assert other.hash == baseline.hash  # the recipe is untouched
    assert other.matches(baseline)
    assert other.differences(baseline) == ()


def test_a_pose_just_inside_the_tolerance_still_matches(tmp_path, capture_doc):
    baseline = capture_protocol([_write(tmp_path, capture_doc)])
    nudged = copy.deepcopy(capture_doc)
    first = next(iter(nudged["pose_ticks"]))
    nudged["pose_ticks"][first] = (
        int(nudged["pose_ticks"][first]) + POSE_TOLERANCE_TICKS
    )
    assert capture_protocol([_write(tmp_path, nudged, "b.json")]).matches(baseline)


def test_a_moved_pose_does_change_the_hash(tmp_path, capture_doc):
    """The run-0001 failure mode: same script, different pose, different fits."""
    baseline = capture_protocol([_write(tmp_path, capture_doc)])
    moved = copy.deepcopy(capture_doc)
    first = next(iter(moved["pose_ticks"]))
    moved["pose_ticks"][first] = int(moved["pose_ticks"][first]) + 300
    other = capture_protocol([_write(tmp_path, moved, "b.json")])
    # The recipe is identical — only the pose moved, which is exactly the case
    # a hash-only pin misses and run-0001 fell into.
    assert other.hash == baseline.hash
    assert not other.matches(baseline)
    assert any("held pose differs" in d for d in other.differences(baseline))


@pytest.mark.parametrize(
    "field, value, expected",
    [
        ("amp_ticks", 150, "amplitude"),
        ("duration_s", 12.0, "duration"),
        ("target_rate_hz", 40.0, "rate"),
        ("kind", "schroeder_multisine_v2", "waveform"),
        ("multisine_hz", [0.2, 0.4, 0.8, 1.2, 2.0], "tones"),
        ("center_ticks", 2000, "center"),
    ],
)
def test_every_commanded_quantity_is_registered(
    tmp_path, capture_doc, field, value, expected
):
    baseline = capture_protocol([_write(tmp_path, capture_doc)])
    changed = copy.deepcopy(capture_doc)
    changed["excitation"][field] = value
    other = capture_protocol([_write(tmp_path, changed, "b.json")])
    assert not other.matches(baseline), field
    assert any(expected in d for d in other.differences(baseline)), field
    # Recipe fields move the hash; the center is a position, so it does not.
    if field != "center_ticks":
        assert other.hash != baseline.hash, field
    else:
        assert other.hash == baseline.hash


def test_a_pose_past_the_tolerance_is_reported_with_its_size(tmp_path, capture_doc):
    baseline = capture_protocol([_write(tmp_path, capture_doc)])
    nudged = copy.deepcopy(capture_doc)
    first = next(iter(nudged["pose_ticks"]))
    delta = 3 * POSE_TOLERANCE_TICKS
    nudged["pose_ticks"][first] = int(nudged["pose_ticks"][first]) + delta
    other = capture_protocol([_write(tmp_path, nudged, "b.json")])
    assert not other.matches(baseline)
    assert any(
        f"servo {first}: {delta} ticks" in d for d in other.differences(baseline)
    )


def test_a_capture_without_excitation_metadata_is_refused(tmp_path, capture_doc):
    blind = copy.deepcopy(capture_doc)
    blind.pop("excitation")
    with pytest.raises(CalibrationError, match="no excitation metadata"):
        capture_protocol([_write(tmp_path, blind)])


def test_differences_are_empty_for_an_identical_protocol():
    a = capture_protocol(_CAPTURES)
    assert a.differences(capture_protocol(_CAPTURES)) == ()


def test_a_missing_joint_is_reported(tmp_path):
    full = capture_protocol(_CAPTURES)
    partial = capture_protocol(_CAPTURES[:2])
    assert any("absent now" in d for d in partial.differences(full))
    assert any("absent before" in d for d in full.differences(partial))


def test_from_dict_refuses_a_foreign_schema():
    d = capture_protocol(_CAPTURES).to_dict()
    d["schema"] = "something-else"
    with pytest.raises(CalibrationError, match="unrecognized schema"):
        CaptureProtocol.from_dict(d)


# --- the store's drift warnings --------------------------------------------


def _report(protocol, *, serial="LK-TEST", score=70):
    """A minimal report carrying a protocol (the store only reads these)."""
    from jaxterity.checkup.report import CheckupReport

    return CheckupReport(
        schema="jaxterity-checkup-v2",
        serial=serial,
        robot="so100",
        created_utc="2026-07-29T00:00:00Z",
        engine_version="test",
        baseline="test",
        joints=(),
        battery=None,
        gates=(),
        score={"total": score},
        score_formula="1",
        attestation_handle="deadbeef",
        protocol=protocol,
    )


def test_one_protocol_across_runs_warns_about_nothing(tmp_path):
    store = CheckupStore(tmp_path)
    protocol = capture_protocol(_CAPTURES)
    for _ in range(3):
        store.save(_report(protocol))
    assert store.drift_warnings("LK-TEST") == ()
    assert store.protocol_hashes("LK-TEST") == [protocol.hash] * 3


def test_a_protocol_change_is_named_at_its_boundary(tmp_path, capture_doc):
    store = CheckupStore(tmp_path)
    first = capture_protocol([_write(tmp_path, capture_doc)])
    changed = copy.deepcopy(capture_doc)
    changed["excitation"]["amp_ticks"] = 150
    second = capture_protocol([_write(tmp_path, changed, "b.json")])
    store.save(_report(first))
    store.save(_report(first))
    store.save(_report(second))
    warnings = store.drift_warnings("LK-TEST")
    assert len(warnings) == 1
    assert "runs 1-2 → run 3" in warnings[0]
    assert "amplitude 200 → 150 ticks" in warnings[0]
    assert "mixes drift with method change" in warnings[0]


def test_unpinned_runs_are_called_out(tmp_path):
    store = CheckupStore(tmp_path)
    store.save(_report(None))  # a pre-pinning run
    store.save(_report(capture_protocol(_CAPTURES)))
    warnings = store.drift_warnings("LK-TEST")
    assert any("no capture protocol" in w for w in warnings)
    assert store.protocol_hashes("LK-TEST")[0] is None


def test_a_single_run_cannot_warn(tmp_path):
    store = CheckupStore(tmp_path)
    store.save(_report(capture_protocol(_CAPTURES)))
    assert store.drift_warnings("LK-TEST") == ()


def test_drift_series_still_works_alongside_the_pin(tmp_path):
    store = CheckupStore(tmp_path)
    protocol = capture_protocol(_CAPTURES)
    store.save(_report(protocol, score=70))
    store.save(_report(protocol, score=80))
    series = store.drift("LK-TEST")
    assert series["score.total"] == [70.0, 80.0]
    assert np.all(np.isfinite(series["score.total"]))
