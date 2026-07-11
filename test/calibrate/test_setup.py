# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""End-to-end first-run setup: discovery, health, calibration, output.

Exercises :func:`run_setup` against the seeded simulator for both the bench
SO-101 and the LeKiwi layouts, plus the discovery/health stages, the missing-
servo and scale-drift warning paths, the on-disk artifacts, and the
``jaxterity setup --simulate`` CLI entry point. No hardware, no network.
"""

from __future__ import annotations

import json

import pytest

from jaxterity.calibrate import layout as _layout
from jaxterity.calibrate import setup as _setup
from jaxterity.calibrate import writer as _writer
from jaxterity.calibrate.link import LoopbackLink

#: No-op sleep for full-speed poll loops.
_NO_SLEEP = lambda _seconds: None  # noqa: E731

#: Recovery budget in encoder counts.
TICK_TOL = 2.0


@pytest.fixture(scope="module")
def lekiwi_limits():
    """Arm limits for the LeKiwi layout, read once from the zoo URDF."""
    return _setup.load_arm_limits(_layout.LEKIWI)


def _planted_offset(servo_id: int, resolution: int = 4096) -> float:
    """The offset `simulated_link` plants for a given servo (test mirror)."""
    return resolution / 2 + (servo_id - 3) * 25


def test_run_setup_lekiwi_recovers_all_offsets(lekiwi_limits):
    layout = _layout.LEKIWI
    link = _setup.simulated_link(layout, lekiwi_limits)
    with link:
        result = _setup.run_setup(
            link, layout, arm_limits=lekiwi_limits, sleep=_NO_SLEEP
        )

    assert result.ok
    assert result.discovery.ok
    assert result.discovery.found_ids == tuple(range(1, 10))
    assert result.health.ok
    # Every joint calibrated; offsets recovered to the planted ground truth.
    by_id = {c.servo_id: c for c in result.calibrations}
    assert set(by_id) == set(range(1, 10))
    for spec in layout.arm_joints:
        cal = by_id[spec.servo_id]
        assert cal.method == "touch_off"
        assert cal.offset_ticks == pytest.approx(
            _planted_offset(spec.servo_id), abs=TICK_TOL
        )
    for spec in layout.wheel_joints:
        assert by_id[spec.servo_id].method == "wheel_zero"


def test_run_setup_so101_arm_only(lekiwi_limits):
    layout = _layout.SO101
    link = _setup.simulated_link(layout, lekiwi_limits)
    with link:
        result = _setup.run_setup(
            link, layout, arm_limits=lekiwi_limits, sleep=_NO_SLEEP
        )
    assert result.ok
    assert len(result.calibrations) == 6
    assert all(c.method == "touch_off" for c in result.calibrations)


def test_missing_servo_warns_and_skips(lekiwi_limits):
    layout = _layout.SO101
    link = _setup.simulated_link(layout, lekiwi_limits)
    del link._servos[3]  # elbow_flex falls off the bus
    with link:
        result = _setup.run_setup(
            link, layout, arm_limits=lekiwi_limits, sleep=_NO_SLEEP
        )
    assert not result.ok
    assert 3 in result.discovery.missing_ids
    assert any("elbow_flex" in w for w in result.warnings)
    assert all(c.servo_id != 3 for c in result.calibrations)


def test_scale_drift_warns(lekiwi_limits):
    """A servo whose stops are too close (a loose horn) trips the self-check."""
    layout = _layout.SO101
    link = _setup.simulated_link(layout, lekiwi_limits)
    # Squeeze shoulder_pan's stops to ~70% of the true span.
    servo = link._servos[1]
    mid = 0.5 * (servo.stop_lo + servo.stop_hi)
    half = 0.35 * (servo.stop_hi - servo.stop_lo)
    servo.stop_lo, servo.stop_hi = mid - half, mid + half
    servo.position = mid
    with link:
        result = _setup.run_setup(
            link, layout, arm_limits=lekiwi_limits, sleep=_NO_SLEEP
        )
    assert any("scale" in w and "shoulder_pan" in w for w in result.warnings)


def test_single_pose_mode(lekiwi_limits):
    layout = _layout.SO101
    link = _setup.simulated_link(layout, lekiwi_limits)
    with link:
        result = _setup.run_setup(
            link,
            layout,
            arm_limits=lekiwi_limits,
            mode="single-pose",
            sleep=_NO_SLEEP,
        )
    assert all(c.method == "single_pose" for c in result.calibrations)


def test_health_flags_low_voltage_and_heat():
    link = LoopbackLink()
    link.add_servo(1, position=2048.0, voltage_v=9.2, temperature_c=30.0)
    link.add_servo(2, position=2048.0, voltage_v=12.0, temperature_c=70.0)
    report = _setup.health_check(link, (1, 2))
    assert not report.ok
    assert any("low voltage" in w for w in report.warnings)
    assert any("hot" in w for w in report.warnings)


def test_discovery_reports_unexpected():
    layout = _layout.SO101
    link = LoopbackLink()
    for sid in (1, 2, 3, 4, 5, 6, 99):  # 99 is a stray
        link.add_servo(sid, position=2048.0)
    report = _setup.discover(link, layout)
    assert report.unexpected_ids == (99,)
    assert not report.ok


def test_native_artifact_roundtrips(tmp_path, lekiwi_limits):
    layout = _layout.LEKIWI
    link = _setup.simulated_link(layout, lekiwi_limits)
    with link:
        result = _setup.run_setup(
            link,
            layout,
            arm_limits=lekiwi_limits,
            handle="deadbeef",
            sleep=_NO_SLEEP,
        )
    path = _writer.write_native(tmp_path / "calibration.json", result)
    loaded = json.loads(path.read_text())
    assert loaded["robot"] == "lekiwi"
    assert loaded["robot_handle"] == "deadbeef"
    assert len(loaded["calibrations"]) == 9
    assert loaded["ok"] is True


def test_lerobot_artifact_shape(tmp_path, lekiwi_limits):
    layout = _layout.SO101
    link = _setup.simulated_link(layout, lekiwi_limits)
    with link:
        result = _setup.run_setup(
            link, layout, arm_limits=lekiwi_limits, sleep=_NO_SLEEP
        )
    d = _writer.lerobot_calibration_dict(result.calibrations)
    assert set(d) == {s.name for s in layout.arm_joints}
    entry = d["shoulder_pan"]
    assert set(entry) == {"id", "drive_mode", "homing_offset", "range_min", "range_max"}
    assert entry["id"] == 1
    assert entry["range_min"] < entry["range_max"]


def test_cli_simulate_smoke(tmp_path):
    from jaxterity.cli import main

    rc = main(["setup", "--robot", "lekiwi", "--simulate", "--out", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "calibration.json").exists()
    assert (tmp_path / "lekiwi_lerobot.json").exists()
