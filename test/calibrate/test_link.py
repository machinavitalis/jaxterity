# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""The transport abstraction: LoopbackLink mechanics + FeetechLink guard.

The simulator is what the rest of the calibrate suite stands on, so these tests
pin its contract directly: scanning, torque gating, motion toward a goal,
clamping at a hard stop with a strain reading, and release. The real
:class:`FeetechLink` is hardware-only; here we only check it refuses to operate
before it is opened.
"""

from __future__ import annotations

from jaxterity.calibrate.link import FeetechLink, LoopbackLink


def test_scan_returns_sorted_ids():
    link = LoopbackLink()
    for sid in (6, 1, 3):
        link.add_servo(sid, position=2048.0)
    assert link.scan() == (1, 3, 6)


def test_no_motion_without_torque():
    link = LoopbackLink(sim_dt=0.1)
    link.add_servo(1, position=1000.0, stop_lo=0.0, stop_hi=4095.0)
    # Reading repeatedly must not move an un-torqued servo.
    for _ in range(5):
        assert link.read_position(1) == 1000


def test_moves_toward_goal_and_stops_at_hard_stop():
    link = LoopbackLink(sim_dt=0.1)
    link.add_servo(1, position=1000.0, stop_lo=0.0, stop_hi=1500.0)
    link.start_move(1, 4095, speed_ticks_per_s=400.0, max_load=0.3)
    last = link.read_position(1)
    for _ in range(200):
        pos = link.read_position(1)
        if pos == last and link.read_status(1).load != 0.0:
            break
        last = pos
    status = link.read_status(1)
    assert status.position_ticks == 1500  # seated against the hard stop
    assert abs(status.load) >= 0.3 * 0.8  # straining there
    assert not status.moving


def test_set_torque_off_releases_load():
    link = LoopbackLink(sim_dt=0.1)
    link.add_servo(1, position=100.0, stop_lo=0.0, stop_hi=200.0)
    link.start_move(1, 4095, speed_ticks_per_s=400.0, max_load=0.3)
    for _ in range(50):
        link.read_position(1)
    link.set_torque(1, False)
    assert link.read_status(1).load == 0.0


def test_feetech_link_refuses_before_open():
    link = FeetechLink("/dev/null", (1, 2))
    import pytest

    with pytest.raises(RuntimeError, match="not open"):
        link.read_position(1)
