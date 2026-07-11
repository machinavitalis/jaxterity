# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""LeRobot calibration interop + the ±2047 homing-offset overflow fix.

The headline test is the regression for HuggingFace lerobot issue #3193: a
reference count near the encoder edge makes LeRobot's naive ``2048 - mid`` exceed
the signed 11-bit register and crash; our converter must instead emit a writable,
wrapped offset for the same input.
"""

from __future__ import annotations

import pytest

from jaxterity.calibrate.lerobot_format import (
    homing_offset_for,
    homing_offset_overflows,
    to_lerobot_calibration,
    to_lerobot_entry,
    wrap_signed,
)
from jaxterity.calibrate.types import JointCalibration

RES = 4096
LIMIT = RES // 2 - 1  # 2047


def _cal(joint, sid, *, offset, rmin=None, rmax=None, sign=1):
    return JointCalibration(
        joint=joint,
        servo_id=sid,
        sign=sign,
        ticks_per_rad=651.9,
        offset_ticks=offset,
        range_min_ticks=rmin,
        range_max_ticks=rmax,
    )


@pytest.mark.parametrize("v", [0, 1, 2047, 2048, 2049, 4095, 4096, -1, 6000, -6000])
def test_wrap_signed_in_range(v):
    w = wrap_signed(v, RES)
    assert -RES // 2 <= w < RES // 2


def test_emitted_homing_offset_never_overflows():
    # Reference counts spanning the whole encoder, including the edges that crash
    # stock LeRobot. Our offset must always be writable.
    for mid in range(0, RES, 37):
        off = homing_offset_for(mid, RES)
        assert -LIMIT <= off <= LIMIT, f"mid={mid} -> {off} out of register range"


def test_reproduces_issue_3193_then_fixes_it():
    # A continuous joint whose fitted zero sits at the encoder boundary: the
    # naive centring correction 2048-mid exceeds the ±2047 register and stock
    # LeRobot crashes ("Magnitude ... exceeds 2047").
    mid = 0.0
    assert homing_offset_overflows(mid, RES)  # stock LeRobot would crash here
    off = homing_offset_for(mid, RES)
    assert -LIMIT <= off <= LIMIT  # ours stays writable
    # same physical centring to within the single clamped count
    assert abs(wrap_signed(int(round(RES // 2 - mid)), RES) - off) <= 1


def test_entry_shape_and_drive_mode():
    e = to_lerobot_entry(_cal("j", 3, offset=2000, rmin=900, rmax=3100, sign=-1))
    assert set(e) == {"id", "drive_mode", "homing_offset", "range_min", "range_max"}
    assert e["id"] == 3
    assert e["drive_mode"] == 1  # sign -1
    assert e["range_min"] < e["range_max"]


def test_continuous_joint_gets_full_turn_range():
    # No measured stops (a continuous wrist_roll / wheel) — the joint LeRobot's
    # own calibration mishandles. We give it a full-turn range and a valid offset.
    e = to_lerobot_entry(_cal("wrist_roll", 5, offset=1934, rmin=None, rmax=None))
    assert e["range_min"] == 0 and e["range_max"] == RES - 1
    assert -LIMIT <= e["homing_offset"] <= LIMIT


def test_full_mapping_keys_and_safety():
    cals = [
        _cal("shoulder_pan", 1, offset=3035, rmin=1784, rmax=4095),  # edge case
        _cal("gripper", 6, offset=1934, rmin=None, rmax=None),  # continuous
    ]
    d = to_lerobot_calibration(cals)
    assert set(d) == {"shoulder_pan", "gripper"}
    for e in d.values():
        assert -LIMIT <= e["homing_offset"] <= LIMIT
