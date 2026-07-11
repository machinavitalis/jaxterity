# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Offset-calibration math: touch-off, single-pose, wheel-zero.

These run entirely against :class:`LoopbackLink`, which plants a hidden true
offset and two hard stops per servo, so each test asserts the routine *recovers*
the planted ground truth. Tolerances are in encoder counts: this is integer
hardware bookkeeping, not a solver/dtype-dependent simulation quantity, so the
project tolerance table does not apply — a ±2-count budget covers the rounding
of the midpoint anchor.
"""

from __future__ import annotations

import math

import pytest

from jaxterity.calibrate.layout import STS3215_TICKS_PER_RAD, JointSpec
from jaxterity.calibrate.link import LoopbackLink
from jaxterity.calibrate.offset import single_pose, touch_off, wheel_zero
from jaxterity.errors import CalibrationError

#: Recovery budget in encoder counts (midpoint-anchor rounding).
TICK_TOL = 2.0

#: A no-op sleep so the poll loop runs at full speed in tests.
_NO_SLEEP = lambda _seconds: None  # noqa: E731


def _arm_link(
    spec: JointSpec, offset: float, limits: tuple[float, float]
) -> LoopbackLink:
    """A one-servo simulator with a planted offset and the implied hard stops."""
    lower, upper = limits
    t_lo = offset + spec.sign * spec.ticks_per_rad * lower
    t_hi = offset + spec.sign * spec.ticks_per_rad * upper
    stop_lo, stop_hi = sorted((t_lo, t_hi))
    link = LoopbackLink(sim_dt=0.08)
    link.add_servo(
        spec.servo_id,
        position=0.5 * (stop_lo + stop_hi),
        stop_lo=stop_lo,
        stop_hi=stop_hi,
    )
    return link


@pytest.mark.parametrize(
    "sign, limits, planted",
    [
        pytest.param(1, (-1.92, 1.92), 2048.0, id="symmetric-pos"),
        pytest.param(-1, (-1.92, 1.92), 2100.0, id="symmetric-neg"),
        pytest.param(1, (-0.174, 1.745), 2000.0, id="asymmetric-gripper"),
        pytest.param(1, (-2.74, 2.84), 2050.0, id="wide-wrist-roll"),
    ],
)
def test_touch_off_recovers_offset(sign, limits, planted):
    spec = JointSpec("j", 1, "arm", sign=sign)
    link = _arm_link(spec, planted, limits)
    cal = touch_off(link, spec, limits, sleep=_NO_SLEEP)
    assert cal.offset_ticks == pytest.approx(planted, abs=TICK_TOL)
    assert cal.method == "touch_off"
    # The measured scale matches the datasheet (no horn slip in the sim).
    assert cal.measured_ticks_per_rad == pytest.approx(STS3215_TICKS_PER_RAD, rel=0.02)
    # The fitted map round-trips q=0 back to the planted offset.
    assert cal.rad_to_ticks(0.0) == pytest.approx(planted, abs=TICK_TOL)


def test_touch_off_leaves_servo_limp():
    """After a touch-off the joint torque is released (safe to handle)."""
    spec = JointSpec("j", 1, "arm")
    limits = (-1.0, 1.0)
    link = _arm_link(spec, 2048.0, limits)
    touch_off(link, spec, limits, sleep=_NO_SLEEP)
    assert link.read_status(1).load == 0.0
    assert not link._servos[1].torque_on


def test_touch_off_rejects_degenerate_limits():
    spec = JointSpec("j", 1, "arm")
    link = _arm_link(spec, 2048.0, (-1.0, 1.0))
    with pytest.raises(CalibrationError, match="non-increasing"):
        touch_off(link, spec, (1.0, 1.0), sleep=_NO_SLEEP)


def test_touch_off_without_stop_raises():
    """A servo with no hard stops (a wheel) never stalls → a clear error."""
    spec = JointSpec("wheel", 7, "wheel")
    link = LoopbackLink(sim_dt=0.2)
    link.add_servo(7, position=2048.0)  # no stops
    with pytest.raises(CalibrationError, match="never reached a hard stop"):
        touch_off(link, spec, (-1.0, 1.0), sleep=_NO_SLEEP)


@pytest.mark.parametrize("q_ref", [0.0, 0.5, -1.2])
def test_single_pose_recovers_offset(q_ref):
    spec = JointSpec("j", 3, "arm", sign=-1)
    planted = 2010.0
    # Place the servo at the tick the planted map assigns to q_ref.
    pos = planted + spec.sign * spec.ticks_per_rad * q_ref
    link = LoopbackLink()
    link.add_servo(3, position=pos)
    cal = single_pose(link, spec, q_ref)
    assert cal.offset_ticks == pytest.approx(planted, abs=TICK_TOL)
    assert cal.method == "single_pose"
    assert cal.range_min_ticks is None  # one pose constrains only the offset


def test_wheel_zero_records_current_position():
    spec = JointSpec("wheel_left", 7, "wheel")
    link = LoopbackLink()
    link.add_servo(7, position=1234.0)
    cal = wheel_zero(link, spec)
    assert cal.offset_ticks == pytest.approx(1234.0)
    assert cal.method == "wheel_zero"
    assert cal.ticks_to_rad(1234.0) == pytest.approx(0.0)


def test_ticks_rad_roundtrip():
    spec = JointSpec("j", 1, "arm", sign=-1)
    cal = touch_off(
        _arm_link(spec, 2048.0, (-1.5, 1.5)), spec, (-1.5, 1.5), sleep=_NO_SLEEP
    )
    for q in (-1.0, 0.0, 0.7):
        assert cal.ticks_to_rad(cal.rad_to_ticks(q)) == pytest.approx(q, abs=1e-6)
    # Direction is honoured: with sign=-1, increasing angle means fewer ticks.
    assert cal.rad_to_ticks(1.0) < cal.rad_to_ticks(0.0)
    assert math.isclose(cal.ticks_per_rad, STS3215_TICKS_PER_RAD)
