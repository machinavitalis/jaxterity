# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Model-aware offset calibration: find each joint's tick ↔ radian map.

This is the heart of the "magic" setup: because Jaxterity already knows the
robot (the zoo URDF gives every joint's angular limits and the layout gives the
servo resolution), calibration does not need the operator to sweep every joint
by hand the way model-free LeRobot does. It only has to pin one unknown per
joint — the encoder *offset* (which raw count corresponds to ``q = 0``).

Two strategies, both producing a :class:`~jaxterity.calibrate.types.JointCalibration`:

* :func:`touch_off` — hands-free. Drive the motorised joint gently (current
  limited) to each mechanical hard stop; the stops sit at the URDF limits, so
  the two readings fix the offset and *measure* the scale as a self-check. This
  is the SO-101/LeKiwi default — both arms are motorised.
* :func:`single_pose` — for a joint that cannot drive itself: read one known
  reference pose and solve the offset from the nominal scale and direction.

Both leave the per-model ``sign`` (encoder direction vs URDF axis) to the
layout: two hard stops fix the offset and the scale magnitude but *not* the
direction, so direction is a one-time per-model constant (see
:mod:`jaxterity.calibrate.layout`). The scale comes from the datasheet
(``ticks_per_rad``); the touch-off's *measured* scale is reported only to flag a
loose horn or a wrong layout, never to silently override the datasheet.
"""

from __future__ import annotations

import time
from typing import Callable

from ..errors import CalibrationError
from .layout import JointSpec
from .link import ActiveLink
from .types import JointCalibration

#: Touch-off move speed (counts/second). Deliberately slow — a hard stop should
#: be met as a nudge, not an impact.
TOUCH_OFF_SPEED_TICKS_PER_S = 400.0

#: Touch-off torque ceiling as a fraction of stall. Low enough to protect the
#: STS3215's plastic gears when pressing into a stop.
TOUCH_OFF_MAX_LOAD = 0.30

#: Seconds between polls while waiting for a stall.
POLL_DT = 0.02

#: Give up looking for a stop after this many polls (≈ 16 s at POLL_DT). Sized
#: to clear the longest SO-101 joint (wrist_roll, ~5.6 rad) at the gentle
#: touch-off speed with margin.
MAX_POLLS = 800

#: A move counts as plateaued when consecutive readings stay within this many
#: counts of each other.
PLATEAU_TICKS = 2

#: Number of consecutive plateaued reads (with high load) that confirm a stop.
PLATEAU_COUNT = 5

#: Fraction of the load ceiling that confirms the servo is pressing a stop.
STALL_FRACTION = 0.8

#: Relative tolerance between measured and nominal scale before warning.
SCALE_WARN_RTOL = 0.10


def _drive_to_stop(
    link: ActiveLink,
    servo_id: int,
    direction: int,
    *,
    sleep: Callable[[float], None],
) -> int:
    """Drive ``servo_id`` to a mechanical hard stop and return its position.

    Commands a gentle, current-limited move past the end of travel in
    ``direction`` (+1 toward the maximum count, -1 toward zero) and polls until
    the servo stalls (stops advancing while reporting load near the ceiling).

    Args
    ----
    link : ActiveLink
        The open transport.
    servo_id : int
        Target servo.
    direction : int
        +1 to seek the maximum-count stop, -1 to seek the zero-count stop.
    sleep : Callable[[float], None]
        Sleep function (injected so tests can pass a no-op).

    Returns
    -------
    int
        The encoder count at the stop.

    Raises
    ------
    CalibrationError
        If no stall is detected within :data:`MAX_POLLS` polls.

    Units
    -----
    Returns encoder counts.
    """
    goal = (link.resolution - 1) if direction > 0 else 0
    link.start_move(
        servo_id,
        goal,
        speed_ticks_per_s=TOUCH_OFF_SPEED_TICKS_PER_S,
        max_load=TOUCH_OFF_MAX_LOAD,
    )
    last = link.read_position(servo_id)
    stable = 0
    for _ in range(MAX_POLLS):
        sleep(POLL_DT)
        pos = link.read_position(servo_id)
        status = link.read_status(servo_id)
        plateaued = abs(pos - last) <= PLATEAU_TICKS
        straining = abs(status.load) >= STALL_FRACTION * TOUCH_OFF_MAX_LOAD
        if plateaued and straining:
            stable += 1
            if stable >= PLATEAU_COUNT:
                link.set_torque(servo_id, False)
                return pos
        else:
            stable = 0
        last = pos
    link.set_torque(servo_id, False)
    raise CalibrationError(
        f"Touch-off for servo {servo_id} never reached a hard stop in "
        f"direction {direction:+d} ({MAX_POLLS} polls). The joint may be "
        "obstructed, the load ceiling may be too low to seat the stop, or the "
        "servo is not moving (check power and torque). Fix: clear the joint's "
        "path, confirm the servo answers `scan`, and retry."
    )


def touch_off(
    link: ActiveLink,
    spec: JointSpec,
    limits: tuple[float, float],
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> JointCalibration:
    """Calibrate one arm joint hands-free by touching both hard stops.

    Drives the joint gently to each mechanical stop, reads the encoder there,
    and — knowing from the URDF that those stops sit at ``limits`` — solves the
    offset and cross-checks the scale. The per-model ``sign`` from ``spec``
    resolves which stop is the lower-angle one. The offset is anchored at the
    midpoint of travel so any residual scale error is split evenly across the
    range.

    Args
    ----
    link : ActiveLink
        The open transport.
    spec : JointSpec
        The joint's wiring/convention (servo ID, sign, ticks_per_rad).
    limits : tuple[float, float]
        ``(lower, upper)`` angular limits from the URDF, radians, ``lower < upper``.
    sleep : Callable[[float], None]
        Sleep function (default :func:`time.sleep`; tests pass a no-op).

    Returns
    -------
    JointCalibration
        The fitted affine map, with the two stop readings and the measured
        scale recorded for the self-check.

    Raises
    ------
    CalibrationError
        If ``limits`` is degenerate, or a stop is never reached.

    Units
    -----
    ``limits`` in radians; readings and the returned offsets in encoder counts.
    """
    lower, upper = limits
    if not upper > lower:
        raise CalibrationError(
            f"touch_off: joint {spec.name!r} has non-increasing limits "
            f"(lower={lower}, upper={upper}). Fix: check the URDF <limit> tag."
        )

    t_at_min = _drive_to_stop(link, spec.servo_id, -1, sleep=sleep)
    t_at_max = _drive_to_stop(link, spec.servo_id, +1, sleep=sleep)
    range_min, range_max = sorted((t_at_min, t_at_max))

    span_ticks = range_max - range_min
    span_rad = upper - lower
    measured_tpr = span_ticks / span_rad if span_rad else float("nan")

    # Midpoint anchor: the mid-travel count maps to the mid-travel angle.
    mid_tick = 0.5 * (range_min + range_max)
    mid_angle = 0.5 * (lower + upper)
    offset = mid_tick - spec.sign * spec.ticks_per_rad * mid_angle

    return JointCalibration(
        joint=spec.name,
        servo_id=spec.servo_id,
        sign=spec.sign,
        ticks_per_rad=spec.ticks_per_rad,
        offset_ticks=offset,
        range_min_ticks=range_min,
        range_max_ticks=range_max,
        measured_ticks_per_rad=measured_tpr,
        method="touch_off",
    )


def single_pose(
    link: ActiveLink,
    spec: JointSpec,
    q_ref: float,
    *,
    samples: int = 5,
) -> JointCalibration:
    """Calibrate one joint from a single known reference pose.

    For a joint that cannot drive itself to its stops (e.g. a purpose-built
    passive leader). The operator holds the joint at the known angle ``q_ref``;
    this averages a few position reads and solves the offset from the nominal
    scale and direction. It cannot measure the scale, so no self-check is
    available — trust in the nominal ``ticks_per_rad`` and ``sign`` is required.

    Args
    ----
    link : ActiveLink
        The open transport.
    spec : JointSpec
        The joint's wiring/convention.
    q_ref : float
        The known reference angle the joint is held at, radians.
    samples : int
        Number of position reads to average (default 5).

    Returns
    -------
    JointCalibration
        The fitted affine map (``range_*`` and ``measured_ticks_per_rad`` are
        ``None`` — a single pose constrains only the offset).

    Units
    -----
    ``q_ref`` in radians; the returned offset in encoder counts.
    """
    if samples < 1:
        raise CalibrationError(
            f"single_pose: samples must be ≥ 1, got {samples}. Fix: pass a "
            "positive count (5 is a good default)."
        )
    reads = [link.read_position(spec.servo_id) for _ in range(samples)]
    t_ref = sum(reads) / len(reads)
    offset = t_ref - spec.sign * spec.ticks_per_rad * q_ref
    return JointCalibration(
        joint=spec.name,
        servo_id=spec.servo_id,
        sign=spec.sign,
        ticks_per_rad=spec.ticks_per_rad,
        offset_ticks=offset,
        method="single_pose",
    )


def wheel_zero(link: ActiveLink, spec: JointSpec) -> JointCalibration:
    """Record a continuous (wheel) servo's current count as its zero.

    A continuous-rotation drive servo has no hard stops and is velocity- not
    position-controlled, so there is no offset to find in the arm-joint sense.
    This captures the present count as a nominal zero and carries the layout's
    direction convention forward, so a controller has a consistent sign per
    wheel. Confirm the direction by commanding a slow forward roll on first run.

    Args
    ----
    link : ActiveLink
        The open transport.
    spec : JointSpec
        The wheel's wiring/convention.

    Returns
    -------
    JointCalibration
        An offset-at-current-position map, ``method="wheel_zero"``.

    Units
    -----
    The returned offset is in encoder counts.
    """
    t = link.read_position(spec.servo_id)
    return JointCalibration(
        joint=spec.name,
        servo_id=spec.servo_id,
        sign=spec.sign,
        ticks_per_rad=spec.ticks_per_rad,
        offset_ticks=float(t),
        method="wheel_zero",
    )


__all__ = ["touch_off", "single_pose", "wheel_zero"]
