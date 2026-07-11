# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""LeRobot per-motor calibration interop — with the ±2047 overflow fixed.

LeRobot stores a Feetech motor's calibration as ``{homing_offset, range_min,
range_max, drive_mode}`` and writes ``homing_offset`` into the servo's signed
11-bit *Position Correction* register, whose range is ``[-2047, 2047]``. Its
``set_half_turn_homings`` computes ``homing_offset = 2048 - present_position`` so
a reference pose reads mid-scale — which **overflows and crashes** when the
encoder zero is far from centre (``"Magnitude 2073 exceeds 2047"``; HuggingFace
lerobot issue #3193), and mishandles continuous joints that have no end stops.

This module emits the same on-disk shape so a calibration we produce drops
straight into LeRobot, but **wraps the homing offset into range** instead of
overflowing (correct for a single-turn encoder, where the correction is modular)
and handles continuous joints explicitly. It is the antidote to the bug, not a
re-implementation of the bug.
"""

from __future__ import annotations

from typing import Sequence

from .types import JointCalibration


def wrap_signed(value: int, resolution: int) -> int:
    """Wrap ``value`` into the signed half-open range ``[-res/2, res/2)``.

    For a single-turn encoder the Position-Correction register is modular, so a
    raw offset of ``2048 - mid`` that lands outside ``±2047`` is not an error —
    it is the same physical correction one turn away. Wrapping (not clamping)
    preserves the mapping while keeping the value writable.

    Args
    ----
    value : int
        The raw (possibly out-of-range) offset in encoder counts.
    resolution : int
        Encoder counts per revolution.

    Returns
    -------
    int
        The equivalent offset in ``[-resolution/2, resolution/2)``.
    """
    half = resolution // 2
    return ((value + half) % resolution) - half


def homing_offset_for(mid_tick: float, resolution: int = 4096) -> int:
    """The LeRobot homing offset that centres ``mid_tick`` at mid-scale.

    Uses LeRobot's convention (the servo *adds* the correction to its raw
    reading, so a reference reads ``resolution/2``), then wraps into the signed
    11-bit register range so it never overflows.

    Args
    ----
    mid_tick : float
        The raw count to map to mid-scale (typically the middle of travel).
    resolution : int
        Encoder counts per revolution (default 4096).

    Returns
    -------
    int
        A homing offset in ``[-(resolution/2 - 1), resolution/2 - 1]`` — always
        writable. The single unreachable count at the very extreme is clamped
        (≤ 1 count ≈ 0.09° error), never an overflow.
    """
    limit = resolution // 2 - 1  # the register holds magnitude ≤ 2047
    wrapped = wrap_signed(int(round(resolution // 2 - mid_tick)), resolution)
    return max(-limit, min(limit, wrapped))


def to_lerobot_entry(cal: JointCalibration, *, resolution: int = 4096) -> dict:
    """Translate one :class:`JointCalibration` into a LeRobot motor entry.

    Args
    ----
    cal : JointCalibration
        Our fitted calibration for the joint.
    resolution : int
        Encoder counts per revolution.

    Returns
    -------
    dict
        ``{id, drive_mode, homing_offset, range_min, range_max}`` in LeRobot's
        shape. ``homing_offset`` is guaranteed in ``[-resolution/2,
        resolution/2)`` (the bug LeRobot hits cannot occur). Continuous joints
        (no measured range) get a full-turn range and a centred homing offset.
    """
    if cal.range_min_ticks is not None and cal.range_max_ticks is not None:
        rmin, rmax = sorted((int(cal.range_min_ticks), int(cal.range_max_ticks)))
        mid = 0.5 * (rmin + rmax)
    else:  # continuous / no usable stop: centre on the fitted zero, full turn
        rmin, rmax = 0, resolution - 1
        mid = cal.offset_ticks
    return {
        "id": cal.servo_id,
        "drive_mode": 0 if cal.sign >= 0 else 1,
        "homing_offset": homing_offset_for(mid, resolution),
        "range_min": rmin,
        "range_max": rmax,
    }


def to_lerobot_calibration(
    calibrations: Sequence[JointCalibration], *, resolution: int = 4096
) -> dict:
    """Build the full LeRobot per-motor calibration mapping.

    Args
    ----
    calibrations : sequence of JointCalibration
        The fitted joints, in motor order.
    resolution : int
        Encoder counts per revolution.

    Returns
    -------
    dict
        ``{joint_name: entry}`` ready to serialise as a LeRobot calibration file.
        Every ``homing_offset`` is in the writable signed-11-bit range.
    """
    return {c.joint: to_lerobot_entry(c, resolution=resolution) for c in calibrations}


def homing_offset_overflows(mid_tick: float, resolution: int = 4096) -> bool:
    """True if LeRobot's *naive* ``2048 - mid`` would overflow the ±2047 register.

    Diagnoses the issue-#3193 condition for a given reference count — useful for
    a regression test or for warning a user calibrating with stock LeRobot.

    Args
    ----
    mid_tick : float
        The reference count LeRobot would centre.
    resolution : int
        Encoder counts per revolution.

    Returns
    -------
    bool
        Whether the unwrapped offset leaves ``[-(resolution/2 - 1), resolution/2
        - 1]``.
    """
    naive = resolution // 2 - mid_tick
    limit = resolution // 2 - 1
    return naive < -limit or naive > limit


__all__ = [
    "wrap_signed",
    "homing_offset_for",
    "to_lerobot_entry",
    "to_lerobot_calibration",
    "homing_offset_overflows",
]
