# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Serialise a setup run to disk: the native artifact and a LeRobot file.

Two outputs, with a deliberate split of authority:

* **Native** (``calibration.json``) — Jaxterity's own, complete record: every
  :class:`~jaxterity.calibrate.types.JointCalibration` field, the discovery and
  health reports, and the anchoring robot's attestation handle. This is the
  source of truth and round-trips losslessly.
* **LeRobot** (``<id>.json``) — a best-effort translation into the per-motor
  calibration schema LeRobot's SO-101/LeKiwi teleop reads, so the same bring-up
  also drops you straight into the LeRobot workflow. LeRobot's schema has
  shifted across releases; this writer targets the per-motor
  ``{id, drive_mode, homing_offset, range_min, range_max}`` shape and flags that
  you should confirm it against your installed version.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from .types import JointCalibration, SetupResult

#: LeRobot centres a joint's travel on this raw count via its homing offset.
_LEROBOT_CENTER = 2048


def native_dict(result: SetupResult) -> dict:
    """Build the JSON-serialisable native artifact for a setup run.

    Args
    ----
    result : SetupResult
        The completed run.

    Returns
    -------
    dict
        A nested dict mirroring ``result`` (the authoritative on-disk form).
    """
    d = result.discovery
    h = result.health
    return {
        "schema": "jaxterity.calibrate/v1",
        "robot": result.robot_name,
        "robot_handle": result.robot_handle,
        "discovery": {
            "found_ids": list(d.found_ids),
            "expected_ids": list(d.expected_ids),
            "missing_ids": list(d.missing_ids),
            "unexpected_ids": list(d.unexpected_ids),
            "ok": d.ok,
        },
        "health": {
            "statuses": [
                {
                    "servo_id": s.servo_id,
                    "position_ticks": s.position_ticks,
                    "voltage_v": s.voltage_v,
                    "temperature_c": s.temperature_c,
                    "load": s.load,
                    "moving": s.moving,
                }
                for s in h.statuses
            ],
            "warnings": list(h.warnings),
            "ok": h.ok,
        },
        "calibrations": [c.to_dict() for c in result.calibrations],
        "warnings": list(result.warnings),
        "ok": result.ok,
    }


def write_native(path: str | Path, result: SetupResult) -> Path:
    """Write the native calibration artifact to ``path`` as JSON.

    Args
    ----
    path : str or Path
        Output file.
    result : SetupResult
        The completed run.

    Returns
    -------
    Path
        ``path`` as a :class:`pathlib.Path`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(native_dict(result), indent=2) + "\n")
    return path


def _lerobot_entry(cal: JointCalibration) -> dict:
    """One per-motor LeRobot calibration entry from a JointCalibration."""
    drive_mode = 0 if cal.sign >= 0 else 1
    if cal.range_min_ticks is not None and cal.range_max_ticks is not None:
        range_min = int(cal.range_min_ticks)
        range_max = int(cal.range_max_ticks)
        mid = 0.5 * (range_min + range_max)
    else:
        # No measured travel (single-pose / wheel): centre on the fitted zero
        # and leave the range at the full encoder span.
        range_min, range_max = 0, 0
        mid = cal.offset_ticks
    homing_offset = int(round(mid - _LEROBOT_CENTER))
    return {
        "id": cal.servo_id,
        "drive_mode": drive_mode,
        "homing_offset": homing_offset,
        "range_min": range_min,
        "range_max": range_max,
    }


def lerobot_calibration_dict(calibrations: Sequence[JointCalibration]) -> dict:
    """Translate calibrations into LeRobot's per-motor calibration mapping.

    Args
    ----
    calibrations : sequence of JointCalibration
        The fitted joints, in motor order.

    Returns
    -------
    dict
        ``{joint_name: {id, drive_mode, homing_offset, range_min, range_max}}``
        in LeRobot's calibration-file shape. Confirm against your ``lerobot``
        version before relying on it (see module docstring).
    """
    return {cal.joint: _lerobot_entry(cal) for cal in calibrations}


def write_lerobot(path: str | Path, calibrations: Sequence[JointCalibration]) -> Path:
    """Write a LeRobot-format per-motor calibration file to ``path``.

    Args
    ----
    path : str or Path
        Output file (LeRobot keys these by ``robot.id``, e.g. ``my_kiwi.json``).
    calibrations : sequence of JointCalibration
        The fitted joints.

    Returns
    -------
    Path
        ``path`` as a :class:`pathlib.Path`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lerobot_calibration_dict(calibrations), indent=2) + "\n")
    return path


__all__ = [
    "native_dict",
    "write_native",
    "lerobot_calibration_dict",
    "write_lerobot",
]
