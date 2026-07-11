# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""The first-run bring-up routine behind ``jaxterity setup``.

This orchestrates the "magic" you run once on a new SO-101 / LeKiwi: scan the
servo bus and match it to the expected layout (:func:`discover`), read each
servo's health (:func:`health_check`), then calibrate every joint's tick ↔
radian map (:func:`run_setup`) — hands-free touch-off for the motorised arm
joints, a recorded zero for the wheels. It anchors to the zoo URDF's joint
limits (:func:`load_arm_limits`), so the operator never has to hand-sweep a
joint the way model-free LeRobot calibration requires.

It is transport-agnostic: hand it a :class:`~jaxterity.calibrate.link.FeetechLink`
on real hardware or a :class:`~jaxterity.calibrate.link.LoopbackLink` (built by
:func:`simulated_link`) to dry-run the whole flow with no robot attached.
"""

from __future__ import annotations

import time
from typing import Callable, Mapping

from . import offset as _offset
from .layout import RobotLayout
from .link import ActiveLink, LoopbackLink
from .types import (
    DiscoveryReport,
    HealthReport,
    JointCalibration,
    ServoStatus,
    SetupResult,
)

#: Below this bus voltage, flag the servo (a sagging or disconnected supply).
DEFAULT_MIN_VOLTAGE = 10.0

#: Above this servo temperature (°C), flag it (warm before we even start).
DEFAULT_MAX_TEMPERATURE = 55.0


def discover(link: ActiveLink, layout: RobotLayout) -> DiscoveryReport:
    """Scan the bus and compare it to the layout's expected servo IDs.

    Args
    ----
    link : ActiveLink
        The open transport.
    layout : RobotLayout
        The expected layout.

    Returns
    -------
    DiscoveryReport
        Found / expected / missing / unexpected IDs.
    """
    found = tuple(link.scan())
    expected = layout.expected_ids
    found_set, expected_set = set(found), set(expected)
    return DiscoveryReport(
        found_ids=tuple(sorted(found_set)),
        expected_ids=expected,
        missing_ids=tuple(sorted(expected_set - found_set)),
        unexpected_ids=tuple(sorted(found_set - expected_set)),
    )


def health_check(
    link: ActiveLink,
    servo_ids: tuple[int, ...],
    *,
    min_voltage: float = DEFAULT_MIN_VOLTAGE,
    max_temperature: float = DEFAULT_MAX_TEMPERATURE,
) -> HealthReport:
    """Read each servo's status and flag voltage/temperature problems.

    Args
    ----
    link : ActiveLink
        The open transport.
    servo_ids : tuple[int, ...]
        The IDs to poll (typically ``discovery.found_ids``).
    min_voltage : float
        Warn below this bus voltage, volts.
    max_temperature : float
        Warn above this temperature, °C.

    Returns
    -------
    HealthReport
        One :class:`ServoStatus` per servo plus any threshold warnings.

    Units
    -----
    Voltage in volts, temperature in degrees Celsius.
    """
    statuses: list[ServoStatus] = []
    warnings: list[str] = []
    for sid in servo_ids:
        status = link.read_status(sid)
        statuses.append(status)
        if status.voltage_v < min_voltage:
            warnings.append(
                f"servo {sid}: low voltage {status.voltage_v:.1f} V "
                f"(< {min_voltage:.1f} V) — check the battery/supply."
            )
        if status.temperature_c > max_temperature:
            warnings.append(
                f"servo {sid}: hot {status.temperature_c:.0f} °C "
                f"(> {max_temperature:.0f} °C) — let it cool before calibrating."
            )
    return HealthReport(statuses=tuple(statuses), warnings=tuple(warnings))


def load_arm_limits(layout: RobotLayout) -> dict[str, tuple[float, float]]:
    """Read each arm joint's ``(lower, upper)`` angular limits from the zoo URDF.

    Args
    ----
    layout : RobotLayout
        The layout whose ``urdf_robot`` supplies the model.

    Returns
    -------
    dict[str, tuple[float, float]]
        ``{joint_name: (lower, upper)}`` in radians, for the arm joints.

    Raises
    ------
    KeyError
        If a layout arm joint is absent from the URDF (a layout/URDF mismatch).

    Units
    -----
    Radians.
    """
    from .. import zoo

    robot = zoo.load(layout.urdf_robot)
    joint_map = robot.joint_map
    limits: dict[str, tuple[float, float]] = {}
    for spec in layout.arm_joints:
        joint = joint_map[spec.name]
        limits[spec.name] = (float(joint.limits.lower), float(joint.limits.upper))
    return limits


def robot_handle(layout: RobotLayout) -> str:
    """Return the attestation handle of the layout's anchoring robot model.

    Args
    ----
    layout : RobotLayout
        The layout whose ``urdf_robot`` supplies the model.

    Returns
    -------
    str
        The robot's attestation handle, recorded in the calibration artifact
        so a calibration is traceable to the exact model it was anchored to.
    """
    from .. import zoo

    return zoo.load(layout.urdf_robot).attestation_handle


def simulated_link(
    layout: RobotLayout,
    arm_limits: Mapping[str, tuple[float, float]],
    *,
    sim_dt: float = 0.08,
) -> LoopbackLink:
    """Build a :class:`LoopbackLink` seeded from a layout and the URDF limits.

    Plants a deterministic hidden offset on every servo (and the two hard stops
    implied by the joint limits on the arm servos), so a dry run of
    :func:`run_setup` exercises the real touch-off code path and recovers the
    planted offsets. Used by ``jaxterity setup --simulate`` and the test suite.

    Args
    ----
    layout : RobotLayout
        The layout to simulate.
    arm_limits : Mapping[str, tuple[float, float]]
        ``{joint_name: (lower, upper)}`` radians for the arm joints.
    sim_dt : float
        Simulated seconds integrated per position read.

    Returns
    -------
    LoopbackLink
        A ready, populated simulator.

    Units
    -----
    ``arm_limits`` in radians.
    """
    link = LoopbackLink(sim_dt=sim_dt)
    for spec in layout.joints:
        # A small, per-servo offset near centre keeps both stops inside the
        # encoder span for every SO-101 joint while still being non-trivial.
        planted_offset = link.resolution / 2 + (spec.servo_id - 3) * 25
        if spec.kind == "arm":
            lower, upper = arm_limits[spec.name]
            t_lo = planted_offset + spec.sign * spec.ticks_per_rad * lower
            t_hi = planted_offset + spec.sign * spec.ticks_per_rad * upper
            stop_lo, stop_hi = sorted((t_lo, t_hi))
            link.add_servo(
                spec.servo_id,
                position=0.5 * (stop_lo + stop_hi),
                stop_lo=stop_lo,
                stop_hi=stop_hi,
            )
        else:  # continuous wheel: no stops
            link.add_servo(spec.servo_id, position=planted_offset)
    return link


def run_setup(
    link: ActiveLink,
    layout: RobotLayout,
    *,
    arm_limits: Mapping[str, tuple[float, float]],
    mode: str = "touch-off",
    reference_pose: Mapping[str, float] | None = None,
    handle: str | None = None,
    sleep: Callable[[float], None] = time.sleep,
    min_voltage: float = DEFAULT_MIN_VOLTAGE,
    max_temperature: float = DEFAULT_MAX_TEMPERATURE,
) -> SetupResult:
    """Run the full first-time bring-up: discover, health-check, calibrate.

    Scans the bus, reads health, then calibrates each present joint — arm joints
    by ``mode`` (``"touch-off"`` hands-free, or ``"single-pose"`` from a held
    reference), wheels by recording a zero. Joints whose servo is missing are
    skipped with a warning; a touch-off whose measured scale drifts from the
    datasheet adds a warning (a loose horn or wrong layout sign).

    Args
    ----
    link : ActiveLink
        The open transport (real or simulated). Not closed here — the caller
        owns its lifecycle.
    layout : RobotLayout
        The robot layout to bring up.
    arm_limits : Mapping[str, tuple[float, float]]
        ``{joint_name: (lower, upper)}`` radians (from :func:`load_arm_limits`).
    mode : str
        ``"touch-off"`` (default) or ``"single-pose"`` for the arm joints.
    reference_pose : Mapping[str, float] | None
        For ``single-pose`` mode, the held reference angle per joint, radians.
        Defaults to all zeros.
    handle : str | None
        The anchoring robot's attestation handle, recorded for traceability.
    sleep : Callable[[float], None]
        Sleep function (tests pass a no-op).
    min_voltage, max_temperature : float
        Health thresholds (see :func:`health_check`).

    Returns
    -------
    SetupResult
        The complete run, ready to print or serialise.

    Raises
    ------
    CalibrationError
        Propagated from a touch-off that never reaches a stop (e.g. an
        obstructed joint).

    Units
    -----
    ``arm_limits`` and ``reference_pose`` in radians; thresholds in volts / °C.
    """
    discovery = discover(link, layout)
    health = health_check(
        link,
        discovery.found_ids,
        min_voltage=min_voltage,
        max_temperature=max_temperature,
    )
    reference_pose = reference_pose or {}
    found = set(discovery.found_ids)

    calibrations: list[JointCalibration] = []
    warnings: list[str] = []
    for spec in layout.joints:
        if spec.servo_id not in found:
            warnings.append(
                f"joint {spec.name!r} (servo {spec.servo_id}) not on the bus — "
                "skipped. Re-seat its cable and rerun."
            )
            continue
        if spec.kind == "wheel":
            calibrations.append(_offset.wheel_zero(link, spec))
            continue
        if mode == "single-pose":
            q_ref = float(reference_pose.get(spec.name, 0.0))
            cal = _offset.single_pose(link, spec, q_ref)
        else:
            cal = _offset.touch_off(link, spec, arm_limits[spec.name], sleep=sleep)
            if cal.measured_ticks_per_rad is not None:
                drift = abs(cal.measured_ticks_per_rad / cal.ticks_per_rad - 1.0)
                if drift > _offset.SCALE_WARN_RTOL:
                    warnings.append(
                        f"joint {spec.name!r}: measured scale "
                        f"{cal.measured_ticks_per_rad:.1f} counts/rad differs "
                        f"{drift * 100:.0f}% from the datasheet "
                        f"{cal.ticks_per_rad:.1f} — check for a loose horn or a "
                        "wrong layout sign before trusting this joint."
                    )
        calibrations.append(cal)

    return SetupResult(
        robot_name=layout.name,
        discovery=discovery,
        health=health,
        calibrations=tuple(calibrations),
        robot_handle=handle,
        warnings=tuple(warnings),
    )


__all__ = [
    "DEFAULT_MIN_VOLTAGE",
    "DEFAULT_MAX_TEMPERATURE",
    "discover",
    "health_check",
    "load_arm_limits",
    "robot_handle",
    "simulated_link",
    "run_setup",
]
