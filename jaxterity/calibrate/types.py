# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Plain-data records produced by the first-run setup routine.

This module owns the small, JAX-free dataclasses the ``jaxterity setup`` flow
passes around and serialises: a per-servo :class:`ServoStatus` snapshot, a
per-joint :class:`JointCalibration` (the affine raw-tick ↔ radian map), and the
:class:`DiscoveryReport` / :class:`HealthReport` / :class:`SetupResult`
aggregates the CLI prints and writes to disk.

These are deliberately not ``equinox.Module`` PyTrees: they are bookkeeping for
a hardware bring-up procedure, not identifiable simulation parameters, so they
carry no JAX leaves and import nothing heavy (the routine runs on a Raspberry
Pi next to the servo bus). The affine map a :class:`JointCalibration` holds is a
*kinematic* offset calibration — distinct from the *dynamic* parameters
(mass/inertia/damping) that :class:`jaxterity.sysid.SysidResult` fits.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class ServoStatus:
    """A one-shot health snapshot of a single smart servo.

    Args
    ----
    servo_id : int
        The servo's bus ID.
    position_ticks : int
        Present position, raw encoder counts (0 .. ``resolution - 1``).
    voltage_v : float
        Bus voltage at the servo, volts.
    temperature_c : float
        Winding/case temperature, degrees Celsius.
    load : float
        Present load as a signed fraction of stall, in ``[-1, 1]`` (sign is the
        torque direction; magnitude near 1 means the servo is straining).
    moving : bool
        True if the servo reports itself in motion.
    """

    servo_id: int
    position_ticks: int
    voltage_v: float
    temperature_c: float
    load: float
    moving: bool


@dataclasses.dataclass(frozen=True)
class JointCalibration:
    """The affine raw-tick ↔ radian map for one joint, plus how it was found.

    The map is ``ticks = offset_ticks + sign * ticks_per_rad * q`` (so
    ``q = sign * (ticks - offset_ticks) / ticks_per_rad``), where ``q`` is the
    URDF joint angle in radians. ``offset_ticks`` is the encoder reading at
    ``q = 0``; ``sign`` (+1/-1) is the encoder direction relative to the URDF
    joint axis; ``ticks_per_rad`` is the servo's resolution (a datasheet
    constant). ``range_min_ticks`` / ``range_max_ticks`` are the two mechanical
    hard-stop readings (when found by touch-off), and ``measured_ticks_per_rad``
    is the scale implied by those stops — a self-check against the nominal.

    Args
    ----
    joint : str
        URDF joint name (e.g. ``"shoulder_pan"``).
    servo_id : int
        The servo driving this joint.
    sign : int
        Encoder direction relative to the URDF axis (+1 or -1).
    ticks_per_rad : float
        Nominal encoder resolution, counts per radian.
    offset_ticks : float
        Encoder reading at ``q = 0`` rad.
    range_min_ticks, range_max_ticks : int | None
        The two hard-stop readings (sorted), or ``None`` for the single-pose
        method and for continuous (wheel) joints.
    measured_ticks_per_rad : float | None
        Scale implied by the two stops, or ``None`` when not measured.
    method : str
        How the offset was obtained: ``"touch_off"``, ``"single_pose"``, or
        ``"wheel_zero"``.
    """

    joint: str
    servo_id: int
    sign: int
    ticks_per_rad: float
    offset_ticks: float
    range_min_ticks: int | None = None
    range_max_ticks: int | None = None
    measured_ticks_per_rad: float | None = None
    method: str = "touch_off"

    def ticks_to_rad(self, ticks: float) -> float:
        """Convert a raw encoder reading to the URDF joint angle.

        Args
        ----
        ticks : float
            Raw encoder counts.

        Returns
        -------
        float
            Joint angle ``q`` in radians.

        Units
        -----
        Input in encoder counts; output in radians.
        """
        return self.sign * (ticks - self.offset_ticks) / self.ticks_per_rad

    def rad_to_ticks(self, q: float) -> float:
        """Convert a URDF joint angle to a raw encoder command.

        Args
        ----
        q : float
            Joint angle in radians.

        Returns
        -------
        float
            Raw encoder counts.

        Units
        -----
        Input in radians; output in encoder counts.
        """
        return self.offset_ticks + self.sign * self.ticks_per_rad * q

    def to_dict(self) -> dict:
        """A JSON-serialisable view of this calibration (native artifact)."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class DiscoveryReport:
    """The result of scanning the bus and matching it to the expected layout.

    Args
    ----
    found_ids : tuple[int, ...]
        Servo IDs that answered the scan (sorted).
    expected_ids : tuple[int, ...]
        Servo IDs the layout expects (sorted).
    missing_ids : tuple[int, ...]
        Expected but absent — almost always a wiring/power fault.
    unexpected_ids : tuple[int, ...]
        Present but not in the layout — a stray or mis-assigned ID.
    """

    found_ids: tuple[int, ...]
    expected_ids: tuple[int, ...]
    missing_ids: tuple[int, ...]
    unexpected_ids: tuple[int, ...]

    @property
    def ok(self) -> bool:
        """True when the bus exactly matches the expected layout."""
        return not self.missing_ids and not self.unexpected_ids


@dataclasses.dataclass(frozen=True)
class HealthReport:
    """Per-servo status plus any threshold violations found.

    Args
    ----
    statuses : tuple[ServoStatus, ...]
        One snapshot per responding servo.
    warnings : tuple[str, ...]
        Human-readable threshold violations (low voltage, hot servo, …).
    """

    statuses: tuple[ServoStatus, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """True when no health warnings were raised."""
        return not self.warnings


@dataclasses.dataclass(frozen=True)
class SetupResult:
    """Everything the first-run setup produced, ready to print or serialise.

    Args
    ----
    robot_name : str
        The layout/robot this run targeted (e.g. ``"lekiwi"``).
    discovery : DiscoveryReport
        Bus-scan outcome.
    health : HealthReport
        Health-check outcome.
    calibrations : tuple[JointCalibration, ...]
        One per calibrated joint, in layout order.
    robot_handle : str | None
        The attestation handle of the robot the calibration is anchored to,
        for traceability (``None`` if the model was not loaded).
    warnings : tuple[str, ...]
        Aggregate non-fatal caveats (e.g. a joint whose measured scale drifted
        from nominal).
    """

    robot_name: str
    discovery: DiscoveryReport
    health: HealthReport
    calibrations: tuple[JointCalibration, ...]
    robot_handle: str | None = None
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """True when discovery, health, and calibration all passed clean."""
        return self.discovery.ok and self.health.ok and not self.warnings


__all__ = [
    "ServoStatus",
    "JointCalibration",
    "DiscoveryReport",
    "HealthReport",
    "SetupResult",
]
