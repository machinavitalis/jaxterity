# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Typed telemetry schema (ADR-009).

Every sysid recipe consumes telemetry, so the schema is fixed here up front
before any recipe can diverge on column names, units, frames, or sample rates.
Telemetry is typed Pydantic v2 models; each data channel declares its SI unit
and (where applicable) its frame via a :class:`ChannelSpec`. Time is a
strictly-increasing ``float64`` array in seconds and the sample rate is
explicit, never inferred.

Construction validates structure (shapes, dtype, monotonic time) and raises
:class:`~jaxterity.errors.SchemaError` on violation; :meth:`Telemetry.validate`
returns a non-fatal :class:`ValidationReport` of data-quality warnings (NaN
stretches, sample-rate drift, gaps). :meth:`Telemetry.hash` is the stable
content hash the attestation chain consumes (ADR-008: SHA-256 of the data).
"""

from __future__ import annotations

import dataclasses
import hashlib
import struct
from typing import ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from ..errors import SchemaError


@dataclasses.dataclass(frozen=True)
class ChannelSpec:
    """Metadata for one telemetry data channel.

    Args
    ----
    name : str
        The model field holding the channel's array.
    unit : str
        The SI unit the channel is stored in (e.g. ``"rad"``, ``"m/s^2"``).
    frame : str | None
        The frame the channel is expressed in, or ``None`` if frame-agnostic.
    description : str
        Human/agent-readable description.
    """

    name: str
    unit: str
    frame: str | None = None
    description: str = ""


@dataclasses.dataclass
class ValidationReport:
    """Structured, non-fatal data-quality findings from :meth:`Telemetry.validate`."""

    warnings: list[str] = dataclasses.field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True iff there are no warnings."""
        return not self.warnings

    def __bool__(self) -> bool:
        return self.ok

    def summary(self) -> str:
        """A one-line human/agent summary."""
        if self.ok:
            return "telemetry OK (no warnings)"
        return f"{len(self.warnings)} warning(s): " + "; ".join(self.warnings)


class Telemetry(BaseModel):
    """Base class for typed telemetry. Subclasses declare data channels.

    Fields
    ------
    time : numpy.ndarray
        Strictly-increasing ``float64`` timestamps in seconds, shape ``(n,)``.
    sample_rate_hz : float
        The nominal sample rate in hertz (explicit, not inferred).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    time: np.ndarray
    sample_rate_hz: float

    #: Per-channel metadata; overridden by each subclass.
    CHANNELS: ClassVar[tuple[ChannelSpec, ...]] = ()

    @model_validator(mode="after")
    def _validate_structure(self) -> "Telemetry":
        t = self.time
        if not isinstance(t, np.ndarray) or t.ndim != 1:
            raise SchemaError(
                f"{type(self).__name__}.time must be a 1-D numpy array, got "
                f"{type(t).__name__} with ndim "
                f"{getattr(t, 'ndim', 'n/a')}. Fix: pass time as a 1-D float64 array."
            )
        if t.dtype != np.float64:
            raise SchemaError(
                f"{type(self).__name__}.time must be float64 seconds, got "
                f"{t.dtype}. Fix: convert with .astype(np.float64) at load time."
            )
        if t.size >= 2 and not np.all(np.diff(t) > 0):
            raise SchemaError(
                f"{type(self).__name__}.time must be strictly increasing "
                "(monotonic clock). Fix: sort/deduplicate timestamps at the "
                "loader boundary."
            )
        if not (self.sample_rate_hz > 0):
            raise SchemaError(
                f"{type(self).__name__}.sample_rate_hz must be > 0, got "
                f"{self.sample_rate_hz}."
            )
        n = t.size
        for ch in self.CHANNELS:
            arr = getattr(self, ch.name)
            if arr is None:
                continue  # optional channel left unset (e.g. BusVoltage.current)
            if not isinstance(arr, np.ndarray):
                raise SchemaError(
                    f"{type(self).__name__}.{ch.name} must be a numpy array."
                )
            if arr.shape[0] != n:
                raise SchemaError(
                    f"{type(self).__name__}.{ch.name} has {arr.shape[0]} rows but "
                    f"time has {n}. Every channel is sampled on the same clock."
                )
        return self

    @property
    def n_samples(self) -> int:
        """Number of time samples."""
        return int(self.time.size)

    def channel_arrays(self) -> dict[str, np.ndarray]:
        """Return ``{channel_name: array}`` for every declared, present channel.

        Optional channels left unset (``None``) are omitted.
        """
        return {
            ch.name: np.asarray(getattr(self, ch.name))
            for ch in self.CHANNELS
            if getattr(self, ch.name) is not None
        }

    def hash(self) -> bytes:
        """Stable SHA-256 content hash over the data (ADR-008).

        Returns
        -------
        bytes
            A 32-byte digest, identical across machines for identical data
            (channels hashed in name order, with units and frames).
        """
        h = hashlib.sha256()
        h.update(type(self).__name__.encode())
        h.update(np.ascontiguousarray(self.time, dtype=np.float64).tobytes())
        h.update(struct.pack("<d", float(self.sample_rate_hz)))
        for ch in sorted(self.CHANNELS, key=lambda c: c.name):
            raw = getattr(self, ch.name)
            if raw is None:
                continue  # optional channel left unset — not part of the content
            arr = np.ascontiguousarray(raw)
            h.update(ch.name.encode())
            h.update(ch.unit.encode())
            h.update((ch.frame or "").encode())
            h.update(str(arr.dtype).encode())
            h.update(arr.tobytes())
        return h.digest()

    def validate(self) -> ValidationReport:  # type: ignore[override]
        """Return non-fatal data-quality warnings (NaN stretches, rate drift, gaps)."""
        report = ValidationReport()
        t = self.time
        if t.size >= 2:
            dt = np.diff(t)
            nominal = 1.0 / self.sample_rate_hz
            median_dt = float(np.median(dt))
            if abs(median_dt - nominal) > 0.1 * nominal:
                report.warnings.append(
                    f"median sample interval {median_dt:.4g}s differs from "
                    f"1/sample_rate_hz={nominal:.4g}s by >10%"
                )
            if float(np.max(dt)) > 5.0 * median_dt:
                report.warnings.append(
                    f"largest time gap {float(np.max(dt)):.4g}s exceeds 5x the "
                    "median interval (dropped samples?)"
                )
        for name, arr in self.channel_arrays().items():
            n_nan = int(np.count_nonzero(~np.isfinite(arr)))
            if n_nan:
                report.warnings.append(
                    f"channel {name!r} has {n_nan} non-finite value(s)"
                )
        return report


class JointTelemetry(Telemetry):
    """Per-joint positions, velocities, and torques.

    Arrays are shape ``(n, n_joints)`` aligned with ``joint_names``.
    """

    joint_names: tuple[str, ...]
    q: np.ndarray
    qd: np.ndarray
    tau: np.ndarray

    CHANNELS: ClassVar[tuple[ChannelSpec, ...]] = (
        ChannelSpec("q", "rad", description="joint positions"),
        ChannelSpec("qd", "rad/s", description="joint velocities"),
        ChannelSpec("tau", "N*m", description="joint torques"),
    )


class MagnetometerTelemetry(Telemetry):
    """3-axis magnetometer telemetry."""

    frame: str
    mag: np.ndarray  # (n, 3) in T (Tesla) or normalized

    CHANNELS: ClassVar[tuple[ChannelSpec, ...]] = (
        ChannelSpec("mag", "T", frame="mag", description="magnetic field reading"),
    )


class IMUTelemetry(Telemetry):
    """Accelerometer + gyroscope (+ optional magnetometer) in one frame."""

    frame: str
    accel: np.ndarray  # (n, 3)
    gyro: np.ndarray  # (n, 3)

    CHANNELS: ClassVar[tuple[ChannelSpec, ...]] = (
        ChannelSpec("accel", "m/s^2", frame="imu", description="linear acceleration"),
        ChannelSpec("gyro", "rad/s", frame="imu", description="angular velocity"),
    )


class FTSensorTelemetry(Telemetry):
    """Six-axis force/torque in one frame."""

    frame: str
    force: np.ndarray  # (n, 3)
    torque: np.ndarray  # (n, 3)

    CHANNELS: ClassVar[tuple[ChannelSpec, ...]] = (
        ChannelSpec("force", "N", frame="sensor", description="force"),
        ChannelSpec("torque", "N*m", frame="sensor", description="torque"),
    )


class MocapTelemetry(Telemetry):
    """Marker positions in one frame, shape ``(n, n_markers, 3)``."""

    frame: str
    marker_names: tuple[str, ...]
    positions: np.ndarray

    CHANNELS: ClassVar[tuple[ChannelSpec, ...]] = (
        ChannelSpec("positions", "m", frame="mocap", description="marker positions"),
    )


class ActuatorCommandTelemetry(Telemetry):
    """Commanded actuator inputs, shape ``(n, n_actuators)``.

    ``command_unit`` records the SI unit of the command channel (e.g. ``"V"``
    for voltage, ``"A"`` for current, ``"N*m"`` for torque).
    """

    actuator_names: tuple[str, ...]
    command_unit: str
    command: np.ndarray

    CHANNELS: ClassVar[tuple[ChannelSpec, ...]] = (
        ChannelSpec("command", "varies", description="commanded actuator input"),
    )


class ActuatorStepTelemetry(Telemetry):
    """Single-actuator step-response telemetry.

    The data shape every position-servo recipe consumes: commanded position
    setpoint and measured position sampled on one clock for one actuator. The
    canonical excitation is a sequence of position steps wide enough to engage
    the saturation and small enough to engage the deadband.

    Fields
    ------
    actuator_name : str
        Identifier of the single actuator the trace belongs to (provenance).
    setpoint : numpy.ndarray, shape ``(n,)``, units rad
        Commanded position, one sample per ``time`` entry.
    position : numpy.ndarray, shape ``(n,)``, units rad
        Measured position, one sample per ``time`` entry.
    """

    actuator_name: str
    setpoint: np.ndarray
    position: np.ndarray

    CHANNELS: ClassVar[tuple[ChannelSpec, ...]] = (
        ChannelSpec("setpoint", "rad", description="commanded position"),
        ChannelSpec("position", "rad", description="measured position"),
    )


class TorqueStepTelemetry(Telemetry):
    """Single-actuator torque-step telemetry (torque-commanded drives).

    The counterpart of :class:`ActuatorStepTelemetry` for actuators commanded
    in *torque* rather than position — the quasi-direct-drive family. Separate
    rather than reinterpreted because the command channel's unit differs, and
    a unit that depends on which actuator you plugged in is not a unit.

    Fields
    ------
    actuator_name : str
        Identifier of the single actuator the trace belongs to (provenance).
    tau_command : numpy.ndarray, shape ``(n,)``, units N·m
        Commanded joint torque.
    position : numpy.ndarray, shape ``(n,)``, units rad
        Measured joint position.
    velocity : numpy.ndarray | None, shape ``(n,)``, units rad/s
        Measured joint velocity, when the drive reports it.
    """

    actuator_name: str
    tau_command: np.ndarray
    position: np.ndarray
    velocity: np.ndarray | None = None

    CHANNELS: ClassVar[tuple[ChannelSpec, ...]] = (
        ChannelSpec("tau_command", "N*m", description="commanded joint torque"),
        ChannelSpec("position", "rad", description="measured joint position"),
        ChannelSpec("velocity", "rad/s", description="measured joint velocity"),
    )


class PneumaticTelemetry(Telemetry):
    """Double-acting pneumatic cylinder telemetry.

    Chamber pressures are the state a pneumatic actuator's identification
    turns on: the valve command and the piston motion are both observable
    without them, but the orifice conductance and the chamber dead volumes are
    not.

    Fields
    ------
    actuator_name : str
        Identifier of the single cylinder the trace belongs to (provenance).
    valve_command : numpy.ndarray, shape ``(n,)``, dimensionless in [-1, 1]
        Valve spool command: positive fills chamber A and vents B.
    pressure_a, pressure_b : numpy.ndarray, shape ``(n,)``, units Pa
        Absolute chamber pressures.
    position : numpy.ndarray, shape ``(n,)``, units m
        Piston position from the chamber-A end cap.
    force : numpy.ndarray | None, shape ``(n,)``, units N
        Measured rod force, when a load cell is fitted.
    """

    actuator_name: str
    valve_command: np.ndarray
    pressure_a: np.ndarray
    pressure_b: np.ndarray
    position: np.ndarray
    force: np.ndarray | None = None

    CHANNELS: ClassVar[tuple[ChannelSpec, ...]] = (
        ChannelSpec("valve_command", "-", description="valve spool command"),
        ChannelSpec("pressure_a", "Pa", description="chamber A absolute pressure"),
        ChannelSpec("pressure_b", "Pa", description="chamber B absolute pressure"),
        ChannelSpec("position", "m", description="piston position"),
        ChannelSpec("force", "N", description="rod force (optional)"),
    )


class EncoderTelemetry(Telemetry):
    """Encoder reading vs reference truth.

    Records both the encoder's measured position and a synchronized "truth"
    reading (typically from a higher-resolution reference encoder or an
    instrumented fixture). The recipe identifies quantization step, drift,
    and noise density from the (truth − measured) residual.

    Fields
    ------
    encoder_name : str
        Identifier of the single encoder the trace belongs to (provenance).
    truth : numpy.ndarray, shape ``(n,)``, units rad
        Reference angular position (high-resolution source).
    measured : numpy.ndarray, shape ``(n,)``, units rad
        Encoder reading.
    """

    encoder_name: str
    truth: np.ndarray
    measured: np.ndarray

    CHANNELS: ClassVar[tuple[ChannelSpec, ...]] = (
        ChannelSpec("truth", "rad", description="reference angular position"),
        ChannelSpec("measured", "rad", description="encoder reading"),
    )


class CurrentSenseTelemetry(Telemetry):
    """Current-sense reading vs reference truth.

    The measurement-channel analogue of :class:`EncoderTelemetry`: the sensor
    under test against a reference instrument (a calibrated clamp meter or a
    precision shunt), so the channel's own gain, offset and bandwidth are
    identifiable separately from whatever the current is doing.

    Fields
    ------
    channel_name : str
        Identifier of the single current channel (provenance).
    truth : numpy.ndarray, shape ``(n,)``, units A
        Reference current.
    measured : numpy.ndarray, shape ``(n,)``, units A
        Sensor reading, already converted from counts to amperes.
    """

    channel_name: str
    truth: np.ndarray
    measured: np.ndarray

    CHANNELS: ClassVar[tuple[ChannelSpec, ...]] = (
        ChannelSpec("truth", "A", description="reference current"),
        ChannelSpec("measured", "A", description="current-sense reading"),
    )


class TemperatureSensorTelemetry(Telemetry):
    """Temperature reading vs reference truth.

    Pairs a servo's own (usually coarsely quantized) temperature report with a
    reference probe. The residual identifies the channel's offset, its
    quantization step, and its thermal lag — the three things that otherwise
    get absorbed into a thermal *model* fit and corrupt it.

    Fields
    ------
    channel_name : str
        Identifier of the single temperature channel (provenance).
    truth : numpy.ndarray, shape ``(n,)``, units K
        Reference temperature.
    measured : numpy.ndarray, shape ``(n,)``, units K
        Sensor reading, converted to kelvin.
    """

    channel_name: str
    truth: np.ndarray
    measured: np.ndarray

    CHANNELS: ClassVar[tuple[ChannelSpec, ...]] = (
        ChannelSpec("truth", "K", description="reference temperature"),
        ChannelSpec("measured", "K", description="temperature-sensor reading"),
    )


class BusLatencyTelemetry(Telemetry):
    """A signal as sent and as received across a transport.

    The trace a latency/jitter/drop identification consumes: the same logical
    signal on both ends of a bus, sampled on one clock. Both channels are
    declared dimensionless because the transport does not care what it carries
    — normalize the payload before constructing this, and record what it was in
    ``signal_name``.

    Fields
    ------
    signal_name : str
        What the payload is (e.g. ``"goal_position"``) — provenance, since the
        channels themselves are unit-free.
    sent : numpy.ndarray, shape ``(n,)``, dimensionless
        The signal at the sender.
    received : numpy.ndarray, shape ``(n,)``, dimensionless
        The signal at the receiver.
    """

    signal_name: str
    sent: np.ndarray
    received: np.ndarray

    CHANNELS: ClassVar[tuple[ChannelSpec, ...]] = (
        ChannelSpec("sent", "-", description="normalized payload at the sender"),
        ChannelSpec("received", "-", description="normalized payload at the receiver"),
    )


class SeriesElasticTelemetry(Telemetry):
    """Series-elastic actuator telemetry.

    Records the motor-side input torque, motor-side and load-side angular
    positions for one series-elastic actuator. From these the spring
    deflection ``q_motor − q_load`` is recoverable algebraically, which is
    what the identification recipe regresses against.

    Fields
    ------
    actuator_name : str
        Identifier of the single actuator the trace belongs to (provenance).
    tau_input : numpy.ndarray, shape ``(n,)``, units N·m
        Torque applied to the motor side.
    q_motor : numpy.ndarray, shape ``(n,)``, units rad
        Motor-side angular position.
    q_load : numpy.ndarray, shape ``(n,)``, units rad
        Load-side angular position.
    """

    actuator_name: str
    tau_input: np.ndarray
    q_motor: np.ndarray
    q_load: np.ndarray

    CHANNELS: ClassVar[tuple[ChannelSpec, ...]] = (
        ChannelSpec("tau_input", "N*m", description="motor-side input torque"),
        ChannelSpec("q_motor", "rad", description="motor-side angular position"),
        ChannelSpec("q_load", "rad", description="load-side angular position"),
    )


class TransmissionTelemetry(Telemetry):
    """Two-port transmission telemetry.

    Records the motor-side and joint-side torques and velocities for one
    transmission. Identifying a gear ratio, efficiency, and backlash from this
    data set is what the ``sysid.actuator`` recipe uses (transmissions
    fit into the same recipe family as motors — the actuator/transmission
    distinction is structural, not procedural).

    Fields
    ------
    transmission_name : str
        Identifier of the single transmission the trace belongs to (provenance).
    tau_motor, tau_joint : numpy.ndarray, shape ``(n,)``, units N·m
        Motor-side and joint-side torques.
    v_motor, v_joint : numpy.ndarray, shape ``(n,)``, units rad/s
        Motor-side and joint-side angular velocities.
    """

    transmission_name: str
    tau_motor: np.ndarray
    v_motor: np.ndarray
    tau_joint: np.ndarray
    v_joint: np.ndarray

    CHANNELS: ClassVar[tuple[ChannelSpec, ...]] = (
        ChannelSpec("tau_motor", "N*m", description="motor-side torque"),
        ChannelSpec("v_motor", "rad/s", description="motor-side angular velocity"),
        ChannelSpec("tau_joint", "N*m", description="joint-side torque"),
        ChannelSpec("v_joint", "rad/s", description="joint-side angular velocity"),
    )


class BacklashTelemetry(Telemetry):
    """Dual-encoder transmission telemetry for positional backlash.

    The data shape a *positional* backlash fit needs and a single encoder
    cannot supply: the motor-side angle and the joint-side angle measured at
    the same instants, plus the torque actually transmitted. Backlash is the
    lost motion between the two angles, so a trace with only one of them
    identifies nothing — this schema is the dual-encoder gate made explicit.

    Fields
    ------
    transmission_name : str
        Identifier of the single transmission the trace belongs to (provenance).
    theta_motor : numpy.ndarray, shape ``(n,)``, units rad
        Motor-side angle, *already referred to the joint side* (i.e. divided by
        the gear ratio). Keeping both angles in joint coordinates is what makes
        their difference the mesh deflection.
    theta_joint : numpy.ndarray, shape ``(n,)``, units rad
        Joint-side angle from the output encoder.
    tau_joint : numpy.ndarray, shape ``(n,)``, units N·m
        Torque transmitted to the joint.
    """

    transmission_name: str
    theta_motor: np.ndarray
    theta_joint: np.ndarray
    tau_joint: np.ndarray

    CHANNELS: ClassVar[tuple[ChannelSpec, ...]] = (
        ChannelSpec(
            "theta_motor", "rad", description="motor angle referred to the joint side"
        ),
        ChannelSpec("theta_joint", "rad", description="joint-side angle"),
        ChannelSpec("tau_joint", "N*m", description="transmitted joint torque"),
    )


class LinearTransmissionTelemetry(Telemetry):
    """Rotary→linear transmission telemetry (lead screw, rack, linear belt).

    The two-port trace of a transmission whose output is a *translation*: the
    unit contract differs on each side, which is exactly why this is a separate
    schema from :class:`TransmissionTelemetry` rather than a reinterpretation
    of it.

    Fields
    ------
    transmission_name : str
        Identifier of the single transmission the trace belongs to (provenance).
    tau_motor : numpy.ndarray, shape ``(n,)``, units N·m
        Motor-side torque.
    v_motor : numpy.ndarray, shape ``(n,)``, units rad/s
        Motor-side angular velocity.
    force_load : numpy.ndarray, shape ``(n,)``, units N
        Load-side force along the axis of travel.
    v_load : numpy.ndarray, shape ``(n,)``, units m/s
        Load-side linear velocity along the axis of travel.
    """

    transmission_name: str
    tau_motor: np.ndarray
    v_motor: np.ndarray
    force_load: np.ndarray
    v_load: np.ndarray

    CHANNELS: ClassVar[tuple[ChannelSpec, ...]] = (
        ChannelSpec("tau_motor", "N*m", description="motor-side torque"),
        ChannelSpec("v_motor", "rad/s", description="motor-side angular velocity"),
        ChannelSpec("force_load", "N", description="load-side axial force"),
        ChannelSpec("v_load", "m/s", description="load-side linear velocity"),
    )


class ActuatorChirpTelemetry(Telemetry):
    """Single-actuator chirp-excitation telemetry.

    The data shape every current-controlled actuator recipe consumes: commanded
    current setpoint and measured velocity (and optionally current) sampled on
    one clock for one actuator, under a chirp excitation that sweeps frequency
    to identify the electrical and mechanical time constants.

    Fields
    ------
    actuator_name : str
        Identifier of the single actuator the trace belongs to (provenance).
    setpoint : numpy.ndarray, shape ``(n,)``, units A
        Commanded current setpoint, one sample per ``time`` entry.
    velocity : numpy.ndarray, shape ``(n,)``, units rad/s
        Measured rotor velocity, one sample per ``time`` entry.
    current : numpy.ndarray, shape ``(n,)``, units A
        Measured motor current, one sample per ``time`` entry.
    """

    actuator_name: str
    setpoint: np.ndarray
    velocity: np.ndarray
    current: np.ndarray

    CHANNELS: ClassVar[tuple[ChannelSpec, ...]] = (
        ChannelSpec("setpoint", "A", description="commanded current setpoint"),
        ChannelSpec("velocity", "rad/s", description="measured rotor velocity"),
        ChannelSpec("current", "A", description="measured motor current"),
    )


class ActuatorRampTelemetry(Telemetry):
    """Single-actuator voltage-ramp telemetry for WPILib SysId identification.

    The data shape every per-actuator sysid recipe (DC motor, servo, BLDC; the
    formal ``sysid.actuator`` recipe lands later) consumes:
    applied voltage and rotor velocity sampled on one clock for one actuator.
    Acceleration is not stored — it is computed at fit time from ``velocity`` by
    central difference (this is how WPILib SysId itself processes its logs).

    Fields
    ------
    actuator_name : str
        Identifier of the single actuator the trace belongs to (provenance).
    command : numpy.ndarray, shape ``(n,)``, units V
        Applied voltage command, one sample per ``time`` entry.
    velocity : numpy.ndarray, shape ``(n,)``, units rad/s
        Measured rotor velocity, one sample per ``time`` entry.
    """

    actuator_name: str
    command: np.ndarray
    velocity: np.ndarray

    CHANNELS: ClassVar[tuple[ChannelSpec, ...]] = (
        ChannelSpec("command", "V", description="applied voltage command"),
        ChannelSpec("velocity", "rad/s", description="measured rotor velocity"),
    )


class ThermalTelemetry(Telemetry):
    """Per-servo temperature trace (servo-thermal model).

    The data shape the servo-thermal model consumes: winding/case temperature
    sampled on one clock for one or more servos. The driving torque is *not* a
    channel here — it is the actuator torque (a :class:`JointTelemetry` channel
    or an actuator command), passed alongside this trace at fit time.

    Fields
    ------
    servo_names : tuple[str, ...]
        Identifiers of the servos, aligned with ``temperature`` columns.
    temperature : numpy.ndarray, shape ``(n, n_servos)``, units K
        Measured winding/case temperature, one column per servo.
    """

    servo_names: tuple[str, ...]
    temperature: np.ndarray

    CHANNELS: ClassVar[tuple[ChannelSpec, ...]] = (
        ChannelSpec(
            "temperature", "K", description="per-servo winding/case temperature"
        ),
    )

    @model_validator(mode="after")
    def _validate_columns(self) -> "ThermalTelemetry":
        temp = self.temperature
        if temp.ndim != 2 or temp.shape[1] != len(self.servo_names):
            raise SchemaError(
                f"ThermalTelemetry.temperature must be 2-D (n, n_servos) with "
                f"one column per servo_name; got shape {temp.shape} for "
                f"{len(self.servo_names)} servo(s). Fix: stack per-servo "
                "temperature columns in servo_names order."
            )
        return self


class BusVoltageTelemetry(Telemetry):
    """Shared-bus terminal-voltage trace, optional per-servo current (LeKiwi).

    The data shape the battery-sag model consumes: the shared battery's terminal
    (bus) voltage sampled on one clock, under a known servo load. Per-servo
    current is an optional auxiliary channel — when a power monitor provides it
    the fit can cross-check the torque→current map; when absent the load is
    reconstructed from the driving torque passed at fit time.

    Fields
    ------
    voltage : numpy.ndarray, shape ``(n,)``, units V
        Measured shared-bus terminal voltage.
    servo_names : tuple[str, ...]
        Identifiers of the servos sharing the bus, aligned with ``current``
        columns when present. May be empty if ``current`` is not provided.
    current : numpy.ndarray | None, shape ``(n, n_servos)``, units A
        Optional measured per-servo current draw.
    """

    voltage: np.ndarray
    servo_names: tuple[str, ...] = ()
    current: np.ndarray | None = None

    CHANNELS: ClassVar[tuple[ChannelSpec, ...]] = (
        ChannelSpec("voltage", "V", description="shared-bus terminal voltage"),
        ChannelSpec("current", "A", description="per-servo current draw (optional)"),
    )

    @model_validator(mode="after")
    def _validate_current(self) -> "BusVoltageTelemetry":
        if self.voltage.ndim != 1:
            raise SchemaError(
                f"BusVoltageTelemetry.voltage must be 1-D (n,), got shape "
                f"{self.voltage.shape}. The bus voltage is a single shared rail."
            )
        if self.current is not None:
            cur = self.current
            if cur.ndim != 2 or cur.shape[1] != len(self.servo_names):
                raise SchemaError(
                    f"BusVoltageTelemetry.current must be 2-D (n, n_servos) with "
                    f"one column per servo_name; got shape {cur.shape} for "
                    f"{len(self.servo_names)} servo(s). Fix: supply servo_names "
                    "matching the current columns, or omit current."
                )
        return self


__all__ = [
    "ChannelSpec",
    "ValidationReport",
    "Telemetry",
    "JointTelemetry",
    "IMUTelemetry",
    "FTSensorTelemetry",
    "MagnetometerTelemetry",
    "MocapTelemetry",
    "ActuatorCommandTelemetry",
    "ActuatorChirpTelemetry",
    "ActuatorRampTelemetry",
    "ActuatorStepTelemetry",
    "EncoderTelemetry",
    "SeriesElasticTelemetry",
    "TransmissionTelemetry",
    "BacklashTelemetry",
    "LinearTransmissionTelemetry",
    "TorqueStepTelemetry",
    "PneumaticTelemetry",
    "CurrentSenseTelemetry",
    "TemperatureSensorTelemetry",
    "BusLatencyTelemetry",
    "ThermalTelemetry",
    "BusVoltageTelemetry",
]
