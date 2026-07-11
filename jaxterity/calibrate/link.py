# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""The servo-bus transport abstraction the setup routine drives.

:class:`ActiveLink` is the minimal set of operations the bring-up needs from a
chain of smart servos: scan the bus, read position/health, enable torque, and
command a slow current-limited move. Two implementations back it:

* :class:`LoopbackLink` — a deterministic, dependency-free *simulator* of a
  Feetech chain (each servo has a hidden offset, direction, and two mechanical
  hard stops). It makes the whole calibration logic gate-testable with no
  hardware.
* :class:`FeetechLink` — the *real* serial transport, wrapping LeRobot's
  ``FeetechMotorsBus``. It is conservative by construction (torque disabled
  until a move, low speed, a load ceiling) and is the one piece of this package
  that needs hardware-in-the-loop validation before it can be trusted; it is
  marked as such and never runs in tests.

The routine talks only to :class:`ActiveLink`, so the same discovery / health /
touch-off code path runs against the simulator in CI and against a real LeKiwi
on the bench.
"""

from __future__ import annotations

import abc
import dataclasses

from .layout import STS3215_RESOLUTION
from .types import ServoStatus

#: Default poll/step the simulator advances per :meth:`read_position`. Real
#: hardware ignores it (the servo moves on its own clock); the simulator uses it
#: to integrate motion so a poll loop converges deterministically.
_SIM_DT = 0.02


class ActiveLink(abc.ABC):
    """A live connection to a chain of smart servos.

    Implementations talk to real hardware or simulate it. Positions are raw
    encoder counts in ``[0, resolution)``; loads are signed fractions of stall
    in ``[-1, 1]``. All methods address servos by their integer bus ID.

    Use as a context manager so the bus is always closed::

        with FeetechLink("/dev/ttyACM0", (1, 2, 3)) as link:
            ids = link.scan()
    """

    #: Encoder counts per revolution for the servos on this bus.
    resolution: int = STS3215_RESOLUTION

    # --- lifecycle ------------------------------------------------------

    def open(self) -> None:
        """Open the transport. Idempotent; the default is a no-op."""

    def close(self) -> None:
        """Close the transport. Idempotent; the default is a no-op."""

    def __enter__(self) -> "ActiveLink":
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- queries --------------------------------------------------------

    @abc.abstractmethod
    def scan(self) -> tuple[int, ...]:
        """Ping the bus and return the IDs that answered, sorted."""

    @abc.abstractmethod
    def read_position(self, servo_id: int) -> int:
        """Return the present position of ``servo_id`` in raw encoder counts."""

    @abc.abstractmethod
    def read_status(self, servo_id: int) -> ServoStatus:
        """Return a full :class:`ServoStatus` snapshot for ``servo_id``."""

    # --- commands -------------------------------------------------------

    @abc.abstractmethod
    def set_torque(self, servo_id: int, on: bool) -> None:
        """Enable or disable the servo's holding/driving torque."""

    @abc.abstractmethod
    def start_move(
        self,
        servo_id: int,
        goal_ticks: int,
        *,
        speed_ticks_per_s: float,
        max_load: float,
    ) -> None:
        """Command a current-limited move toward ``goal_ticks``.

        The servo drives toward the goal at ``speed_ticks_per_s`` and is
        expected to stall (stop advancing, report load near ``max_load``) when
        it meets a mechanical hard stop short of the goal. The caller polls
        :meth:`read_position` / :meth:`read_status` to detect the stall.

        Args
        ----
        servo_id : int
            Target servo.
        goal_ticks : int
            Desired position in raw counts (typically commanded *past* a stop
            so the servo seeks the stop).
        speed_ticks_per_s : float
            Move speed, counts per second. Keep this low for touch-off.
        max_load : float
            Load ceiling as a fraction of stall in ``(0, 1]`` — the servo backs
            off rather than exceeding it, protecting the gears.

        Units
        -----
        Positions in encoder counts, speed in counts/second, load a unit-free
        fraction of stall.
        """


# --------------------------------------------------------------------------- #
# Simulator
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class _SimServo:
    """Hidden ground-truth state of one simulated servo."""

    servo_id: int
    position: float
    stop_lo: float | None  # None for continuous (wheel) servos
    stop_hi: float | None
    goal: float
    speed: float = 0.0
    max_load: float = 0.5
    torque_on: bool = False
    voltage_v: float = 12.0
    temperature_c: float = 30.0
    load: float = 0.0
    resolution: int = STS3215_RESOLUTION

    def advance(self, dt: float) -> None:
        """Integrate motion for ``dt`` seconds, clamping at any hard stop."""
        if not self.torque_on or self.speed <= 0.0:
            self.load = 0.0
            return
        step = self.speed * dt
        delta = self.goal - self.position
        moved = max(-step, min(step, delta))
        target = self.position + moved
        lo = 0.0 if self.stop_lo is None else self.stop_lo
        hi = float(self.resolution - 1) if self.stop_hi is None else self.stop_hi
        clamped = max(lo, min(hi, target))
        blocked = clamped != target
        self.position = clamped
        if blocked:
            # Pushing into a stop: straining at the load ceiling, not moving.
            direction = 1.0 if delta > 0 else -1.0
            self.load = direction * self.max_load
        elif abs(delta) <= 1e-6:
            self.load = 0.0
        else:
            self.load = (1.0 if delta > 0 else -1.0) * 0.1 * self.max_load

    @property
    def moving(self) -> bool:
        return self.torque_on and abs(self.load) < self.max_load and self.speed > 0


class LoopbackLink(ActiveLink):
    """A deterministic simulator of a Feetech servo chain (no hardware).

    Each servo carries a *hidden* true offset, direction, and (for arm joints)
    two mechanical hard stops, so the touch-off routine has something real to
    discover. Motion is integrated on :meth:`read_position` (by ``sim_dt``), so
    a poll loop converges in a fixed number of reads independent of wall-clock
    time — the property that makes the calibration tests fast and reproducible.

    Build one with :meth:`add_servo` (or the higher-level
    :func:`jaxterity.calibrate.setup.simulated_link`, which seeds it from a
    layout and the zoo URDF limits).

    Args
    ----
    resolution : int
        Encoder counts per revolution (default STS3215's 4096).
    sim_dt : float
        Seconds of simulated motion integrated per position read.
    """

    def __init__(
        self, *, resolution: int = STS3215_RESOLUTION, sim_dt: float = _SIM_DT
    ) -> None:
        self.resolution = resolution
        self._sim_dt = sim_dt
        self._servos: dict[int, _SimServo] = {}

    def add_servo(
        self,
        servo_id: int,
        *,
        position: float,
        stop_lo: float | None = None,
        stop_hi: float | None = None,
        voltage_v: float = 12.0,
        temperature_c: float = 30.0,
    ) -> None:
        """Register a simulated servo with its hidden ground-truth state.

        Args
        ----
        servo_id : int
            Bus ID.
        position : float
            Initial present position, raw counts.
        stop_lo, stop_hi : float | None
            Hard-stop positions in counts, or ``None`` for a continuous
            (wheel) servo with no stops.
        voltage_v, temperature_c : float
            Simulated health readings.
        """
        self._servos[servo_id] = _SimServo(
            servo_id=servo_id,
            position=position,
            stop_lo=stop_lo,
            stop_hi=stop_hi,
            goal=position,
            voltage_v=voltage_v,
            temperature_c=temperature_c,
            resolution=self.resolution,
        )

    def _get(self, servo_id: int) -> _SimServo:
        try:
            return self._servos[servo_id]
        except KeyError:
            raise KeyError(
                f"LoopbackLink has no servo with ID {servo_id}. "
                f"Registered IDs: {sorted(self._servos)}."
            ) from None

    def scan(self) -> tuple[int, ...]:
        return tuple(sorted(self._servos))

    def read_position(self, servo_id: int) -> int:
        servo = self._get(servo_id)
        servo.advance(self._sim_dt)
        return int(round(servo.position))

    def read_status(self, servo_id: int) -> ServoStatus:
        servo = self._get(servo_id)
        return ServoStatus(
            servo_id=servo_id,
            position_ticks=int(round(servo.position)),
            voltage_v=servo.voltage_v,
            temperature_c=servo.temperature_c,
            load=servo.load,
            moving=servo.moving,
        )

    def set_torque(self, servo_id: int, on: bool) -> None:
        servo = self._get(servo_id)
        servo.torque_on = on
        if not on:
            servo.speed = 0.0
            servo.load = 0.0

    def start_move(
        self,
        servo_id: int,
        goal_ticks: int,
        *,
        speed_ticks_per_s: float,
        max_load: float,
    ) -> None:
        servo = self._get(servo_id)
        servo.torque_on = True
        servo.goal = float(goal_ticks)
        servo.speed = float(speed_ticks_per_s)
        servo.max_load = float(max_load)


# --------------------------------------------------------------------------- #
# Real hardware
# --------------------------------------------------------------------------- #


class FeetechLink(ActiveLink):
    """Real Feetech servo-bus transport, wrapping LeRobot's ``FeetechMotorsBus``.

    This is the on-device path: it opens the USB-TTL bus, scans for servos, and
    reads/writes the STS3215 control registers through LeRobot's motor bus
    (LeRobot is the established, tested Feetech driver — Jaxterity does not
    reimplement servo comms). Moves are conservative by construction: torque is
    left disabled until a move is commanded, speed is whatever the caller passes
    (the setup routine keeps it low), and a torque limit is written so a
    touch-off presses into a hard stop gently rather than stripping the gears.

    .. warning::

       This transport has **not** been validated against physical hardware in
       this repository; the simulator (:class:`LoopbackLink`) is what the test
       suite exercises. LeRobot's motor-bus register names have changed across
       releases, so confirm the mapping below against your installed ``lerobot``
       version on first bring-up, and keep a hand on the power switch the first
       time you run a real touch-off.

    Args
    ----
    port : str
        Serial device, e.g. ``"/dev/ttyACM0"`` (Linux) or
        ``"/dev/tty.usbmodemXXXX"`` (macOS). Find it with ``lerobot-find-port``.
    servo_ids : tuple[int, ...]
        The IDs to register on the bus.
    resolution : int
        Encoder counts per revolution (default STS3215's 4096).
    """

    def __init__(
        self,
        port: str,
        servo_ids: tuple[int, ...],
        *,
        resolution: int = STS3215_RESOLUTION,
    ) -> None:
        self.port = port
        self.servo_ids = tuple(servo_ids)
        self.resolution = resolution
        self._bus: object | None = None

    def _bus_or_raise(self) -> object:
        if self._bus is None:
            raise RuntimeError(
                "FeetechLink is not open. Use it as a context manager "
                "(`with FeetechLink(...) as link:`) or call .open() first."
            )
        return self._bus

    def open(self) -> None:
        """Construct and connect the underlying LeRobot Feetech bus.

        Raises
        ------
        jaxterity.errors.CalibrationError
            If ``lerobot`` (with Feetech support) is not importable. The message
            says how to install it.
        """
        from ..errors import CalibrationError

        try:  # pragma: no cover - requires the optional lerobot dependency
            from lerobot.motors import Motor, MotorNormMode
            from lerobot.motors.feetech import FeetechMotorsBus
        except ImportError as exc:  # pragma: no cover
            raise CalibrationError(
                "FeetechLink needs LeRobot's Feetech driver to talk to the bus, "
                f"but importing it failed ({exc}). Install it on the device with "
                "`pip install 'lerobot[feetech]'` (the same package the SO-101 / "
                "LeKiwi tutorial installs), or run `jaxterity setup --simulate` "
                "to exercise the flow without hardware."
            ) from exc

        # STS3215 in default (single-turn, 0..4095) position mode. Names mirror
        # the SO-101/LeKiwi tutorial's motor table.
        motors = {
            f"id_{sid}": Motor(sid, "sts3215", MotorNormMode.RANGE_0_100)
            for sid in self.servo_ids
        }
        bus = FeetechMotorsBus(port=self.port, motors=motors)
        bus.connect()
        self._bus = bus

    def close(self) -> None:  # pragma: no cover - requires hardware
        if self._bus is not None:
            try:
                self._bus.disconnect()  # type: ignore[attr-defined]
            finally:
                self._bus = None

    def _name(self, servo_id: int) -> str:
        return f"id_{servo_id}"

    def scan(self) -> tuple[int, ...]:  # pragma: no cover - requires hardware
        bus = self._bus_or_raise()
        found = bus.broadcast_ping()  # type: ignore[attr-defined]
        return tuple(sorted(int(i) for i in found))

    def read_position(self, servo_id: int) -> int:  # pragma: no cover - hardware
        bus = self._bus_or_raise()
        return int(bus.read("Present_Position", self._name(servo_id)))  # type: ignore[attr-defined]

    def read_status(self, servo_id: int) -> ServoStatus:  # pragma: no cover - hardware
        bus = self._bus_or_raise()
        name = self._name(servo_id)
        raw_load = int(bus.read("Present_Load", name))  # type: ignore[attr-defined]
        # STS3215 Present_Load: 0..1000 magnitude with bit 10 the sign.
        sign = -1.0 if raw_load & 0x400 else 1.0
        load = sign * min(1.0, (raw_load & 0x3FF) / 1000.0)
        return ServoStatus(
            servo_id=servo_id,
            position_ticks=int(bus.read("Present_Position", name)),  # type: ignore[attr-defined]
            voltage_v=float(bus.read("Present_Voltage", name)) / 10.0,  # type: ignore[attr-defined]
            temperature_c=float(bus.read("Present_Temperature", name)),  # type: ignore[attr-defined]
            load=load,
            moving=bool(bus.read("Moving", name)),  # type: ignore[attr-defined]
        )

    def set_torque(self, servo_id: int, on: bool) -> None:  # pragma: no cover - hw
        bus = self._bus_or_raise()
        bus.write("Torque_Enable", self._name(servo_id), 1 if on else 0)  # type: ignore[attr-defined]

    def start_move(  # pragma: no cover - requires hardware
        self,
        servo_id: int,
        goal_ticks: int,
        *,
        speed_ticks_per_s: float,
        max_load: float,
    ) -> None:
        bus = self._bus_or_raise()
        name = self._name(servo_id)
        # Cap the torque first so the press into a stop is gentle, then move.
        bus.write("Torque_Limit", name, int(max(1, min(1000, round(max_load * 1000)))))  # type: ignore[attr-defined]
        bus.write("Torque_Enable", name, 1)  # type: ignore[attr-defined]
        bus.write("Goal_Speed", name, int(max(1, round(speed_ticks_per_s))))  # type: ignore[attr-defined]
        bus.write("Goal_Position", name, int(goal_ticks))  # type: ignore[attr-defined]


__all__ = ["ActiveLink", "LoopbackLink", "FeetechLink"]
