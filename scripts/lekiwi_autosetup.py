#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Single-shot first-run bring-up for a LeKiwi / SO-101, on the device.

This is a self-contained port of ``jaxterity setup`` meant to run on the
Raspberry Pi next to the robot, so it depends only on the Feetech servo SDK
(``scservo_sdk``, installed by ``lekiwi_pi_install.sh``) — not on the full
Jaxterity / JAX / MuJoCo stack. It does, in order:

1. **Discover** — scan the servo bus and match it to the expected LeKiwi layout
   (arm IDs 1-6, wheels 7-9).
2. **Health** — read each servo's voltage and temperature.
3. **Calibrate** *(only with ``--execute``)* — find each arm joint's raw-tick to
   radian map hands-free by gently touching both mechanical hard stops. The
   stops sit at the URDF joint limits (embedded below), so only the per-joint
   offset has to be found — no manual joint-by-joint sweep. Wheels get a
   recorded zero.

SAFETY — read this before using ``--execute``:
  * Without ``--execute`` the script is **read-only**: it pings and reads, and
    NOTHING moves. Always run it read-only first to confirm the bus is wired and
    healthy and that comms work against your installed SDK.
  * ``--execute`` MOVES the arm slowly into each hard stop with a capped torque.
    This driving path has not been validated on hardware in this repo — run it
    the first time with the workspace clear and a hand on the power switch, and
    consider ``--only <joint>`` to bring up one joint at a time.
  * The per-joint ``sign`` (encoder direction vs the URDF axis) is a per-model
    constant defaulted to +1 here; confirm it once for your unit.

Outputs (with ``--out DIR``): ``calibration.json`` (this tool's native record)
and ``<robot>_lerobot.json`` (a LeRobot-shaped per-motor calibration; confirm
against your LeRobot version before relying on it).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------- #
# Feetech STS3215 facts
# --------------------------------------------------------------------------- #

RESOLUTION = 4096  # 12-bit single-turn encoder, 0..4095 over 360°
TICKS_PER_RAD = RESOLUTION / (2.0 * math.pi)
DEFAULT_BAUD = 1_000_000  # Feetech bus default

# STS/SMS control-table addresses (stable across SDK versions).
ADDR_TORQUE_ENABLE = 40
ADDR_GOAL_ACC = 41
ADDR_GOAL_POSITION = 42
ADDR_GOAL_SPEED = 46
ADDR_TORQUE_LIMIT = 48
ADDR_PRESENT_POSITION = 56
ADDR_PRESENT_LOAD = 60
ADDR_PRESENT_VOLTAGE = 62
ADDR_PRESENT_TEMPERATURE = 63

# Touch-off tuning (gentle: a stop is met as a nudge, not an impact).
TOUCH_OFF_SPEED = 300  # raw goal-speed units (low)
TOUCH_OFF_ACC = 10
TOUCH_OFF_MAX_LOAD = 0.30  # fraction of stall → torque-limit register
POLL_DT = 0.03
MAX_POLLS = 700
PLATEAU_TICKS = 3
PLATEAU_COUNT = 6
STALL_FRACTION = 0.7
SCALE_WARN_RTOL = 0.12


@dataclasses.dataclass(frozen=True)
class JointSpec:
    name: str
    servo_id: int
    kind: str  # "arm" | "wheel"
    sign: int = 1
    limits: tuple[float, float] | None = None  # radians, arm joints only


# SO-101 arm limits read from the vendored URDF (so101_new_calib.urdf).
_ARM = (
    JointSpec("shoulder_pan", 1, "arm", limits=(-1.91986, 1.91986)),
    JointSpec("shoulder_lift", 2, "arm", limits=(-1.74533, 1.74533)),
    JointSpec("elbow_flex", 3, "arm", limits=(-1.69, 1.69)),
    JointSpec("wrist_flex", 4, "arm", limits=(-1.65806, 1.65806)),
    JointSpec("wrist_roll", 5, "arm", limits=(-2.74385, 2.84121)),
    JointSpec("gripper", 6, "arm", limits=(-0.174533, 1.74533)),
)

LAYOUTS: dict[str, tuple[JointSpec, ...]] = {
    "so101": _ARM,
    "lekiwi": (
        *_ARM,
        JointSpec("wheel_left", 7, "wheel"),
        JointSpec("wheel_back", 8, "wheel"),
        JointSpec("wheel_right", 9, "wheel"),
    ),
}

#: Joints with no reliable hard stop to seed against (the gripper's open side is
#: compliant) — skipped by ``--from-seed`` and calibrated separately.
NO_SEED_JOINTS = {"gripper"}


# --------------------------------------------------------------------------- #
# Bus transport
# --------------------------------------------------------------------------- #


class Bus:
    """Thin wrapper over ``scservo_sdk`` for the registers this tool needs."""

    def __init__(
        self, port: str, baud: int = DEFAULT_BAUD, *, protocol_end: int = 0
    ) -> None:
        try:
            from scservo_sdk import PacketHandler, PortHandler
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "scservo_sdk not found. Install it with "
                "`pip install feetech-servo-sdk` (or run lekiwi_pi_install.sh)."
            ) from exc
        self._port = PortHandler(port)
        # STS/SMS series (the STS3215) use protocol_end=0 (little-endian words);
        # SCS-series servos would use 1. PacketHandler sets the global endianness
        # and returns the low-level handler whose ops take the port explicitly.
        self._ph = PacketHandler(protocol_end)
        if not self._port.openPort():
            raise SystemExit(f"failed to open serial port {port!r}.")
        if not self._port.setBaudRate(baud):
            raise SystemExit(f"failed to set baud {baud} on {port!r}.")

    def ping(self, sid: int) -> bool:
        from scservo_sdk import COMM_SUCCESS

        _model, comm, err = self._ph.ping(self._port, sid)
        return comm == COMM_SUCCESS and err == 0

    def scan(self, candidate_ids: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(sid for sid in candidate_ids if self.ping(sid))

    def _read1(self, sid: int, addr: int) -> int:
        val, _comm, _err = self._ph.read1ByteTxRx(self._port, sid, addr)
        return int(val)

    def _read2(self, sid: int, addr: int) -> int:
        val, _comm, _err = self._ph.read2ByteTxRx(self._port, sid, addr)
        return int(val)

    def _write1(self, sid: int, addr: int, value: int) -> None:
        self._ph.write1ByteTxRx(self._port, sid, addr, int(value))

    def _write2(self, sid: int, addr: int, value: int) -> None:
        self._ph.write2ByteTxRx(self._port, sid, addr, int(value))

    def position(self, sid: int) -> int:
        return self._read2(sid, ADDR_PRESENT_POSITION) & 0x0FFF

    def voltage(self, sid: int) -> float:
        return self._read1(sid, ADDR_PRESENT_VOLTAGE) / 10.0

    def temperature(self, sid: int) -> float:
        return float(self._read1(sid, ADDR_PRESENT_TEMPERATURE))

    def load(self, sid: int) -> float:
        raw = self._read2(sid, ADDR_PRESENT_LOAD)
        sign = -1.0 if raw & 0x400 else 1.0
        return sign * min(1.0, (raw & 0x3FF) / 1000.0)

    def set_torque(self, sid: int, on: bool) -> None:
        self._write1(sid, ADDR_TORQUE_ENABLE, 1 if on else 0)

    def start_move(self, sid: int, goal: int, *, speed: int, max_load: float) -> None:
        # Cap torque first so a press into a stop is gentle, then command.
        self._write2(sid, ADDR_TORQUE_LIMIT, max(1, min(1000, round(max_load * 1000))))
        self._write1(sid, ADDR_TORQUE_ENABLE, 1)
        self._write1(sid, ADDR_GOAL_ACC, TOUCH_OFF_ACC)
        self._write2(sid, ADDR_GOAL_SPEED, max(1, speed))
        self._write2(sid, ADDR_GOAL_POSITION, max(0, min(RESOLUTION - 1, goal)))

    def relax_all(self, ids: tuple[int, ...]) -> None:
        for sid in ids:
            try:
                self.set_torque(sid, False)
            except Exception:  # noqa: BLE001 — best effort on shutdown
                pass

    def close(self) -> None:
        try:
            self._port.closePort()
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


DEBUG = False
BOUNDARY_MARGIN = 8  # ticks from 0/4095 that count as "at the encoder ceiling"

#: Optional callable polled *during* a touch-off sweep (L-013). It takes no args
#: and returns ``None`` to continue or a reason string to abort — used by the
#: executor to watch the *held* joints and stop the sweep before a sagging arm
#: drives the gripper into the floor. Left ``None`` for standalone use.
SWEEP_WATCH = None


class SweepAborted(RuntimeError):
    """Raised when ``SWEEP_WATCH`` vetoes an in-progress sweep (e.g. a held joint
    sagged). Carries the human-readable ``reason``."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _drive_to_stop(
    bus: Bus, sid: int, direction: int, *, max_load: float = TOUCH_OFF_MAX_LOAD
) -> tuple[int, bool]:
    """Drive a servo gently toward an end of travel.

    Returns ``(pos, found)``: ``found`` is True if a real mechanical hard stop
    was met (the servo strains at the torque cap while no longer advancing);
    False if the joint instead ran to the encoder's 0/4095 boundary with low
    load — meaning that stop lies beyond the single-turn encoder's range, which
    is normal when the servo zero is off-centre and is NOT an error.

    ``max_load`` is the torque cap (fraction of stall). Lower = gentler and less
    structural deflection at the stop (see :func:`touch_off_compliant`).
    """
    goal = (RESOLUTION - 1) if direction > 0 else 0
    boundary = (RESOLUTION - 1 - BOUNDARY_MARGIN) if direction > 0 else BOUNDARY_MARGIN
    bus.start_move(sid, goal, speed=TOUCH_OFF_SPEED, max_load=max_load)
    last = bus.position(sid)
    stall = settle = 0
    for i in range(MAX_POLLS):
        time.sleep(POLL_DT)
        pos = bus.position(sid)
        load = bus.load(sid)
        # in-sweep watchdog: let the caller veto if a held joint is sagging
        # (L-013) before the arm grazes the floor. Throttled to keep the bus free.
        if SWEEP_WATCH is not None and i % 4 == 0:
            reason = SWEEP_WATCH()
            if reason:
                bus.set_torque(sid, False)
                raise SweepAborted(reason)
        plateaued = abs(pos - last) <= PLATEAU_TICKS
        if DEBUG and i % 10 == 0:
            print(
                f"      [{sid} dir{direction:+d}] poll {i:3d} pos={pos:4d} "
                f"load={load:+.3f} stall={stall} settle={settle}",
                flush=True,
            )
        # Real hard stop: straining at the cap and no longer advancing.
        if plateaued and abs(load) >= STALL_FRACTION * max_load:
            stall += 1
            if stall >= PLATEAU_COUNT:
                bus.set_torque(sid, False)
                if DEBUG:
                    print(
                        f"      [{sid} dir{direction:+d}] HARD STOP pos={pos}",
                        flush=True,
                    )
                return pos, True
        else:
            stall = 0
        # Encoder boundary with low load: the stop is beyond the encoder range.
        at_boundary = (pos >= boundary) if direction > 0 else (pos <= boundary)
        if at_boundary and plateaued:
            settle += 1
            if settle >= PLATEAU_COUNT:
                bus.set_torque(sid, False)
                if DEBUG:
                    print(
                        f"      [{sid} dir{direction:+d}] encoder boundary pos={pos} "
                        "(no stop in range)",
                        flush=True,
                    )
                return pos, False
        else:
            settle = 0
        last = pos
    bus.set_torque(sid, False)
    raise RuntimeError(
        f"servo {sid}: motion in direction {direction:+d} neither reached a stop "
        "nor the encoder boundary (it kept moving). Check the joint is free."
    )


# --------------------------------------------------------------------------- #
# #3 Stop-compliance handling (deflection ~ proportional to contact force)
# --------------------------------------------------------------------------- #

#: Two torque caps for the deflection probe (fraction of stall).
COMPLIANCE_FORCES = (0.20, 0.34)


def stop_at_zero_force(bus: Bus, sid: int, direction: int) -> tuple[int, bool]:
    """Find a hard-stop tick extrapolated to *zero* contact force.

    A hard stop is not perfectly rigid: the structure compresses by an amount
    roughly proportional to the contact force, so a single firm push reads a few
    counts past the true stop (a known servo-homing error source). This touches
    the stop at two torque caps and linearly extrapolates the reading to zero
    force, cancelling that deflection. Falls back to the single reading if the
    direction only reaches the encoder boundary.

    Returns
    -------
    tuple[int, bool]
        ``(tick, found)`` — the deflection-corrected stop count and whether a
        real stop (not the encoder boundary) was met.
    """
    f1, f2 = COMPLIANCE_FORCES
    p1, found1 = _drive_to_stop(bus, sid, direction, max_load=f1)
    if not found1:
        return p1, False
    p2, _ = _drive_to_stop(bus, sid, direction, max_load=f2)
    # pos(f) ≈ pos0 + k·f  →  pos0 = p1 - (p2 - p1)·f1/(f2 - f1)
    pos0 = p1 - (p2 - p1) * f1 / (f2 - f1)
    return int(round(pos0)), True


# --------------------------------------------------------------------------- #
# #4 Seed verification — the "within one revolution" disambiguation
# --------------------------------------------------------------------------- #


def seed_tolerance_ticks(resolution: int = RESOLUTION) -> int:
    """How far the operator may be off the seed and still be disambiguated.

    A single-turn absolute encoder maps uniquely within one revolution, so the
    operator only needs to be within ~half a turn of the intended landmark for
    the reading to resolve to the right one (the ABB "within one revolution"
    mastering principle — precision is the encoder's job, not the human's).
    """
    return resolution // 2 - 1


def _circular_diff(a: float, b: float, resolution: int = RESOLUTION) -> float:
    d = (a - b) % resolution
    return d - resolution if d > resolution // 2 else d


def verify_seed(
    measured_ticks: dict,
    expected_ticks: dict,
    *,
    resolution: int = RESOLUTION,
    tol: int | None = None,
) -> list:
    """Check the operator's seed pose is within the disambiguation tolerance.

    Returns ``(joint, off_by_ticks, ok)`` per joint; ``ok=False`` means that
    joint is more than ~half a turn off — the wrong stop/revolution was set, so
    re-ask before calibrating to a wrong landmark.
    """
    tol = seed_tolerance_ticks(resolution) if tol is None else tol
    out = []
    for j, exp in expected_ticks.items():
        if j not in measured_ticks:
            continue
        d = _circular_diff(measured_ticks[j], exp, resolution)
        out.append((j, d, abs(d) <= tol))
    return out


def touch_off(bus: Bus, spec: JointSpec) -> dict:
    """Calibrate one arm joint by touching its hard stop(s).

    Drives both directions. Each reachable hard stop sits at a known URDF limit
    (the ``-1`` direction reaches the minimum-tick end, ``+1`` the maximum-tick
    end; which limit that is depends on ``sign``), so even a single reachable
    stop pins the offset. With both reachable, the implied scale is also
    cross-checked against the datasheet.
    """
    assert spec.limits is not None
    lower, upper = spec.limits
    t_neg, found_neg = _drive_to_stop(bus, spec.servo_id, -1)
    t_pos, found_pos = _drive_to_stop(bus, spec.servo_id, +1)

    # Map each reachable stop to the URDF limit it represents.
    stops: list[tuple[int, float]] = []
    if found_neg:
        stops.append((t_neg, lower if spec.sign > 0 else upper))
    if found_pos:
        stops.append((t_pos, upper if spec.sign > 0 else lower))
    if not stops:
        raise RuntimeError(
            f"{spec.name}: no hard stop reachable within the encoder range in "
            "either direction — cannot anchor the offset. Check the joint moves "
            "freely and that its zero is not doubly off the encoder span."
        )

    # ticks = offset + sign*tpr*angle  ->  offset = t - sign*tpr*a
    offsets = [t - spec.sign * TICKS_PER_RAD * a for (t, a) in stops]
    offset = sum(offsets) / len(offsets)

    measured_tpr: float | None = None
    if len(stops) == 2:
        (t0, a0), (t1, a1) = stops
        measured_tpr = abs((t1 - t0) / (a1 - a0)) if a1 != a0 else None

    # Usable range = the fitted map's ticks at each limit, clamped to the encoder.
    def _clamp(t: float) -> int:
        return int(max(0, min(RESOLUTION - 1, round(t))))

    pred = sorted(
        (
            offset + spec.sign * TICKS_PER_RAD * lower,
            offset + spec.sign * TICKS_PER_RAD * upper,
        )
    )
    return {
        "joint": spec.name,
        "servo_id": spec.servo_id,
        "sign": spec.sign,
        "ticks_per_rad": TICKS_PER_RAD,
        "offset_ticks": offset,
        "range_min_ticks": _clamp(pred[0]),
        "range_max_ticks": _clamp(pred[1]),
        "measured_ticks_per_rad": measured_tpr,
        "stops_found": len(stops),
        "method": "touch_off",
    }


def wheel_zero(bus: Bus, spec: JointSpec) -> dict:
    """Record a continuous wheel servo's current count as its zero."""
    return {
        "joint": spec.name,
        "servo_id": spec.servo_id,
        "sign": spec.sign,
        "ticks_per_rad": TICKS_PER_RAD,
        "offset_ticks": float(bus.position(spec.servo_id)),
        "range_min_ticks": None,
        "range_max_ticks": None,
        "measured_ticks_per_rad": None,
        "method": "wheel_zero",
    }


def hold_position(bus: Bus, sid: int, max_load: float | None = None) -> None:
    """Lock a servo at its current position (torque on, goal = present).

    Used to keep the distal joints folded while a proximal joint sweeps, so the
    limp arm cannot droop under gravity and lengthen into a collision. Pass
    ``max_load`` to set a *stiff* hold torque (a gentle approach cap leaves the
    held joint compliant enough to sag ~20° and drop the gripper to the floor —
    L-005); the swept joint stays gentle, only the held ones are stiffened.
    """
    pos = bus.position(sid)
    if max_load is not None:
        bus._write2(sid, ADDR_TORQUE_LIMIT, max(1, min(1000, round(max_load * 1000))))
    bus._write1(sid, ADDR_TORQUE_ENABLE, 1)
    bus._write2(sid, ADDR_GOAL_POSITION, pos)


def calibrate_from_seed(
    bus: Bus, spec: JointSpec, *, samples: int = 7
) -> tuple[dict, str]:
    """Calibrate a joint held at one of its hard stops — inferring *which* stop.

    The arm is already at the seed pose, so a joint sits against one of its two
    limits, but the operator can't easily tell which. We read the encoder and
    compute the offset for *both* limits; only the physically valid stop puts the
    q=0 tick inside the encoder range, so we pick that one. No motor motion, so
    it cannot collide. The direction (sign) is assumed and stays unverified —
    confirm it in teleop (a reversed joint just needs its sign flipped).

    Returns ``(calibration, note)`` where ``note`` is a non-empty warning string
    when the stop could not be resolved cleanly.
    """
    lo, hi = spec.limits  # type: ignore[misc]
    t = sum(bus.position(spec.servo_id) for _ in range(samples)) / samples
    cands = [(q, t - spec.sign * TICKS_PER_RAD * q) for q in (lo, hi)]
    in_range = [(q, o) for q, o in cands if 0 <= o < RESOLUTION]
    note = ""
    if len(in_range) == 1:
        q_at, off = in_range[0]
    elif not in_range:
        q_at, off = min(cands, key=lambda c: abs(c[1] - RESOLUTION / 2))
        note = "no stop puts q=0 in the encoder range (range wraps); extrapolated"
    else:
        q_at, off = min(in_range, key=lambda c: abs(c[1] - RESOLUTION / 2))
        note = "both stops plausible — could not disambiguate; verify in teleop"

    # Derive the usable travel range from the model limits + the fitted offset
    # (clamped to the encoder), so the LeRobot file is properly bounded.
    def _clamp(x: float) -> int:
        return int(max(0, min(RESOLUTION - 1, round(x))))

    edges = sorted(off + spec.sign * TICKS_PER_RAD * q for q in (lo, hi))
    return {
        "joint": spec.name,
        "servo_id": spec.servo_id,
        "sign": spec.sign,
        "ticks_per_rad": TICKS_PER_RAD,
        "offset_ticks": off,
        "range_min_ticks": _clamp(edges[0]),
        "range_max_ticks": _clamp(edges[1]),
        "measured_ticks_per_rad": None,
        "seed_stop_rad": q_at,
        "method": "single_pose_seed",
    }, note


def _wrap_signed(value: int, resolution: int = RESOLUTION) -> int:
    """Wrap ``value`` into the signed half-open range ``[-res/2, res/2)``."""
    half = resolution // 2
    return ((value + half) % resolution) - half


def _homing_offset(mid_tick: float, resolution: int = RESOLUTION) -> int:
    """LeRobot homing offset centring ``mid_tick``, kept register-writable.

    LeRobot writes this into the servo's signed 11-bit Position-Correction
    register (magnitude ≤ 2047) and its naive ``2048 - mid`` overflows when the
    encoder zero is off-centre — the "Magnitude ... exceeds 2047" crash (lerobot
    issue #3193). For a single-turn encoder the correction is modular, so we
    *wrap* (then clamp the one extreme count) instead of overflowing.
    """
    limit = resolution // 2 - 1
    wrapped = _wrap_signed(round(resolution // 2 - mid_tick), resolution)
    return max(-limit, min(limit, wrapped))


def lerobot_entry(cal: dict) -> dict:
    """One LeRobot-shaped per-motor calibration entry — overflow-safe.

    Mirrors LeRobot's ``{homing_offset, range_min, range_max, drive_mode}`` but
    keeps ``homing_offset`` in the writable signed-11-bit range (no issue-#3193
    crash) and gives a continuous joint (no measured stops) a full-turn range
    instead of LeRobot's mishandling. Confirm the homing-offset *sign* against
    your installed LeRobot version before relying on it.
    """
    drive_mode = 0 if cal["sign"] >= 0 else 1
    if cal["range_min_ticks"] is not None:
        rmin, rmax = sorted((int(cal["range_min_ticks"]), int(cal["range_max_ticks"])))
        mid = 0.5 * (rmin + rmax)
    else:  # continuous / no usable stop: full turn, centre on the fitted zero
        rmin, rmax = 0, RESOLUTION - 1
        mid = cal["offset_ticks"]
    return {
        "id": cal["servo_id"],
        "drive_mode": drive_mode,
        "homing_offset": _homing_offset(mid),
        "range_min": rmin,
        "range_max": rmax,
    }


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #


def run(args: argparse.Namespace) -> int:
    global DEBUG
    DEBUG = bool(getattr(args, "debug", False))
    specs = LAYOUTS[args.robot]
    expected = tuple(sorted(s.servo_id for s in specs))
    print(
        f"lekiwi-autosetup — robot={args.robot} port={args.port} "
        f"({'EXECUTE (will move)' if args.execute else 'read-only'})"
    )

    bus = Bus(args.port, args.baud)
    warnings: list[str] = []
    try:
        # --- discover -----------------------------------------------------
        found = bus.scan(expected)
        missing = tuple(s for s in expected if s not in found)
        print("\ndiscovery:")
        print(f"  found    : {list(found)}")
        print(f"  expected : {list(expected)}")
        if missing:
            print(f"  MISSING  : {list(missing)}  — re-seat cables / check power")
            warnings.append(f"missing servos {list(missing)}")

        # --- health -------------------------------------------------------
        print("\nhealth:")
        statuses = []
        for sid in found:
            v, t = bus.voltage(sid), bus.temperature(sid)
            statuses.append(
                {
                    "servo_id": sid,
                    "voltage_v": v,
                    "temperature_c": t,
                    "position_ticks": bus.position(sid),
                }
            )
            flag = ""
            if v < args.min_voltage:
                flag += f"  LOW VOLTAGE (<{args.min_voltage} V)"
                warnings.append(f"servo {sid} low voltage {v:.1f} V")
            if t > args.max_temp:
                flag += f"  HOT (>{args.max_temp} °C)"
                warnings.append(f"servo {sid} hot {t:.0f} °C")
            print(
                f"  servo {sid}: {v:4.1f} V, {t:3.0f} °C, pos={bus.position(sid)}{flag}"
            )

        # --- calibrate ----------------------------------------------------
        calibrations: list[dict] = []
        if getattr(args, "from_seed", False):
            print(
                "\ncalibration from the current seed pose (READING ONLY — no motion):"
            )
            for spec in specs:
                if spec.servo_id not in found:
                    warnings.append(f"{spec.name}: servo {spec.servo_id} absent")
                    continue
                if spec.kind == "wheel":
                    cal = wheel_zero(bus, spec)
                elif spec.name in NO_SEED_JOINTS:
                    warnings.append(f"{spec.name}: no reliable stop — skipped")
                    continue
                else:
                    cal, note = calibrate_from_seed(bus, spec)
                    if note:
                        warnings.append(f"{spec.name}: {note}")
                calibrations.append(cal)
                stop = cal.get("seed_stop_rad")
                at = f"  (at {stop:+.2f} rad stop)" if stop is not None else ""
                print(
                    f"  {spec.name:14s} id={spec.servo_id} "
                    f"offset={cal['offset_ticks']:8.1f} ticks  ({cal['method']}){at}"
                )
        elif not args.execute:
            print(
                "\nread-only run complete — nothing moved. To calibrate from the "
                "seed pose (no motion) use --from-seed, or --execute to touch off."
            )
        else:
            print("\ncalibration (touching hard stops, gently):")
            if getattr(args, "hold_others", False) and args.only:
                held = [
                    s
                    for s in specs
                    if s.kind == "arm" and s.name != args.only and s.servo_id in found
                ]
                for s in held:
                    hold_position(bus, s.servo_id)
                print(f"  holding folded: {[s.name for s in held]}")
            for spec in specs:
                if spec.servo_id not in found:
                    warnings.append(
                        f"{spec.name}: servo {spec.servo_id} absent — skipped"
                    )
                    continue
                if args.only and spec.name != args.only:
                    continue
                print(f"  → {spec.name} (id {spec.servo_id}) ...", flush=True)
                if spec.kind == "wheel":
                    cal = wheel_zero(bus, spec)
                else:
                    cal = touch_off(bus, spec)
                    m = cal["measured_ticks_per_rad"]
                    if m and abs(m / TICKS_PER_RAD - 1.0) > SCALE_WARN_RTOL:
                        warnings.append(
                            f"{spec.name}: measured scale {m:.0f} vs datasheet "
                            f"{TICKS_PER_RAD:.0f} counts/rad — loose horn / wrong sign?"
                        )
                    if cal.get("stops_found") == 1:
                        warnings.append(
                            f"{spec.name}: only one hard stop is within the encoder "
                            "range — offset anchored from it, but the scale is the "
                            "datasheet value and the direction (sign) is unverified."
                        )
                calibrations.append(cal)
                stops = cal.get("stops_found")
                tag = (
                    f" ({cal['method']}, {stops} stop{'s' if stops != 1 else ''})"
                    if stops is not None
                    else f"  ({cal['method']})"
                )
                rng = (
                    f"  range[{cal['range_min_ticks']}..{cal['range_max_ticks']}]"
                    if cal["range_min_ticks"] is not None
                    else ""
                )
                print(
                    f"  {spec.name:14s} id={spec.servo_id} offset="
                    f"{cal['offset_ticks']:8.1f} ticks{tag}{rng}"
                )

        # --- write --------------------------------------------------------
        if args.out and (calibrations or not args.execute):
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            native = {
                "schema": "lekiwi-autosetup/v1",
                "robot": args.robot,
                "discovery": {
                    "found": list(found),
                    "expected": list(expected),
                    "missing": list(missing),
                },
                "health": statuses,
                "calibrations": calibrations,
                "warnings": warnings,
            }
            (out / "calibration.json").write_text(json.dumps(native, indent=2) + "\n")
            if calibrations:
                lr = {c["joint"]: lerobot_entry(c) for c in calibrations}
                (out / f"{args.robot}_lerobot.json").write_text(
                    json.dumps(lr, indent=2) + "\n"
                )
            print(f"\nwrote {out / 'calibration.json'}")

        if warnings:
            print("\nWARNINGS:")
            for w in warnings:
                print(f"  ! {w}")
            return 1
        print("\nall clear.")
        return 0
    finally:
        bus.relax_all(expected)  # leave every servo limp
        bus.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lekiwi_autosetup.py",
        description="First-run bring-up + offset calibration for a LeKiwi / SO-101.",
    )
    p.add_argument(
        "--port",
        required=True,
        help="Servo-bus serial device (find it with `lerobot-find-port`).",
    )
    p.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    p.add_argument("--robot", choices=tuple(LAYOUTS), default="lekiwi")
    p.add_argument(
        "--execute",
        action="store_true",
        help="MOVE the arm to calibrate. Omit for a read-only check.",
    )
    p.add_argument(
        "--from-seed",
        action="store_true",
        help="Calibrate by READING the arm at the seed pose (no motion). The "
        "arm must already be folded to the all-stops seed.",
    )
    p.add_argument(
        "--hold-others",
        action="store_true",
        help="During a single-joint --execute sweep, lock the other arm joints "
        "at their current (folded) position so they can't droop into a collision.",
    )
    p.add_argument(
        "--only",
        default=None,
        help="Calibrate just this one joint (e.g. shoulder_pan).",
    )
    p.add_argument("--out", default=None, help="Directory to write calibration files.")
    p.add_argument(
        "--debug",
        action="store_true",
        help="Print position/load every few polls during touch-off.",
    )
    p.add_argument("--min-voltage", type=float, default=10.0)
    p.add_argument("--max-temp", type=float, default=55.0)
    return p


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
