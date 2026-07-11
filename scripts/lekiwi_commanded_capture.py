#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors
"""On-device commanded-excitation capture for a LeKiwi / SO-101 STS3215 joint.

This is the **split-capture** half of the active-sysid path: the
Raspberry Pi has the Feetech servo bus but not the JAX/MJX stack, so it runs this
lightweight, dependency-light script (``scservo_sdk`` only) to *command* an
excitation and record telemetry; the fit then runs on a workstation via
:func:`jaxterity.calibrate.load_commanded_capture` +
:func:`jaxterity.calibrate.identify_closed_loop`.

WHY COMMANDED, NOT FREE-COAST.  A geared Feetech STS3215 is **not back-drivable**:
a torque-off "free coast" decelerates to rest in ~170 ms, nothing like the
frictionless observer, and gravity-loaded joints cannot torque-off safely at all.
So this script keeps **torque ON the whole time** and excites *through the servo's
own position inner loop*: it streams a band-limited multisine of goal *positions*
and records the servo's response (position, speed, load) plus bus voltage and
temperature. The servo's PRESENT_LOAD is the inner loop's PWM-duty command — a
proxy for motor current, hence joint torque (duty x bus-voltage, back-EMF
neglected at the low excitation speeds). The workstation fit turns that telemetry
into joint friction/damping + a load->torque scale via an inverse-dynamics
(equation-error) observer. NO joint is ever torque-off.

SAFETY.  Pure position mode, torque held on. The goal trajectory is clamped to a
guard band strictly inside the calibrated tick range; the loop aborts and restores
a stationary hold the instant the measured position leaves the band. A slow,
guarded ramp moves the joint to the excitation center before the multisine starts.
``--dry-run`` records the response with NO goal writes (the joint just holds), for
a no-motion bus check. Only the one ``--servo-id`` is ever commanded.

Usage (on the Pi)::

    python lekiwi_commanded_capture.py --servo-id 1 --out /tmp/cmd_id1.json \
        --calib calibration_final.json

Output JSON conforms to the schema
:func:`jaxterity.calibrate.load_commanded_capture` reads.
"""

from __future__ import annotations

import argparse
import json
import math
import time

from scservo_sdk import COMM_SUCCESS, PacketHandler, PortHandler

# --- STS3215 control table (little-endian / SCS_END = 0) ---
ADDR_MODE = 33  # 0 = position, 1 = wheel/constant-speed
ADDR_TORQUE_ENABLE = 40
ADDR_GOAL_POSITION = 42  # 2 bytes
ADDR_PRESENT_POSITION = 56  # 2 bytes, 0..4095
ADDR_PRESENT_SPEED = 58  # 2 bytes, signed-magnitude (bit15 = sign)
ADDR_PRESENT_LOAD = 60  # 2 bytes, signed-magnitude (bit10 = sign), 0..1000 (duty .1%)
ADDR_PRESENT_VOLTAGE = 62  # 1 byte, decivolts
ADDR_PRESENT_TEMPERATURE = 63  # 1 byte, deg C
BAUD = 1_000_000
PROTO = 0  # STS/SMS little-endian

# --- conservative safety / excitation defaults ---
GUARD_MARGIN_TICKS = 80  # stay this far inside [range_min, range_max]
RAMP_TICKS_PER_S = 600  # slow approach to the excitation center (~0.9 rad/s)
SETTLE_S = 0.4  # hold at center before exciting
# Band-limited multisine (Hz): low f pins gravity/friction, higher f adds q'' for
# the inertia/load-scale reference. Kept <= 2.5 Hz so the inner loop can track.
MULTISINE_HZ = (0.25, 0.5, 0.9, 1.4, 2.1)


def _signed_mag(raw: int, sign_bit: int) -> int:
    mask = 1 << sign_bit
    return -(raw & (mask - 1)) if (raw & mask) else (raw & (mask - 1))


def _multisine(t: float, period: float, amp_ticks: float) -> float:
    """Schroeder-phased multisine, peak-normalized to +/- amp_ticks.

    Frequencies are snapped to integer multiples of 1/period so the signal is
    exactly periodic over the capture (clean spectral content for the fit).
    """
    ks = [max(1, round(f * period)) for f in MULTISINE_HZ]
    n = len(ks)
    # Schroeder phases minimize crest factor for a flat multisine.
    phases = [-math.pi * i * (i + 1) / n for i in range(n)]
    raw = sum(math.sin(2 * math.pi * k * t / period + p) for k, p in zip(ks, phases))
    # Peak of a unit-amplitude n-tone Schroeder multisine ~ sqrt(2n); normalize
    # conservatively so the commanded swing never exceeds amp_ticks.
    return amp_ticks * raw / math.sqrt(2.0 * n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--servo-id", type=int, default=1)
    ap.add_argument(
        "--calib",
        default="calibration_final.json",
        help="per-joint calibration JSON (the LeKiwi autosetup output)",
    )
    ap.add_argument("--out", default="/tmp/cmd_capture.json")
    ap.add_argument(
        "--center-ticks",
        type=int,
        default=-1,
        help="excitation center (default: mid safety band)",
    )
    ap.add_argument(
        "--amp-ticks",
        type=int,
        default=200,
        help="multisine half-amplitude in ticks (~0.31 rad @ 652 t/rad)",
    )
    ap.add_argument("--duration", type=float, default=14.0, help="excitation seconds")
    ap.add_argument("--rate-hz", type=float, default=80.0, help="target sample rate")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="record the response with NO goal writes (no motion)",
    )
    args = ap.parse_args()

    calib = json.load(open(args.calib))
    cj = next(c for c in calib["calibrations"] if c["servo_id"] == args.servo_id)
    if cj.get("range_min_ticks") is None or cj.get("range_max_ticks") is None:
        print(
            f"ABORT: joint {cj['joint']!r} has no tick range (wheel?); not excitable."
        )
        return 2
    rmin, rmax = int(cj["range_min_ticks"]), int(cj["range_max_ticks"])
    lo, hi = rmin + GUARD_MARGIN_TICKS, rmax - GUARD_MARGIN_TICKS
    center = args.center_ticks if args.center_ticks >= 0 else (lo + hi) // 2
    amp = int(args.amp_ticks)
    if not (lo <= center - amp and center + amp <= hi):
        print(
            f"ABORT: center {center} +/- {amp} leaves band [{lo},{hi}]; "
            "lower --amp-ticks or move --center-ticks."
        )
        return 2

    ph, pk = PortHandler(args.port), PacketHandler(PROTO)
    assert ph.openPort() and ph.setBaudRate(BAUD), "could not open bus"
    sid = args.servo_id

    def rpos() -> int:
        v, c, _ = pk.read2ByteTxRx(ph, sid, ADDR_PRESENT_POSITION)
        return v if c == COMM_SUCCESS else -1

    def rsig(addr: int, two: bool, sign_bit: int | None) -> int:
        if two:
            v, c, _ = pk.read2ByteTxRx(ph, sid, addr)
        else:
            v, c, _ = pk.read1ByteTxRx(ph, sid, addr)
        if c != COMM_SUCCESS:
            return 0
        return _signed_mag(v, sign_bit) if sign_bit is not None else v

    # Record the full arm pose once (the other joints hold static during a
    # single-joint excitation; shoulder_pan's effective inertia and every joint's
    # gravity torque depend on it, so the workstation fit needs the real pose, not
    # a default). Arm servos are ids 1..6; wheels (7..9) are excluded.
    arm_ids = [
        c["servo_id"]
        for c in calib["calibrations"]
        if c.get("range_min_ticks") is not None
    ]
    pose_ticks = {}
    for aid in arm_ids:
        v, cc, _ = pk.read2ByteTxRx(ph, aid, ADDR_PRESENT_POSITION)
        if cc == COMM_SUCCESS:
            pose_ticks[str(aid)] = v

    pos0 = rpos()
    print(
        f"id{sid} ({cj['joint']}) pos0={pos0}; band [{lo},{hi}]; "
        f"center={center} amp={amp} ({'DRY-RUN' if args.dry_run else 'COMMANDED'})"
    )
    if pos0 < 0:
        print("ABORT: servo did not answer position read.")
        ph.closePort()
        return 2

    def hold(at: int) -> None:
        at = max(lo, min(hi, at))
        pk.write1ByteTxRx(ph, sid, ADDR_TORQUE_ENABLE, 0)
        pk.write1ByteTxRx(ph, sid, ADDR_MODE, 0)  # position mode
        pk.write1ByteTxRx(ph, sid, ADDR_TORQUE_ENABLE, 1)
        pk.write2ByteTxRx(ph, sid, ADDR_GOAL_POSITION, at)

    samples = []  # (t, goal, pos, speed_raw, load_raw, volt_dV, temp_C)
    aborted = False
    try:
        # 1. enter position mode, torque ON, slow guarded ramp to the center.
        if not args.dry_run:
            hold(pos0)
            time.sleep(0.05)
            step = max(1, int(RAMP_TICKS_PER_S / max(args.rate_hz, 1.0)))
            cur = pos0
            # A joint may legitimately start just *outside* its calibrated range
            # (e.g. a parked wrist); tolerate that while ramping IN, guarding only
            # against running further out than where it started. The inner band is
            # enforced during the excitation itself.
            app_lo = min(rmin, pos0) - 30
            app_hi = max(rmax, pos0) + 30
            while cur != center:
                cur += step if center > cur else -step
                if (step > 0 and cur > center) or (step < 0 and cur < center):
                    cur = center
                cur = max(lo, min(hi, cur))
                pk.write2ByteTxRx(ph, sid, ADDR_GOAL_POSITION, cur)
                p = rpos()
                if not (app_lo <= p <= app_hi):
                    print(f"GUARD tripped during approach at {p}; aborting.")
                    aborted = True
                    break
                time.sleep(1.0 / args.rate_hz)
            time.sleep(SETTLE_S)

        # 2. commanded multisine about the center (torque stays ON throughout).
        dt = 1.0 / args.rate_hz
        t0 = time.perf_counter()
        next_tick = t0
        while not aborted:
            t = time.perf_counter() - t0
            if t >= args.duration:
                break
            goal = center
            if not args.dry_run:
                goal = int(round(center + _multisine(t, args.duration, amp)))
                goal = max(lo, min(hi, goal))
                pk.write2ByteTxRx(ph, sid, ADDR_GOAL_POSITION, goal)
            p = rpos()
            spd = rsig(ADDR_PRESENT_SPEED, True, 15)
            load = rsig(ADDR_PRESENT_LOAD, True, 10)
            volt = rsig(ADDR_PRESENT_VOLTAGE, False, None)
            temp = rsig(ADDR_PRESENT_TEMPERATURE, False, None)
            samples.append((t, goal, p, spd, load, volt, temp))
            # During the commanded multisine the joint must stay in the inner band;
            # in dry-run nothing is commanded, so just record wherever it sits.
            if not args.dry_run and not (lo <= p <= hi):
                print(f"GUARD tripped during excitation at {p}; restoring.")
                aborted = True
                break
            next_tick += dt
            sleep = next_tick - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
    finally:
        if not args.dry_run:
            hold(center)  # leave the joint holding a safe, stationary pose
        ph.closePort()

    if not samples:
        print("no samples captured.")
        return 3
    ts = [s[0] for s in samples]
    rate = (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 else 0.0
    out = {
        "schema": "lekiwi-commanded-capture-v1",
        "joint": cj["joint"],
        "servo_id": sid,
        "dry_run": args.dry_run,
        "aborted": aborted,
        "excitation": {
            "kind": "schroeder_multisine",
            "center_ticks": center,
            "amp_ticks": amp,
            "duration_s": args.duration,
            "multisine_hz": list(MULTISINE_HZ),
            "target_rate_hz": args.rate_hz,
        },
        "calibration": {
            k: cj[k]
            for k in (
                "sign",
                "ticks_per_rad",
                "offset_ticks",
                "range_min_ticks",
                "range_max_ticks",
            )
        },
        # full arm pose (servo_id -> ticks) held during the single-joint excitation
        "pose_ticks": pose_ticks,
        "t": ts,
        "goal_ticks": [s[1] for s in samples],
        "pos_ticks": [s[2] for s in samples],
        "speed_raw": [s[3] for s in samples],
        "load_raw": [s[4] for s in samples],
        "volt_dV": [s[5] for s in samples],
        "temp_C": [s[6] for s in samples],
    }
    json.dump(out, open(args.out, "w"))
    travel = max(out["pos_ticks"]) - min(out["pos_ticks"])
    tag = " [ABORTED]" if aborted else ""
    print(
        f"captured {len(samples)} samples @ ~{rate:.0f} Hz; travel {travel} ticks "
        f"({travel * 360.0 / 4096.0:.1f} deg){tag} -> {args.out}"
    )
    return 3 if aborted else 0


if __name__ == "__main__":
    raise SystemExit(main())
