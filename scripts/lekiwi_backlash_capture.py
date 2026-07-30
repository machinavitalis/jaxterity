#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""On-device reversal sweep: the lost motion the checkup never measured.

Backlash is a headline wear mechanism for geared hobby servos, and nothing in the
checkup looks for it. This captures the raw material: a **slow, bidirectional**
goal-position sweep at a pinned pose, per joint, logging commanded goal against
measured position and load. :mod:`jaxterity.checkup.backlash` reads it.

What this can and cannot establish, stated up front because the difference
matters more than the number:

* The measurement is the **command-to-measurement hysteresis loop** — sweep the
  goal up, sweep it back, and the measured position does not retrace its path.
* Its width is **total lost motion referred to the output**, an *upper bound* on
  gear play. Three things live inside it and one encoder cannot separate them:
  gear backlash, the servo's own position-loop deadband, and compliance.
* A width that **grows** over a unit's life is real wear evidence — the deadband
  and compliance terms are constant. The absolute value is not a gear-play
  number, and this script does not report one.
* Separating gear play needs a second angle reference (an external encoder or a
  dial indicator and a human). That is bench work.

Two protocol rules the analysis enforces, so the capture had better honour them:

* **Both directions**, or there is no loop and no width to measure.
* **Slowly.** Quasi-static, or the loop width picks up velocity lag from the
  position loop and inertia — a dynamic effect that inflates the number with no
  wear behind it. ``--ticks-per-s`` defaults deliberately low.

Only one joint moves at a time, at its pinned pose; the rest hold. Sweeps stay
inside the calibration's guard band. Torque is released on every exit path,
including SIGTERM. Schema ``lekiwi-backlash-capture-v1``.

Dependency-light (``scservo_sdk`` only), like its sibling capture scripts.
"""

from __future__ import annotations

import argparse
import json
import signal
import time

from scservo_sdk import COMM_SUCCESS, PacketHandler, PortHandler

ADDR_TORQUE_ENABLE = 40
ADDR_GOAL_POSITION = 42
ADDR_PRESENT_POSITION = 56
ADDR_PRESENT_LOAD = 60  # 2 bytes, signed-magnitude (bit10 = sign), duty .1%
ADDR_PRESENT_VOLTAGE = 62
ADDR_PRESENT_TEMPERATURE = 63
BAUD = 1_000_000
PROTO = 0

SCHEMA = "lekiwi-backlash-capture-v1"

#: Default sweep half-range, ticks either side of the pinned centre. Wide enough
#: that the loop is many times the encoder quantum, narrow enough to stay well
#: inside a guard band and to keep the arm near its vetted pose.
DEFAULT_SPAN_TICKS = 250
#: Default sweep rate, ticks/s. Slow on purpose — see the module docstring on why
#: a fast sweep inflates the measured width. At ~650 ticks/rad this is about
#: 0.06 rad/s.
DEFAULT_TICKS_PER_S = 40.0
#: Guard band kept clear of each end of the calibrated range, ticks. Matches the
#: other capture scripts' guard.
GUARD_TICKS = 80


def _signed_mag(raw: int, sign_bit: int) -> int:
    mask = 1 << sign_bit
    return -(raw & (mask - 1)) if (raw & mask) else (raw & (mask - 1))


def _terminate_as_interrupt(signum, frame):
    """Turn SIGTERM into KeyboardInterrupt so the torque-release cleanup runs."""
    raise KeyboardInterrupt(f"signal {signum}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--calib", default="calibration_final.json")
    ap.add_argument("--out", default="/tmp/backlash_capture.json")
    ap.add_argument(
        "--servo-ids",
        default="1,2,3,4,5",
        help="arm servos to sweep (wheels have no position loop to test)",
    )
    ap.add_argument("--span-ticks", type=int, default=DEFAULT_SPAN_TICKS)
    ap.add_argument("--ticks-per-s", type=float, default=DEFAULT_TICKS_PER_S)
    ap.add_argument("--rate-hz", type=float, default=50.0)
    ap.add_argument(
        "--settle-s",
        type=float,
        default=1.0,
        help="hold at each turnaround before reversing, s",
    )
    ap.add_argument(
        "--cycles", type=int, default=1, help="up-and-back cycles per joint"
    )
    args = ap.parse_args()
    signal.signal(signal.SIGTERM, _terminate_as_interrupt)

    ids = [int(x) for x in args.servo_ids.split(",") if x.strip()]
    try:
        calib = json.load(open(args.calib))
    except (OSError, ValueError) as exc:
        print(f"ABORT: cannot read {args.calib}: {exc}")
        return 2
    rows = {int(c["servo_id"]): c for c in calib.get("calibrations", [])}
    missing = [sid for sid in ids if sid not in rows]
    if missing:
        print(f"ABORT: no calibration for servo(s) {missing}; cannot bound the sweep.")
        return 2

    port = PortHandler(args.port)
    if not port.openPort() or not port.setBaudRate(BAUD):
        print(f"ABORT: cannot open {args.port} at {BAUD}.")
        return 3
    ph = PacketHandler(PROTO)

    def read1(sid, addr):
        v, comm, err = ph.read1ByteTxRx(port, sid, addr)
        return v if comm == COMM_SUCCESS and err == 0 else None

    def read2(sid, addr):
        v, comm, err = ph.read2ByteTxRx(port, sid, addr)
        return v if comm == COMM_SUCCESS and err == 0 else None

    samples: list[tuple] = []
    energized: set[int] = set()
    aborted = False
    t0 = time.perf_counter()

    def sweep(sid: int, start: int, end: int, phase: str) -> None:
        """Ramp the goal from *start* to *end* at the bounded rate, logging."""
        period = 1.0 / args.rate_hz
        step = args.ticks_per_s * period
        span = abs(end - start)
        n_steps = max(1, int(round(span / max(step, 1e-9))))
        direction = 1 if end >= start else -1
        next_tick = time.perf_counter()
        for i in range(n_steps + 1):
            goal = int(round(start + direction * min(i * step, span)))
            ph.write2ByteTxRx(port, sid, ADDR_GOAL_POSITION, goal)
            load_raw = read2(sid, ADDR_PRESENT_LOAD)
            samples.append(
                (
                    time.perf_counter() - t0,
                    sid,
                    phase,
                    goal,
                    read2(sid, ADDR_PRESENT_POSITION),
                    None if load_raw is None else _signed_mag(load_raw, 10),
                    read1(sid, ADDR_PRESENT_VOLTAGE),
                    read1(sid, ADDR_PRESENT_TEMPERATURE),
                )
            )
            next_tick += period
            time.sleep(max(0.0, next_tick - time.perf_counter()))

    try:
        for sid in ids:
            row = rows[sid]
            centre = read2(sid, ADDR_PRESENT_POSITION)
            if centre is None:
                print(f"ABORT: servo {sid} did not answer; is it on the bus?")
                return 3
            lo = int(row["range_min_ticks"]) + GUARD_TICKS
            hi = int(row["range_max_ticks"]) - GUARD_TICKS
            # Keep the whole sweep inside the guard band: shrink the span rather
            # than clamp mid-sweep, so both branches cover the same range and the
            # analysis has an overlap to compare.
            span = min(args.span_ticks, centre - lo, hi - centre)
            if span < 40:
                print(
                    f"  servo {sid} ({row.get('joint', '?')}): only {span} ticks "
                    "of room inside the guard band — skipped. Re-pose this joint "
                    "away from its limit if you need its lost motion."
                )
                continue
            print(
                f"  servo {sid} ({row.get('joint', '?')}): sweeping "
                f"{centre - span}..{centre + span} at {args.ticks_per_s:.0f} ticks/s"
            )
            ph.write1ByteTxRx(port, sid, ADDR_TORQUE_ENABLE, 1)
            energized.add(sid)
            ph.write2ByteTxRx(port, sid, ADDR_GOAL_POSITION, centre - span)
            time.sleep(max(args.settle_s, span / max(args.ticks_per_s, 1e-9)))
            for _ in range(max(1, args.cycles)):
                sweep(sid, centre - span, centre + span, "forward")
                time.sleep(args.settle_s)
                sweep(sid, centre + span, centre - span, "reverse")
                time.sleep(args.settle_s)
            ph.write2ByteTxRx(port, sid, ADDR_GOAL_POSITION, centre)
            time.sleep(args.settle_s)
            ph.write1ByteTxRx(port, sid, ADDR_TORQUE_ENABLE, 0)
            energized.discard(sid)
    except KeyboardInterrupt:
        aborted = True
        print("\ninterrupted — releasing torque on every energized servo.")
    finally:
        for sid in sorted(energized) or ids:
            ph.write1ByteTxRx(port, sid, ADDR_TORQUE_ENABLE, 0)
        port.closePort()

    if not samples:
        print("no samples captured.")
        return 3
    swept = sorted({s[1] for s in samples})
    times = [s[0] for s in samples]
    rate = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0
    out = {
        "schema": SCHEMA,
        "servo_ids": swept,
        "joints": {str(sid): rows[sid].get("joint", "") for sid in swept},
        "aborted": aborted,
        "excitation": {
            "kind": "slow_reversal_sweep",
            "span_ticks": args.span_ticks,
            "ticks_per_s": args.ticks_per_s,
            "settle_s": args.settle_s,
            "cycles": args.cycles,
            "target_rate_hz": args.rate_hz,
        },
        "achieved_rate_hz": rate,
        "t": times,
        "servo_id": [s[1] for s in samples],
        "phase": [s[2] for s in samples],
        "goal_ticks": [s[3] for s in samples],
        "pos_ticks": [s[4] for s in samples],
        "load_raw": [s[5] for s in samples],
        "volt_dV": [s[6] for s in samples],
        "temp_C": [s[7] for s in samples],
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh)
    print(f"wrote {args.out}: {len(samples)} samples ({rate:.1f} Hz achieved)")
    print(
        "Analyse with jaxterity.checkup.reversal_lost_motion. The width it "
        "reports is total lost motion, an upper bound on gear play — the servo's "
        "position-loop deadband and compliance are inside it."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
