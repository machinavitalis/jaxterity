#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""On-device wheel-servo health capture: the three ids the checkup ignores.

The per-unit checkup fits the six arm joints and says nothing at all about the
drive wheels (ids 7-9), which is half the mechanism on a LeKiwi. This is their
capture: a speed staircase in both directions, then a coast-down, on each wheel
in turn, logging commanded speed against measured speed, load duty, bus voltage
and temperature.

WHEELS OFF THE GROUND. This script spins the drive wheels, so on the floor it
*drives the robot*. Run it with the base on blocks. There is no way for the
script to check that, so it says so loudly and defaults to short, slow spins.

What a free-spinning wheel can and cannot tell you:

* **Speed tracking** — commanded vs measured speed across the staircase. A wheel
  that will not reach its commanded speed unloaded is a wheel with a problem
  (dragging bearing, failing winding, a driver derating itself).
* **No-load duty** — how much torque the servo spends to spin nothing. This is
  the single most diagnostic number here: three nominally identical wheels
  should draw nearly the same, and the outlier is the suspect.
* **Coast-down** — with the goal speed zeroed and torque released, the decay
  time constant is the wheel's own internal drag (bearing + gearbox + magnetic).
  Note what this is *not*: with the wheels off the ground there is no ground
  contact, so this is **not** rolling resistance. It is the no-contact baseline
  that a later on-floor coast-down must be compared against — the difference of
  the two is the rolling resistance proper
  (:func:`jaxterity.physics.compliance.fit_coast_down`).
* **Symmetry** — the comparison across the three wheels, which needs no
  reference unit and no absolute calibration, so it is the health signal that
  works on day one.

A genuine **stall test** is deliberately absent: stalling a wheel requires
physically holding it, which a script cannot do and must not fake by driving
into a limit. Hold a wheel by hand during a staircase segment if you want the
stall duty, and note it separately.

Dependency-light (``scservo_sdk`` only), like its sibling capture scripts.
Schema ``lekiwi-wheel-capture-v1``. Torque and the original mode are restored on
every exit path, including SIGTERM.
"""

from __future__ import annotations

import argparse
import json
import signal
import time

from scservo_sdk import COMM_SUCCESS, PacketHandler, PortHandler

ADDR_MODE = 33  # 0 = position, 1 = wheel/constant-speed
ADDR_TORQUE_ENABLE = 40
ADDR_GOAL_SPEED = 46  # 2 bytes, signed-magnitude (bit15 = sign)
ADDR_PRESENT_POSITION = 56
ADDR_PRESENT_SPEED = 58  # 2 bytes, signed-magnitude (bit15 = sign)
ADDR_PRESENT_LOAD = 60  # 2 bytes, signed-magnitude (bit10 = sign), duty .1%
ADDR_PRESENT_VOLTAGE = 62  # 1 byte, decivolts
ADDR_PRESENT_TEMPERATURE = 63  # 1 byte, deg C
BAUD = 1_000_000
PROTO = 0

SCHEMA = "lekiwi-wheel-capture-v1"
#: Conservative default staircase (raw speed units). The wheels reach a few
#: hundred rpm at 1000+; these stay well inside that so an unexpectedly loaded
#: wheel cannot run away.
DEFAULT_STEPS = (200, 400, 600)


def _signed_mag(raw: int, sign_bit: int) -> int:
    mask = 1 << sign_bit
    return -(raw & (mask - 1)) if (raw & mask) else (raw & (mask - 1))


def _encode_speed(speed: int) -> int:
    raw = abs(int(speed)) & 0x7FFF
    return raw | 0x8000 if speed < 0 else raw


def _terminate_as_interrupt(signum, frame):
    """Turn SIGTERM into KeyboardInterrupt so the stop/restore cleanup runs."""
    raise KeyboardInterrupt(f"signal {signum}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument(
        "--calib",
        default="calibration_final.json",
        help="per-joint calibration JSON (for wheel id -> name)",
    )
    ap.add_argument("--out", default="/tmp/wheel_capture.json")
    ap.add_argument("--servo-ids", default="7,8,9")
    ap.add_argument(
        "--steps",
        default=",".join(str(s) for s in DEFAULT_STEPS),
        help="comma-separated raw speed steps, applied forward then reverse",
    )
    ap.add_argument("--step-s", type=float, default=2.0, help="hold per step")
    ap.add_argument("--coast-s", type=float, default=3.0, help="logged coast-down")
    ap.add_argument("--rate-hz", type=float, default=50.0)
    ap.add_argument(
        "--on-blocks",
        action="store_true",
        help="REQUIRED acknowledgement that the wheels are off the ground",
    )
    args = ap.parse_args()
    signal.signal(signal.SIGTERM, _terminate_as_interrupt)

    if not args.on_blocks:
        print(
            "ABORT: this spins the drive wheels, which drives the robot if they "
            "are on the floor. Put the base on blocks and pass --on-blocks."
        )
        return 2

    ids = [int(x) for x in args.servo_ids.split(",") if x.strip()]
    steps = [int(x) for x in args.steps.split(",") if x.strip()]
    names = {}
    try:
        calib = json.load(open(args.calib))
        names = {
            int(c["servo_id"]): c.get("joint", "")
            for c in calib.get("calibrations", [])
        }
    except (OSError, ValueError):
        pass  # names are cosmetic; the capture is keyed by id

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

    original_mode: dict[int, int] = {}
    samples: list[tuple] = []
    aborted = False
    t0 = time.perf_counter()

    def log_for(sid: int, seconds: float, phase: str, commanded: int) -> None:
        period = 1.0 / args.rate_hz
        start = time.perf_counter()
        next_tick = start
        while time.perf_counter() - start < seconds:
            speed_raw = read2(sid, ADDR_PRESENT_SPEED)
            load_raw = read2(sid, ADDR_PRESENT_LOAD)
            samples.append(
                (
                    time.perf_counter() - t0,
                    sid,
                    phase,
                    commanded,
                    None if speed_raw is None else _signed_mag(speed_raw, 15),
                    None if load_raw is None else _signed_mag(load_raw, 10),
                    read1(sid, ADDR_PRESENT_VOLTAGE),
                    read1(sid, ADDR_PRESENT_TEMPERATURE),
                    read2(sid, ADDR_PRESENT_POSITION),
                )
            )
            next_tick += period
            time.sleep(max(0.0, next_tick - time.perf_counter()))

    try:
        for sid in ids:
            mode = read1(sid, ADDR_MODE)
            if mode is None:
                print(f"ABORT: servo {sid} did not answer; is it on the bus?")
                return 3
            original_mode[sid] = mode
            print(f"wheel {sid} ({names.get(sid, '?')}): mode {mode} -> 1 (wheel)")
            ph.write1ByteTxRx(port, sid, ADDR_TORQUE_ENABLE, 0)
            ph.write1ByteTxRx(port, sid, ADDR_MODE, 1)
            ph.write1ByteTxRx(port, sid, ADDR_TORQUE_ENABLE, 1)

            for signed in [s for s in steps] + [-s for s in steps]:
                ph.write2ByteTxRx(port, sid, ADDR_GOAL_SPEED, _encode_speed(signed))
                log_for(sid, args.step_s, "staircase", signed)
            # Coast: zero the goal and release torque, then watch it wind down.
            ph.write2ByteTxRx(port, sid, ADDR_GOAL_SPEED, 0)
            ph.write1ByteTxRx(port, sid, ADDR_TORQUE_ENABLE, 0)
            log_for(sid, args.coast_s, "coast", 0)
    except KeyboardInterrupt:
        aborted = True
        print("\ninterrupted — stopping wheels and restoring modes.")
    finally:
        for sid in ids:
            ph.write2ByteTxRx(port, sid, ADDR_GOAL_SPEED, 0)
            ph.write1ByteTxRx(port, sid, ADDR_TORQUE_ENABLE, 0)
            if sid in original_mode:
                ph.write1ByteTxRx(port, sid, ADDR_MODE, original_mode[sid])
        port.closePort()

    if not samples:
        print("no samples captured.")
        return 3
    times = [s[0] for s in samples]
    rate = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0
    out = {
        "schema": SCHEMA,
        "servo_ids": ids,
        "joints": {str(sid): names.get(sid, "") for sid in ids},
        "aborted": aborted,
        "excitation": {
            "kind": "wheel_speed_staircase",
            "steps": steps,
            "step_s": args.step_s,
            "coast_s": args.coast_s,
            "target_rate_hz": args.rate_hz,
        },
        "achieved_rate_hz": rate,
        "t": times,
        "servo_id": [s[1] for s in samples],
        "phase": [s[2] for s in samples],
        "commanded_speed": [s[3] for s in samples],
        "speed_raw": [s[4] for s in samples],
        "load_raw": [s[5] for s in samples],
        "volt_dV": [s[6] for s in samples],
        "temp_C": [s[7] for s in samples],
        "pos_ticks": [s[8] for s in samples],
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh)
    print(f"wrote {args.out}: {len(samples)} samples ({rate:.1f} Hz achieved)")
    for sid in ids:
        rows = [s for s in samples if s[1] == sid and s[2] == "staircase"]
        if not rows:
            continue
        top = max(steps)
        at_top = [r for r in rows if r[3] == top]
        speeds = [abs(r[4]) for r in at_top if r[4] is not None]
        duty = [abs(r[5]) for r in at_top if r[5] is not None]
        print(
            f"  wheel {sid} ({names.get(sid, '?')}) at cmd {top}: "
            f"speed {sorted(speeds)[len(speeds) // 2] if speeds else '?'}, "
            f"duty {sorted(duty)[len(duty) // 2] if duty else '?'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
