#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""On-device thermal capture: hold a vetted pose and log the servos warming up.

WHY A SEPARATE MODE. The commanded-excitation captures are 14-16 s long and a
servo's thermal time constant is *minutes*, so those traces carry no usable
thermal signal at all — running ``fit_servo_thermal`` on them returns a
non-significant fit, which is the honest outcome and the reason the twin's
thermal defaults are still representative rather than measured. What the fit
needs is the opposite experiment: a long, boring, **static** hold in which the
servos dissipate a steady torque and the temperature register (1 degC quantized)
is given enough rise to resolve.

Mechanically this is the safest motion the rig performs. Nothing accelerates:
the arm ramps to the capture plan's already-vetted pose at the plan's bounded
speed, torque stays enabled, and then the arm just *holds* while gravity loads
the shoulder. There is no excitation, so there is no tip-over hazard beyond the
static margin the plan already cleared -- unlike the multisine captures, whose
commanded accelerations are the thing the ZMP gate exists to screen.

The protocol, and what each part is for:

1. **Ramp to the plan's pose** at ``transit_ticks_per_s`` (the plan validated
   exactly this motion shape from a worst-case start).
2. **Baseline** for ``--baseline-s``: hold with torque enabled and log. Gives the
   fit its ambient/initial condition instead of assuming one, and catches a rig
   that was already warm from a previous run.
3. **Hold** for ``--hold-s``: same pose, still logging. The rise is the signal.
4. **Cool** for ``--cool-s`` with torque *disabled*: the decay identifies the
   cooling time constant independently of the heating gain, which is what breaks
   the ``k_pwr``/``R_th`` correlation a rise-only trace leaves behind.

Sampling is 1 Hz by default -- the register has 1 degC resolution and a
minutes-scale constant, so faster sampling buys nothing but bus traffic.

Dependency-light by design (``scservo_sdk`` only), like its sibling capture
scripts: it runs on the Pi with no jaxterity install. The output schema is
``lekiwi-thermal-capture-v1``.

SAFETY. Refuses to run without a plan (the pose has to be one something vetted),
refuses a plan schema it does not recognize, clamps every commanded tick into
the calibration's guard band, and disables torque on exit -- including on
Ctrl-C, so an aborted run never leaves the arm holding indefinitely.
"""

from __future__ import annotations

import argparse
import json
import signal
import time

from scservo_sdk import COMM_SUCCESS, PacketHandler, PortHandler

ADDR_TORQUE_ENABLE = 40
ADDR_GOAL_POSITION = 42  # 2 bytes
ADDR_PRESENT_POSITION = 56  # 2 bytes, 0..4095
ADDR_PRESENT_LOAD = 60  # 2 bytes, signed-magnitude (bit10 = sign), duty .1%
ADDR_PRESENT_VOLTAGE = 62  # 1 byte, decivolts
ADDR_PRESENT_TEMPERATURE = 63  # 1 byte, deg C
BAUD = 1_000_000
PROTO = 0  # STS/SMS little-endian

GUARD_MARGIN_TICKS = 80
PLAN_SCHEMA = "jaxterity-checkup-plan-v2"
SCHEMA = "lekiwi-thermal-capture-v1"


def _signed_mag(raw: int, sign_bit: int) -> int:
    mask = 1 << sign_bit
    return -(raw & (mask - 1)) if (raw & mask) else (raw & (mask - 1))



def _terminate_as_interrupt(signum, frame):
    """Turn SIGTERM into KeyboardInterrupt so the torque-off cleanup runs.

    Hardware finding (2026-07-29): `pkill` on a capture skipped the `finally`
    and left the arm holding position under torque (a servo sat at 48 degC until
    it was released by hand). A capture that can be killed must still put the
    arm down.
    """
    raise KeyboardInterrupt(f"signal {signum}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument(
        "--calib",
        default="calibration_final.json",
        help="per-joint calibration JSON (the LeKiwi autosetup output)",
    )
    ap.add_argument(
        "--plan",
        required=True,
        help="stability-vetted capture plan (jaxterity checkup --plan-out); the "
        "hold pose comes from it, so an unvetted pose cannot be held",
    )
    ap.add_argument("--out", default="/tmp/thermal_capture.json")
    ap.add_argument(
        "--baseline-s", type=float, default=60.0, help="logged hold before heating"
    )
    ap.add_argument(
        "--hold-s", type=float, default=300.0, help="torque-enabled heating hold"
    )
    ap.add_argument(
        "--cool-s",
        type=float,
        default=180.0,
        help="torque-disabled cooling tail (identifies the cooling constant)",
    )
    ap.add_argument("--rate-hz", type=float, default=1.0)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="log only: no torque enable, no goal writes, no motion",
    )
    args = ap.parse_args()
    signal.signal(signal.SIGTERM, _terminate_as_interrupt)

    plan = json.load(open(args.plan))
    if plan.get("schema") != PLAN_SCHEMA:
        print(
            f"ABORT: unrecognized plan schema {plan.get('schema')!r} (expected "
            f"{PLAN_SCHEMA}). Re-run `jaxterity checkup --plan-out` to re-vet."
        )
        return 2
    pose = {int(sid): int(ticks) for sid, ticks in plan["pose_ticks"].items()}
    if not pose:
        print("ABORT: plan carries no pose_ticks.")
        return 2

    calib = json.load(open(args.calib))
    rows = {int(c["servo_id"]): c for c in calib["calibrations"]}
    ids = sorted(pose)
    joints = {sid: rows[sid].get("joint", f"servo{sid}") for sid in ids if sid in rows}

    def guard(sid: int, ticks: int) -> int:
        row = rows[sid]
        lo = int(row["range_min_ticks"]) + GUARD_MARGIN_TICKS
        hi = int(row["range_max_ticks"]) - GUARD_MARGIN_TICKS
        return max(lo, min(hi, int(ticks)))

    port = PortHandler(args.port)
    if not port.openPort() or not port.setBaudRate(BAUD):
        print(f"ABORT: cannot open {args.port} at {BAUD}.")
        return 3
    ph = PacketHandler(PROTO)

    def read1(sid: int, addr: int):
        v, comm, err = ph.read1ByteTxRx(port, sid, addr)
        return v if comm == COMM_SUCCESS and err == 0 else None

    def read2(sid: int, addr: int):
        v, comm, err = ph.read2ByteTxRx(port, sid, addr)
        return v if comm == COMM_SUCCESS and err == 0 else None

    def torque(sid: int, on: bool) -> None:
        ph.write1ByteTxRx(port, sid, ADDR_TORQUE_ENABLE, 1 if on else 0)

    samples: list[tuple] = []
    phases: list[dict] = []
    t0 = time.perf_counter()

    def log_for(seconds: float, phase: str) -> None:
        """Poll every servo at the sample rate for ``seconds``, tagging the phase."""
        start = time.perf_counter()
        phases.append({"phase": phase, "start_s": start - t0, "duration_s": seconds})
        period = 1.0 / args.rate_hz
        next_tick = start
        while time.perf_counter() - start < seconds:
            now = time.perf_counter()
            row = [now - t0, phase]
            for sid in ids:
                pos = read2(sid, ADDR_PRESENT_POSITION)
                load_raw = read2(sid, ADDR_PRESENT_LOAD)
                row.extend(
                    [
                        pos,
                        None if load_raw is None else _signed_mag(load_raw, 10),
                        read1(sid, ADDR_PRESENT_VOLTAGE),
                        read1(sid, ADDR_PRESENT_TEMPERATURE),
                    ]
                )
            samples.append(tuple(row))
            next_tick += period
            time.sleep(max(0.0, next_tick - time.perf_counter()))

    aborted = False
    try:
        if not args.dry_run:
            # Ramp the whole arm to the plan's vetted pose at the plan's bounded
            # speed — the same motion shape the plan validated.
            for sid in ids:
                torque(sid, True)
            time.sleep(0.2)
            current = {sid: (read2(sid, ADDR_PRESENT_POSITION) or 0) for sid in ids}
            target = {sid: guard(sid, pose[sid]) for sid in ids}
            speed = float(plan["transit_ticks_per_s"])
            loop_hz = 50.0
            step = max(1, int(speed / loop_hz))
            print(f"ramping to the plan pose at {speed:.0f} ticks/s ...", flush=True)
            while True:
                done = True
                for sid in ids:
                    delta = target[sid] - current[sid]
                    if abs(delta) > step:
                        current[sid] += step if delta > 0 else -step
                        done = False
                    else:
                        current[sid] = target[sid]
                    ph.write2ByteTxRx(port, sid, ADDR_GOAL_POSITION, current[sid])
                if done:
                    break
                time.sleep(1.0 / loop_hz)
            time.sleep(1.0)

        print(f"baseline {args.baseline_s:.0f}s ...", flush=True)
        log_for(args.baseline_s, "baseline")
        print(f"heating hold {args.hold_s:.0f}s ...", flush=True)
        log_for(args.hold_s, "hold")
        if not args.dry_run:
            for sid in ids:
                torque(sid, False)
        print(f"cooling {args.cool_s:.0f}s (torque off) ...", flush=True)
        log_for(args.cool_s, "cool")
    except KeyboardInterrupt:
        aborted = True
        print("\ninterrupted — disabling torque.")
    finally:
        if not args.dry_run:
            for sid in ids:
                torque(sid, False)

    if not samples:
        print("no samples captured.")
        return 3
    times = [s[0] for s in samples]
    rate = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0
    out = {
        "schema": SCHEMA,
        "servo_ids": ids,
        "joints": {str(sid): joints.get(sid, "") for sid in ids},
        "dry_run": args.dry_run,
        "aborted": aborted,
        "excitation": {
            # A static hold: no waveform, so no waveform version. Recorded
            # explicitly rather than left absent, so a protocol pin cannot
            # mistake this for a multisine capture.
            "kind": "static_hold",
            "baseline_s": args.baseline_s,
            "hold_s": args.hold_s,
            "cool_s": args.cool_s,
            "target_rate_hz": args.rate_hz,
        },
        "plan": {
            "schema": plan.get("schema"),
            "serial": plan.get("serial"),
            "excitation_kind": plan.get("excitation_kind"),
            "transit_ticks_per_s": plan.get("transit_ticks_per_s"),
        },
        "pose_ticks": {str(sid): pose[sid] for sid in ids},
        "calibration": {
            str(sid): {
                k: rows[sid][k]
                for k in (
                    "sign",
                    "ticks_per_rad",
                    "offset_ticks",
                    "range_min_ticks",
                    "range_max_ticks",
                )
            }
            for sid in ids
            if sid in rows
        },
        "achieved_rate_hz": rate,
        "t": times,
        "phase": [s[1] for s in samples],
        # Per-servo channels, in servo_ids order.
        "pos_ticks": {
            str(sid): [s[2 + 4 * i] for s in samples] for i, sid in enumerate(ids)
        },
        "load_raw": {
            str(sid): [s[3 + 4 * i] for s in samples] for i, sid in enumerate(ids)
        },
        "volt_dV": {
            str(sid): [s[4 + 4 * i] for s in samples] for i, sid in enumerate(ids)
        },
        "temp_C": {
            str(sid): [s[5 + 4 * i] for s in samples] for i, sid in enumerate(ids)
        },
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh)
    span = times[-1] - times[0]
    print(
        f"wrote {args.out}: {len(samples)} samples over {span:.0f}s "
        f"({rate:.2f} Hz achieved)"
    )
    for sid in ids:
        temps = [v for v in out["temp_C"][str(sid)] if v is not None]
        if not temps:
            continue
        # Report a ROBUST rise, not max-minus-first. The Feetech temperature
        # register throws occasional wild single samples (a 30 degC servo
        # reading 39), and a max-based summary turns one bad sample into a
        # reported +8 degC rise -- which is how a human ends up writing down a
        # number the despiked trace does not support. Medians of the first and
        # last 10 samples; the raw max is shown separately, labelled, when it
        # disagrees.
        head = sorted(temps[: min(10, len(temps))])[min(10, len(temps)) // 2]
        tail = sorted(temps[-min(10, len(temps)) :])[min(10, len(temps)) // 2]
        line = (
            f"  {joints.get(sid, sid):16s} {head:3d} -> {tail:3d} degC "
            f"(rise {tail - head:+d})"
        )
        if max(temps) - tail > 4:
            line += f"  [raw max {max(temps)} -- likely register glitch, despiked away]"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
