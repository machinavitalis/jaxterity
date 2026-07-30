#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors
"""On-device aggregate multi-servo load capture for the LeKiwi battery sag.

The single-joint commanded capture (``lekiwi_commanded_capture.py``) barely moves
the shared bus voltage — one servo's current is small. To identify the battery's
Thévenin parameters (``V_oc``, ``R_int`` in ``V_bus = V_oc - R_int * I_total``) we
need *several* servos drawing current at once. This script commands a synchronized
band-limited multisine on a set of arm servos (torque held on, position mode,
per-joint guard bands) and records, each tick, every commanded servo's PWM-duty
``load`` (the per-servo torque/current proxy) plus the **shared bus voltage** —
the aggregate-load trace the workstation fits BatterySag against.

SAFETY mirrors the single-joint script: position mode, torque on; each servo is
clamped to a guard band inside its calibrated range and the whole capture aborts
to a stationary hold the instant any commanded servo leaves its band. Only the
``--servo-ids`` listed are ever commanded; each excites about its *current*
position (read at start), so it must already be inside its band.

Usage (on the Pi)::

    python lekiwi_load_capture.py --servo-ids 1,2,3 --amp-ticks 180 --duration 16 \
        --out /tmp/load_trace.json --calib calibration_final.json
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import time

from scservo_sdk import COMM_SUCCESS, PacketHandler, PortHandler

ADDR_MODE = 33
ADDR_TORQUE_ENABLE = 40
ADDR_GOAL_POSITION = 42
ADDR_PRESENT_POSITION = 56
ADDR_PRESENT_LOAD = 60  # signed-mag bit10, 0..1000 (duty .1%)
ADDR_PRESENT_VOLTAGE = 62  # decivolts
BAUD = 1_000_000
PROTO = 0

GUARD_MARGIN_TICKS = 80
SETTLE_S = 0.3
# Distinct multisine tones per servo (Hz) so the aggregate current is broadband
# and the servos do not all peak together (a fairer, higher-duty aggregate load).
# Deliberately a different set from the single-joint capture's; the offline gate
# mirrors this set (jaxterity.checkup.stability.LOAD_MULTISINE_HZ).
MULTISINE_HZ = (0.3, 0.6, 1.0, 1.5, 2.2)
# Per-servo phase step (rad): staggers the servos so they do not all peak
# together. The ZMP gate deliberately validates the *in-phase* case instead,
# which rocks the base harder — erring safe, and documented on both sides.
PHASE_STEP = 1.7
# Waveform version (see the commanded-capture script): v2 normalizes by the
# sum's measured peak instead of the sqrt(2n) estimate this tone set overshoots.
MULTISINE_KIND = "schroeder_multisine_v2"


def _signed_mag(raw: int, sign_bit: int) -> int:
    mask = 1 << sign_bit
    return -(raw & (mask - 1)) if (raw & mask) else (raw & (mask - 1))


def _raw_multisine(t: float, period: float, phase: float) -> float:
    """Un-normalized Schroeder sum with a per-servo phase offset."""
    ks = [max(1, round(f * period)) for f in MULTISINE_HZ]
    n = len(ks)
    phases = [-math.pi * i * (i + 1) / n + phase for i in range(n)]
    return sum(math.sin(2 * math.pi * k * t / period + p) for k, p in zip(ks, phases))


_PEAK_CACHE: dict[tuple[float, float], float] = {}


def _multisine_peak(period: float, phase: float) -> float:
    """True peak |value| of the raw sum over one period (1 kHz scan, cached).

    Per (period, phase) because the phase offset moves the peak. sqrt(2n) is a
    crest *estimate* and this tone set overshoots it — measuring is exact.
    """
    key = (period, phase)
    if key not in _PEAK_CACHE:
        steps = max(int(period * 1000.0), 1000)
        _PEAK_CACHE[key] = max(
            abs(_raw_multisine(i * period / steps, period, phase)) for i in range(steps)
        )
    return _PEAK_CACHE[key]


def _multisine(t: float, period: float, amp: float, phase: float) -> float:
    """Per-servo waveform, true-peak-normalized so the swing stays inside +/-amp."""
    return amp * _raw_multisine(t, period, phase) / _multisine_peak(period, phase)



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
    ap.add_argument("--servo-ids", default="1,2,3", help="comma-separated servo ids")
    ap.add_argument(
        "--calib",
        default="calibration_final.json",
        help="per-joint calibration JSON (the LeKiwi autosetup output)",
    )
    ap.add_argument("--out", default="/tmp/load_trace.json")
    ap.add_argument("--amp-ticks", type=int, default=180)
    ap.add_argument("--duration", type=float, default=16.0)
    ap.add_argument("--rate-hz", type=float, default=60.0)
    ap.add_argument(
        "--plan",
        default=None,
        help="stability-vetted capture plan JSON (jaxterity checkup --plan-out); "
        "uses its 'load' entry and ramps the arm to the vetted pose first",
    )
    args = ap.parse_args()
    signal.signal(signal.SIGTERM, _terminate_as_interrupt)

    plan = None
    if args.plan:
        plan = json.load(open(args.plan))
        if plan.get("schema") != "jaxterity-checkup-plan-v2":
            print(
                f"ABORT: unrecognized plan schema {plan.get('schema')!r} "
                "(expected jaxterity-checkup-plan-v2). Re-run `jaxterity "
                "checkup --plan-out` to re-vet against the current waveform."
            )
            return 2
        entry = next((e for e in plan["entries"] if e["kind"] == "load"), None)
        if entry is None or not entry.get("feasible"):
            print("ABORT: plan has no feasible 'load' entry. Re-plan.")
            return 2
        plan_kind = plan.get("excitation_kind", "schroeder_multisine")
        if plan_kind != MULTISINE_KIND:
            print(
                f"ABORT: plan was vetted for waveform {plan_kind!r} but this "
                f"script commands {MULTISINE_KIND!r}. Re-vet the plan."
            )
            return 2
        args.servo_ids = ",".join(str(s) for s in entry["servo_ids"])
        args.amp_ticks = int(entry["amp_ticks"])
        args.duration = float(entry["duration_s"])

    calib = json.load(open(args.calib))
    cal_by_id = {c["servo_id"]: c for c in calib["calibrations"]}
    ids = [int(x) for x in args.servo_ids.split(",")]
    amp = int(args.amp_ticks)

    ph, pk = PortHandler(args.port), PacketHandler(PROTO)
    assert ph.openPort() and ph.setBaudRate(BAUD), "could not open bus"

    def rpos(sid: int) -> int:
        v, c, _ = pk.read2ByteTxRx(ph, sid, ADDR_PRESENT_POSITION)
        return v if c == COMM_SUCCESS else -1

    def rload(sid: int) -> int:
        v, c, _ = pk.read2ByteTxRx(ph, sid, ADDR_PRESENT_LOAD)
        return _signed_mag(v, 10) if c == COMM_SUCCESS else 0

    def rvolt(sid: int) -> int:
        v, c, _ = pk.read1ByteTxRx(ph, sid, ADDR_PRESENT_VOLTAGE)
        return v if c == COMM_SUCCESS else 0

    # With a vetted plan: ramp the WHOLE arm to the plan's pose first at the
    # bounded transit speed, so the trace excites about the vetted pose rather
    # than wherever the arm happened to be.
    if plan is not None:
        rows = {
            int(c["servo_id"]): c
            for c in calib["calibrations"]
            if c.get("range_min_ticks") is not None
        }
        pose = plan["pose_ticks"]
        ramp_ids = sorted(int(s) for s in pose)
        cur, guard = {}, {}
        ok = True
        for aid in ramp_ids:
            v, cc, _ = pk.read2ByteTxRx(ph, aid, ADDR_PRESENT_POSITION)
            if cc != COMM_SUCCESS:
                print(f"ABORT: servo {aid} did not answer during pose ramp.")
                return 2
            cur[aid] = v
            row = rows.get(aid, {})
            guard[aid] = (
                min(int(row.get("range_min_ticks", v)), v) - 30,
                max(int(row.get("range_max_ticks", v)), v) + 30,
            )
            pk.write1ByteTxRx(ph, aid, ADDR_TORQUE_ENABLE, 0)
            pk.write1ByteTxRx(ph, aid, ADDR_MODE, 0)
            pk.write1ByteTxRx(ph, aid, ADDR_TORQUE_ENABLE, 1)
            pk.write2ByteTxRx(ph, aid, ADDR_GOAL_POSITION, cur[aid])
        loop_hz = 30.0
        step = max(1, int(float(plan["transit_ticks_per_s"]) / loop_hz))
        # Ramp load servos to their vetted excitation center and everything else
        # to the plan pose, so the trace starts where it was validated to. The
        # loop's exit condition MUST test the same targets it drives toward —
        # testing the pose while driving to the centers never terminates, which
        # is how an earlier revision of this line span the arm under torque
        # until it was killed.
        targets = {
            aid: int((entry.get("center_ticks") or {}).get(str(aid), pose[str(aid)]))
            for aid in ramp_ids
        }
        # Iteration cap: at `step` ticks per pass the longest legal travel needs
        # a few hundred passes; anything beyond that is a logic error, not a slow
        # ramp, and must not keep driving the servos.
        max_passes = int(4096 / max(step, 1)) + 200
        passes = 0
        while ok and any(cur[a] != targets[a] for a in ramp_ids):
            passes += 1
            if passes > max_passes:
                print(
                    f"ABORT: pose ramp did not converge in {max_passes} passes "
                    f"(targets {targets}, at {cur}) — refusing to keep driving."
                )
                ok = False
                break
            for aid in ramp_ids:
                tgt = targets[aid]
                if cur[aid] == tgt:
                    continue
                nxt = cur[aid] + (step if tgt > cur[aid] else -step)
                if (tgt > cur[aid] and nxt > tgt) or (tgt < cur[aid] and nxt < tgt):
                    nxt = tgt
                cur[aid] = nxt
                pk.write2ByteTxRx(ph, aid, ADDR_GOAL_POSITION, nxt)
                p, cc, _ = pk.read2ByteTxRx(ph, aid, ADDR_PRESENT_POSITION)
                if cc == COMM_SUCCESS and not (guard[aid][0] <= p <= guard[aid][1]):
                    print(f"GUARD tripped during pose ramp (servo {aid} at {p}).")
                    ok = False
                    break
            time.sleep(1.0 / loop_hz)
        if not ok:
            return 2
        time.sleep(SETTLE_S)

    bands, centers = {}, {}
    for sid in ids:
        cj = cal_by_id[sid]
        if cj.get("range_min_ticks") is None:
            print(f"ABORT: servo {sid} ({cj['joint']}) has no range (wheel?).")
            return 2
        lo = int(cj["range_min_ticks"]) + GUARD_MARGIN_TICKS
        hi = int(cj["range_max_ticks"]) - GUARD_MARGIN_TICKS
        # With a plan, the center is the one the plan VETTED — not wherever the
        # servo happens to be sitting. Hardware finding (2026-07-29): deriving
        # the center from the live position re-invents a motion nobody vetted,
        # and when the pose leaves a servo on its band edge the amplitude
        # collapses to a couple of ticks and the guard trips on sample one.
        planned = None
        if plan is not None:
            planned = (entry.get("center_ticks") or {}).get(str(sid))
        c0 = int(planned) if planned is not None else rpos(sid)
        if not (lo <= c0 <= hi):
            print(
                f"ABORT: servo {sid} center {c0} outside band [{lo},{hi}]"
                + (
                    " (the plan's vetted center does not fit this calibration's "
                    "guard band — re-plan against this unit's calibration)"
                    if planned is not None
                    else "; move it into range first."
                )
            )
            return 2
        # shrink amplitude per joint so center +/- amp stays in band
        a = min(amp, c0 - lo, hi - c0)
        if planned is not None and a < amp:
            print(
                f"ABORT: servo {sid} center {c0} leaves only {a} ticks inside "
                f"[{lo},{hi}] but the plan vetted {amp}. Running a smaller swing "
                "than was vetted would silently change the experiment — re-plan."
            )
            return 2
        bands[sid] = (lo, hi)
        centers[sid] = (c0, a)
        pk.write1ByteTxRx(ph, sid, ADDR_TORQUE_ENABLE, 0)
        pk.write1ByteTxRx(ph, sid, ADDR_MODE, 0)
        pk.write1ByteTxRx(ph, sid, ADDR_TORQUE_ENABLE, 1)
        print(f"  servo {sid} ({cj['joint']}): center {c0}, amp {a}, band [{lo},{hi}]")

    time.sleep(SETTLE_S)
    samples = []  # (t, {sid: load}, volt_dV, {sid: pos})
    aborted = False
    dt = 1.0 / args.rate_hz
    t0 = time.perf_counter()
    next_tick = t0
    try:
        while not aborted:
            t = time.perf_counter() - t0
            if t >= args.duration:
                break
            for k, sid in enumerate(ids):
                c0, a = centers[sid]
                goal = int(round(c0 + _multisine(t, args.duration, a, k * PHASE_STEP)))
                lo, hi = bands[sid]
                pk.write2ByteTxRx(ph, sid, ADDR_GOAL_POSITION, max(lo, min(hi, goal)))
            loads, poss = {}, {}
            for sid in ids:
                poss[sid] = rpos(sid)
                loads[sid] = rload(sid)
            volt = rvolt(ids[0])
            samples.append((t, loads, volt, poss))
            for sid in ids:
                lo, hi = bands[sid]
                if not (lo <= poss[sid] <= hi):
                    print(f"GUARD tripped: servo {sid} at {poss[sid]}; aborting.")
                    aborted = True
                    break
            next_tick += dt
            s = next_tick - time.perf_counter()
            if s > 0:
                time.sleep(s)
    finally:
        # Re-centre, then RELEASE. Leaving the arm holding under torque after a
        # capture bakes a thermal confound into whatever runs next and cooks the
        # loaded servo for as long as nobody notices.
        for sid in ids:
            pk.write2ByteTxRx(ph, sid, ADDR_GOAL_POSITION, centers[sid][0])
        time.sleep(0.3)
        for sid in ids:
            pk.write1ByteTxRx(ph, sid, ADDR_TORQUE_ENABLE, 0)
        ph.closePort()

    if not samples:
        print("no samples captured.")
        return 3
    ts = [s[0] for s in samples]
    out = {
        "schema": "lekiwi-load-capture-v1",
        "servo_ids": ids,
        "joints": {sid: cal_by_id[sid]["joint"] for sid in ids},
        "aborted": aborted,
        "excitation": {
            # The per-servo phase offsets are what make this a *staggered*
            # multisine rather than the synchronized one the old label claimed;
            # the waveform kind now carries the normalization version too.
            "kind": MULTISINE_KIND,
            "staggered": True,
            "phase_step_rad": PHASE_STEP,
            "amp_ticks": amp,
            "duration_s": args.duration,
            "multisine_hz": list(MULTISINE_HZ),
        },
        "calibration": {
            sid: {
                k: cal_by_id[sid][k]
                for k in (
                    "sign",
                    "ticks_per_rad",
                    "offset_ticks",
                    "range_min_ticks",
                    "range_max_ticks",
                )
            }
            for sid in ids
        },
        "t": ts,
        "load_raw": {sid: [s[1][sid] for s in samples] for sid in ids},
        "pos_ticks": {sid: [s[3][sid] for s in samples] for sid in ids},
        "volt_dV": [s[2] for s in samples],
    }
    json.dump(out, open(args.out, "w"))
    volts = [s[2] for s in samples]
    lo_v, hi_v = min(volts) / 10, max(volts) / 10
    tag = " [ABORTED]" if aborted else ""
    print(
        f"captured {len(samples)} samples; bus {lo_v:.1f}-{hi_v:.1f} V "
        f"(sag {hi_v - lo_v:.1f} V){tag} -> {args.out}"
    )
    return 3 if aborted else 0


if __name__ == "__main__":
    raise SystemExit(main())
