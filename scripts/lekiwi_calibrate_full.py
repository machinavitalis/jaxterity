#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Full-range validation calibration: play the sim-verified choreography (LeKiwi).

With the signs known, the arm can be auto-posed into each joint's own
collision-free stance (the poses ``calib_choreography_lekiwi.py`` found and
verified in sim), so every joint can be swept its *whole* range to both stops —
the end-to-end validation. For each joint: pose the other joints into the stance
(gentle, with an arrival check), hold them, sweep the joint to both stops
(measuring offset/scale/range), and return to the compact home. Load is capped
throughout and a joint that stalls unexpectedly aborts the run.

Reads the choreography (angles, radians) + the current calibration (offset/sign
per joint, to convert angles->ticks). Writes the validated calibration + report.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lekiwi_autosetup as las
from lekiwi_autosetup import (
    ADDR_GOAL_POSITION,
    ADDR_GOAL_SPEED,
    ADDR_TORQUE_ENABLE,
    ADDR_TORQUE_LIMIT,
    DEFAULT_BAUD,
    LAYOUTS,
    RESOLUTION,
    TICKS_PER_RAD,
    Bus,
    SweepAborted,
    touch_off,
)

SWEEP_SAG_TOL = 70  # ticks (~6°): a held joint drifting more than this mid-sweep
# is sagging toward the floor -> abort before contact (L-013)
TEMP_ABORT_C = 55  # servo temperature that aborts the run (Feetech throttles /
# cuts torque near 65-70 C; stop well before, L-014b)


def settle_and_relax(bus: Bus, ids, dt=0.35) -> None:
    """Lower the arm gently before cutting torque, so it descends instead of
    crashing down at the end of the run (L-014d/e). Ramps the torque cap down in
    FINE steps that ease *through* the shoulder's gravity-hold torque (~0.42):
    re-anchoring the goal at the present position each step, the joint holds above
    gravity, then sags a little per step as the cap crosses below it — a smooth
    descent rather than the coarse ramp's sudden drop. Ends limp."""
    caps = [0.55, 0.46, 0.42, 0.38, 0.34, 0.30, 0.26, 0.22, 0.18, 0.14, 0.10, 0.06]
    for cap in caps:
        lim = max(1, min(1000, round(cap * 1000)))
        for sid in ids:
            try:
                bus._write2(sid, ADDR_TORQUE_LIMIT, lim)
                bus._write1(sid, ADDR_TORQUE_ENABLE, 1)
                bus._write2(sid, ADDR_GOAL_POSITION, bus.position(sid))
                bus._write2(sid, ADDR_GOAL_SPEED, 150)  # slow, gentle descent
            except Exception:  # noqa: BLE001 — best effort during shutdown
                pass
        time.sleep(dt)
    bus.relax_all(tuple(ids))


def hottest(bus: Bus, ids) -> tuple:
    """(name, temp) of the warmest servo among ``ids`` — a cheap overload/heat
    sentinel. A strained joint heats up before it trips its torque protection;
    polling this lets the run stop safely instead of collapsing mid-dance."""
    names = {s.servo_id: s.name for s in LAYOUTS["lekiwi"]}
    worst = ("", -1.0)
    for sid in ids:
        try:
            t = bus.temperature(sid)
        except Exception:  # noqa: BLE001 — a dropped read shouldn't kill the run
            continue
        if t > worst[1]:
            worst = (names.get(sid, str(sid)), t)
    return worst


LAY2MODEL = {
    "shoulder_pan": "Rotation",
    "shoulder_lift": "Pitch",
    "elbow_flex": "Elbow",
    "wrist_flex": "Wrist_Pitch",
    "wrist_roll": "Wrist_Roll",
}
ARM = list(LAY2MODEL)
ID = {s.name: s.servo_id for s in LAYOUTS["lekiwi"] if s.kind == "arm"}

POSE_SPEED = 500
POSE_MAX_LOAD = 0.55  # approach cap. MUST exceed the heaviest joint's gravity-
# hold torque (~0.42 for the shoulder) — at the old 0.35
# the shoulder maxed the cap on every transit, so the
# monitor cried "contact" on pure gravity (L-014c). 0.55
# leaves headroom so saturation now means a real push.
HOLD_MAX_LOAD = 0.95  # stiff cap once posed, so held joints don't sag (L-005);
# proximal joints bear the whole cantilever during a
# distal sweep, so the hold must be near-rigid (brief)
SAT_LOAD = 0.50  # absolute load that counts as "fighting something" —
# above gravity (~0.42), below the 0.55 approach cap
ARRIVE_TOL = 130  # ticks
STANCE_TOL = 50  # ticks (~4.4°): max realized-vs-commanded stance error
MOVE_TIMEOUT = 200  # poll iterations (×50 ms) ≈ 10 s for big moves
SWEEPABLE = ["wrist_roll", "wrist_flex", "elbow_flex", "shoulder_lift"]  # pan wraps


def load_cal(path: Path) -> dict:
    doc = json.loads(path.read_text())
    return {
        c["joint"]: {"offset": c["offset_ticks"], "sign": c["sign"]}
        for c in doc["calibrations"]
        if c["joint"] in ARM
    }


def to_tick(cal: dict, joint: str, q: float) -> int:
    t = cal[joint]["offset"] + cal[joint]["sign"] * TICKS_PER_RAD * q
    return max(0, min(RESOLUTION - 1, int(round(t))))


def _set_goal(bus: Bus, sid: int, tick: int) -> None:
    bus._write2(sid, ADDR_TORQUE_LIMIT, int(POSE_MAX_LOAD * 1000))
    bus._write1(sid, ADDR_TORQUE_ENABLE, 1)
    bus._write2(sid, ADDR_GOAL_SPEED, POSE_SPEED)
    bus._write2(sid, ADDR_GOAL_POSITION, tick)


#: Report of the most recent ``move_to``: servos whose torque saturated near the
#: cap while moving (sustained >=90% of POSE_MAX_LOAD for >=4 polls), plus the
#: pack voltage at rest and its minimum under load. Saturation mid-transit means
#: the move was fighting something; the voltage sag tells *what* — a depleted
#: battery sags hard under load, a mechanical obstacle does not (L-010/L-011).
LAST_MOVE: dict = {"saturated": {}, "v_rest": None, "v_min": None}
VOLT_SAG_BATTERY = 0.8  # V drop under load that points at a weak pack


def move_to(bus: Bus, targets: dict) -> bool:
    """Move several servos to their target ticks *simultaneously* (matches the
    sim-verified poses), returning False if any stalls short of its target.
    Monitors PRESENT_LOAD and pack voltage throughout — the transit itself is
    instrumented, not just the endpoint (L-010); see ``LAST_MOVE``."""
    for sid, tick in targets.items():
        _set_goal(bus, sid, tick)
    rep = next(iter(targets))  # representative servo for voltage
    v_rest = bus.voltage(rep)
    v_min = v_rest
    last = {sid: bus.position(sid) for sid in targets}
    stalls = {sid: 0 for sid in targets}
    sat = {sid: 0 for sid in targets}
    LAST_MOVE.update(saturated={}, v_rest=v_rest, v_min=v_rest)
    for i in range(MOVE_TIMEOUT):
        time.sleep(0.05)
        if i % 10 == 0:  # voltage every ~0.5 s (bus-cheap)
            v_min = min(v_min, bus.voltage(rep))
        done = True
        for sid, tick in targets.items():
            pos = bus.position(sid)
            if abs(bus.load(sid)) >= SAT_LOAD:  # genuine push, not gravity (L-014c)
                sat[sid] += 1
            if abs(pos - tick) > ARRIVE_TOL:
                done = False
                if abs(pos - last[sid]) <= 2:
                    stalls[sid] += 1
                    if stalls[sid] >= 12:  # stuck short of target -> blocked
                        print(f"    ! servo {sid} stalled at {pos} (target {tick})")
                        LAST_MOVE.update(
                            saturated={s: n for s, n in sat.items() if n >= 4},
                            v_rest=v_rest,
                            v_min=v_min,
                        )
                        return False
                else:
                    stalls[sid] = 0
            last[sid] = pos
        if done:
            break
    LAST_MOVE.update(
        saturated={s: n for s, n in sat.items() if n >= 4}, v_rest=v_rest, v_min=v_min
    )
    miss = {
        sid: (bus.position(sid), t)
        for sid, t in targets.items()
        if abs(bus.position(sid) - t) > ARRIVE_TOL
    }
    if miss:
        print(f"    ! timed out; not arrived: {miss}")
    return not miss


def classify_stall() -> tuple[str, str]:
    """Disambiguate the last move's saturation by the bus signals alone (no
    model needed on the robot): a depleted pack sags hard under load, a
    mechanical obstacle (wall/floor/another link) does not. Returns
    (cause, human_detail) where cause is 'battery' | 'contact' | 'ok'."""
    sat = LAST_MOVE.get("saturated") or {}
    if not sat:
        return "ok", ""
    vr, vm = LAST_MOVE.get("v_rest"), LAST_MOVE.get("v_min")
    if vr is not None and vm is not None and (vr - vm) >= VOLT_SAG_BATTERY:
        return "battery", (
            f"pack sagged {vr - vm:.1f} V under load "
            f"({vr:.1f}->{vm:.1f} V) — charge the battery"
        )
    return "contact", (
        f"voltage held steady ({vr:.1f}->{vm:.1f} V) while torque "
        f"saturated — the arm pushed something (wall/floor/object)"
    )


def hold_at(bus: Bus, targets: dict, max_load: float = HOLD_MAX_LOAD) -> None:
    """Hold servos *at their target ticks* with a stiff torque cap.

    Holding at the present position (``hold_position``) freezes any gravity
    shortfall left by the gentle approach — the joint then sits just inside
    ARRIVE_TOL but outside STANCE_TOL forever. Re-commanding the target under a
    near-rigid cap closes that gap."""
    for sid, tick in targets.items():
        bus._write2(sid, ADDR_TORQUE_LIMIT, max(1, min(1000, round(max_load * 1000))))
        bus._write1(sid, ADDR_TORQUE_ENABLE, 1)
        bus._write2(sid, ADDR_GOAL_SPEED, POSE_SPEED)
        bus._write2(sid, ADDR_GOAL_POSITION, tick)
    time.sleep(0.6)


def _held_deviation(bus: Bus, targets: dict) -> dict:
    """Joints whose realized position is off their commanded stance > STANCE_TOL."""
    names = {ID[j]: j for j in ARM}
    return {
        names[sid]: bus.position(sid) - t
        for sid, t in targets.items()
        if abs(bus.position(sid) - t) > STANCE_TOL
    }


def guarded_touch_off(bus: Bus, spec, held: dict):
    """``touch_off`` with the held joints watched *throughout* the sweep (L-013):
    if any sags past SWEEP_SAG_TOL the sweep aborts before the arm reaches the
    floor. Each check also re-commands the sagging joint back to its target (a
    cheap active re-assert). Returns ``(result_or_None, abort_reason_or_None)``."""
    names = {ID[j]: j for j in ARM}

    def watch():
        worst = None
        for sid, t in held.items():
            dev = bus.position(sid) - t
            if abs(dev) > SWEEP_SAG_TOL:
                bus._write2(sid, ADDR_GOAL_POSITION, t)  # re-assert the hold
                worst = f"{names.get(sid, sid)} sagged {dev:+d} ticks mid-sweep"
        return worst

    las.SWEEP_WATCH = watch
    try:
        return touch_off(bus, spec), None
    except SweepAborted as e:
        return None, e.reason
    finally:
        las.SWEEP_WATCH = None


def pose_stance(bus: Bus, cal: dict, target: str, stance_model: dict):
    """Pose every arm joint except ``target`` into the stance and verify it.

    Returns the ``{servo_id: target_tick}`` map of held joints on success (so the
    caller can re-check it after the sweep), or ``None`` if the stance can't be
    reached or doesn't hold — in which case we skip the joint rather than sweep
    from a pose that isn't where sim thinks it is (L-005)."""
    targets = {
        ID[j]: to_tick(cal, j, stance_model[LAY2MODEL[j]]) for j in ARM if j != target
    }
    if not move_to(bus, targets):
        print("    ! stance not reached — aborting this joint")
        return None
    # stiff hold AT TARGET so the cantilevered arm can't sag into the floor
    # mid-sweep (and any gentle-approach shortfall is pulled closed)
    hold_at(bus, targets)
    dev = _held_deviation(bus, targets)
    if dev:
        print(f"    ! realized stance off commanded by >{STANCE_TOL} ticks: {dev}")
        print("    ! skipping sweep (gripper may not be where sim expects)")
        return None
    return targets


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    p.add_argument("--cal", default="./lekiwi/out/calibration_final.json")
    p.add_argument(
        "--choreo", default="./lekiwi/calib_choreography_lekiwi.json"
    )
    p.add_argument("--only", default=None, help="run one joint (safety check)")
    p.add_argument("--out", default="./lekiwi/out")
    args = p.parse_args(argv)

    cal = load_cal(Path(args.cal))
    choreo = json.loads(Path(args.choreo).read_text())
    home = choreo["home"]
    steps = {s["joint"]: s for s in choreo["steps"] if s.get("feasible")}
    todo = [args.only] if args.only else SWEEPABLE

    bus = Bus(args.port, args.baud)
    results = {}
    try:
        home_t = {ID[j]: to_tick(cal, j, home[LAY2MODEL[j]]) for j in ARM}
        print("moving to compact home ...", flush=True)
        if not move_to(bus, home_t):
            print("  ! could not reach home — aborting")
            return 1
        hold_at(bus, home_t)
        for joint in todo:
            spec = next(s for s in LAYOUTS["lekiwi"] if s.name == joint)
            print(f"\n=== {joint}: pose stance -> full sweep ===", flush=True)
            held = pose_stance(bus, cal, joint, steps[joint]["stance"])
            if held is None:
                continue
            bus.set_torque(spec.servo_id, False)
            cal_j = touch_off(bus, spec)  # full both-stop sweep, in tick space
            # did the held joints stay put through the sweep? if not, the gripper
            # wandered and the swept stops may be false (floor/self) -> flag it.
            post = _held_deviation(bus, held)
            cal_j["stance_held"] = not post
            if post:
                print(f"  ! held joints drifted during sweep: {post} -> result suspect")
            results[joint] = cal_j
            m = cal_j.get("measured_ticks_per_rad")
            drift = abs(m / TICKS_PER_RAD - 1) * 100 if m else None
            print(
                f"  {joint}: offset={cal_j['offset_ticks']:.1f} "
                f"range[{cal_j['range_min_ticks']}..{cal_j['range_max_ticks']}] "
                f"stops={cal_j.get('stops_found')}"
                + (f" scale_drift={drift:.0f}%" if drift is not None else "")
            )
            move_to(bus, home_t)  # everything back to the compact home
            hold_at(bus, home_t)
        if args.out and results:
            Path(args.out + "/validation_sweeps.json").write_text(
                json.dumps({k: v for k, v in results.items()}, indent=2) + "\n"
            )
            print(f"\nwrote {args.out}/validation_sweeps.json")
        print("\ndone.")
        return 0
    finally:
        bus.relax_all(tuple(ID.values()))
        bus.close()


if __name__ == "__main__":
    raise SystemExit(main())
