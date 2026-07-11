#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Verify the LeKiwi's omni-wheel assignment + direction (preassembled robots).

The three drive servos are detected on the bus like any motor, but the bus
cannot tell you *which physical wheel* a given ID drives or which way it spins —
that depends on the wiring, which is unknown on a preassembled unit. There is no
external sensor here, so the only reliable check is to spin one wheel at a time
and look. This script drives that procedure.

Wheels are continuous-rotation (velocity-controlled): there is **no offset /
homing calibration** like the arm joints — no stops, no zero. What they need is
(1) the correct ID -> wheel assignment and (2) the direction sign per wheel for
the omni-drive kinematics. This produces both.

SAFETY: put the base on a stand / lift it so the wheels spin FREELY before
spinning anything, or it will drive off. Speeds here are low; one wheel moves at
a time. It runs against the real bus through ``scservo_sdk`` only (no JAX/MuJoCo).

Modes
-----
* ``spin --id N [--reverse] [--seconds T]`` — spin one motor briefly, then stop
  and restore its mode. Use this to drive the identification by hand (or have an
  operator watch and report which wheel turned).
* ``identify`` — interactive: spins each candidate in turn and asks which wheel
  it is and which way it turned; writes the verified map.
* ``save --map "7=left:+,8=back:+,9=right:-" --out DIR`` — record a map you
  determined via ``spin`` without the interactive prompts.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from lekiwi_autosetup import ADDR_TORQUE_ENABLE, DEFAULT_BAUD, Bus

#: STS3215 operating-mode register. 0 = position servo, 1 = wheel (speed) mode.
ADDR_MODE = 33
#: Wheel target-speed register (same address as the position-mode goal speed).
ADDR_GOAL_SPEED = 46

#: Default LeKiwi wheel IDs and the SeeedStudio convention (what we verify
#: against). ``angle_deg`` is each wheel's mounting angle for the omni kinematics
#: (0° = robot +x/forward, CCW positive) — confirm against your chassis.
DEFAULT_WHEEL_IDS = (7, 8, 9)
CONVENTION = {
    7: ("wheel_left", 120.0),
    8: ("wheel_back", 240.0),
    9: ("wheel_right", 0.0),
}

#: Gentle default spin (raw speed units) and duration.
SPIN_SPEED = 280
SPIN_SECONDS = 2.5


def set_mode(bus: Bus, sid: int, mode: int) -> None:
    bus._write1(sid, ADDR_MODE, mode)


def read_mode(bus: Bus, sid: int) -> int:
    return bus._read1(sid, ADDR_MODE)


def spin(bus: Bus, sid: int, speed: int) -> None:
    """Spin servo ``sid`` at signed ``speed`` (wheel mode). Negative = reverse."""
    bus._write1(sid, ADDR_TORQUE_ENABLE, 1)
    raw = abs(int(speed)) & 0x7FFF
    if speed < 0:
        raw |= 0x8000  # STS sign bit
    bus._write2(sid, ADDR_GOAL_SPEED, raw)


def stop(bus: Bus, sid: int) -> None:
    bus._write2(sid, ADDR_GOAL_SPEED, 0)
    bus._write1(sid, ADDR_TORQUE_ENABLE, 0)


def spin_once(bus: Bus, sid: int, *, speed: int, seconds: float, reverse: bool) -> None:
    """Briefly spin one wheel, then stop and restore its original mode."""
    original = read_mode(bus, sid)
    set_mode(bus, sid, 1)  # wheel/speed mode
    try:
        spin(bus, sid, -speed if reverse else speed)
        time.sleep(seconds)
    finally:
        stop(bus, sid)
        set_mode(bus, sid, original)  # leave it as we found it


def _candidates(bus: Bus, ids) -> tuple[int, ...]:
    present = bus.scan(tuple(ids))
    missing = tuple(i for i in ids if i not in present)
    if missing:
        print(f"  note: wheel IDs not on the bus: {list(missing)}")
    return present


def cmd_spin(args) -> int:
    bus = Bus(args.port, args.baud)
    try:
        print(
            f"LIFT THE BASE so the wheels spin freely. Spinning motor {args.id} "
            f"{'REVERSE' if args.reverse else 'forward'} for {args.seconds:.1f}s ...",
            flush=True,
        )
        spin_once(
            bus, args.id, speed=args.speed, seconds=args.seconds, reverse=args.reverse
        )
        print("  done. Which wheel turned, and which way? (record with `save`)")
        return 0
    finally:
        stop(bus, args.id)
        bus.close()


def cmd_identify(args) -> int:
    bus = Bus(args.port, args.baud)
    mapping = {}
    try:
        input("LIFT THE BASE so all 3 wheels spin freely, then press Enter ...")
        for sid in _candidates(bus, args.ids):
            input(f"\nReady to spin motor {sid}? Watch the wheels, press Enter ...")
            spin_once(bus, sid, speed=args.speed, seconds=args.seconds, reverse=False)
            wheel = (
                input(
                    f"  Which wheel turned for motor {sid}? "
                    "[l]eft / [b]ack / [r]ight / [s]kip: "
                )
                .strip()
                .lower()
            )
            if wheel.startswith("s") or not wheel:
                continue
            name = {"l": "wheel_left", "b": "wheel_back", "r": "wheel_right"}.get(
                wheel[0], "wheel_" + wheel
            )
            d = (
                input(
                    "  At positive speed, did it drive that side FORWARD? "
                    "[y]es / [n]o: "
                )
                .strip()
                .lower()
            )
            sign = 1 if d.startswith("y") else -1
            mapping[sid] = {"wheel": name, "direction_sign": sign}
            print(f"  recorded: motor {sid} -> {name}, sign {sign:+d}")
        _report(mapping)
        if args.out:
            _write(Path(args.out), mapping, args.port)
        return 0
    finally:
        for sid in args.ids:
            try:
                stop(bus, sid)
            except Exception:  # noqa: BLE001
                pass
        bus.close()


def _norm_wheel(name: str) -> str:
    """Normalise a short wheel name (``left``) to the canonical ``wheel_left``."""
    name = name.strip().lower()
    short = {
        "l": "wheel_left",
        "left": "wheel_left",
        "b": "wheel_back",
        "back": "wheel_back",
        "r": "wheel_right",
        "right": "wheel_right",
    }
    return short.get(name, name if name.startswith("wheel_") else "wheel_" + name)


def cmd_save(args) -> int:
    mapping = {}
    for tok in args.map.split(","):
        tok = tok.strip()
        if not tok:
            continue
        sid_s, rest = tok.split("=")
        wheel, sgn = rest.split(":")
        mapping[int(sid_s)] = {
            "wheel": _norm_wheel(wheel),
            "direction_sign": -1 if sgn.strip() in ("-", "rev", "r") else 1,
        }
    _report(mapping)
    _write(Path(args.out), mapping, args.port)
    return 0


def _report(mapping: dict) -> None:
    print("\nverified wheel assignment:")
    for sid, info in sorted(mapping.items()):
        exp = CONVENTION.get(sid, ("?", None))[0]
        flag = "" if info["wheel"] == exp else f"  (convention expects {exp}!)"
        sign = info["direction_sign"]
        print(f"  motor {sid} -> {info['wheel']:12s} sign {sign:+d}{flag}")
    seen = [i["wheel"] for i in mapping.values()]
    dupes = {w for w in seen if seen.count(w) > 1}
    if dupes:
        print(f"  WARNING: two motors mapped to the same wheel: {dupes}")


def _write(out: Path, mapping: dict, port: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema": "lekiwi-wheels/v1",
        "port": port,
        "wheels": {
            str(sid): {
                "wheel": info["wheel"],
                "direction_sign": info["direction_sign"],
                "mounting_angle_deg": CONVENTION.get(sid, (None, None))[1],
            }
            for sid, info in sorted(mapping.items())
        },
        "note": "Continuous-rotation drive servos: no offset/homing calibration. "
        "mounting_angle_deg feeds the omni-drive kinematics; confirm vs the chassis.",
    }
    path = out / "wheels.json"
    path.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"\nwrote {path}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("spin", help="Spin one motor briefly (drive the ID check).")
    s.add_argument("--id", type=int, required=True)
    s.add_argument("--speed", type=int, default=SPIN_SPEED)
    s.add_argument("--seconds", type=float, default=SPIN_SECONDS)
    s.add_argument("--reverse", action="store_true")
    s.set_defaults(func=cmd_spin)

    i = sub.add_parser("identify", help="Interactive: spin each wheel and identify.")
    i.add_argument(
        "--ids",
        type=lambda v: tuple(int(x) for x in v.split(",")),
        default=DEFAULT_WHEEL_IDS,
    )
    i.add_argument("--speed", type=int, default=SPIN_SPEED)
    i.add_argument("--seconds", type=float, default=SPIN_SECONDS)
    i.add_argument("--out", default=None)
    i.set_defaults(func=cmd_identify)

    v = sub.add_parser("save", help="Record a map found via `spin`.")
    v.add_argument("--map", required=True, help='e.g. "7=left:+,8=back:+,9=right:-"')
    v.add_argument("--out", default="wheels_out")
    v.set_defaults(func=cmd_save)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
