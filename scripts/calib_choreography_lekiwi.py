#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Derive + verify the per-joint full-range calibration choreography (LeKiwi).

For each joint we want a *stance* of the other joints from which that joint can
sweep its whole range collision-free. With the signs now known the arm can be
auto-posed, so this finds those stances in sim, routes between them through a
compact home (hub-and-spoke, easy to verify), and checks that **every** motion —
each transit and each full sweep — stays collision-free. Output is a
choreography (angles in radians) the hardware executor plays.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import calib_planner as cp
import numpy as np
from calib_planner_lekiwi import LekiwiCollisionModel

ORDER = ["wrist_roll", "wrist_flex", "elbow_flex", "shoulder_lift", "shoulder_pan"]
LAY2MODEL = {
    "shoulder_pan": "Rotation",
    "shoulder_lift": "Pitch",
    "elbow_flex": "Elbow",
    "wrist_flex": "Wrist_Pitch",
    "wrist_roll": "Wrist_Roll",
}

#: Required clearance of the whole arm above the *real* floor across every pose,
#: sweep, AND transit (L-005, L-010). This budget must absorb: held-joint sag
#: under a rigid hold (~1-2 cm), floor covering standing above the wheel-contact
#: plane the floor is derived from (~1-2 cm), and transient tracking lag during
#: torque-limited transits (several cm at low battery). 8 cm keeps headroom for
#: all three without killing coverage.
FLOOR_CLEARANCE = 0.08  # metres

#: Extra fingertip reach past the model's jaw mesh, folded into the floor check.
#: The Ekumen model is the *SO-100* (assets/so_arm100/...Moving_Jaw.stl) while
#: this hardware is an *SO-101* — but the redesigned SO-101 jaw has the same reach
#: (model 10.2 cm vs measured ~10 cm tip-to-wrist-roll), so the gripper is NOT
#: under-modeled. This margin therefore represents only the floor covering standing
#: above the wheel-contact plane (~1.5 cm). Override per surface via the env var.
FINGER_MARGIN = float(os.environ.get("LEKIWI_FINGER_MARGIN", "0.015"))

#: Keep the *wrist* near extended (so the gripper points along the forearm, not
#: down at the floor — the operator's insight), but let the BODY (shoulder/elbow)
#: stay compact. Neutralizing the whole arm extends it horizontally and overloads
#: the shoulder (it then can't hold the stance and saturates transits, L-014b); a
#: folded body is low-load and reachable, and the fingertip-aware floor check —
#: not the wrist bend — is what keeps the end-effector off the floor now.
WRIST_NEUTRAL_TOL = 0.40  # rad (~23 deg): max |Wrist_Pitch| in any stance/home
NEUTRAL_WEIGHT = {
    "Wrist_Pitch": 4.0,
    "Wrist_Roll": 1.0,
    "Elbow": 0.0,
    "Pitch": 0.0,
    "Rotation": 0.0,
}


def naturalness_cost(cm, q):
    """Weighted squared deviation from neutral — only the WRIST is penalized
    (body weights are 0); lower means the gripper points straight, not down."""
    c = 0.0
    for k, (lo, hi) in cm.jnt_range.items():
        mid = 0.5 * (lo + hi)
        c += NEUTRAL_WEIGHT.get(k, 0.0) * (q[k] - mid) ** 2
    return c


def safe_transit(cm, qa, qb, moving_joint, steps=24):
    """Is the straight interp qa->qb collision-free (moving_joint held at qa)?"""
    for t in np.linspace(0, 1, steps):
        q = {k: qa[k] + (qb[k] - qa[k]) * t for k in qb}
        q[moving_joint] = qa[moving_joint]
        if not cm.is_safe(q):
            return False
    return True


def sweep_min_clearance(cm, stance, joint, lo, hi, steps=21):
    """Lowest the arm gets above the floor while ``joint`` sweeps [lo, hi], in m.

    Bigger = more sag headroom: a stance whose gripper stays high tolerates more
    real-world droop before touching the floor (L-005), so we prefer it among
    equal-coverage candidates."""
    worst = 1e9
    for a in np.linspace(lo, hi, steps):
        q = dict(stance)
        q[joint] = float(a)
        cm.set_q(q)
        worst = min(worst, cm.min_real_z() - cm._floor_z)  # incl. fingertip reach
    return worst


def main() -> int:
    cm = LekiwiCollisionModel(
        Path("/tmp/lekiwi_ek/packages/lekiwi_sim/lekiwi_sim/assets"),
        floor_clearance=FLOOR_CLEARANCE,
        finger_margin=FINGER_MARGIN,
    )
    rng = np.random.default_rng(1)
    print(
        f"real floor z = {cm._floor_z * 100:+.1f} cm; required fingertip "
        f"clearance = {FLOOR_CLEARANCE * 100:.0f} cm; finger margin = "
        f"{FINGER_MARGIN * 100:.0f} cm past the mesh"
    )

    # compact, collision-free hub the arm returns to between joints: the most
    # COMPACT (highest-clearance, lowest-load) pose whose WRIST is near-extended
    cp.ARM_JOINTS = tuple(cm.jnt_range)
    home = None
    for _ in range(12000):
        q = {k: float(rng.uniform(*cm.jnt_range[k])) for k in cm.jnt_range}
        if abs(q["Wrist_Pitch"]) > WRIST_NEUTRAL_TOL or not cm.is_safe(q):
            continue
        z = cm.min_real_z()
        if home is None or z > home[0]:
            home = (z, q)
    home = home[1]
    cm.set_q(home)  # restore home state before reading its clearance
    print(
        "home:",
        {k: round(v, 2) for k, v in home.items()},
        f"(Wrist_Pitch {home['Wrist_Pitch'] * 57.3:+.0f} deg, "
        f"fingertip {cm.min_real_z() * 100:+.1f} cm)",
    )

    steps = []
    for lay in ORDER:
        j = LAY2MODEL[lay]
        best = None  # (coverage, -transit_cost, stance, arc)
        for _ in range(3000):
            stance = dict(home)
            for k in cm.jnt_range:
                if k != j:
                    stance[k] = float(rng.uniform(*cm.jnt_range[k]))
            # keep the wrist near-extended (gripper not aimed down); the body is
            # left free to fold compactly for low shoulder load (L-014b)
            if j != "Wrist_Pitch" and abs(stance["Wrist_Pitch"]) > WRIST_NEUTRAL_TOL:
                continue
            arc = cp._largest_safe_arc(cm, stance, j, k=41)
            if arc is None or arc[2] < 0.985:
                continue
            if not safe_transit(cm, home, stance, j):
                continue
            cost = sum((stance[k] - home[k]) ** 2 for k in cm.jnt_range if k != j)
            clear = sweep_min_clearance(cm, stance, j, arc[0], arc[1])
            # full coverage first, then the most COMPACT body (max clearance =
            # low shoulder load = reachable), then least transit. Wrist already
            # near-extended; fingertip-aware is_safe keeps the gripper off floor.
            key = (round(arc[2], 3), round(clear, 3), -cost)
            if best is None or key > best[0]:
                best = (key, stance, arc, clear)
        if best is None:
            print(f"  {lay:14s} NO full-range transit-safe stance found")
            steps.append({"joint": lay, "feasible": False})
            continue
        _, stance, arc, clear = best
        steps.append(
            {
                "joint": lay,
                "model_joint": j,
                "feasible": True,
                "stance": {k: round(v, 4) for k, v in stance.items()},
                "sweep_lo": round(arc[0], 4),
                "sweep_hi": round(arc[1], 4),
                "coverage": round(arc[2], 3),
                "min_floor_clearance_m": round(clear, 4),
            }
        )
        print(
            f"  {lay:14s} full-range stance found "
            f"(cov {arc[2] * 100:.0f}%, clearance {clear * 100:.1f} cm, "
            f"Wrist_Pitch {stance['Wrist_Pitch'] * 57.3:+.0f} deg), transit OK"
        )

    # final end-to-end verification of the whole sequence
    print(
        "\nverifying the full sequence (home -> stance -> sweep -> home, each joint):"
    )
    ok = True
    for s in steps:
        if not s.get("feasible"):
            ok = False
            continue
        j = s["model_joint"]
        stance = {k: v for k, v in s["stance"].items()}
        t1 = safe_transit(cm, home, stance, j)
        arc = cp._largest_safe_arc(cm, stance, j, k=61)
        sweep_ok = arc is not None and arc[2] >= 0.985
        t2 = safe_transit(cm, stance, home, j)
        print(f"  {s['joint']:14s} transit_out={t1} sweep={sweep_ok} transit_back={t2}")
        ok &= t1 and sweep_ok and t2

    out = {
        "robot": "lekiwi",
        "home": {k: round(v, 4) for k, v in home.items()},
        "order": ORDER,
        "steps": steps,
        "verified": ok,
        "floor_z_m": round(cm._floor_z, 4),
        "floor_clearance_m": FLOOR_CLEARANCE,
        "finger_margin_m": FINGER_MARGIN,
        "objective": "neutral-pose (extended wrist) + fingertip-aware floor",
    }
    Path("examples/media/calib_choreography_lekiwi.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    print(f"\nfull sequence collision-free: {ok}")
    print("wrote examples/media/calib_choreography_lekiwi.json")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
