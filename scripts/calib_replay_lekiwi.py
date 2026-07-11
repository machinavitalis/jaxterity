#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Render the full sim replay of the LeKiwi calibration choreography to mp4.

Plays exactly what the hardware does — home -> (per joint: transit to stance,
sweep the joint to both stops, transit back) — in the real Ekumen model with the
derived floor and the gripper held CLOSED, from a low side camera so the
gripper/floor proximity is visible. Also tracks a *real-fingertip* proxy (a point
extended ~10 cm past the modeled jaw along the finger axis, since the model's
fingers are shorter than the real SO-101's) and reports the lowest point of the
whole dance and where it occurs. Writes the mp4 + the lowest-clearance frame PNG.
"""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from calib_planner_lekiwi import LekiwiCollisionModel

ASSETS = Path("/tmp/lekiwi_ek/packages/lekiwi_sim/lekiwi_sim/assets")
FINGER = 0.10  # m: real fingertip reach past the modeled jaw origin (local -y)
W, H, FPS = 720, 540, 30


def fingertip_z(cm, q) -> float:
    m, d = cm.model, cm.data
    cm.set_q(q)
    z = 1e9
    for bn in ("Moving_Jaw", "Fixed_Jaw"):
        b = m.body(bn).id
        R = d.xmat[b].reshape(3, 3)
        z = min(z, float((d.xpos[b] + R @ np.array([0, -FINGER, 0]))[2]))
    return z - cm._floor_z


def lerp(a, b, t):
    return {k: a[k] + (b[k] - a[k]) * t for k in b}


def main() -> int:
    cm = LekiwiCollisionModel(ASSETS, floor_clearance=0.0)
    m, d = cm.model, cm.data
    ch = json.loads((Path("examples/media/calib_choreography_lekiwi.json")).read_text())
    home = ch["home"]
    steps = [s for s in ch["steps"] if s.get("feasible")]

    # build the played trajectory (list of (label, q dict))
    traj = []
    traj += [("home", dict(home))] * 10
    for s in steps:
        j, lo, hi = s["model_joint"], s["sweep_lo"], s["sweep_hi"]
        stance = s["stance"]
        for t in np.linspace(0, 1, 20):  # transit out
            traj.append((f"to {s['joint']} stance", lerp(home, stance, t)))
        start = stance[j]
        seq = (
            list(np.linspace(start, lo, 18))
            + list(np.linspace(lo, hi, 30))
            + list(np.linspace(hi, start, 18))
        )
        for a in seq:  # the both-stop sweep
            q = dict(stance)
            q[j] = float(a)
            traj.append((f"sweep {s['joint']}", q))
        for t in np.linspace(0, 1, 20):  # transit back
            traj.append(("home", lerp(stance, home, t)))

    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.08, 0.0, 0.10]
    cam.distance, cam.azimuth, cam.elevation = 1.15, 70, -8
    opt = mujoco.MjvOption()
    r = mujoco.Renderer(m, H, W)

    out = Path("examples/media/calib_replay_lekiwi.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(out), fps=FPS, codec="libx264", quality=8, macro_block_size=None
    )
    lowest = (1e9, None, None)
    for i, (label, q) in enumerate(traj):
        fz = fingertip_z(cm, q)  # also sets q
        if fz < lowest[0]:
            lowest = (fz, label, dict(q))
        r.update_scene(d, cam, opt)
        frame = r.render()
        writer.append_data(frame)
        if i == 0:
            first = frame
    writer.close()

    # render + save the lowest-clearance frame
    cm.set_q(lowest[2])
    r.update_scene(d, cam, opt)
    imageio.imwrite("examples/media/calib_replay_lowest.png", r.render())
    imageio.imwrite("examples/media/calib_replay_first.png", first)

    print(f"wrote {out}  ({len(traj)} frames, {len(traj) / FPS:.1f} s)")
    print(
        f"LOWEST real-fingertip clearance over the whole dance: "
        f"{lowest[0] * 100:+.1f} cm  during '{lowest[1]}'"
    )
    print(
        f"  (modeled floor at z={cm._floor_z * 100:+.2f} cm; floor covering sits "
        "~1-2 cm above that)"
    )
    print("wrote examples/media/calib_replay_lowest.png + _first.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
