#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Infer each joint's sign from observed image motions + the model (host-side).

``lekiwi_vision_sign.py`` measures, per joint, the *image-space* direction the
arm moves for a +tick nudge. This closes the loop: the MuJoCo model predicts the
gripper's *3D* motion for a +angle nudge, and we solve for the single unknown
camera orientation that best aligns all the predicted 3D motions with the
observed 2D ones. No manual camera extrinsics — the camera pose is recovered
from the motions themselves. Each joint's sign is then the sign of the
(observed · predicted) alignment at that camera pose.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

ASSETS = Path("/tmp/lekiwi_ek/packages/lekiwi_sim/lekiwi_sim/assets")

# layout joint -> model joint
JMAP = {
    "shoulder_pan": "Rotation",
    "shoulder_lift": "Pitch",
    "elbow_flex": "Elbow",
    "wrist_flex": "Wrist_Pitch",
    "wrist_roll": "Wrist_Roll",
    "gripper": "Jaw",
}
# the folded pose the probes were taken from
SEED = {
    "Rotation": -1.920,
    "Pitch": -1.747,
    "Elbow": -1.657,
    "Wrist_Pitch": 1.660,
    "Wrist_Roll": 2.790,
    "Jaw": 0.3,
}

#: observed image motion for a +tick nudge: (dx, dy, pixels). image y is DOWN.
#: shoulder_pan was probed at -280 (it was on a stop) so its +tick vector is the
#: negation of that down+right reading.
OBS = {
    "shoulder_pan": (-15.0, -17.0, 7607),
    "shoulder_lift": (-42.0, -9.0, 21732),
    "elbow_flex": (18.0, -1.0, 8491),
    "wrist_flex": (-66.0, -3.0, 10326),
    "wrist_roll": (-11.0, -3.0, 289),  # yellow-sticker centroid (clean gripper point)
    "gripper": (-19.0, -3.0, 2897),
}


def model_motions():
    """3D gripper-tip motion for a +angle nudge of each joint, at the seed pose."""
    scene = ASSETS / "calib_scene.xml"  # written earlier by the LeKiwi planner
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    tip = m.body("Moving_Jaw").id

    def tip_pos(q):
        for jn, val in q.items():
            d.qpos[m.joint(jn).qposadr[0]] = val
        mujoco.mj_forward(m, d)
        return d.xpos[tip].copy()

    base = dict(SEED)
    p0 = tip_pos(base)
    out = {}
    for lay, mj in JMAP.items():
        q = dict(base)
        q[mj] = base[mj] + 0.06
        out[lay] = tip_pos(q) - p0
    return out


def rot(az, el, roll):
    """Camera world->image rotation from azimuth/elevation/roll (radians)."""
    ca, sa = np.cos(az), np.sin(az)
    ce, se = np.cos(el), np.sin(el)
    cr, sr = np.cos(roll), np.sin(roll)
    Rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, ce, -se], [0, se, ce]])
    Rr = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Rr @ Rx @ Rz


def main() -> int:
    d3d = model_motions()
    joints = list(OBS)
    obs = {j: np.array(OBS[j][:2], float) for j in joints}
    obs = {j: v / (np.linalg.norm(v) + 1e-9) for j, v in obs.items()}
    w = {j: OBS[j][2] for j in joints}

    # search camera orientation maximizing |observed . projected| (sign-agnostic)
    best = (-1, None)
    grid = [
        (az, el, roll)
        for az in np.linspace(0, 2 * np.pi, 48, endpoint=False)
        for el in np.linspace(-1.0, 1.0, 13)
        for roll in np.linspace(-0.5, 0.5, 7)
    ]
    for az, el, roll in grid:
        R = rot(az, el, roll)
        score = 0.0
        for j in joints:
            pj = (R @ d3d[j])[:2]
            n = np.linalg.norm(pj)
            if n < 1e-6:
                continue
            score += w[j] * abs(float(obs[j] @ (pj / n)))
        if score > best[0]:
            best = (score, (az, el, roll))
    az, el, roll = best[1]
    R = rot(az, el, roll)

    print(
        f"recovered camera (az={np.degrees(az):.0f} el={np.degrees(el):.0f} "
        f"roll={np.degrees(roll):.0f}); alignment score {best[0]:.0f}"
    )
    print("\nper-joint sign (and alignment quality):")
    signs = {}
    for j in joints:
        pj = (R @ d3d[j])[:2]
        n = np.linalg.norm(pj)
        align = float(obs[j] @ (pj / (n + 1e-9)))
        s = 1 if align >= 0 else -1
        signs[j] = s
        conf = "strong" if abs(align) > 0.7 else ("ok" if abs(align) > 0.4 else "WEAK")
        print(f"  {j:14s} sign {s:+d}   (cos={align:+.2f}, {conf})")
    print("\nsigns =", {j: signs[j] for j in joints})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
