#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Collision-aware calibration planner on the *real* LeKiwi MuJoCo model.

Same idea as ``calib_planner.py`` (a-priori, per-embodiment plan + sim dry-run
video) but using the open-source Ekumen LeKiwi model — real meshes and the
actual mobile base with its electronics stack — instead of capsule proxies. So
the clearances are the true ones: the arm's links against the LeKiwi's own
battery/SBC/camera on top of the base, and against the floor the base stands on.

Model: Ekumen-OS/lekiwi (``packages/lekiwi_sim/.../assets``), Apache-2.0.

Only the arm moves during calibration, so a contact is a real collision iff it
involves a *moving arm body* and the two bodies are not structurally near in the
kinematic tree. Floor and base-electronics contacts with a moving arm link are
exactly the dangers we plan around.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import calib_planner as cp  # reuse find_home / plan_choreography / render_dryrun
import mujoco
import numpy as np

#: SO-ARM100 arm joints in the Ekumen model, distal->proximal calibration order.
LEKIWI_ARM = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll")
ORDER = ["Wrist_Roll", "Wrist_Pitch", "Elbow", "Pitch", "Rotation"]

#: Bodies that move when an arm joint moves (everything below the first arm
#: hinge). The arm's static ``Base`` and the whole mobile base are excluded.
MOVING_ARM_BODIES = {
    "Rotation_Pitch",
    "Upper_Arm",
    "Lower_Arm",
    "Wrist_Pitch_Roll",
    "Fixed_Jaw",
    "Moving_Jaw",
    "wrist_camera_mount_link",
    "wrist_camera_link",
}

SAFETY_MARGIN = 0.008  # m clearance buffer (real meshes → can be tight)
NEAR_LINK_HOPS = 2


def _calib_scene(assets: Path) -> Path:
    """Write a minimal scene that includes the LeKiwi + a named floor, no cubes."""
    scene = assets / "calib_scene.xml"
    scene.write_text(
        '<mujoco model="lekiwi_calib">\n'
        '  <include file="lekiwi/lekiwi.xml"/>\n'
        '  <visual><global offwidth="1280" offheight="960"/>'
        '<headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3"/></visual>\n'
        "  <worldbody>\n"
        '    <geom name="floor" pos="0 0 -0.1" size="0 0 0.05" type="plane" '
        'rgba="0.8 0.8 0.84 1"/>\n'
        '    <light pos="0.4 0.4 1.4" dir="-0.2 -0.2 -1" diffuse="0.8 0.8 0.8"/>\n'
        '    <camera name="iso" pos="0.62 -0.58 0.52" '
        'xyaxes="0.7 0.7 0 -0.33 0.33 0.88"/>\n'
        "  </worldbody>\n"
        "</mujoco>\n"
    )
    return scene


class LekiwiCollisionModel:
    """The real LeKiwi model with the same query surface as ``CollisionModel``."""

    def __init__(
        self,
        assets: Path,
        floor_clearance: float = 0.0,
        jaw_open: bool = False,
        finger_margin: float = 0.0,
    ):
        scene = _calib_scene(assets)
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        m = self.model
        m.geom_margin[:] = SAFETY_MARGIN  # clearance buffer for contact reporting
        # Model fidelity (L-011): the real SO-101 gripper opens wider than the
        # model's Jaw range, so clearance computed with the jaw closed under-reads
        # if the real jaw is open. Calibration holds the jaw *closed* through all
        # arm motion (so closed is correct), but ``jaw_open=True`` parks the jaw
        # at its model-max as a worst-case check that the plan survives a slipped
        # hold. (Widen the model's Jaw range upstream for a fully honest splay.)
        if "Jaw" in [m.joint(i).name for i in range(m.njnt)]:
            ja = m.joint("Jaw")
            self.data.qpos[ja.qposadr[0]] = ja.range[1] if jaw_open else ja.range[0]
        self.qadr = {j: m.joint(j).qposadr[0] for j in LEKIWI_ARM}
        self.jnt_range = {j: tuple(m.joint(j).range) for j in LEKIWI_ARM}
        self._world = 0
        self._moving = {m.body(n).id for n in MOVING_ARM_BODIES}
        # arm geoms (for measuring how low the arm dips, ignoring the base/wheels)
        self._arm_geoms = [
            g for g in range(m.ngeom) if m.geom_bodyid[g] in self._moving
        ]
        # graph distance between bodies along the kinematic tree
        adj: dict[int, set[int]] = {i: set() for i in range(m.nbody)}
        for b in range(1, m.nbody):
            p = int(m.body_parentid[b])
            adj[b].add(p)
            adj[p].add(b)
        self._hops = cp.CollisionModel._all_pairs_hops(adj, m.nbody)

        # --- floor fidelity (L-005) -------------------------------------- #
        # The hard-coded floor plane (z=-0.1) lets the gripper pass through
        # 10 cm of phantom air. Derive the *real* floor from where the static
        # base/wheels rest, snap the plane there, and require the arm to keep a
        # clearance margin above it (a sag/compliance budget) — not merely "not
        # touching". This makes the pre-HW plan honest about the floor.
        mujoco.mj_forward(m, self.data)
        self._floor_geom = int(m.geom("floor").id)
        base_geoms = [
            g
            for g in range(m.ngeom)
            if m.geom_bodyid[g] not in self._moving
            and g != self._floor_geom
            and m.geom_bodyid[g] != self._world
        ]
        self._floor_z = (
            min(float(self.data.geom_xpos[g, 2]) for g in base_geoms)
            if base_geoms
            else -0.1
        )
        m.geom_pos[self._floor_geom, 2] = self._floor_z  # snap plane to real floor
        self.floor_clearance = float(floor_clearance)

        # The model's jaw finger geoms reach ~10 cm, but the real SO-101 fingers
        # (closed) run a bit longer; ``finger_margin`` adds that extra reach as a
        # virtual point past each jaw body along its finger axis (local -y), so
        # the floor check accounts for the part of the gripper the mesh omits
        # (L-013). Set it to (real finger length - ~0.10 m). 0 = trust the mesh.
        self.finger_margin = float(finger_margin)
        self._jaw_bodies = [
            m.body(n).id
            for n in ("Moving_Jaw", "Fixed_Jaw")
            if n in [m.body(i).name for i in range(m.nbody)]
        ]

    def _watched(self, ba, bb) -> bool:
        if ba == bb:
            return False
        # only collisions that involve a moving arm link matter
        if ba not in self._moving and bb not in self._moving:
            return False
        return self._hops.get(frozenset((ba, bb)), 99) > NEAR_LINK_HOPS

    def _bad_contacts(self):
        m, d = self.model, self.data
        out = []
        for i in range(d.ncon):
            c = d.contact[i]
            if self._watched(m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]):
                out.append(c.dist)
        return out

    def set_q(self, q: dict) -> None:
        for name, adr in self.qadr.items():
            if name in q:
                self.data.qpos[adr] = q[name]
        mujoco.mj_forward(self.model, self.data)

    def is_safe(self, q: dict) -> bool:
        self.set_q(q)
        if self._bad_contacts():
            return False
        # keep the whole arm (incl. the real fingertip reach) a margin above floor
        return self.min_real_z() >= self._floor_z + self.floor_clearance

    def min_geom_z(self) -> float:
        """Lowest point of any *arm* geom (how far the arm dips), in metres."""
        zs = [float(self.data.geom_xpos[g, 2]) for g in self._arm_geoms]
        return min(zs) if zs else 0.0

    def _fingertip_zs(self) -> list:
        """World z of a virtual fingertip ``finger_margin`` past each jaw body."""
        out = []
        for b in self._jaw_bodies:
            r = self.data.xmat[b].reshape(3, 3)
            out.append(
                float(
                    (self.data.xpos[b] + r @ np.array([0.0, -self.finger_margin, 0.0]))[
                        2
                    ]
                )
            )
        return out

    def min_real_z(self) -> float:
        """Lowest point of the arm OR the real fingertip (modeled jaw + margin)."""
        z = self.min_geom_z()
        if self.finger_margin > 0 and self._jaw_bodies:
            z = min(z, min(self._fingertip_zs()))
        return z


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--assets", default="/tmp/lekiwi_ek/packages/lekiwi_sim/lekiwi_sim/assets"
    )
    p.add_argument("--out", default="examples/media/calib_plan_lekiwi")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    cm = LekiwiCollisionModel(Path(args.assets))

    cp.ARM_JOINTS = LEKIWI_ARM  # find_home reads this module global
    print("finding a compact collision-free home on the real LeKiwi ...", flush=True)
    home = cp.find_home(cm, rng, n=6000)
    # Face the arm forward in the seed pose (base yaw is calibrated last anyway);
    # keep it only if still collision-free.
    forward = {**home, "Rotation": 0.0}
    if cm.is_safe(forward):
        home = forward
    cm.set_q(home)
    print(
        "  home:",
        {k: round(v, 3) for k, v in home.items()},
        f"(lowest arm point z = {cm.min_geom_z() * 100:.1f} cm)",
    )

    print("planning collision-free stances (distal -> proximal) ...", flush=True)
    steps = cp.plan_choreography(cm, ORDER, home, rng, n_candidates=300)
    for s in steps:
        if s.get("feasible"):
            print(
                f"  {s['joint']:12s} cover={s['range_coverage'] * 100:5.1f}%  "
                f"both_stops={s['reaches_both_limits']}  "
                f"depends_on={s['depends_on']}"
            )
        else:
            print(f"  {s['joint']:12s} INFEASIBLE")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(out) + ".json").write_text(
        json.dumps(
            {
                "robot": "lekiwi",
                "model_source": "Ekumen-OS/lekiwi",
                "home": {k: round(v, 5) for k, v in home.items()},
                "order": ORDER,
                "safety_margin_m": SAFETY_MARGIN,
                "steps": steps,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {out}.json")

    print("rendering sim dry-run video on the real model ...", flush=True)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.05, 0.0, 0.18]
    cam.distance = 1.55
    cam.azimuth = 135
    cam.elevation = -20
    video = cp.render_dryrun(cm, steps, home, Path(str(out) + ".mp4"), camera=cam)
    print(f"wrote {video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
