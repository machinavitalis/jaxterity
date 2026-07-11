#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Model-driven, embodiment-agnostic calibration planner.

Given *any* MuJoCo scene (a model with collision geometry + a floor), this
derives the whole calibration plan from the kinematic tree — no per-robot
hardcoding:

* **arm joints** = the limited hinge/slide joints (minus declared grippers),
* **collision rule** = a contact is real iff the two bodies are more than
  ``NEAR_LINK_HOPS`` apart in the body tree (this single rule, taken over the
  tree *including the world body*, catches self-collision, base, and floor at
  once — structurally-adjacent links share a joint and are ignored),
* **calibration order** = distal→proximal, by body depth, so each proximal
  sweep can tuck the already-calibrated distal links,
* **seed pose** = a compact, collision-free landmark the operator sets once,
  snapped onto joint limits where possible so it is easy to hit by hand.

It reuses the SO-101 planner's stance search and renderer (``calib_planner``),
and also renders a "fold from an arbitrary pose to the seed" clip for the human.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import calib_planner as cp
import mujoco
import numpy as np

NEAR_LINK_HOPS = 2
SAFETY_MARGIN = 0.008


class GenericModel:
    """Any compiled MuJoCo arm scene, exposing the ``CollisionModel`` surface."""

    def __init__(self, scene_path: str, *, gripper_joints=(), exclude_joints=()):
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        m = self.model
        m.geom_margin[:] = SAFETY_MARGIN
        skip = set(gripper_joints) | set(exclude_joints)
        # arm joints: limited hinge(3)/slide(2) with a real range, not a gripper
        self.arm_joints = []
        for i in range(m.njnt):
            j = m.joint(i)
            if (
                int(j.type[0]) in (2, 3)
                and int(j.limited[0])
                and j.range[1] > j.range[0]
            ):
                if j.name not in skip:
                    self.arm_joints.append(j.name)
        self.qadr = {j: m.joint(j).qposadr[0] for j in self.arm_joints}
        self.jnt_range = {j: tuple(m.joint(j).range) for j in self.arm_joints}
        # body tree (include world=0), all-pairs hop distance
        adj: dict[int, set[int]] = {i: set() for i in range(m.nbody)}
        for b in range(1, m.nbody):
            p = int(m.body_parentid[b])
            adj[b].add(p)
            adj[p].add(b)
        self._hops = cp.CollisionModel._all_pairs_hops(adj, m.nbody)
        self._depth = {b: self._hops.get(frozenset((0, b)), 0) for b in range(m.nbody)}
        # bodies that move when an arm joint moves = subtree below each arm joint
        self._arm_bodies = set()
        for jn in self.arm_joints:
            cb = int(m.jnt_bodyid[m.joint(jn).id])
            self._arm_bodies |= self._subtree(cb)
        self._arm_geoms = [
            g for g in range(m.ngeom) if m.geom_bodyid[g] in self._arm_bodies
        ]
        # distal -> proximal: deepest body first
        self.order = sorted(
            self.arm_joints,
            key=lambda jn: self._depth[int(m.jnt_bodyid[m.joint(jn).id])],
            reverse=True,
        )
        # Allowed-Collision Matrix (MoveIt/SRDF style): geom pairs that are in
        # contact in (almost) every sampled configuration are structural, not
        # faults — disable them. Derived from the model, not hand-listed.
        self._acm = self._build_acm()

    def _build_acm(self, samples: int = 500, always: float = 0.98) -> set:
        d = self.data
        rng = np.random.default_rng(12345)
        poses = [{j: 0.0 for j in self.arm_joints}]
        poses += [
            {j: float(rng.uniform(*self.jnt_range[j])) for j in self.arm_joints}
            for _ in range(samples)
        ]
        counts: dict = {}
        for q in poses:
            self.set_q(q)
            for i in range(d.ncon):
                c = d.contact[i]
                key = frozenset((int(c.geom1), int(c.geom2)))
                counts[key] = counts.get(key, 0) + 1
        n = len(poses)
        return {k for k, v in counts.items() if v / n >= always}

    def _subtree(self, root: int) -> set:
        m = self.model
        out, stack = set(), [root]
        while stack:
            b = stack.pop()
            out.add(b)
            for c in range(m.nbody):
                if int(m.body_parentid[c]) == b and c not in out:
                    stack.append(c)
        return out

    def _watched_geoms(self, g1, g2) -> bool:
        """A contact is real unless the pair is directly-adjacent or in the ACM."""
        if frozenset((g1, g2)) in self._acm:
            return False
        b1, b2 = int(self.model.geom_bodyid[g1]), int(self.model.geom_bodyid[g2])
        if b1 == b2:
            return False
        return self._hops.get(frozenset((b1, b2)), 99) > 1  # not parent-child

    def _bad_contacts(self):
        d = self.data
        return [
            d.contact[i].dist
            for i in range(d.ncon)
            if self._watched_geoms(int(d.contact[i].geom1), int(d.contact[i].geom2))
        ]

    def self_touch_angle(self, joint: str, stance: dict, direction: int = 1, *, k=90):
        """Angle at which sweeping ``joint`` first makes a *self* contact.

        For a joint with no usable hard stop (continuous, or a stop beyond the
        encoder range), the first link-on-link self-contact is a known geometric
        reference — the runtime drives until that contact (a load spike) and
        reads the encoder there, exactly as it would at a mechanical stop. Floor
        contacts are excluded so the reference is independent of how the robot is
        mounted. Returns the angle, or ``None`` if no self-contact in range.
        """
        lo, hi = self.jnt_range[joint]
        target = hi if direction > 0 else lo
        for a in np.linspace(stance[joint], target, k):
            q = dict(stance)
            q[joint] = float(a)
            self.set_q(q)
            d = self.data
            for i in range(d.ncon):
                c = d.contact[i]
                g1, g2 = int(c.geom1), int(c.geom2)
                b1, b2 = (
                    int(self.model.geom_bodyid[g1]),
                    int(self.model.geom_bodyid[g2]),
                )
                if 0 in (b1, b2):
                    continue  # skip the floor — we want self-touch
                if self._watched_geoms(g1, g2):
                    return float(a)
        return None

    def set_q(self, q: dict) -> None:
        for name, adr in self.qadr.items():
            if name in q:
                self.data.qpos[adr] = q[name]
        mujoco.mj_forward(self.model, self.data)

    def is_safe(self, q: dict) -> bool:
        self.set_q(q)
        return not self._bad_contacts()

    def min_geom_z(self) -> float:
        zs = [float(self.data.geom_xpos[g, 2]) for g in self._arm_geoms]
        return min(zs) if zs else 0.0


def seed_pose(gm: GenericModel, rng) -> tuple[dict, list]:
    """A collision-free seed that puts as many joints as possible *on a hard stop*.

    Every candidate value is something a human can set precisely by hand — a hard
    stop (best: push until it stops) or "straight/in line" (the joint's zero) —
    never an arbitrary middle. A joint on a stop also hands us its offset for
    free (the stop is a known angle). Picks the collision-free combination with
    the most joints on stops, then the most lifted.

    Returns ``(pose, notes)`` with ``notes`` = ``(joint, value, instruction,
    kind)`` and ``kind`` in {``"stop"``, ``"straight"``}.
    """
    import itertools

    joints = list(gm.arm_joints)
    options = {}
    for j in joints:
        lo, hi = gm.jnt_range[j]
        opts = [
            (lo, "push to its hard stop (firmly, as shown)", "stop"),
            (hi, "push to its hard stop (firmly, as shown)", "stop"),
        ]
        if lo < 0.0 < hi:  # zero is reachable -> "straight / in line"
            opts.append(
                (0.0, "set straight, in line (approx OK - auto-tuned)", "straight")
            )
        options[j] = opts

    best = None
    for combo in itertools.product(*[range(len(options[j])) for j in joints]):
        pose = {j: options[j][combo[k]][0] for k, j in enumerate(joints)}
        if not gm.is_safe(pose):
            continue
        n_stop = sum(options[j][combo[k]][2] == "stop" for k, j in enumerate(joints))
        gm.set_q(pose)
        score = (n_stop, gm.min_geom_z())
        if best is None or score > best[0]:
            best = (score, combo)

    if best is None:  # nothing all-landmark is safe -> fall back to compact home
        cp.ARM_JOINTS = tuple(joints)
        home = cp.find_home(gm, rng, n=6000)
        notes = [
            (
                j,
                round(float(home[j]), 3),
                "roughly like the picture (auto-tuned)",
                "approx",
            )
            for j in joints
        ]
        return {j: float(home[j]) for j in joints}, notes

    combo = best[1]
    pose = {j: float(options[j][combo[k]][0]) for k, j in enumerate(joints)}
    notes = [
        (j, round(pose[j], 3), options[j][combo[k]][1], options[j][combo[k]][2])
        for k, j in enumerate(joints)
    ]
    return pose, notes


def plan_fold(gm: GenericModel, start: dict, seed: dict, *, per=26):
    """A collision-free joint-space path from ``start`` to ``seed``.

    Tries a direct interpolation; if any frame collides, folds joints one at a
    time distal->proximal (tuck the hand first), which keeps the swept volume
    small and almost always stays clear.
    """

    def interp(a, b, t):
        return {k: a[k] + (b[k] - a[k]) * t for k in a}

    direct = [interp(start, seed, t) for t in np.linspace(0, 1, per * len(gm.order))]
    if all(gm.is_safe(q) for q in direct):
        return direct, "direct"
    path, cur = [], dict(start)
    for j in gm.order:  # distal first
        for t in np.linspace(0, 1, per):
            q = dict(cur)
            q[j] = cur[j] + (seed[j] - cur[j]) * float(t)
            path.append(q)
        cur[j] = seed[j]
    return path, "staged distal->proximal"


def _annot(frame, lines):
    import cv2

    img = np.ascontiguousarray(frame[..., ::-1])
    y = 36
    for i, (t, c) in enumerate(lines):
        s = 0.85 if i == 0 else 0.62
        cv2.putText(
            img, t, (18, y), cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 0), 4, cv2.LINE_AA
        )
        cv2.putText(img, t, (18, y), cv2.FONT_HERSHEY_SIMPLEX, s, c, 1, cv2.LINE_AA)
        y += 32
    return np.ascontiguousarray(img[..., ::-1])


def render_fold(
    gm: GenericModel, start, seed, notes, path, out: Path, cam, *, fps=30, w=900, h=720
):
    """Instructional clip: open on the TARGET pose, preview the hand-fold to it,
    then orbit and hold the target so the operator can replicate it.

    The arm is uncalibrated and cannot move itself; the fold is a *preview* of
    what the operator does by hand. The clip therefore shows the target first
    and last (held + orbited), with the transient fold clearly labelled.
    """
    import math

    import imageio.v2 as imageio

    r = mujoco.Renderer(gm.model, h, w)
    frames = []

    short = {
        "stop": "PUSH IT TO ITS STOP",
        "straight": "set straight / in line",
        "approx": "~ as shown (auto-tuned)",
    }
    seed_lines = [("SET THE ARM TO THIS POSE - BY HAND", (90, 235, 120))]
    for j, _, _, kind in notes[:6]:
        col = (120, 235, 140) if kind == "stop" else (205, 215, 240)
        seed_lines.append((f"   {j}:  {short.get(kind, kind)}", col))

    def frame(lines, n=1):
        r.update_scene(gm.data, camera=cam)
        img = _annot(r.render(), lines)
        frames.extend([img] * n)

    # 1) open on the TARGET pose — make the ask unmistakable
    gm.set_q(seed)
    frame(
        [
            ("This is the TARGET calibration pose.", (255, 230, 120)),
            ("YOU set the arm to it by hand (preview below).", (205, 225, 255)),
            (
                "The arm does NOT move itself - it isn't calibrated yet.",
                (240, 150, 150),
            ),
        ],
        fps * 2,
    )

    # 2) preview the hand-fold from wherever the arm is now, to the target
    for q in path:
        gm.set_q(q)
        frame([("preview: you hand-fold the arm to the target ...", (255, 230, 120))])

    # 3) the answer: orbit + hold the target so it reads from every side
    base_az = float(cam.azimuth)
    for k in range(fps * 5):
        cam.azimuth = base_az + 38.0 * math.sin(2 * math.pi * k / (fps * 5))
        gm.set_q(seed)
        frame(seed_lines)
    cam.azimuth = base_az

    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out, frames, fps=fps, quality=8, macro_block_size=None)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", required=True, help="MuJoCo scene xml (model + floor).")
    p.add_argument(
        "--gripper-joints", default="", help="comma-separated joints to skip"
    )
    p.add_argument("--out", default="examples/media/calib_generic")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fold-video", action="store_true")
    p.add_argument(
        "--cam",
        default="0.0,-0.1,0.15,0.7,140,-18",
        help="lookat_x,y,z,distance,azimuth,elevation",
    )
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    grip = tuple(s for s in args.gripper_joints.split(",") if s)
    gm = GenericModel(args.scene, gripper_joints=grip)
    print(f"model: {args.scene}")
    print(f"  detected arm joints : {gm.arm_joints}")
    print(f"  calibration order   : {gm.order}")

    pose, notes = seed_pose(gm, rng)
    n_stop = sum(k == "stop" for _, _, _, k in notes)
    print(
        f"  seed pose (set once by hand): {n_stop}/{len(notes)} joints on a hard stop"
    )
    for j, v, instr, kind in notes:
        print(f"    {j:16s} = {v:+.3f} rad   -> {instr}")

    steps = cp.plan_choreography(gm, gm.order, pose, rng, n_candidates=250)
    print("  per-joint sweep coverage from the planned stances:")
    for s in steps:
        if s.get("feasible"):
            print(
                f"    {s['joint']:16s} {s['range_coverage'] * 100:5.1f}%  "
                f"both_stops={s['reaches_both_limits']}"
            )
        else:
            print(f"    {s['joint']:16s} infeasible")

    print(f"  ACM: {len(gm._acm)} structural geom-pairs auto-disabled (derived)")
    # self-touch fallback: a model-derived reference angle for a no-stop joint
    demo_j = gm.order[len(gm.order) // 2]
    ang = gm.self_touch_angle(demo_j, pose, +1)
    if ang is None:
        ang = gm.self_touch_angle(demo_j, pose, -1)
    print(
        f"  self-touch reference for {demo_j}: "
        + (f"{ang:+.3f} rad" if ang is not None else "none in range")
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(out) + ".json").write_text(
        json.dumps(
            {
                "scene": args.scene,
                "arm_joints": gm.arm_joints,
                "order": gm.order,
                "seed_pose": {k: round(v, 5) for k, v in pose.items()},
                "seed_notes": [[j, v, instr, kind] for j, v, instr, kind in notes],
                "steps": steps,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"  wrote {out}.json")

    if args.fold_video:
        lx, ly, lz, dist, az, el = (float(x) for x in args.cam.split(","))
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [lx, ly, lz]
        cam.distance, cam.azimuth, cam.elevation = dist, az, el
        # an arbitrary, collision-free middle start pose
        start = None
        for _ in range(2000):
            cand = {j: float(rng.uniform(*gm.jnt_range[j])) for j in gm.arm_joints}
            if gm.is_safe(cand):
                start = cand
                break
        start = start or {j: 0.0 for j in gm.arm_joints}
        path, mode = plan_fold(gm, start, pose)
        print(f"  fold path: {mode}, {len(path)} frames")
        vid = render_fold(
            gm, start, pose, notes, path, Path(str(out) + "_fold.mp4"), cam
        )
        print(f"  wrote {vid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
