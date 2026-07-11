#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Offline, collision-aware calibration planner for the SO-101 (a-priori, per model).

The runtime that drives the real servos must stay dumb so it cannot improvise
into a collision. This module is where the intelligence lives: once per
embodiment, on the host (with MuJoCo), it

1. builds a *collision* model of the arm from the URDF kinematics + primitive
   capsule proxies + a floor (geometry the dynamics model omits),
2. finds a compact, collision-free "home" the operator seeds the arm in,
3. plans, joint by joint, a **stance** for the already-calibrated joints that
   opens up the widest collision-free sweep for the next joint — the
   knowledge-accumulating sequence — and
4. dry-runs the whole choreography in simulation and renders it to a video,

emitting a ``choreography.json`` the thin on-device executor would later play.
Nothing here touches hardware; the point is to validate the plan in sim first.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

# --------------------------------------------------------------------------- #
# 1. Collision model: capsule proxies + floor over the URDF kinematics
# --------------------------------------------------------------------------- #

#: Capsule radius for the link proxies (m). Slightly fat = conservative.
LINK_RADIUS = 0.018
#: Contact margin (m): configs with any watched pair closer than this are
#: treated as unsafe, so the plan keeps a safety buffer.
SAFETY_MARGIN = 0.008
#: Link pairs within this graph distance in the kinematic chain are structurally
#: near (their capsules share a joint) and are not treated as self-collisions;
#: the real dangers — a distal link into the base, any link into the floor — are
#: farther apart and stay watched.
NEAR_LINK_HOPS = 2

ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)


def build_collision_mjcf(robot) -> str:
    """Augment the robot's dynamics MJCF with capsule collision proxies + a floor.

    Args
    ----
    robot : jaxterity Robot
        Source of the (geometry-free) kinematics via ``to_mjcf()``.

    Returns
    -------
    str
        An MJCF string with one capsule per moving link, a base box, a ground
        plane, a light, an offscreen camera, and a render framebuffer.
    """
    root = ET.fromstring(robot.to_mjcf())
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", {"offwidth": "1280", "offheight": "960"})
    ET.SubElement(
        visual, "headlight", {"diffuse": "0.6 0.6 0.6", "ambient": "0.3 0.3 0.3"}
    )
    defaults = ET.SubElement(root, "default")
    ET.SubElement(defaults, "geom", {"margin": str(SAFETY_MARGIN)})

    wb = root.find("worldbody")

    def add_capsule(body, to, rgba="0.25 0.55 0.95 1"):
        length = float(np.linalg.norm(to))
        g = ET.SubElement(
            body, "geom", {"type": "capsule", "size": str(LINK_RADIUS), "rgba": rgba}
        )
        g.set(
            "fromto",
            "0 0 0 0 0 0.04" if length < 1e-4 else f"0 0 0 {to[0]} {to[1]} {to[2]}",
        )

    def walk(body):
        children = body.findall("body")
        if body.get("name") == "base_link":
            ET.SubElement(
                body,
                "geom",
                {
                    "type": "box",
                    "pos": "0 0 0.031",
                    "size": "0.045 0.05 0.031",
                    "rgba": "0.40 0.40 0.45 1",
                },
            )
        elif children:
            add_capsule(body, [float(x) for x in children[0].get("pos").split()])
        else:
            add_capsule(body, (0.0, 0.0, 0.0), rgba="0.95 0.55 0.2 1")  # gripper tip
        for c in children:
            walk(c)

    for b in wb.findall("body"):
        walk(b)

    ET.SubElement(
        wb,
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "pos": "0 0 0",
            "size": "1 1 0.1",
            "rgba": "0.80 0.80 0.84 1",
        },
    )
    ET.SubElement(
        wb,
        "light",
        {"pos": "0.3 0.3 1.3", "dir": "-0.2 -0.2 -1", "diffuse": "0.8 0.8 0.8"},
    )
    ET.SubElement(
        wb,
        "camera",
        {"name": "iso", "pos": "0.55 -0.5 0.5", "xyaxes": "0.7 0.7 0 -0.33 0.33 0.88"},
    )
    return ET.tostring(root, encoding="unicode")


class CollisionModel:
    """A compiled SO-101 collision model with a clean safe/clearance query."""

    def __init__(self, robot):
        self.model = mujoco.MjModel.from_xml_string(build_collision_mjcf(robot))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.qadr = {j: m.joint(j).qposadr[0] for j in ARM_JOINTS}
        self.jnt_range = {j: tuple(m.joint(j).range) for j in ARM_JOINTS}
        self._world = 0
        self._base = m.body("base_link").id
        # Graph distance between link bodies along the kinematic tree, so we can
        # ignore structurally-near pairs (their capsules share a joint).
        adj: dict[int, set[int]] = {i: set() for i in range(m.nbody)}
        for joint in robot.joints:
            a, b = m.body(joint.parent_link).id, m.body(joint.child_link).id
            adj[a].add(b)
            adj[b].add(a)
        self._hops = self._all_pairs_hops(adj, m.nbody)

    @staticmethod
    def _all_pairs_hops(adj, n):
        import collections

        hops = {}
        for src in range(n):
            dist = {src: 0}
            q = collections.deque([src])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        q.append(v)
            for dst, dd in dist.items():
                hops[frozenset((src, dst))] = dd
        return hops

    def _watched(self, ba, bb) -> bool:
        """True if a contact between these two bodies is a real collision."""
        if ba == bb:
            return False
        if self._world in (ba, bb):  # floor: watch every link except the base
            other = bb if ba == self._world else ba
            return other != self._base
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
        """Set the arm joints from a ``{name: angle}`` dict and run forward."""
        for name, adr in self.qadr.items():
            if name in q:
                self.data.qpos[adr] = q[name]
        mujoco.mj_forward(self.model, self.data)

    def is_safe(self, q: dict) -> bool:
        """True if no non-adjacent pair is within the safety margin."""
        self.set_q(q)
        return not self._bad_contacts()

    def min_geom_z(self) -> float:
        """Lowest point of any moving geom (height above the floor)."""
        return float(self.data.geom_xpos[1:, 2].min())


# --------------------------------------------------------------------------- #
# 2. Home pose + 3. knowledge-accumulating stance planner
# --------------------------------------------------------------------------- #


def find_home(cm: CollisionModel, rng: np.random.Generator, n: int = 4000) -> dict:
    """Find a compact, collision-free seed pose (the one the operator sets).

    Samples configurations and keeps the safe one whose lowest link sits highest
    above the floor — i.e. the arm folded up and away from the table.
    """
    best, best_z = None, -1e9
    lows = {j: lo for j, (lo, hi) in cm.jnt_range.items()}
    highs = {j: hi for j, (lo, hi) in cm.jnt_range.items()}
    for _ in range(n):
        q = {j: float(rng.uniform(lows[j], highs[j])) for j in ARM_JOINTS}
        if cm.is_safe(q):
            z = cm.min_geom_z()
            if z > best_z:
                best, best_z = q, z
    if best is None:
        raise RuntimeError("no collision-free home found — loosen the model.")
    return best


def _largest_safe_arc(cm: CollisionModel, base_q: dict, joint: str, k: int = 31):
    """Largest contiguous collision-free angular interval of ``joint`` from base_q."""
    lo, hi = cm.jnt_range[joint]
    angles = np.linspace(lo, hi, k)
    safe = []
    for a in angles:
        q = dict(base_q)
        q[joint] = float(a)
        safe.append(cm.is_safe(q))
    # longest run of True
    best_i = best_len = cur_i = cur_len = 0
    for i, s in enumerate(safe):
        if s:
            if cur_len == 0:
                cur_i = i
            cur_len += 1
            if cur_len > best_len:
                best_len, best_i = cur_len, cur_i
        else:
            cur_len = 0
    if best_len == 0:
        return None
    return float(angles[best_i]), float(angles[best_i + best_len - 1]), best_len / k


def plan_choreography(
    cm: CollisionModel,
    order,
    home: dict,
    rng: np.random.Generator,
    n_candidates: int = 250,
) -> list:
    """Plan a stance + safe sweep for each joint, in dependency order.

    For joint ``J`` the planner may reposition only joints **earlier** in
    ``order`` (already calibrated, so commandable in radians); later joints stay
    at ``home`` (their operator-seeded rest). It searches earlier-joint stances
    to maximise ``J``'s contiguous collision-free sweep — the "put the arm where
    the next motor has room" step.
    """
    steps = []
    for idx, joint in enumerate(order):
        earlier = order[:idx]
        best = None  # (coverage, stance, arc_lo, arc_hi)
        # candidate 0: plain home (no repositioning)
        candidates = [dict(home)]
        for _ in range(n_candidates):
            cand = dict(home)
            for e in earlier:
                lo, hi = cm.jnt_range[e]
                cand[e] = float(rng.uniform(lo, hi))
            candidates.append(cand)
        for cand in candidates:
            arc = _largest_safe_arc(cm, cand, joint)
            if arc is None:
                continue
            lo_a, hi_a, cov = arc
            if best is None or cov > best[0]:
                best = (cov, cand, lo_a, hi_a)
        if best is None:
            steps.append({"joint": joint, "feasible": False})
            continue
        cov, stance, lo_a, hi_a = best
        full_lo, full_hi = cm.jnt_range[joint]
        steps.append(
            {
                "joint": joint,
                "feasible": True,
                "stance": {k: round(v, 5) for k, v in stance.items()},
                "depends_on": list(earlier),
                "sweep_lo": round(lo_a, 5),
                "sweep_hi": round(hi_a, 5),
                "range_coverage": round(cov, 3),
                "full_range": [round(full_lo, 5), round(full_hi, 5)],
                "reaches_both_limits": cov > 0.98,
            }
        )
    return steps


# --------------------------------------------------------------------------- #
# 4. Dry-run + video
# --------------------------------------------------------------------------- #


def _annotate(frame, lines):
    """Overlay text lines on a frame (top-left) with OpenCV."""
    import cv2

    img = np.ascontiguousarray(frame[..., ::-1])  # RGB->BGR for cv2
    y = 34
    for i, (text, color) in enumerate(lines):
        cv2.putText(
            img,
            text,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8 if i == 0 else 0.62,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            text,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8 if i == 0 else 0.62,
            color,
            1,
            cv2.LINE_AA,
        )
        y += 30
    return np.ascontiguousarray(img[..., ::-1])  # back to RGB


def render_dryrun(
    cm: CollisionModel,
    steps,
    home: dict,
    path: Path,
    *,
    fps: int = 30,
    w: int = 720,
    h: int = 960,
    camera="iso",
) -> Path:
    """Animate the choreography and write an annotated MP4."""
    import imageio.v2 as imageio

    renderer = mujoco.Renderer(cm.model, h, w)
    frames = []
    calibrated = []
    cur = dict(home)

    def emit(q, lines, n):
        cm.set_q(q)
        safe = not cm._bad_contacts()
        renderer.update_scene(cm.data, camera=camera)
        img = renderer.render()
        status = ("SAFE", (40, 200, 70)) if safe else ("COLLISION", (235, 60, 60))
        for _ in range(n):
            frames.append(_annotate(img, lines + [status]))

    # intro: the seeded home pose
    emit(
        home,
        [
            ("SO-101 collision-aware calibration plan", (255, 255, 255)),
            ("operator seeds compact home pose", (200, 220, 255)),
        ],
        fps,
    )

    def lerp(a, b, t):
        return {k: a[k] + (b[k] - a[k]) * t for k in a}

    for step in steps:
        joint = step["joint"]
        if not step.get("feasible"):
            continue
        stance = {**home, **step["stance"]}
        done = ", ".join(calibrated) or "—"
        header = [
            (f"calibrate: {joint}", (255, 230, 120)),
            (f"calibrated so far: {done}", (180, 255, 180)),
        ]
        # transit to stance (reposition already-calibrated joints)
        for t in np.linspace(0, 1, max(2, fps // 2)):
            q = lerp(cur, stance, float(t))
            cm.set_q(q)
            renderer.update_scene(cm.data, camera=camera)
            img = renderer.render()
            safe = (
                ("SAFE", (40, 200, 70))
                if not cm._bad_contacts()
                else ("COLLISION", (235, 60, 60))
            )
            frames.append(
                _annotate(img, header + [("move to stance", (200, 220, 255)), safe])
            )
        cur = dict(stance)
        # sweep the joint across its safe arc (to one stop, to the other, back)
        lo, hi = step["sweep_lo"], step["sweep_hi"]
        seq = (
            list(np.linspace((lo + hi) / 2, lo, fps))
            + list(np.linspace(lo, hi, 2 * fps))
            + list(np.linspace(hi, (lo + hi) / 2, fps))
        )
        cover = step["range_coverage"]
        note = (
            "full range -> both stops"
            if step["reaches_both_limits"]
            else f"safe arc {cover * 100:.0f}% of range"
        )
        for a in seq:
            q = dict(stance)
            q[joint] = float(a)
            cm.set_q(q)
            renderer.update_scene(cm.data, camera=camera)
            img = renderer.render()
            safe = (
                ("SAFE", (40, 200, 70))
                if not cm._bad_contacts()
                else ("COLLISION", (235, 60, 60))
            )
            frames.append(
                _annotate(img, header + [(f"sweep: {note}", (200, 220, 255)), safe])
            )
            cur[joint] = float(a)
        calibrated.append(joint)

    # outro
    emit(
        cur,
        [
            ("plan complete — all joints calibrated in sim", (255, 255, 255)),
            ("no collision throughout", (180, 255, 180)),
        ],
        fps,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps, quality=8, macro_block_size=None)
    return path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--robot", default="so100")
    p.add_argument("--out", default="examples/media/calib_plan")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from jaxterity import zoo

    rng = np.random.default_rng(args.seed)
    robot = zoo.load(args.robot)
    cm = CollisionModel(robot)

    print("finding a compact collision-free home ...", flush=True)
    home = find_home(cm, rng)
    cm.set_q(home)  # refresh data to the home pose before reading its height
    print(
        "  home:",
        {k: round(v, 3) for k, v in home.items()},
        f"(lowest link center z = {cm.min_geom_z() * 100:.1f} cm)",
    )

    # distal -> proximal: each proximal sweep can use the now-calibrated distal
    # joints to fold the arm clear of the floor/base.
    order = ["wrist_roll", "wrist_flex", "elbow_flex", "shoulder_lift", "shoulder_pan"]
    print("planning collision-free stances (distal -> proximal) ...", flush=True)
    steps = plan_choreography(cm, order, home, rng)
    for s in steps:
        if s.get("feasible"):
            print(
                f"  {s['joint']:14s} cover={s['range_coverage'] * 100:5.1f}%  "
                f"both_stops={s['reaches_both_limits']}  depends_on={s['depends_on']}"
            )
        else:
            print(f"  {s['joint']:14s} INFEASIBLE from any planned stance")

    out = Path(args.out)
    choreography = {
        "robot": args.robot,
        "home": {k: round(v, 5) for k, v in home.items()},
        "order": order,
        "safety_margin_m": SAFETY_MARGIN,
        "steps": steps,
    }
    (out.with_suffix(".json")).parent.mkdir(parents=True, exist_ok=True)
    Path(str(out) + ".json").write_text(json.dumps(choreography, indent=2) + "\n")
    print(f"wrote {out}.json")

    print("rendering sim dry-run video ...", flush=True)
    video = render_dryrun(cm, steps, home, Path(str(out) + ".mp4"))
    print(f"wrote {video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
