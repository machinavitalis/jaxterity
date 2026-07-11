# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Post-hoc MuJoCo rendering of jaxterity trajectories with real meshes.

The tutorials run their dynamics on jaxterity's **mesh-free** zoo models (a
differentiable MJX plant) and render the resulting ``qpos(t)`` through the
**full menagerie** models (with visual meshes) vendored under
``examples/assets/<robot>_mjcf/``. MuJoCo does *no physics* here — each frame
writes a pose into ``data.qpos`` and calls ``mj_forward`` (kinematics only),
exactly the post-hoc-render recipe from the project's ``mujoco-author`` guide.

The render model and the dynamics model are different MJCFs, so this maps one
``qpos`` layout onto the other by **joint name** (free/floating joints by type),
which is robust even if the two models order or name their joints differently.

Public surface:

- :class:`TrajectoryRenderer` — load a showcase scene, render a ``qpos`` trajectory.
- :func:`write_gif` / :func:`write_mp4` — compact, notebook-friendly media.
- :func:`add_arrow` — a visual-only force/thrust arrow overlay.
"""

from __future__ import annotations

import pathlib

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image as PILImage

ASSETS = pathlib.Path(__file__).resolve().parent / "assets"

_FREE = int(mujoco.mjtJoint.mjJNT_FREE)
_BALL = int(mujoco.mjtJoint.mjJNT_BALL)
#: qpos width per MuJoCo joint type (free=0, ball=1, slide=2, hinge=3).
_QW = {
    _FREE: 7,
    _BALL: 4,
    int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
    int(mujoco.mjtJoint.mjJNT_HINGE): 1,
}


def build_qpos_remap(src: mujoco.MjModel, dst: mujoco.MjModel) -> np.ndarray:
    """Index array ``idx`` such that ``q_dst = q_src[idx]`` (by joint name/type).

    Free/floating joints are matched positionally (their 7 qpos copied as a
    block); every other joint is matched by name. Raises if a ``dst`` joint has
    no counterpart in ``src``.
    """
    idx = np.full(dst.nq, -1, dtype=int)
    src_free = [j for j in range(src.njnt) if int(src.jnt_type[j]) == _FREE]
    src_named = {
        src.joint(j).name: (int(src.jnt_qposadr[j]), _QW[int(src.jnt_type[j])])
        for j in range(src.njnt)
        if int(src.jnt_type[j]) != _FREE
    }
    for j in range(dst.njnt):
        jt = int(dst.jnt_type[j])
        da = int(dst.jnt_qposadr[j])
        if jt == _FREE:
            if not src_free:
                raise ValueError("dst has a free joint but src has none.")
            sa = int(src.jnt_qposadr[src_free[0]])
            idx[da : da + 7] = np.arange(sa, sa + 7)
        else:
            name = dst.joint(j).name
            if name not in src_named:
                raise ValueError(f"dst joint {name!r} has no src counterpart.")
            sa, w = src_named[name]
            idx[da : da + w] = np.arange(sa, sa + w)
    if (idx < 0).any():
        raise ValueError("incomplete qpos remap (unmapped dst qpos entries).")
    return idx


def add_arrow(scene, frm, to, *, rgba=(0.95, 0.25, 0.15, 1.0), width=0.01):
    """Append a visual-only arrow from ``frm`` to ``to`` to a render scene."""
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3),
        np.zeros(3),
        np.zeros(9),
        np.asarray(rgba, np.float32),
    )
    mujoco.mjv_connector(
        g,
        mujoco.mjtGeom.mjGEOM_ARROW,
        width,
        np.asarray(frm, float),
        np.asarray(to, float),
    )
    scene.ngeom += 1


class TrajectoryRenderer:
    """Render a jaxterity ``qpos`` trajectory through a full-mesh showcase scene.

    Args
    ----
    scene : str
        Name under ``examples/assets/`` (e.g. ``"unitree_g1_mjcf"``) or a path
        to a scene ``.xml``. The scene's model supplies the meshes.
    src_mjcf : str
        The dynamics model's MJCF (``robot.to_mjcf()``) whose ``qpos`` layout the
        trajectory uses. Used to build the name/type remap onto the scene model.
    size : (int, int)
        ``(width, height)`` in pixels. Default ``(960, 720)``.
    """

    def __init__(self, scene: str, src_mjcf: str, *, size=(960, 720)) -> None:
        path = scene if scene.endswith(".xml") else str(ASSETS / scene / "scene.xml")
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        self._src = mujoco.MjModel.from_xml_string(src_mjcf)
        self._idx = build_qpos_remap(self._src, self.model)
        self._w, self._h = size
        self.renderer = mujoco.Renderer(self.model, height=self._h, width=self._w)
        # For floor grounding: the free joint's base-z qpos index (if any) and the
        # non-plane geoms whose extent defines the robot's lowest point.
        free = [
            j for j in range(self.model.njnt) if int(self.model.jnt_type[j]) == _FREE
        ]
        self._base_zidx = int(self.model.jnt_qposadr[free[0]]) + 2 if free else None
        plane = int(mujoco.mjtGeom.mjGEOM_PLANE)
        self._solid_geoms = [
            g for g in range(self.model.ngeom) if int(self.model.geom_type[g]) != plane
        ]

    def _lowest_z(self) -> float:
        """World-frame z of the robot's lowest point (tight rotated-AABB bound)."""
        m, d = self.model, self.data
        lo = float("inf")
        for g in self._solid_geoms:
            rot = d.geom_xmat[g].reshape(3, 3)
            ctr, half = m.geom_aabb[g, 0:3], m.geom_aabb[g, 3:6]
            z = d.geom_xpos[g][2] + rot[2] @ ctr - np.abs(rot[2]) @ half
            lo = min(lo, z)
        return lo

    def render(
        self,
        qpos_traj,
        *,
        camera,
        track_body=None,
        stride=1,
        overlay=None,
        ground_floor=False,
    ):
        """Render frames for ``qpos_traj`` (``(T, nq_src)``, src layout).

        Args
        ----
        camera : dict
            ``{distance, azimuth, elevation, lookat}`` (free camera).
        track_body : str, optional
            If given, the camera ``lookat`` follows this body's world position
            each frame (so a flying/falling robot stays framed).
        stride : int
            Render every ``stride``-th pose.
        overlay : callable(scene, data, q) -> None, optional
            Add visual-only geoms (arrows, markers) per frame.
        ground_floor : bool
            For a floating-base robot, shift the base height each frame so the
            robot's lowest point rests on the floor (``z = 0``). A *kinematic*
            trajectory (set joint angles, no contact) otherwise floats or
            penetrates the floor — e.g. a squat. No-op for a fixed-base robot.
        """
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = camera["distance"]
        cam.azimuth = camera["azimuth"]
        cam.elevation = camera["elevation"]
        base_lookat = np.asarray(camera["lookat"], float)
        bid = self.model.body(track_body).id if track_body else None
        do_ground = ground_floor and self._base_zidx is not None

        frames = []
        for q in np.asarray(qpos_traj)[::stride]:
            self.data.qpos[:] = q[self._idx]
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)
            if do_ground:
                self.data.qpos[self._base_zidx] -= self._lowest_z()
                mujoco.mj_forward(self.model, self.data)
            cam.lookat[:] = self.data.xpos[bid] if bid is not None else base_lookat
            self.renderer.update_scene(self.data, camera=cam)
            if overlay is not None:
                overlay(self.renderer.scene, self.data, q)
            frames.append(self.renderer.render())
        return frames


def write_gif(frames, path, *, width=480, fps=20, stride=1):
    """Write a compact looping GIF (downscaled) — renders in every notebook viewer."""
    small = []
    for f in frames[::stride]:
        h = int(round(width * f.shape[0] / f.shape[1]))
        small.append(np.asarray(PILImage.fromarray(f).resize((width, h))))
    imageio.mimsave(path, small, fps=fps, loop=0)
    return path


def write_mp4(frames, path, *, fps=30):
    """Write an MP4 (full resolution)."""
    imageio.mimsave(path, frames, fps=fps, quality=8, macro_block_size=None)
    return path
