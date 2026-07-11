# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Render the three Jaxterity tutorial GIFs into ``examples/assets/``.

The SO-101 GIFs render the **upstream TheRobotStudio/SO-ARM100 MuJoCo model**
(Apache-2.0) vendored under ``examples/assets/so101_mjcf/`` -- meshes, joints,
inertials, all of it -- so the tutorial shows the real arm rather than capsule
proxies. The kinematic chain and joint order match the Jaxterity-imported
``zoo.load("so100")`` model exactly (same source URDF), so swapping in different
damping at render time mirrors what Jaxterity would simulate.

The Dzhanibekov GIF stays Jaxterity-native (it is a synthetic free body, not a
real robot); cosmetic proxy geometry is still injected by :mod:`examples._viz`.

GIFs produced:

* ``so101_arm.gif``        -- the real SO-101 arm swinging and settling.
* ``dzhanibekov.gif``      -- a torque-free asymmetric free body doing the
  tennis-racket / Dzhanibekov flip in zero gravity.
* ``sysid_before_after.gif`` -- the same arm, uncalibrated (damping 0.0,
  overshoots) vs calibrated (damping 0.12, settles), side by side.

Run:  ``python examples/render_tutorial_gifs.py``  (needs the ``[mujoco]`` extra).
"""

from __future__ import annotations

import pathlib
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import jax.numpy as jnp
import mujoco
import numpy as np
from _viz import add_scene
from PIL import Image, ImageDraw

from jaxterity.robot import (
    Frame,
    FramedQuantity,
    Joint,
    JointLimits,
    JointType,
    Link,
    Position,
    Robot,
    Transform,
)

ASSETS = pathlib.Path(__file__).with_name("assets")
SO101_SCENE = ASSETS / "so101_mjcf" / "scene_showcase.xml"
TIMESTEP = 0.002
FPS = 25

# Shared SO-101 view: front-right, slightly elevated, framing the arm.
_ARM_CAM = (0.40, -0.40, 0.40, "1 1 0 -0.4 0.4 1")
_ARM_START = [0.8, -1.0, 1.2, -0.6, 0.5, 0.6]  # a folded, off-balance pose

# Sysid coefficients, matching examples/sysid_so101.py.
_FITTED = ("shoulder_lift", "elbow_flex")  # joint names the sysid fits damping for
_CALIBRATED = 0.12


def _so101_model(joint_damping: dict[str, float]):
    """Load the vendored SO-101 showcase scene and override per-joint damping.

    ``joint_damping`` maps joint name -> damping in N*m*s/rad. Joints not in the
    map keep their upstream default (``sts3215`` class, ~0.6).
    """
    model = mujoco.MjModel.from_xml_path(str(SO101_SCENE))
    for jname, damp in joint_damping.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            raise ValueError(f"joint {jname!r} not found in vendored SO-101 MJCF")
        dof = model.jnt_dofadr[jid]
        model.dof_damping[dof] = damp
    model.opt.timestep = TIMESTEP
    return model


def _rollout(model, data, duration, size, *, camera="showcase"):
    """Step ``model`` to ``duration`` and return RGB frames at ``FPS``."""
    model.opt.timestep = TIMESTEP
    frames = []
    with mujoco.Renderer(model, size, size) as renderer:
        while data.time < duration:
            if len(frames) < data.time * FPS:
                renderer.update_scene(data, camera=camera)
                frames.append(renderer.render())
            mujoco.mj_step(model, data)
        renderer.update_scene(data, camera=camera)
        frames.append(renderer.render())
    return frames


def _save_gif(frames, path, *, fps=FPS):
    """Write ``frames`` as a looping GIF and assert it is non-blank."""
    arr = np.asarray(frames)
    assert arr.var() > 0, f"all frames blank for {path.name}"
    imageio.mimsave(path, frames, format="GIF", duration=1.0 / fps, loop=0)
    size = path.stat().st_size
    print(f"  wrote {path}  ({size / 1024:.0f} KiB, {len(frames)} frames)")
    assert size > 0
    return size


# --------------------------------------------------------------------------- #
# 1. SO-101 arm: swing and settle under gravity.
# --------------------------------------------------------------------------- #
_ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def render_arm():
    print("[1/3] so101_arm.gif -- real SO-101 arm swinging and settling")
    # Modest damping on every revolute joint so the swing settles in a few s.
    model = _so101_model({j: 0.05 for j in _ARM_JOINTS})
    data = mujoco.MjData(model)
    data.qpos[:] = _ARM_START
    mujoco.mj_forward(model, data)
    frames = _rollout(model, data, duration=4.0, size=420)
    return _save_gif(frames, ASSETS / "so101_arm.gif")


# --------------------------------------------------------------------------- #
# 2. Dzhanibekov / tennis-racket flip: torque-free asymmetric free body.
# --------------------------------------------------------------------------- #
def _dzhanibekov_robot():
    """A single floating body with asymmetric inertia diag(0.1, 0.2, 0.3)."""
    free = Joint(
        name="root",
        joint_type=JointType.FLOATING,
        parent_link="world",
        child_link="plate",
        axis=FramedQuantity(Position(jnp.array([0.0, 0.0, 1.0])), Frame("world")),
        limits=JointLimits(lower=-1.0, upper=1.0, effort=1.0, velocity=1.0),
        origin=Transform.identity(Frame("world")),
        damping=jnp.asarray(0.0),
        friction=jnp.asarray(0.0),
    )
    plate = Link(
        name="plate",
        body_frame=Frame("plate"),
        mass=jnp.asarray(1.0),
        com=jnp.zeros(3),
        inertia=jnp.diag(jnp.asarray((0.1, 0.2, 0.3))),  # intermediate axis = y
    )
    return Robot(links=(plate,), joints=(free,), name="dzhanibekov", gravity=0.0)


def _dzhanibekov_geoms(mjcf):
    """Add a clearly non-cubic plate plus RGB axis sticks to the free body."""
    root = ET.fromstring(mjcf)
    plate = root.find("worldbody").find("body")
    ET.SubElement(
        plate,
        "geom",
        {
            "type": "box",
            "size": "0.30 0.16 0.03",
            "rgba": "0.30 0.55 0.85 1",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    # x=red, y=green (the unstable intermediate axis), z=blue.
    for fromto, col in [
        ("0 0 0 0.42 0 0", "0.90 0.20 0.20 1"),
        ("0 0 0 0 0.30 0", "0.20 0.80 0.30 1"),
        ("0 0 0 0 0 0.22", "0.25 0.45 0.95 1"),
    ]:
        ET.SubElement(
            plate,
            "geom",
            {
                "type": "capsule",
                "fromto": fromto,
                "size": "0.022",
                "rgba": col,
                "contype": "0",
                "conaffinity": "0",
            },
        )
    return ET.tostring(root, encoding="unicode")


def render_dzhanibekov():
    print("[2/3] dzhanibekov.gif -- torque-free tennis-racket flip (zero g)")
    mjcf = _dzhanibekov_geoms(_dzhanibekov_robot().to_mjcf())
    mjcf = add_scene(mjcf, floor=False, camera=(1.1, -1.1, 0.7, "1 1 0 -0.3 0.3 1"))
    model = mujoco.MjModel.from_xml_string(mjcf)
    data = mujoco.MjData(model)
    data.qvel[3:6] = [0.1, 3.0, 0.05]  # spin mostly about intermediate (y) axis
    mujoco.mj_forward(model, data)
    frames = _rollout(model, data, duration=6.0, size=420)
    return _save_gif(frames, ASSETS / "dzhanibekov.gif")


# --------------------------------------------------------------------------- #
# 3. Sysid payoff: uncalibrated vs calibrated damping, side by side.
# --------------------------------------------------------------------------- #
def _arm_model(damping):
    # All revolute joints get a small bleed damping so non-fitted DoFs settle;
    # the two FITTED joints take the sysid value (0.0 uncalibrated, 0.12 cal).
    damp = {j: 0.05 for j in _ARM_JOINTS}
    for j in _FITTED:
        damp[j] = damping
    model = _so101_model(damp)
    data = mujoco.MjData(model)
    data.qpos[:] = _ARM_START
    mujoco.mj_forward(model, data)
    return model, data


def _label(frame, text, color):
    """Draw a translucent caption band at the top of a frame."""
    img = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    w = img.width
    draw.rectangle([0, 0, w, 26], fill=(20, 22, 28, 200))
    draw.text((8, 7), text, fill=color)
    return np.asarray(img)


def render_sysid():
    print("[3/3] sysid_before_after.gif -- uncalibrated vs calibrated damping")
    mu, du = _arm_model(0.0)
    mc, dc = _arm_model(_CALIBRATED)
    size, duration = 400, 4.0
    frames, qu, qc = [], [], []
    with mujoco.Renderer(mu, size, size) as ru, mujoco.Renderer(mc, size, size) as rc:
        while du.time < duration:
            if len(frames) < du.time * FPS:
                ru.update_scene(du, camera="showcase")
                rc.update_scene(dc, camera="showcase")
                left = _label(
                    ru.render(), "UNCALIBRATED  damping = 0.0", (255, 120, 110)
                )
                right = _label(
                    rc.render(), "CALIBRATED  damping = 0.12", (120, 230, 150)
                )
                frames.append(np.hstack([left, right]))
            qu.append(float(du.qpos[1]))
            qc.append(float(dc.qpos[1]))
            mujoco.mj_step(mu, du)
            mujoco.mj_step(mc, dc)
    # Sanity: the uncalibrated arm explores a much wider angle range than the
    # calibrated one (it overshoots / oscillates instead of settling).
    span_u = float(np.ptp(qu))
    span_c = float(np.ptp(qc[len(qc) // 2 :]))  # late-time span: should be small
    print(
        f"  shoulder_lift angle span: uncalibrated={span_u:.2f} rad, "
        f"calibrated(late)={span_c:.2f} rad"
    )
    assert span_u > span_c, "expected uncalibrated arm to oscillate more"
    return _save_gif(frames, ASSETS / "sysid_before_after.gif")


def main():
    ASSETS.mkdir(exist_ok=True)
    sizes = {
        "so101_arm.gif": render_arm(),
        "dzhanibekov.gif": render_dzhanibekov(),
        "sysid_before_after.gif": render_sysid(),
    }
    print("\nDone. Assets:")
    for name, size in sizes.items():
        print(f"  {ASSETS / name}  ({size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
