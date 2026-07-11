# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Render the trained nanojaxterity cartpole agent balancing, with MuJoCo.

Mission: visualization for humans (post-hoc rendering). The dynamics live in
jaxterity — the trained policy + the functional MJX env produce a pose
trajectory; MuJoCo does *no physics* here, it just draws each pose.

The robot's own MJCF carries only ``<inertial>`` bodies (no visual geoms, lights,
or camera), so this authors a small *scene* around it per the house MuJoCo
recipe: visual geoms for the cart and pole, a checker floor, a directional light
+ headlight ambient, and a named camera. Loads the committed policy
(`nanojaxterity_policy.eqx`) and writes a video next to this file. Targets
mujoco 3.8.

Run::

    python -m examples.nanojaxterity_render
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import equinox as eqx
import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from PIL import Image as PILImage

_HERE = pathlib.Path(__file__).resolve().parent


def write_gif(frames, path, *, stride=5, width=320, fps=12):
    """Write a compact animated GIF (downsampled) from RGB frames.

    GIFs render as plain images in every notebook viewer (Jupyter, VS Code,
    GitHub, nbviewer) and in untrusted notebooks — unlike an HTML5 ``<video>``,
    which several renderers will not play. Kept small (stride + downscale) so the
    notebook stays a reasonable size.
    """
    small = []
    for f in frames[::stride]:
        h = int(round(width * f.shape[0] / f.shape[1]))
        small.append(np.asarray(PILImage.fromarray(f).resize((width, h))))
    imageio.mimsave(path, small, fps=fps, loop=0)
    return path


# Scene additions injected into the robot MJCF (which is geom-/light-/camera-less).
_VISUAL_ASSET = """  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.7 0.7 0.7" specular="0.1 0.1 0.1"/>
    <global offwidth="640" offheight="480"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".2 .3 .4"
             width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="10 10" reflectance=".2"/>
  </asset>
"""
_WORLD_EXTRAS = """    <light name="sun" directional="true" diffuse=".8 .8 .8"
           specular=".2 .2 .2" pos="1 -1 3" dir="-0.3 0.3 -1"/>
    <geom name="floor" type="plane" pos="0 0 -0.15" size="0 0 .05" material="grid"/>
    <camera name="main" pos="0 -2.2 0.5" xyaxes="1 0 0 0 0 1"/>
"""
# Visual geom for the otherwise-geomless cart body (a box). Injected after the
# body's open tag; the pole capsule is built per-call (its length can vary).
_CART_GEOM = (
    '\n      <geom name="cart_g" type="box" size="0.18 0.12 0.07" rgba=".25 .5 .85 1"/>'
)


def build_scene_mjcf(
    robot, pole_length: float = 1.0, pole_rgba: str = ".9 .5 .2 1"
) -> str:
    """Wrap the robot MJCF with visual geoms, lights, a floor, and a camera.

    ``pole_length`` sets the drawn pole capsule (m) so a length-variant robot is
    rendered with a matching rod; ``pole_rgba`` recolours it.
    """
    pole_geom = (
        f'\n        <geom name="pole_g" type="capsule" fromto="0 0 0 0 0 {pole_length}"'
        f' size="0.03" rgba="{pole_rgba}"/>'
    )
    mjcf = robot.to_mjcf()
    mjcf = mjcf.replace(
        '<option gravity="0 0 -9.81"/>',
        '<option gravity="0 0 -9.81"/>\n' + _VISUAL_ASSET,
    )
    mjcf = mjcf.replace("<worldbody>\n", "<worldbody>\n" + _WORLD_EXTRAS)
    cart_open = '<body name="cart" pos="0.0 0.0 0.0">'
    pole_open = '<body name="pole" pos="0.0 0.0 0.0">'
    mjcf = mjcf.replace(cart_open, cart_open + _CART_GEOM)
    mjcf = mjcf.replace(pole_open, pole_open + pole_geom)
    return mjcf


#: Force-arrow length per newton (m/N) and shaft width (m).
_ARROW_SCALE = 0.05
_ARROW_WIDTH = 0.03


def rollout(nj, robot, start, n_steps):
    """Greedy trained-policy rollout; return ``(states (n,4), forces (n,))``.

    ``forces`` is the cart force (N) the policy commanded at each step — what the
    render draws as an arrow so the control effort is visible in real time.
    """
    cfg = nj.Config()
    _, step = nj.make_env(robot, cfg)
    like = nj.ActorCritic(jax.random.PRNGKey(0))
    model = eqx.tree_deserialise_leaves(nj.POLICY_PATH, like)
    force_of = jnp.asarray(nj.ACTIONS) * cfg.force

    @eqx.filter_jit
    def run(model):
        def body(state, _):
            logits, _ = model(state)
            action = jnp.argmax(logits)
            nxt, _, _ = step(state, action)
            return nxt, (nxt, force_of[action])

        s0 = jnp.asarray(start, dtype=float)
        _, (traj, forces) = jax.lax.scan(body, s0, None, length=n_steps)
        return traj, forces

    traj, forces = run(model)
    return np.asarray(traj), np.asarray(forces)


def _add_force_arrow(scene, cart_x, force):
    """Append a red force arrow at the cart to the render scene (visual only)."""
    if abs(force) < 1e-6 or scene.ngeom >= scene.maxgeom:
        return
    length = _ARROW_SCALE * abs(force)
    z = 0.12
    frm = np.array([cart_x, 0.0, z])
    to = np.array([cart_x + np.sign(force) * length, 0.0, z])
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3),
        np.zeros(3),
        np.zeros(9),
        np.array([0.95, 0.25, 0.15, 1.0], np.float32),
    )
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, _ARROW_WIDTH, frm, to)
    scene.ngeom += 1


def render_frames(mjcf, traj, forces, *, size=(640, 480), cam_z=0.5, cam_dist=3.0):
    """Render each pose with the live force arrow, from a fixed camera (post-hoc).

    On the physically-corrected model the agent regulates the cart back to
    centre, so the camera is **fixed** (lookat at the origin) and you watch the
    cartpole settle to upright-and-centred — the pole snaps back at a realistic
    rate while the cart makes a small corrective move and returns. The red arrow
    on the cart is the commanded force each frame (the control effort).
    ``cam_z`` / ``cam_dist`` frame a taller pole when rendering a length variant.
    """
    mj = mujoco.MjModel.from_xml_string(mjcf)
    data = mujoco.MjData(mj)
    width, height = size
    renderer = mujoco.Renderer(mj, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = cam_dist
    cam.azimuth = 90.0
    cam.elevation = -8.0
    cam.lookat[:] = [0.0, 0.0, cam_z]  # fixed — the cart stays near centre
    frames = []
    for s, f in zip(traj, forces):
        data.qpos[0] = float(s[0])
        data.qpos[1] = float(s[1])  # pole θ
        data.qvel[:] = 0.0
        mujoco.mj_forward(mj, data)  # kinematics only — no physics step
        renderer.update_scene(data, camera=cam)
        _add_force_arrow(renderer.scene, float(s[0]), float(f))
        frames.append(renderer.render())
    return frames


def _load_demo():
    spec = importlib.util.spec_from_file_location(
        "nanojaxterity_demo", _HERE / "nanojaxterity.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


# A visible tilt (5.7°). On the corrected model the snappy pole is caught
# quickly; the cart makes a modest corrective move and *stops* (joint friction),
# holding the pole upright — vs the old frictionless glide.
_START = [0.0, 0.10, 0.0, 0.0]
_N_STEPS = 400  # 8 s at dt = 0.02


def _render_episode(nj, robot, start, out, **scene):
    """Roll out the committed policy on ``robot`` and write a video to ``out``."""
    pole_length = scene.pop("pole_length", 1.0)
    pole_rgba = scene.pop("pole_rgba", ".9 .5 .2 1")
    mjcf = build_scene_mjcf(robot, pole_length=pole_length, pole_rgba=pole_rgba)
    traj, forces = rollout(nj, robot, start, _N_STEPS)
    frames = render_frames(mjcf, traj, forces, **scene)
    imageio.mimsave(out, frames, fps=50, quality=8, macro_block_size=None)
    gif = write_gif(frames, out.with_suffix(".gif"))  # notebook-friendly
    peak = float(np.max(np.abs(traj[:, 1])))
    print(f"wrote {out.name} + {gif.name} ({len(frames)} frames); max|θ|={peak:.4f}")


def longer_pole(robot, scale=2.0):
    """Same cartpole with a ``scale``× longer pole (inertia ∝ L², CoM at L/2)."""
    return robot.with_parameters(
        {
            "pole.inertia": jnp.diag(jnp.asarray([0.1 * scale**2 / 12] * 3)),
            "pole.com": jnp.asarray([0.0, 0.0, scale / 2]),
        }
    )


def main(*, preview_only: bool = False) -> None:
    nj = _load_demo()
    robot = nj.calibrated_cartpole()
    if preview_only:
        mjcf = build_scene_mjcf(robot)
        traj, forces = rollout(nj, robot, _START, _N_STEPS)
        frame = render_frames(mjcf, traj[40:41], forces[40:41])[0]
        imageio.imwrite("/tmp/mujoco_preview.png", frame)
        print("wrote /tmp/mujoco_preview.png")
        return

    # Nominal cartpole.
    _render_episode(nj, robot, _START, _HERE / "nanojaxterity.mp4")
    # Robustness, no retraining: the SAME policy on a 2× longer pole (a different
    # physical robot — recoloured purple, taller capsule, taller camera).
    _render_episode(
        nj,
        longer_pole(robot, 2.0),
        [0.0, 0.08, 0.0, 0.0],
        _HERE / "nanojaxterity_long.mp4",
        pole_length=2.0,
        pole_rgba=".55 .35 .85 1",
        cam_z=0.9,
        cam_dist=4.6,
    )


if __name__ == "__main__":
    main()
