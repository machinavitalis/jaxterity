# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""The tutorial render bridge (examples/_robot_render.py) + tutorial integrity.

The bridge post-hoc-renders a jaxterity ``qpos`` trajectory through the vendored
full-mesh menagerie scenes. These tests guard the shared infrastructure the four
tutorial notebooks depend on (the qpos remap and an actual render), and assert
the committed notebooks are present and were executed with outputs.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

pytest.importorskip("mujoco")

# These tests drive the MuJoCo GL renderer, which aborts (core dump) without an
# OpenGL/display context — so they run only on a workstation with a display, not
# on a headless CI runner. Excluded from the PR gate; see pyproject markers.
pytestmark = pytest.mark.render

_REPO = pathlib.Path(__file__).resolve().parents[2]
_EXAMPLES = _REPO / "examples"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture(scope="module")
def R():
    return _load("_robot_render", _EXAMPLES / "_robot_render.py")


def test_qpos_remap_floating_base_is_consistent(R):
    """The G1 floating-base qpos maps onto the mesh scene (nq=36, free + 29)."""
    import mujoco

    from jaxterity import zoo

    src = mujoco.MjModel.from_xml_string(
        zoo.unitree_g1.load(floating_base=True).to_mjcf()
    )
    dst = mujoco.MjModel.from_xml_path(str(R.ASSETS / "unitree_g1_mjcf" / "scene.xml"))
    idx = R.build_qpos_remap(src, dst)
    assert idx.shape == (dst.nq,)
    assert idx.min() >= 0 and idx.max() < src.nq
    # The free-joint block (first 7 qpos) maps as a contiguous run.
    assert list(idx[:7]) == list(range(7))


def test_render_produces_frames(R):
    """Rendering a short SO-100 trajectory yields RGB frames of the right shape."""
    from jaxterity import zoo

    robot = zoo.load("so100")
    qs = np.linspace(0.0, 0.4, 5)[:, None] * np.ones((5, 6))
    rend = R.TrajectoryRenderer("so101_mjcf", robot.to_mjcf(), size=(320, 240))
    frames = rend.render(
        qs, camera=dict(distance=0.7, azimuth=130, elevation=-20, lookat=(0, 0, 0.15))
    )
    assert len(frames) == 5
    assert frames[0].shape == (240, 320, 3)
    assert frames[0].dtype == np.uint8
    assert int(frames[0].sum()) > 0  # not an all-black frame


def test_ground_floor_keeps_feet_on_floor(R):
    """A floating-base kinematic squat penetrates the floor; grounding fixes it."""
    import mujoco

    from jaxterity import zoo

    g1 = zoo.unitree_g1.load(floating_base=True)
    mj = mujoco.MjModel.from_xml_string(g1.to_mjcf())
    qad = {mj.joint(i).name: int(mj.jnt_qposadr[i]) for i in range(mj.njnt)}
    # Lower the pelvis + bend the knees: feet drop through the floor.
    q = np.zeros(mj.nq)
    q[2] = 0.55
    q[3] = 1.0  # unit quaternion
    for j, v in [("left_knee_joint", 1.0), ("right_knee_joint", 1.0)]:
        q[qad[j]] = v
    qs = q[None, :]
    rend = R.TrajectoryRenderer("unitree_g1_mjcf", g1.to_mjcf(), size=(64, 64))
    cam = dict(distance=2.0, azimuth=140, elevation=-8, lookat=(0, 0, 0.5))

    rend.render(qs, camera=cam)  # ungrounded: lowest point is below the floor
    assert rend._lowest_z() < -0.02
    rend.render(qs, camera=cam, ground_floor=True)  # grounded: rests on the floor
    assert abs(rend._lowest_z()) < 1e-6


@pytest.mark.parametrize(
    "slug",
    ["unitree_g1", "crazyflie", "nanolerobot", "rerun_streaming"],
)
def test_tutorial_notebook_is_executed(slug):
    """Each tutorial ships present, valid, executed (code cells have outputs)."""
    nbformat = pytest.importorskip("nbformat")
    nb_path = _EXAMPLES / f"tutorial_{slug}.ipynb"
    assert nb_path.exists(), nb_path
    nb = nbformat.read(nb_path, as_version=4)
    code = [c for c in nb.cells if c.cell_type == "code" and c.source.strip()]
    assert code, f"{slug}: no code cells"
    assert any(c.get("outputs") for c in code), f"{slug}: notebook was not executed"
    gif = _EXAMPLES / "media" / f"tutorial_{slug}.gif"
    assert gif.exists() and gif.stat().st_size > 0, gif
