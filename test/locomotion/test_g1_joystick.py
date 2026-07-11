# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""G1 joystick walking controller.

Three layers of evidence, cheapest first: the vendored scene keeps the
trained topology; the pure-JAX policy matches the upstream ONNX network
bit-closely (onnxruntime golden, recorded at vendoring time); and the
closed loop actually walks — forward displacement under a forward command,
upright the whole way — including on a parameter-perturbed plant (the
drift-study entry point).
"""

from __future__ import annotations

import numpy as np
import pytest

from jaxterity.errors import JaxterityError

pytest.importorskip("mujoco")

from jaxterity.locomotion import (  # noqa: E402  (after the mujoco skip-guard)
    G1JoystickController,
    G1JoystickPolicy,
    load_walk_model,
)
from jaxterity.locomotion.g1_joystick import GOLDEN_PATH  # noqa: E402

#: One shared controller: the rollout scan compiles once per n_steps and the
#: walking tests below deliberately reuse the same length to share it.
_N_TICKS = 75  # 1.5 simulated seconds


@pytest.fixture(scope="module")
def controller() -> G1JoystickController:
    return G1JoystickController()


def test_walk_scene_keeps_trained_topology():
    """The vendored scene is the model the policy was trained against."""
    m = load_walk_model()
    assert (m.nq, m.nv, m.nu) == (36, 35, 29)
    # Mesh-free contact set: the floor plane + two foot boxes, nothing else.
    names = {m.geom(i).name for i in range(m.ngeom)}
    assert names == {"floor", "left_foot", "right_foot"}
    # The control contract's anchors exist.
    m.keyframe("knees_bent"), m.keyframe("home")
    m.sensor("local_linvel_pelvis"), m.sensor("gyro_pelvis")
    m.site("imu_in_pelvis")


def test_policy_matches_onnx_golden():
    """The pure-JAX MLP reproduces the upstream ONNX policy.

    The golden holds onnxruntime outputs for seeded random observations,
    recorded by ``scripts/vendor_g1_walk_scene.py``. Float32 network ⇒
    1e-5 absolute agreement is the honest bar.
    """
    policy = G1JoystickPolicy.load()
    with np.load(GOLDEN_PATH) as golden:
        obs, expected = golden["obs"], golden["actions"]
    ours = np.stack([np.asarray(policy(o)) for o in obs])
    np.testing.assert_allclose(ours, expected, atol=1e-5)
    # Tanh head: actions live in [-1, 1].
    assert np.all(np.abs(ours) <= 1.0)


def test_wrong_model_raises():
    """A model without the trained actuation contract is rejected loudly."""
    import mujoco

    bare = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    with pytest.raises(JaxterityError, match="position actuators"):
        G1JoystickController(model=bare)


def test_g1_walks_forward(controller):
    """Closed loop: a 0.5 m/s forward command produces a forward walk.

    1.5 simulated seconds. The robot must stay upright (pelvis height in the
    walking band the policy was trained around) and cover ground; falling,
    diverging, or walking in place all fail.
    """
    traj = controller.rollout(command=[0.5, 0.0, 0.0], n_steps=_N_TICKS)
    qpos = np.asarray(traj.qpos)
    assert np.all(np.isfinite(qpos))
    pelvis_z = qpos[:, 2]
    assert pelvis_z.min() > 0.6 and pelvis_z.max() < 0.95, (
        pelvis_z.min(),
        pelvis_z.max(),
    )
    forward = float(qpos[-1, 0] - qpos[0, 0])
    assert forward > 0.35, forward  # ~0.65 m commanded; generous startup margin
    # The free-base quaternion stays unit-norm through 750 contact steps.
    assert abs(float(np.linalg.norm(qpos[-1, 3:7])) - 1.0) < 1e-4


def test_g1_stands_under_zero_command(controller):
    """A zero command steps in place: upright, with little net translation."""
    traj = controller.rollout(command=[0.0, 0.0, 0.0], n_steps=_N_TICKS)
    qpos = np.asarray(traj.qpos)
    assert np.all(np.isfinite(qpos))
    assert qpos[:, 2].min() > 0.6
    drift = float(np.linalg.norm(qpos[-1, :2] - qpos[0, :2]))
    assert drift < 0.3, drift


def test_walks_on_perturbed_plant(controller):
    """The same controller walks on a mildly drifted plant (demo entry point).

    +30 % joint friction loss on every DOF — the kind of plant-side
    perturbation the runtime-drift work injects. The trained policy is
    feedback; it must shrug this off, and the rollout must reuse the same
    compiled kernel (model is a traced argument).
    """
    model = controller.mjx_model
    drifted = model.replace(dof_frictionloss=model.dof_frictionloss * 1.3)
    traj = controller.rollout(command=[0.5, 0.0, 0.0], n_steps=_N_TICKS, model=drifted)
    qpos = np.asarray(traj.qpos)
    assert np.all(np.isfinite(qpos))
    assert qpos[:, 2].min() > 0.6
    assert float(qpos[-1, 0] - qpos[0, 0]) > 0.3
