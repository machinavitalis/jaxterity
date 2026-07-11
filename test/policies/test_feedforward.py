# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Feedforward precision tracking — the precision oracle.

Same plant, same trajectory, two self-models: the accurate model's
feedforward tracks inside the tolerance band; the stale (factory-default)
model misses it by a clear margin. This is the scoring task the runtime
drift/calibration demos hinge on.
"""

from __future__ import annotations

import numpy as np
import pytest

from jaxterity.errors import JaxterityError

pytest.importorskip("mujoco")

from jaxterity.policies import (  # noqa: E402
    design_tracking_trajectory,
    feedforward_tracking,
)
from jaxterity.twin import unitree_g1_twin  # noqa: E402

#: The right shoulder performs; the rest of the body is scaffolding. (The
#: task joints must be dynamics-dominated, not friction-dominated — see the
#: module docstring's selection rule; an elbow/wrist ratchets at velocity
#: reversals for any model and stops discriminating.)
_TASK_JOINTS = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
)
#: The plant's true (drifted-from-factory) parameters.
_TRUE_FRICTION = {"right_shoulder_pitch_joint": 0.65}
_TRUE_KT = {
    "right_shoulder_pitch_joint": 0.85,
    "right_shoulder_roll_joint": 0.85,
}
_TASK_OMEGA = 4.0


@pytest.fixture(scope="module")
def setup():
    nominal = unitree_g1_twin("G1-PRECISION")
    names = nominal.servo_names
    plant = nominal.robot.with_parameters(
        {f"{j}.friction": v for j, v in _TRUE_FRICTION.items()}
    )
    kt_true = np.ones(nominal.n_servos)
    for joint, value in _TRUE_KT.items():
        kt_true[names.index(joint)] = value
    trajectory = design_tracking_trajectory(
        nominal.robot,
        _TASK_JOINTS,
        duration_s=2.0,
        frequency_hz=0.5,
        amplitude_fraction=0.35,
    )
    return nominal, plant, kt_true, trajectory


def test_accurate_model_passes_stale_model_fails(setup):
    nominal, plant, kt_true, trajectory = setup

    # The accurate self-model: the plant's true parameters.
    accurate = type(nominal)(
        plant,
        serial=nominal.serial,
        C_th=nominal.C_th,
        R_th=nominal.R_th,
        k_pwr=nominal.k_pwr,
        K_t_scale=kt_true,
        T_amb=nominal.T_amb,
        servo_names=nominal.servo_names,
    )
    good = feedforward_tracking(
        accurate, plant, trajectory, plant_K_t_scale=kt_true, task_omega=_TASK_OMEGA
    )
    # The stale self-model: factory defaults, never updated.
    stale = feedforward_tracking(
        nominal, plant, trajectory, plant_K_t_scale=kt_true, task_omega=_TASK_OMEGA
    )

    # The catch: accurate passes the band, stale misses it by a clear margin.
    assert good.passed, (good.error_max, good.tolerance)
    assert not stale.passed, (stale.error_max, stale.tolerance)
    assert stale.error_max > 2.0 * good.error_max, (
        good.error_max,
        stale.error_max,
    )
    # Both rollouts were sane (the held joints did their scaffolding job).
    assert np.all(np.isfinite(good.q)) and np.all(np.isfinite(stale.q))


def test_trajectory_design_validation(setup):
    nominal, *_ = setup
    with pytest.raises(JaxterityError, match="unknown task joints"):
        design_tracking_trajectory(nominal.robot, ("bogus_joint",))


def test_trajectory_starts_at_rest(setup):
    *_, trajectory = setup
    q0, qd0, _ = (np.asarray(v) for v in trajectory.desired(0.0))
    np.testing.assert_allclose(q0, np.asarray(trajectory.q_rest), atol=1e-12)
    np.testing.assert_allclose(qd0, 0.0, atol=1e-12)
