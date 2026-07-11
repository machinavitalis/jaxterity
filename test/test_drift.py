# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Scheduled plant-side drift.

Validation, interpolation semantics, jit-safety of ``apply_to_mjx``, and the
end-to-end physics oracle: a knee frictionloss ramped to a seizure visibly
locks the joint while the walking controller keeps the (drift-free) gait.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxterity.drift import (
    DriftChannel,
    DriftSchedule,
    actuator_index,
    joint_dof_index,
)
from jaxterity.errors import JaxterityError

pytest.importorskip("mujoco")

from jaxterity.locomotion import G1JoystickController  # noqa: E402


def test_channel_validation():
    with pytest.raises(JaxterityError, match="field"):
        DriftChannel(field="body_mass", index=0, times=(0.0,), values=(1.0,))
    with pytest.raises(JaxterityError, match="strictly increasing"):
        DriftChannel(field="dof_damping", index=0, times=(1.0, 1.0), values=(0.0, 1.0))
    with pytest.raises(JaxterityError, match="matching"):
        DriftChannel(field="dof_damping", index=0, times=(0.0, 1.0), values=(1.0,))
    with pytest.raises(JaxterityError, match="scale"):
        DriftChannel(
            field="actuator_gain", index=0, times=(0.0,), values=(1.0,), mode="set"
        )
    with pytest.raises(JaxterityError, match="same target"):
        DriftSchedule(
            (
                DriftChannel(field="dof_damping", index=3, times=(0.0,), values=(1.0,)),
                DriftChannel(field="dof_damping", index=3, times=(0.0,), values=(2.0,)),
            )
        )


def test_piecewise_linear_with_clamped_ends():
    channel = DriftChannel(
        field="dof_frictionloss",
        index=0,
        times=(10.0, 20.0),
        values=(0.1, 0.5),
    )
    assert float(channel.value(0.0)) == pytest.approx(0.1)  # before: hold first
    assert float(channel.value(15.0)) == pytest.approx(0.3)  # midpoint
    assert float(channel.value(99.0)) == pytest.approx(0.5)  # after: hold last


def test_apply_to_mjx_under_jit():
    """``apply_to_mjx`` is pure and traceable: t can be a traced scalar."""
    controller = G1JoystickController()
    mj_model = controller.mj_model
    knee = joint_dof_index(mj_model, "right_knee_joint")
    knee_act = actuator_index(mj_model, "right_knee_joint")
    schedule = DriftSchedule(
        (
            DriftChannel(
                field="dof_frictionloss",
                index=knee,
                times=(0.0, 100.0),
                values=(0.1, 1.1),
            ),
            DriftChannel(
                field="dof_damping",
                index=knee,
                times=(0.0, 100.0),
                values=(1.0, 1.5),
                mode="scale",
            ),
            DriftChannel(
                field="actuator_gain",
                index=knee_act,
                times=(0.0, 100.0),
                values=(1.0, 0.8),
                mode="scale",
            ),
        )
    )
    nominal = controller.mjx_model

    drifted = jax.jit(schedule.apply_to_mjx)(nominal, jnp.asarray(50.0))
    assert float(drifted.dof_frictionloss[knee]) == pytest.approx(0.6)
    assert float(drifted.dof_damping[knee]) == pytest.approx(
        float(nominal.dof_damping[knee]) * 1.25
    )
    assert float(drifted.actuator_gainprm[knee_act, 0]) == pytest.approx(
        float(nominal.actuator_gainprm[knee_act, 0]) * 0.9
    )
    # kv bias row scales in lockstep with the gain (torque-constant droop).
    assert float(drifted.actuator_biasprm[knee_act, 1]) == pytest.approx(
        float(nominal.actuator_biasprm[knee_act, 1]) * 0.9
    )
    # Untouched entries stay bit-equal; the nominal model is unchanged.
    other = (knee + 1) % nominal.dof_frictionloss.shape[0]
    assert float(drifted.dof_frictionloss[other]) == pytest.approx(
        float(nominal.dof_frictionloss[other])
    )


def test_seized_knee_locks_the_joint_mid_gait():
    """The B3-style fault renders legibly: a seized knee stops articulating.

    Ramp the right-knee frictionloss to 400 N·m (≫ the 139 N·m actuator
    class) by the window epoch and roll the *same* walking controller on the
    drifted plant. The seized knee's range of motion must collapse relative
    to the healthy gait while the healthy plant's knee keeps swinging.
    """
    controller = G1JoystickController()
    mj_model = controller.mj_model
    knee_dof = joint_dof_index(mj_model, "right_knee_joint")
    # Free base: qpos index = dof index + 1 (7 qpos vs 6 dof for the base).
    knee_qpos = knee_dof + 1
    schedule = DriftSchedule(
        (
            DriftChannel(
                field="dof_frictionloss",
                index=knee_dof,
                times=(10.0, 18.0),
                values=(0.1, 400.0),
            ),
        )
    )
    n_ticks = 50  # 1 simulated second of gait

    healthy = controller.rollout(command=[0.4, 0.0, 0.0], n_steps=n_ticks)
    seized_model = schedule.apply_to_mjx(controller.mjx_model, t=20.0)
    seized = controller.rollout(
        command=[0.4, 0.0, 0.0], n_steps=n_ticks, model=seized_model
    )

    healthy_range = float(np.ptp(np.asarray(healthy.qpos)[:, knee_qpos]))
    seized_range = float(np.ptp(np.asarray(seized.qpos)[:, knee_qpos]))
    assert healthy_range > 0.2, healthy_range  # the gait swings the knee
    assert seized_range < 0.5 * healthy_range, (healthy_range, seized_range)
    # Both rollouts stay finite (the fault is legible, not a NaN blow-up).
    assert np.all(np.isfinite(np.asarray(seized.qpos)))
