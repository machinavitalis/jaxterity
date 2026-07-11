# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""S1-S4 layered policy API."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaxterity import policies, zoo
from jaxterity.policies import S3, S4
from jaxterity.wbc import PostureTask

pytest.importorskip("mujoco")


def _cartpole_s4(**kw):
    return S4(zoo.load("cartpole"), **kw)


# --- S4 -------------------------------------------------------------


def test_s4_reads_zero_initial_state():
    s4 = _cartpole_s4()
    q, qd = s4.state()
    np.testing.assert_array_equal(np.asarray(q), np.zeros(q.shape))
    np.testing.assert_array_equal(np.asarray(qd), np.zeros(qd.shape))


def test_s4_apply_torque_caches_command():
    s4 = _cartpole_s4()
    name = s4.joint_names[0]
    s4.apply_torque(name, 1.5)
    assert float(s4._tau[s4._dof_index[name]]) == 1.5


def test_s4_read_encoder_returns_position_by_name():
    s4 = _cartpole_s4()
    name = s4.joint_names[0]
    assert s4.read_encoder(name) == 0.0
    s4.set_state(jnp.array([0.3, 0.0]), jnp.zeros(2))
    assert s4.read_encoder(name) == pytest.approx(0.3, abs=1e-9)


def test_s4_reset_zeros_everything():
    s4 = _cartpole_s4()
    s4.set_state(jnp.array([1.0, -0.5]), jnp.array([2.0, 3.0]))
    s4.reset()
    q, qd = s4.state()
    np.testing.assert_array_equal(np.asarray(q), np.zeros(2))
    np.testing.assert_array_equal(np.asarray(qd), np.zeros(2))


# --- S3 -------------------------------------------------------------


def test_s3_step_wbc_with_posture_task_returns_qdd():
    s4 = _cartpole_s4()
    s3 = S3(
        s4,
        tasks={
            "p": PostureTask(name="p", q_desired=jnp.array([0.1, 0.0]), kp=10.0),
        },
    )
    qdd = s3.step_wbc()
    assert qdd.shape == (2,)
    assert float(qdd[0]) > 0  # pushes slider toward +0.1


def test_s3_set_task_priority_updates_in_place():
    s4 = _cartpole_s4()
    s3 = S3(
        s4,
        tasks={"p": PostureTask(name="p", q_desired=jnp.zeros(2), weight=1.0)},
    )
    s3.set_task_priority("p", 100.0)
    assert s3.tasks["p"].weight == 100.0


def test_s3_set_task_priority_unknown_raises():
    s4 = _cartpole_s4()
    s3 = S3(s4)
    with pytest.raises(KeyError, match="unknown task"):
        s3.set_task_priority("nope", 5.0)


def test_s3_step_mpc_returns_horizon_state():
    s4 = _cartpole_s4()
    s3 = S3(
        s4,
        tasks={"p": PostureTask(name="p", q_desired=jnp.array([0.05, 0.0]), kp=5.0)},
    )
    out = s3.step_mpc(horizon=5, dt=0.01)
    assert out.shape == (4,)  # [q (2), qd (2)]


# --- S2 -------------------------------------------------------------


def test_s2_move_joint_converges_under_posture_dynamics():
    """The slider tracks the setpoint under PD-on-acceleration WBC.

    Tolerances are loose by intent: this asserts the layered S2 → S3 → WBC
    stack runs and *makes progress* toward the goal. Precision tuning
    (higher gains, longer horizon, MPC) is the user's choice and is
    deliberately not pinned by the current tests.
    """
    s4 = _cartpole_s4()
    s3 = S3(s4)
    qs, _ = policies.s2.move_joint(s3, "slider_to_cart", 0.05, max_steps=400, dt=0.005)
    # Started at 0, target 0.05 → end > 0.025 means we've covered > 50% of the gap.
    assert qs[-1][0] > 0.025


def test_s2_track_trajectory_follows_setpoint_sequence():
    s4 = _cartpole_s4()
    s3 = S3(s4)
    n = 50
    times = np.arange(n) * 0.01
    q_des = np.column_stack([np.linspace(0.0, 0.2, n), np.zeros(n)])
    qs, _ = policies.s2.track_trajectory(s3, times, q_des, dt=0.01)
    # Started at 0, ramp ends at 0.2; with a finite-bandwidth PD the tracker
    # lags but should clear half the gap.
    assert qs[-1][0] > 0.1


# --- S1 -------------------------------------------------------------


def test_s1_reach_returns_s1result_with_trajectory():
    s4 = _cartpole_s4()
    s3 = S3(s4)
    result = policies.s1.reach(s3, np.array([0.1, 0.0]), max_steps=300)
    assert isinstance(result, policies.s1.S1Result)
    assert result.trajectory is not None
    assert result.trajectory.shape[1] == 2


def test_s1_pick_returns_not_implemented_summary():
    s4 = _cartpole_s4()
    s3 = S3(s4)
    result = policies.s1.pick(s3, "ball", np.zeros(3))
    assert not result.success
    assert "not implemented" in result.summary


def test_s1_walk_to_returns_not_implemented_summary():
    s4 = _cartpole_s4()
    s3 = S3(s4)
    result = policies.s1.walk_to(s3, np.array([1.0, 2.0]))
    assert not result.success
    assert "not implemented" in result.summary


# --- S4.step actually drives the MJX dynamics ----------------------


def test_s4_step_moves_robot_under_torque():
    """apply_torque + step integrate the Robot's MJX dynamics.

    The cartpole's cart_mass=1 kg + slider_mass=0; a constant 1 N·m on the
    slider for 50 ms gives x ≈ ½·a·t² = 1.25 mm. We check that the slider
    is positively displaced and time advanced.
    """
    s4 = _cartpole_s4()
    assert s4.read_encoder("slider_to_cart") == 0.0
    s4.apply_torque("slider_to_cart", 1.0)
    s4.step(0.05)
    new_pos = s4.read_encoder("slider_to_cart")
    assert new_pos > 0, f"slider didn't move under +tau: got {new_pos}"
    assert s4.time == pytest.approx(0.05, abs=1e-12)


def test_s4_step_negative_torque_moves_negative():
    """Sign convention — negative torque pushes slider negative."""
    s4 = _cartpole_s4()
    s4.apply_torque("slider_to_cart", -1.0)
    s4.step(0.05)
    assert s4.read_encoder("slider_to_cart") < 0


def test_s4_reset_resets_clock_too():
    """reset() should zero the clock as well as q/qd/tau."""
    s4 = _cartpole_s4()
    s4.apply_torque("slider_to_cart", 1.0)
    s4.step(0.05)
    assert s4.time == pytest.approx(0.05)  # the clock advanced by exactly dt
    s4.reset()
    assert s4.time == 0.0


def test_s4_dof_index_unknown_raises_typed_error():
    """Public dof_index() raises KeyError naming the available joints."""
    s4 = _cartpole_s4()
    with pytest.raises(KeyError, match="Known joints"):
        s4.dof_index("nonexistent_joint")


# --- S3.update_task_field ----------------------------------------


def test_s3_update_task_field_replaces_and_invalidates_cache():
    """Public update_task_field swaps a frozen-dataclass field in place
    and invalidates the cached WBCSolver so the next step_wbc rebuilds."""
    import jax.numpy as jnp

    s4 = _cartpole_s4()
    s3 = S3(
        s4,
        tasks={
            "p": PostureTask(name="p", q_desired=jnp.zeros(2), kp=5.0, weight=1.0),
        },
    )
    qdd0 = s3.step_wbc()
    # Now change the target — the WBC qdd should track the new setpoint.
    s3.update_task_field("p", q_desired=jnp.array([0.1, 0.0]))
    qdd1 = s3.step_wbc()
    assert float(qdd1[0]) > float(qdd0[0]) + 0.01, (qdd0, qdd1)


def test_s3_update_task_field_unknown_raises():
    s4 = _cartpole_s4()
    s3 = S3(s4)
    with pytest.raises(KeyError, match="unknown task"):
        s3.update_task_field("nope", weight=2.0)


# --- step_mpc emits DeprecationWarning ----------------------------


def test_step_mpc_alias_warns_but_still_works():
    """step_mpc is a deprecated alias for step_horizon_shooter."""
    import warnings

    import jax.numpy as jnp

    s4 = _cartpole_s4()
    s3 = S3(
        s4,
        tasks={"p": PostureTask(name="p", q_desired=jnp.zeros(2), kp=5.0)},
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        out = s3.step_mpc(horizon=3, dt=0.01)
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert out.shape == (4,)


# --- WBCSolver caching ---------------------------------------------


def test_s3_solver_cached_until_task_set_changes():
    """The solver instance is reused until add/remove/update invalidates."""
    import jax.numpy as jnp

    s4 = _cartpole_s4()
    s3 = S3(
        s4,
        tasks={"p": PostureTask(name="p", q_desired=jnp.zeros(2), kp=5.0)},
    )
    s3.step_wbc()  # populates the cache
    cached = s3._solver
    assert cached is not None
    s3.step_wbc()  # should reuse
    assert s3._solver is cached
    s3.update_task_field("p", weight=2.0)  # invalidates
    assert s3._solver is None
    s3.step_wbc()  # rebuilds
    assert s3._solver is not cached
