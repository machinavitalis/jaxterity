# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""The certified reflex: does the clamp actually hold the envelope?

A safety filter is only worth its complexity if the closed loop it produces
stays inside limits that the unfiltered loop leaves. So the load-bearing tests
here are closed-loop: run the same commanded torque through the twin with and
without the filter and compare what the *unit* did — temperature against its
ceiling, velocity against its limit. The rest cover the structure (rows from
the measured twin, not from kinematics alone), the agreement with the monitor
that vetoes on the same envelope, and the arbitration.

Everything runs on the two-DOF cartpole twin: the filter's cost is dominated by
differentiating the body dynamics, and the physics under test (a thermal
ceiling, a velocity limit) is DOF-count-agnostic.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxterity import zoo
from jaxterity.errors import JaxterityError
from jaxterity.selfcheck import (
    ReflexController,
    SafetyAction,
    SafetyEnvelope,
    SafetyMonitor,
    SimplexSwitch,
)
from jaxterity.twin import Twin

pytest.importorskip("mujoco")

#: Representative thermal leaves for the test twin (J/K, K/W, W/(N·m)²).
_C_TH, _R_TH, _K_PWR = 15.0, 4.0, 6.0
#: Ceiling the reflex must hold, K, and the ambient it starts from.
_CEILING_K = 320.0
_T_AMB = 298.15
#: The sustainable torque this envelope implies: T_amb + k_pwr·R_th·tau² = ceiling.
_SUSTAINABLE_TAU = float(np.sqrt((_CEILING_K - _T_AMB) / (_K_PWR * _R_TH)))


@pytest.fixture(scope="module")
def twin():
    robot = zoo.calibrated("cartpole")
    n = robot.build_system(actuation=True)._nv
    return Twin(
        robot,
        serial="CP-REFLEX",
        C_th=jnp.full(n, _C_TH),
        R_th=jnp.full(n, _R_TH),
        k_pwr=jnp.full(n, _K_PWR),
        T_amb=_T_AMB,
    )


@pytest.fixture(scope="module")
def envelope():
    return SafetyEnvelope(temp_ceiling_k=_CEILING_K, torque_margin_frac=0.9)


@pytest.fixture(scope="module")
def reflex(twin, envelope):
    return ReflexController(twin, envelope)


def _rest(twin):
    nq = twin.robot.build_system(actuation=True)._nq
    return (
        np.zeros(nq),
        np.zeros(twin.n_servos),
        np.full(twin.n_servos, _T_AMB),
    )


def _closed_loop(twin, tau_cmd, *, reflex=None, n_steps=400, dt=0.05):
    """Step the twin under ``tau_cmd``, optionally filtering every tick.

    Returns ``(peak_temperature_K, peak_abs_velocity)``.
    """
    step = twin.compiled_rollout(dt)
    q, qd, T = _rest(twin)
    q, qd, T = jnp.asarray(q), jnp.asarray(qd), jnp.asarray(T)
    peak_T, peak_qd = float(T.max()), 0.0
    for _ in range(n_steps):
        tau = np.asarray(tau_cmd, dtype=float)
        if reflex is not None:
            tau = reflex.filter(q, qd, T, tau).tau
        out = step(q, qd, T, jnp.asarray(tau)[None, :])
        q, qd, T = out.q[-1], out.qd[-1], out.temperature[-1]
        peak_T = max(peak_T, float(T.max()))
        peak_qd = max(peak_qd, float(jnp.max(jnp.abs(qd))))
    return peak_T, peak_qd


# --- structure: rows come from the measured twin ---------------------------


def test_rows_cover_joints_velocity_and_the_measured_thermal_state(reflex, twin):
    names = reflex.row_names
    assert any(n.startswith("joint_upper") for n in names)
    assert any(n.startswith("joint_lower") for n in names)
    assert any(n.startswith("joint_velocity") for n in names)
    # The differentiator: a thermal row per servo, from the twin's own fit.
    assert sum(n.startswith("servo_temperature") for n in names) == twin.n_servos
    assert reflex.n_rows == len(names)


def test_barrier_values_are_positive_when_safe_and_negative_when_out(reflex, twin):
    q, qd, T = _rest(twin)
    safe = reflex.barrier_values(q, qd, T)
    assert all(v > 0 for v in safe.values())
    over = reflex.barrier_values(q, qd, np.full(twin.n_servos, _CEILING_K + 5.0))
    assert over["servo_temperature[servo_0]"] == pytest.approx(-5.0)


def test_state_width_mismatch_is_refused(reflex, twin):
    q, _, T = _rest(twin)
    with pytest.raises(JaxterityError, match="state widths"):
        reflex.filter(q, np.zeros(twin.n_servos + 1), T, np.zeros(twin.n_servos))
    q, qd, T = _rest(twin)
    with pytest.raises(JaxterityError, match="actuated DOFs"):
        reflex.filter(q, qd, T, np.zeros(twin.n_servos + 3))


# --- the clamp --------------------------------------------------------------


def test_a_safe_command_passes_through_untouched(reflex, twin):
    q, qd, T = _rest(twin)
    decision = reflex.filter(q, qd, T, np.full(twin.n_servos, 0.05))
    assert not decision.engaged
    assert decision.feasible
    assert np.allclose(decision.tau, 0.05)


def test_clamp_lands_on_the_analytic_sustainable_torque(reflex, twin):
    """With α = 1/τ_th the thermal row *is* the sustained-hold criterion."""
    q, qd, T = _rest(twin)
    decision = reflex.filter(q, qd, T, np.full(twin.n_servos, 1.5))
    assert decision.engaged
    assert decision.feasible
    assert decision.binding_constraint.startswith("servo_temperature")
    assert np.allclose(decision.tau, _SUSTAINABLE_TAU, atol=5e-3)


def test_clamped_torque_sits_on_the_monitors_veto_threshold(reflex, twin, envelope):
    """The gate and the reflex agree about where the edge is, by construction.

    The filter converges *onto* the boundary, so the returned torque can land a
    solver residual either side of it — which is what ``violation`` /
    ``tolerance`` exist to report. The claim asserted here is the one that
    matters: a hair under the clamp the monitor is content, a few percent over
    it vetoes.
    """
    q, qd, T = _rest(twin)
    tau = reflex.filter(q, qd, T, np.full(twin.n_servos, 1.5)).tau
    monitor = SafetyMonitor(twin, envelope)

    def verdict(action):
        prediction = twin.rollout(q, qd, T, jnp.asarray(action)[None, :], 0.01)
        return monitor.evaluate(prediction, action=action).action

    assert verdict(tau * 0.99) is not SafetyAction.VETO
    assert verdict(tau * 1.05) is SafetyAction.VETO


def test_the_torque_box_is_respected(twin):
    """``torque_margin_frac`` bounds what the filter may command."""
    envelope = SafetyEnvelope(temp_ceiling_k=1e4, torque_margin_frac=0.5)
    reflex = ReflexController(twin, envelope)
    q, qd, T = _rest(twin)
    decision = reflex.filter(q, qd, T, np.full(twin.n_servos, 1e3))
    jmap = twin.robot.joint_map
    effort = np.array(
        [float(jmap[n].limits.effort) for n in reflex._names]  # noqa: SLF001
    )
    finite = effort > 0
    assert np.all(np.abs(decision.tau)[finite] <= 0.5 * effort[finite] + 1e-9)


def test_the_filter_is_deterministic(reflex, twin):
    q, qd, T = _rest(twin)
    first = reflex.filter(q, qd, T, np.full(twin.n_servos, 1.5)).tau
    second = reflex.filter(q, qd, T, np.full(twin.n_servos, 1.5)).tau
    assert np.array_equal(first, second)


# --- closed loop: the claim that matters ------------------------------------


def _held_pose_thermal_loop(twin, tau_cmd, *, reflex=None, n_steps=600, dt=0.1):
    """Heat a *held* unit (position-controlled, body at rest) for ``n_steps·dt``.

    The sustained hold is the scenario the thermal ceiling is about, and holding
    the body fixed isolates the thermal claim from the body's own motion (see
    ``test_a_stop_outranks_the_thermal_budget_and_says_so`` for what happens
    when the two interact).
    """
    q, qd, T = _rest(twin)
    q, qd, T = jnp.asarray(q), jnp.asarray(qd), jnp.asarray(T)
    peak = float(T.max())
    # Build the dynamics closure once and jit the step: eager MJX dispatch
    # (a fresh model per call, op by op) otherwise dominates the loop.
    dynamics = twin.robot.functional_dynamics()
    heat = jax.jit(
        lambda q, qd, T, tau: twin.state_derivative(q, qd, T, tau, dynamics=dynamics)[2]
    )
    for _ in range(n_steps):
        tau = np.asarray(tau_cmd, dtype=float)
        if reflex is not None:
            tau = reflex.filter(q, qd, T, tau).tau
        T = T + dt * heat(q, qd, T, jnp.asarray(tau))
        peak = max(peak, float(T.max()))
    return peak


def test_filtered_hold_holds_the_thermal_ceiling_the_raw_hold_breaks(twin, reflex):
    """Same command, same unit: unfiltered cooks the servo, filtered does not."""
    # 60 s of hold — a servo with a 60 s thermal time constant needs that long
    # to cross a ceiling 22 K above ambient under this command.
    command = np.full(twin.n_servos, 1.5)
    raw_peak = _held_pose_thermal_loop(twin, command)
    filtered_peak = _held_pose_thermal_loop(twin, command, reflex=reflex)
    assert raw_peak > _CEILING_K  # the unfiltered hold crosses the ceiling
    assert filtered_peak <= _CEILING_K  # the filtered one never does
    assert filtered_peak < raw_peak


def test_a_stop_outranks_the_thermal_budget_and_says_so(twin, reflex):
    """When the rows genuinely conflict, the declared priority decides.

    Approaching a joint limit at speed needs more braking torque than the
    sustained thermal budget allows. The reflex stops the unit — an end stop at
    speed is the immediate hazard — spends the thermal budget doing it, and
    reports which row yielded instead of leaving the trade implicit.
    """
    jmap = twin.robot.joint_map
    names = reflex._names  # noqa: SLF001
    upper = float(jmap[names[0]].limits.upper)
    q = np.zeros(twin.robot.build_system(actuation=True)._nq)
    q[0] = upper - 0.2  # right at the stop
    qd = np.zeros(twin.n_servos)
    qd[0] = 3.0  # and still running at it
    decision = reflex.filter(q, qd, np.full(twin.n_servos, _T_AMB), np.zeros(2))
    assert decision.engaged
    assert decision.tau[0] < -_SUSTAINABLE_TAU  # braking harder than the budget
    assert not decision.feasible
    assert decision.binding_constraint.startswith("servo_temperature")
    assert "declared priority" in decision.detail


def test_filtered_loop_holds_the_velocity_limit(twin, envelope, reflex):
    """The mechanical row, closed-loop: a limit MJX itself does not enforce."""
    jmap = twin.robot.joint_map
    limits = np.array(
        [float(jmap[n].limits.velocity) for n in reflex._names]  # noqa: SLF001
    )
    qd_max = float(np.min(limits[limits > 0]))
    command = np.full(twin.n_servos, 1.2)
    _, raw_peak_qd = _closed_loop(twin, command, n_steps=200, dt=0.02)
    _, filtered_peak_qd = _closed_loop(
        twin, command, reflex=reflex, n_steps=200, dt=0.02
    )
    assert raw_peak_qd > qd_max
    assert filtered_peak_qd <= qd_max * 1.05


# --- recovery ---------------------------------------------------------------


def test_recovery_brakes_and_is_itself_filtered(twin, reflex):
    """The reflex's own action opposes motion — and cannot cook the servo."""
    q, _, T = _rest(twin)
    qd = np.array([3.0, 1.0])
    decision = reflex.recover(q, qd, T)
    assert np.all(decision.tau * qd <= 0.0)  # every component opposes motion
    assert "recovery brake" in decision.detail
    # The brake request itself goes through the barriers.
    assert np.max(np.abs(decision.tau)) <= _SUSTAINABLE_TAU + 5e-3


def test_recovery_actually_slows_the_unit(twin, reflex):
    """Braking through the filter reduces speed over a short rollout."""
    step = twin.compiled_rollout(0.02)
    q, _, T = _rest(twin)
    q, qd, T = jnp.asarray(q), jnp.asarray([3.0, 0.0]), jnp.asarray(T)
    speed0 = float(jnp.abs(qd[0]))
    for _ in range(40):
        tau = reflex.recover(q, qd, T).tau
        out = step(q, qd, T, jnp.asarray(tau)[None, :])
        q, qd, T = out.q[-1], out.qd[-1], out.temperature[-1]
    assert float(jnp.abs(qd[0])) < speed0


# --- arbitration ------------------------------------------------------------


def test_simplex_hands_over_on_a_veto_and_passes_through_when_clear(
    twin, envelope, reflex
):
    monitor = SafetyMonitor(twin, envelope)
    switch = SimplexSwitch(monitor, reflex)
    q, qd, T = _rest(twin)

    gentle = np.full(twin.n_servos, 0.05)
    prediction = twin.rollout(q, qd, T, jnp.asarray(gentle)[None, :], 0.01)
    clear = switch.step(prediction, q, qd, T, gentle)
    assert clear.mode == "pass"
    assert np.allclose(clear.tau, gentle)

    hot = np.full(twin.n_servos, 2.0)
    prediction = twin.rollout(q, qd, T, jnp.asarray(hot)[None, :], 0.01)
    vetoed = switch.step(prediction, q, qd, T, hot)
    assert vetoed.mode == "reflex"
    assert vetoed.safety.action is SafetyAction.VETO
    # Userland's command is dropped, not scaled.
    assert not np.allclose(vetoed.tau, hot)
    assert "reflex" in vetoed.detail


def test_simplex_refuses_a_monitor_and_reflex_from_different_units(twin, envelope):
    other = twin.with_serial("CP-OTHER")
    with pytest.raises(JaxterityError, match="different unit versions"):
        SimplexSwitch(SafetyMonitor(other, envelope), ReflexController(twin, envelope))


def test_reflex_needs_the_thermal_nodes_to_match_the_dofs(envelope):
    robot = zoo.calibrated("cartpole")
    n = robot.build_system(actuation=True)._nv
    mismatched = Twin(
        robot,
        serial="CP-BAD",
        C_th=jnp.full(n + 1, _C_TH),
        R_th=jnp.full(n + 1, _R_TH),
        k_pwr=jnp.full(n + 1, _K_PWR),
    )
    with pytest.raises(JaxterityError, match="thermal nodes"):
        ReflexController(mismatched, envelope)


def test_filter_runs_under_jit(reflex, twin):
    """The solve is jittable: static shapes, fixed trip counts, no branches."""
    q, qd, T = _rest(twin)
    x = jnp.concatenate([jnp.asarray(q), jnp.asarray(qd), jnp.asarray(T)])
    solve = jax.jit(reflex._solve_impl)  # noqa: SLF001
    tau, slack, barriers = solve(x, jnp.full(twin.n_servos, 1.5))
    assert np.all(np.isfinite(np.asarray(tau)))
    assert slack.shape[0] == reflex.n_rows
    assert barriers.shape[0] == reflex.n_rows
