# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Servo — metadata, forward sim under step, validate residuals.

The servo model ships with the ``ActuatorStepTelemetry`` schema; the
identification recipe (an iterative nonlinear least squares — step response
through saturation + deadband is not closed-form) is deferred for now.
Validation is a forward-simulation comparison against an
:class:`~jaxterity.data.schema.ActuatorStepTelemetry` trace.
"""

from __future__ import annotations

import jax.numpy as jnp
import jaxonomy
import numpy as np
import pytest
from jaxonomy import SimulatorOptions
from jaxonomy.library import Constant

from jaxterity.actuators import Actuator, Servo
from jaxterity.data import ActuatorStepTelemetry
from jaxterity.testing import tolerances as tol

# --- metadata --------------------------------------------------------------


def test_is_actuator_subclass():
    assert issubclass(Servo, Actuator)


def test_uncalibrated_constructs_with_neutral_defaults():
    s = Servo.uncalibrated()
    # All ODE-denominator parameters (T_max, J, deadband) must be > 0.
    assert float(s.T_max) > 0
    assert float(s.J) > 0
    assert float(s.deadband) > 0


def test_parameter_spec_lists_seven_leaves():
    assert tuple(Servo.uncalibrated().parameter_spec()) == (
        "Kp",
        "Kd",
        "T_max",
        "deadband",
        "J",
        "b",
        "Fc",
    )


def test_sysid_protocol_describes_step_response():
    p = Servo.uncalibrated().sysid_protocol
    assert p.name == "step-response"
    assert p.excitation == "step"
    assert set(p.required_columns) == {"setpoint", "position"}


# --- forward sim (exercises the ODE through the Jaxonomy diagram) ---------


_RK4_OPTS = SimulatorOptions(
    math_backend="jax", ode_solver_method="rk4", max_minor_step_size=0.001
)


def _simulate_step_response(
    *,
    q_set: float,
    Kp: float,
    Kd: float,
    T_max: float = 5.0,
    deadband: float = 1e-4,
    J: float = 1e-3,
    t_end: float = 1.0,
) -> tuple[float, float]:
    """Drive a Servo with a constant setpoint and return its terminal (q, v)."""
    servo = Servo(Kp=Kp, Kd=Kd, T_max=T_max, deadband=deadband, J=J)
    src = Constant(jnp.asarray(q_set), name="q_set_src")
    b = jaxonomy.DiagramBuilder()
    b.add(src)
    b.add(servo)
    b.connect(src.output_ports[0], servo.input_ports[0])
    diagram = b.build()
    res = jaxonomy.simulate(
        diagram, diagram.create_context(), (0.0, t_end), options=_RK4_OPTS
    )
    state = np.asarray(res.context[servo.system_id].continuous_state)
    return float(state[0]), float(state[1])


def test_forward_sim_settles_at_setpoint_with_critically_damped_pd():
    """Under a step setpoint, a PD-controlled servo settles near the setpoint.

    Kp=20, Kd≈2·sqrt(Kp·J)=0.283 → ζ≈1 (critically damped) for J=1e-3 → settling
    time ~0.4 s; by t=1 s we are well-settled. Final position must be within
    the deadband of the setpoint, terminal velocity at the joint-velocity floor.
    """
    Kp, J = 20.0, 1e-3
    Kd = 2.0 * float(np.sqrt(Kp * J))
    q_set = 0.5
    q_end, v_end = _simulate_step_response(q_set=q_set, Kp=Kp, Kd=Kd, J=J)
    # Final position close to setpoint (within deadband + steady-state error).
    assert abs(q_end - q_set) < 5e-3
    # Velocity has decayed; bound on rk4/float64.
    assert abs(v_end) < tol.atol("joint_velocity", "rk4", "float64") * 1e5


def test_forward_sim_holds_at_zero_when_setpoint_equals_initial():
    """With setpoint == q0, the servo stays put — no spurious motion."""
    q_end, v_end = _simulate_step_response(q_set=0.0, Kp=20.0, Kd=0.3, t_end=0.5)
    assert q_end == pytest.approx(0.0, abs=1e-9)
    assert v_end == pytest.approx(0.0, abs=1e-9)


# --- validate --------------------------------------------------------------


def _synthetic_step_telemetry(
    *,
    Kp: float,
    Kd: float,
    T_max: float = 5.0,
    deadband: float = 1e-4,
    J: float = 1e-3,
    b: float = 0.0,
    Fc: float = 0.0,
    setpoints: list[float] | None = None,
    sample_rate_hz: float = 200.0,
    samples_per_step: int = 200,
) -> ActuatorStepTelemetry:
    """Forward-simulate the servo over a sequence of setpoints and return telemetry.

    Uses the same NumPy RK4 inner loop that ``Servo.validate`` uses, so the
    "truth" run and the validation run are bit-equal at matched parameters.
    """
    setpoints = setpoints or [0.0, 0.5, 0.5, -0.2, 0.0]
    dt = 1.0 / sample_rate_hz
    v_eps = 1e-3

    def rhs(q, v, q_set):
        err = q_set - q
        err_eff = err - deadband * np.tanh(err / deadband)
        tau_cmd = Kp * err_eff - Kd * v
        tau_act = T_max * np.tanh(tau_cmd / T_max)
        return v, (tau_act - b * v - Fc * np.tanh(v / v_eps)) / J

    ts, qs, sets = [], [], []
    q, v = 0.0, 0.0
    t = 0.0
    for sp in setpoints:
        for _ in range(samples_per_step):
            ts.append(t)
            qs.append(q)
            sets.append(sp)
            k1q, k1v = rhs(q, v, sp)
            k2q, k2v = rhs(q + 0.5 * dt * k1q, v + 0.5 * dt * k1v, sp)
            k3q, k3v = rhs(q + 0.5 * dt * k2q, v + 0.5 * dt * k2v, sp)
            k4q, k4v = rhs(q + dt * k3q, v + dt * k3v, sp)
            q = q + (dt / 6.0) * (k1q + 2 * k2q + 2 * k3q + k4q)
            v = v + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
            t += dt
    return ActuatorStepTelemetry(
        time=np.asarray(ts, dtype=np.float64),
        sample_rate_hz=sample_rate_hz,
        actuator_name="servo_0",
        setpoint=np.asarray(sets, dtype=np.float64),
        position=np.asarray(qs, dtype=np.float64),
    )


def test_validate_on_matched_telemetry_is_ok():
    """The model evaluated on its own forward-sim trace yields a clean report."""
    Kp, Kd, J = 20.0, 0.3, 1e-3
    telemetry = _synthetic_step_telemetry(Kp=Kp, Kd=Kd, J=J)
    report = Servo(Kp=Kp, Kd=Kd, J=J).validate(telemetry)
    assert report.ok, report.summary()


def test_validate_on_mismatched_params_flags_warning():
    """Substantially-wrong gains produce a non-ok report."""
    telemetry = _synthetic_step_telemetry(Kp=20.0, Kd=0.3, J=1e-3)
    # 4× Kp produces a very different trajectory.
    report = Servo(Kp=80.0, Kd=0.3, J=1e-3).validate(telemetry)
    assert not report.ok


# --- nightly ---------------------------------------------------------------


@pytest.mark.nightly
@pytest.mark.parametrize("solver", tol.SOLVERS)
@pytest.mark.parametrize("dtype", tol.DTYPES)
def test_forward_sim_grid_under_step(solver, dtype):
    """Forward sim settles at setpoint under every (solver, dtype) combo."""
    Kp, J = 20.0, 1e-3
    Kd = 2.0 * float(np.sqrt(Kp * J))
    q_set = 0.5
    opts = SimulatorOptions(
        math_backend="jax", ode_solver_method=solver, max_minor_step_size=0.001
    )
    servo = Servo(Kp=Kp, Kd=Kd, T_max=5.0, deadband=1e-4, J=J)
    src = Constant(jnp.asarray(q_set, dtype=dtype), name="q_set_src")
    b = jaxonomy.DiagramBuilder()
    b.add(src)
    b.add(servo)
    b.connect(src.output_ports[0], servo.input_ports[0])
    diagram = b.build()
    res = jaxonomy.simulate(diagram, diagram.create_context(), (0.0, 1.0), options=opts)
    q_end = float(np.asarray(res.context[servo.system_id].continuous_state)[0])
    assert abs(q_end - q_set) < 1e-2
