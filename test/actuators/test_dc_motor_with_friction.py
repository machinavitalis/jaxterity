# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""DCMotorWithFriction — metadata, forward sim, fit recovery, validate.

The recovery test is necessarily synthetic: no open-source WPILib reference
trace with known truth parameters exists. We pick a smooth velocity profile
``v(t)``, differentiate to obtain ``a(t)``, apply :func:`predict_voltage` to
obtain the consistent ``V(t)``, then run the OLS fitter and assert recovery
within the policy ``actuator_voltage`` tolerance. This tests the
self-consistency of the model and its closed-form least-squares fitter — not
third-party calibration; real FRC log fits land later.
"""

from __future__ import annotations

import jax.numpy as jnp
import jaxonomy
import numpy as np
import pytest
from jaxonomy import SimulatorOptions
from jaxonomy.library import Constant

from jaxterity.actuators import Actuator, DCMotorWithFriction
from jaxterity.actuators.dc_motor import _fit_voltage_ramp, predict_voltage
from jaxterity.data import ActuatorRampTelemetry
from jaxterity.testing import tolerances as tol

# --- metadata --------------------------------------------------------------


def test_is_actuator_subclass():
    assert issubclass(DCMotorWithFriction, Actuator)


def test_uncalibrated_constructs_with_zero_params():
    act = DCMotorWithFriction.uncalibrated()
    assert float(act.Ks) == 0.0
    assert float(act.Kv) == 0.0
    # Ka must be > 0 to keep the ODE well-defined; uncalibrated default is the
    # identifiability-neutral placeholder 1.0 (documented in the class).
    assert float(act.Ka) == 1.0


def test_parameter_spec_lists_ks_kv_ka():
    assert tuple(DCMotorWithFriction.uncalibrated().parameter_spec()) == (
        "Ks",
        "Kv",
        "Ka",
    )


def test_sysid_protocol_describes_voltage_ramp():
    p = DCMotorWithFriction.uncalibrated().sysid_protocol
    assert p.name == "voltage-ramp"
    assert p.excitation == "ramp"
    assert set(p.required_columns) == {"command", "velocity"}
    assert p.sample_rate_hz == 200.0


# --- predict_voltage -------------------------------------------------------


def test_predict_voltage_matches_definition():
    # V = Ks·tanh(v/v_eps) + Kv·v + Ka·a; at v=1.5 ≫ v_eps the tanh saturates to 1.
    # 0.1*1 + 0.5*1.5 + 0.02*2.0 = 0.89
    v = predict_voltage(jnp.asarray(1.5), jnp.asarray(2.0), Ks=0.1, Kv=0.5, Ka=0.02)
    assert float(v) == pytest.approx(0.89, abs=1e-9)


def test_predict_voltage_smooths_sign_at_zero():
    # At v=0, tanh(0/v_eps) = 0 → the Ks term drops out.
    v = predict_voltage(jnp.asarray(0.0), jnp.asarray(2.0), Ks=0.1, Kv=0.5, Ka=0.02)
    assert float(v) == pytest.approx(0.04, abs=1e-12)


# --- forward sim (exercises the ODE through the Jaxonomy diagram) ---------


_RK4_OPTS = SimulatorOptions(
    math_backend="jax", ode_solver_method="rk4", max_minor_step_size=0.001
)


def _simulate_const_voltage(
    V_cmd: float, *, Ks: float, Kv: float, Ka: float, t_end: float = 1.5
) -> float:
    """Drive the actuator with a constant voltage and return its terminal velocity."""
    act = DCMotorWithFriction(Ks=Ks, Kv=Kv, Ka=Ka)
    src = Constant(jnp.asarray(V_cmd), name="V_src")
    b = jaxonomy.DiagramBuilder()
    b.add(src)
    b.add(act)
    b.connect(src.output_ports[0], act.input_ports[0])
    diagram = b.build()
    res = jaxonomy.simulate(
        diagram, diagram.create_context(), (0.0, t_end), options=_RK4_OPTS
    )
    return float(np.asarray(res.context[act.system_id].continuous_state))


def test_forward_sim_approaches_steady_state_velocity():
    """Under constant V, velocity approaches (V − Ks) / Kv; Ka sets the time constant.

    Time constant τ = Ka / Kv. With Ka=0.05, Kv=0.5 → τ=0.1 s; by t=1.5 s
    (15τ) the residual transient + RK4 floor is bounded by
    ``rtol("joint_velocity", "rk4", "float64")``.
    """
    Ks, Kv, Ka = 0.2, 0.5, 0.05
    V = 5.0
    v_ss_expected = (V - Ks) / Kv
    v_terminal = _simulate_const_voltage(V, Ks=Ks, Kv=Kv, Ka=Ka)
    assert v_terminal == pytest.approx(
        v_ss_expected, rel=tol.rtol("joint_velocity", "rk4", "float64")
    )


# --- recovery (round-trip on analytically consistent telemetry) -----------


def _synthetic_ramp_telemetry(
    *,
    Ks: float,
    Kv: float,
    Ka: float,
    sample_rate_hz: float = 200.0,
    duration_s: float = 2.0,
) -> ActuatorRampTelemetry:
    """Generate (V, v) telemetry analytically: pick v(t), compute a(t), V(t).

    Velocity profile is a quadratic ``v(t) = α + β t + γ t²`` so that the OLS
    design matrix ``[sgn(v), v, a]`` has full column rank (a linear ramp
    gives constant ``a`` and rank-deficient OLS). The trace stays in the
    positive-velocity half-plane so the smoothed-sign saturates to +1 and the
    closed-form fitter (which uses raw ``np.sign``) agrees exactly.
    """
    n = int(duration_s * sample_rate_hz) + 1
    t = np.arange(n, dtype=np.float64) / sample_rate_hz
    # v(t) = 0.5 + 1.5 t + 0.8 t² — v(0)=0.5 ≫ v_eps; strictly increasing.
    v = 0.5 + 1.5 * t + 0.8 * t * t
    a = 1.5 + 1.6 * t
    V = Ks * np.sign(v) + Kv * v + Ka * a
    return ActuatorRampTelemetry(
        time=t,
        sample_rate_hz=sample_rate_hz,
        actuator_name="motor_0",
        command=V,
        velocity=v,
    )


def test_round_trip_recovers_truth_parameters_clean():
    """Clean-data OLS recovers truth Ks/Kv/Ka within the actuator_voltage rtol."""
    Ks_true, Kv_true, Ka_true = 0.2, 0.5, 0.05
    telemetry = _synthetic_ramp_telemetry(Ks=Ks_true, Kv=Kv_true, Ka=Ka_true)
    fitted = _fit_voltage_ramp(telemetry)
    r = tol.rtol("actuator_voltage", "rk4", "float64")
    assert float(fitted["Ks"]) == pytest.approx(Ks_true, rel=r)
    assert float(fitted["Kv"]) == pytest.approx(Kv_true, rel=r)
    assert float(fitted["Ka"]) == pytest.approx(Ka_true, rel=r)


# --- validate --------------------------------------------------------------


def test_validate_on_truth_telemetry_has_near_zero_residuals():
    """When the model matches the data, residuals are at the numerical floor."""
    Ks, Kv, Ka = 0.2, 0.5, 0.05
    telemetry = _synthetic_ramp_telemetry(Ks=Ks, Kv=Kv, Ka=Ka)
    report = DCMotorWithFriction(Ks=Ks, Kv=Kv, Ka=Ka).validate(telemetry)
    assert report.ok, report.summary()


def test_validate_on_wrong_truth_flags_large_residuals():
    """Mis-specified Ks/Kv/Ka yields a non-ok report."""
    telemetry = _synthetic_ramp_telemetry(Ks=0.2, Kv=0.5, Ka=0.05)
    # Wrong by 50% — residuals must exceed the per-actuator-voltage atol bound.
    report = DCMotorWithFriction(Ks=0.3, Kv=0.75, Ka=0.075).validate(telemetry)
    assert not report.ok


# --- nightly ---------------------------------------------------------------


@pytest.mark.nightly
def test_round_trip_noisy_recovery_within_heuristic_bound():
    """With Gaussian voltage noise σ=0.05 V, recovered params stay within ~5× σ_OLS."""
    Ks_t, Kv_t, Ka_t = 0.2, 0.5, 0.05
    telemetry = _synthetic_ramp_telemetry(Ks=Ks_t, Kv=Kv_t, Ka=Ka_t, duration_s=5.0)
    rng = np.random.default_rng(0)
    sigma_V = 0.05  # V; representative FRC sensor noise
    noisy_V = telemetry.command + rng.normal(
        scale=sigma_V, size=telemetry.command.shape
    )
    noisy = ActuatorRampTelemetry(
        time=telemetry.time,
        sample_rate_hz=telemetry.sample_rate_hz,
        actuator_name=telemetry.actuator_name,
        command=noisy_V,
        velocity=telemetry.velocity,
    )
    fitted = _fit_voltage_ramp(noisy)
    # OLS σ_β ≈ σ_V / sqrt(n·var(X_col)). For n≈1000 and span(v)≈O(10), the per-
    # parameter σ is ~σ_V/30, ~σ_V/100, ~σ_V/15 for Ks, Kv, Ka. A 5× envelope on
    # σ_V (~0.25 V) is well above all three; tight enough to catch a regression.
    band = 5.0 * sigma_V
    assert abs(float(fitted["Ks"]) - Ks_t) < band
    assert abs(float(fitted["Kv"]) - Kv_t) < band
    assert abs(float(fitted["Ka"]) - Ka_t) < band


@pytest.mark.nightly
@pytest.mark.parametrize("solver", tol.SOLVERS)
@pytest.mark.parametrize("dtype", tol.DTYPES)
def test_forward_sim_grid_reaches_steady_state(solver, dtype):
    """Forward sim approaches steady-state velocity under every (solver, dtype).

    Validates that the ODE callback composes cleanly with each of the three
    integrators jaxonomy ships, at both precisions. The bound is the policy
    ``rtol("joint_velocity", solver, dtype)`` — relative because v_ss is O(10).
    """
    Ks, Kv, Ka = 0.2, 0.5, 0.05
    V = 5.0
    v_ss_expected = (V - Ks) / Kv
    opts = SimulatorOptions(
        math_backend="jax", ode_solver_method=solver, max_minor_step_size=0.001
    )
    act = DCMotorWithFriction(Ks=Ks, Kv=Kv, Ka=Ka)
    src = Constant(jnp.asarray(V, dtype=dtype), name="V_src")
    b = jaxonomy.DiagramBuilder()
    b.add(src)
    b.add(act)
    b.connect(src.output_ports[0], act.input_ports[0])
    diagram = b.build()
    res = jaxonomy.simulate(diagram, diagram.create_context(), (0.0, 1.5), options=opts)
    v_terminal = float(np.asarray(res.context[act.system_id].continuous_state))
    assert v_terminal == pytest.approx(
        v_ss_expected, rel=tol.rtol("joint_velocity", solver, dtype)
    )
