# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""BLDC — metadata, forward sim, validate on matched/mismatched params.

The BLDC model ships with the ``ActuatorChirpTelemetry`` schema. The
chirp-excitation identification recipe (frequency-domain matching through the
current loop is not closed-form) is deferred for now. Validation here is a
forward-simulation comparison: integrate the 3-D ODE [q, v, i] under a
piecewise-constant current setpoint, compare velocity + current residuals.
"""

from __future__ import annotations

import jax.numpy as jnp
import jaxonomy
import numpy as np
import pytest
from jaxonomy import SimulatorOptions
from jaxonomy.library import Constant

from jaxterity.actuators import BLDC, Actuator
from jaxterity.data import ActuatorChirpTelemetry
from jaxterity.testing import tolerances as tol

# --- metadata --------------------------------------------------------------


def test_is_actuator_subclass():
    assert issubclass(BLDC, Actuator)


def test_uncalibrated_constructs_with_neutral_defaults():
    m = BLDC.uncalibrated()
    # All ODE-denominator parameters (R, L, J, Kp_i) must be > 0.
    assert float(m.R) > 0
    assert float(m.L) > 0
    assert float(m.J) > 0
    assert float(m.Kp_i) > 0


def test_parameter_spec_lists_eight_leaves():
    assert tuple(BLDC.uncalibrated().parameter_spec()) == (
        "R",
        "L",
        "Kt",
        "Ke",
        "Kp_i",
        "J",
        "b",
        "Fc",
    )


def test_sysid_protocol_describes_current_chirp():
    p = BLDC.uncalibrated().sysid_protocol
    assert p.name == "current-chirp"
    assert p.excitation == "chirp"
    assert set(p.required_columns) == {"setpoint", "velocity", "current"}
    assert p.sample_rate_hz >= 1000.0  # must resolve electrical time constant


# --- forward sim -----------------------------------------------------------


_RK4_OPTS = SimulatorOptions(
    math_backend="jax", ode_solver_method="rk4", max_minor_step_size=1e-4
)


def _simulate_constant_iref(
    i_ref: float,
    *,
    R: float = 1.0,
    L: float = 1e-3,
    Kt: float = 0.05,
    Ke: float = 0.05,
    Kp_i: float = 10.0,
    J: float = 1e-4,
    t_end: float = 0.5,
) -> tuple[float, float, float]:
    """Drive a BLDC with a constant current setpoint and return terminal (q, v, i)."""
    m = BLDC(R=R, L=L, Kt=Kt, Ke=Ke, Kp_i=Kp_i, J=J)
    src = Constant(jnp.asarray(i_ref), name="i_ref_src")
    b = jaxonomy.DiagramBuilder()
    b.add(src)
    b.add(m)
    b.connect(src.output_ports[0], m.input_ports[0])
    diagram = b.build()
    res = jaxonomy.simulate(
        diagram, diagram.create_context(), (0.0, t_end), options=_RK4_OPTS
    )
    state = np.asarray(res.context[m.system_id].continuous_state)
    return float(state[0]), float(state[1]), float(state[2])


def test_forward_sim_current_tracks_setpoint_at_zero_velocity():
    """With v=0 (no back-EMF), the current loop drives i toward i_ref.

    Steady-state current with P controller: V_app = Kp_i·(i_ref − i) = R·i
    → i = Kp_i·i_ref / (R + Kp_i). With Kp_i=10, R=1 → i = 10/11 · i_ref.
    """
    i_ref = 1.0
    Kp_i, R = 10.0, 1.0
    # Strap J to a huge value so velocity stays ~0 during the test and back-EMF
    # is negligible: focus on the current dynamics.
    _, v, i = _simulate_constant_iref(i_ref, J=1.0, t_end=0.2)
    i_ss_expected = Kp_i * i_ref / (R + Kp_i)
    assert abs(i - i_ss_expected) < 1e-3
    assert abs(v) < 0.5  # didn't run away even with huge J


def test_forward_sim_torque_produces_motion():
    """With i_ref > 0 and finite J, the rotor accelerates."""
    _, v_end, i_end = _simulate_constant_iref(1.0, J=1e-4, t_end=0.1)
    # Both v and i must be substantially non-zero after 100 ms of drive.
    assert v_end > 1.0
    assert i_end > 0.01


# --- validate --------------------------------------------------------------


def _synthetic_chirp_telemetry(
    *,
    R: float = 1.0,
    L: float = 1e-3,
    Kt: float = 0.05,
    Ke: float = 0.05,
    Kp_i: float = 10.0,
    J: float = 1e-4,
    b: float = 0.0,
    Fc: float = 0.0,
    sample_rate_hz: float = 1000.0,
    duration_s: float = 0.5,
    f0: float = 1.0,
    f1: float = 50.0,
    amp: float = 0.5,
) -> ActuatorChirpTelemetry:
    """Forward-simulate the BLDC under a linear chirp i_ref(t)."""
    v_eps = 1e-3
    n = int(duration_s * sample_rate_hz) + 1
    t = np.arange(n, dtype=np.float64) / sample_rate_hz
    # Linear chirp f0 → f1 over duration_s.
    k = (f1 - f0) / duration_s
    iref = amp * np.sin(2 * np.pi * (f0 * t + 0.5 * k * t * t))
    dt = 1.0 / sample_rate_hz

    def rhs(q, v, i, i_ref):
        V_app = Kp_i * (i_ref - i)
        e_back = Ke * v
        didt = (V_app - R * i - e_back) / L
        tau = Kt * i
        dvdt = (tau - b * v - Fc * np.tanh(v / v_eps)) / J
        return v, dvdt, didt

    # Mirror BLDC.validate's substepping rule so the truth trace and a matched-
    # params validate run produce bit-equal trajectories.
    tau_e = L / max(R + Kp_i, 1e-12)
    dt_internal_max = 0.25 * tau_e
    n_sub = max(1, int(np.ceil(dt / dt_internal_max)))
    dts = dt / n_sub

    qs, vs, is_ = np.empty(n), np.empty(n), np.empty(n)
    q, v, i = 0.0, 0.0, 0.0
    qs[0], vs[0], is_[0] = q, v, i
    for kk in range(1, n):
        ir = iref[kk - 1]  # ZOH on the leading edge
        for _ in range(n_sub):
            k1q, k1v, k1i = rhs(q, v, i, ir)
            k2q, k2v, k2i = rhs(
                q + 0.5 * dts * k1q, v + 0.5 * dts * k1v, i + 0.5 * dts * k1i, ir
            )
            k3q, k3v, k3i = rhs(
                q + 0.5 * dts * k2q, v + 0.5 * dts * k2v, i + 0.5 * dts * k2i, ir
            )
            k4q, k4v, k4i = rhs(q + dts * k3q, v + dts * k3v, i + dts * k3i, ir)
            q = q + (dts / 6.0) * (k1q + 2 * k2q + 2 * k3q + k4q)
            v = v + (dts / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
            i = i + (dts / 6.0) * (k1i + 2 * k2i + 2 * k3i + k4i)
        qs[kk], vs[kk], is_[kk] = q, v, i
    return ActuatorChirpTelemetry(
        time=t,
        sample_rate_hz=sample_rate_hz,
        actuator_name="bldc_0",
        setpoint=iref,
        velocity=vs,
        current=is_,
    )


def test_validate_on_matched_telemetry_is_ok():
    """The model on its own forward-sim trace (matched params) reports ok."""
    R, L, Kt, Ke, Kp_i, J = 1.0, 1e-3, 0.05, 0.05, 10.0, 1e-4
    tel = _synthetic_chirp_telemetry(R=R, L=L, Kt=Kt, Ke=Ke, Kp_i=Kp_i, J=J)
    report = BLDC(R=R, L=L, Kt=Kt, Ke=Ke, Kp_i=Kp_i, J=J).validate(tel)
    assert report.ok, report.summary()


def test_validate_on_mismatched_mechanical_params_flags_warning():
    """A different inertia J produces a non-ok report.

    Mechanical mismatches (J, b, Fc) produce divergent trajectories without
    risking the integrator's electrical-stability bound — exactly what a
    real sysid recipe needs to detect at the right side of its fit.
    """
    tel = _synthetic_chirp_telemetry()
    report = BLDC(J=4e-4).validate(tel)  # 4× wrong J
    assert not report.ok


# --- nightly ---------------------------------------------------------------


@pytest.mark.nightly
@pytest.mark.parametrize("solver", tol.SOLVERS)
@pytest.mark.parametrize("dtype", tol.DTYPES)
def test_forward_sim_grid_under_constant_iref(solver, dtype):
    """Forward sim produces sensible (v, i) under every (solver, dtype) combo."""
    opts = SimulatorOptions(
        math_backend="jax", ode_solver_method=solver, max_minor_step_size=1e-4
    )
    m = BLDC()
    src = Constant(jnp.asarray(1.0, dtype=dtype), name="i_ref_src")
    b = jaxonomy.DiagramBuilder()
    b.add(src)
    b.add(m)
    b.connect(src.output_ports[0], m.input_ports[0])
    diagram = b.build()
    res = jaxonomy.simulate(diagram, diagram.create_context(), (0.0, 0.1), options=opts)
    state = np.asarray(res.context[m.system_id].continuous_state)
    v, i = float(state[1]), float(state[2])
    assert v > 0.5  # accelerating under positive torque
    assert i > 0.01  # current loop responded
