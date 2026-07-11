# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""SeriesElasticActuator — metadata, forward sim, validate."""

from __future__ import annotations

import jax.numpy as jnp
import jaxonomy
import numpy as np
from jaxonomy import SimulatorOptions
from jaxonomy.library import Constant

from jaxterity.actuators import Actuator, SeriesElasticActuator
from jaxterity.data import SeriesElasticTelemetry

# --- metadata --------------------------------------------------------------


def test_is_actuator_subclass():
    assert issubclass(SeriesElasticActuator, Actuator)


def test_uncalibrated_constructs_with_neutral_defaults():
    s = SeriesElasticActuator.uncalibrated()
    assert float(s.k) > 0
    assert float(s.J_motor) > 0
    assert float(s.J_load) > 0


def test_parameter_spec_lists_six_leaves():
    assert tuple(SeriesElasticActuator.uncalibrated().parameter_spec()) == (
        "k",
        "c",
        "J_motor",
        "J_load",
        "b_m",
        "b_l",
    )


def test_sysid_protocol_describes_sea_step():
    p = SeriesElasticActuator.uncalibrated().sysid_protocol
    assert p.name == "sea-step"
    assert set(p.required_columns) == {"tau_input", "q_motor", "q_load"}


# --- forward sim -----------------------------------------------------------


_RK4_OPTS = SimulatorOptions(
    math_backend="jax", ode_solver_method="rk4", max_minor_step_size=1e-4
)


def _simulate_step(
    tau: float, *, k: float = 100.0, J_load: float = 1e-2, t_end: float = 0.5
) -> tuple[float, float]:
    """Apply constant motor torque; return terminal (q_motor, q_load)."""
    s = SeriesElasticActuator(k=k, J_load=J_load)
    src = Constant(jnp.asarray(tau), name="tau_src")
    b = jaxonomy.DiagramBuilder()
    b.add(src)
    b.add(s)
    b.connect(src.output_ports[0], s.input_ports[0])
    diagram = b.build()
    res = jaxonomy.simulate(
        diagram, diagram.create_context(), (0.0, t_end), options=_RK4_OPTS
    )
    state = np.asarray(res.context[s.system_id].continuous_state)
    return float(state[0]), float(state[2])


def test_forward_sim_spring_couples_motor_and_load():
    """Under constant torque, motor moves first; load follows via the spring."""
    qm, ql = _simulate_step(1.0, t_end=0.05)
    # Motor accelerates first (smaller inertia), load lags but is dragged along.
    assert qm > 0
    assert ql > 0
    assert qm > ql  # spring deflection is positive (motor leads load)


def test_forward_sim_static_equilibrium_recovers_deflection():
    """Long enough sim → torque balance: τ_input ≈ k · (q_motor − q_load).

    With friction off and infinite-time limit, both inertias are at rest
    and the spring deflection equals τ/k. b_l = 0 means the load drifts at
    constant velocity at infinity — pick a moderate t_end and check the
    instantaneous deflection ≈ τ/k.
    """
    k = 100.0
    qm, ql = _simulate_step(1.0, k=k, t_end=0.2)
    expected_deflection = 1.0 / k
    # Substantial oscillation means we won't hit exact equilibrium — just check
    # the deflection is the right order of magnitude.
    assert 0.5 * expected_deflection < (qm - ql) < 2.0 * expected_deflection


# --- validate --------------------------------------------------------------


def _synthetic_sea_telemetry(
    *,
    k: float,
    c: float,
    J_motor: float,
    J_load: float,
    b_m: float = 0.0,
    b_l: float = 0.0,
    sample_rate_hz: float = 500.0,
    duration_s: float = 0.5,
    tau_amp: float = 1.0,
) -> SeriesElasticTelemetry:
    """Step-train forward sim sharing the substep rule used by validate."""
    n = int(duration_s * sample_rate_hz) + 1
    t = np.arange(n, dtype=np.float64) / sample_rate_hz
    dt = 1.0 / sample_rate_hz
    omega = np.sqrt(max(k / J_load, 1e-12))
    dt_internal_max = 0.1 * (2 * np.pi / omega)
    n_sub = max(1, int(np.ceil(dt / dt_internal_max)))
    dts = dt / n_sub
    # Step every 0.1 s, alternating sign — exercises spring in both directions.
    tau = tau_amp * np.where((t // 0.1) % 2 == 0, 1.0, -1.0)

    def rhs(qm, vm, ql, vl, ti):
        ts = k * (qm - ql) + c * (vm - vl)
        return vm, (ti - ts - b_m * vm) / J_motor, vl, (ts - b_l * vl) / J_load

    qms, qls = np.empty(n), np.empty(n)
    qm = vm = ql = vl = 0.0
    qms[0], qls[0] = qm, ql
    for kk in range(1, n):
        ti = float(tau[kk - 1])
        for _ in range(n_sub):
            k1qm, k1vm, k1ql, k1vl = rhs(qm, vm, ql, vl, ti)
            k2qm, k2vm, k2ql, k2vl = rhs(
                qm + 0.5 * dts * k1qm,
                vm + 0.5 * dts * k1vm,
                ql + 0.5 * dts * k1ql,
                vl + 0.5 * dts * k1vl,
                ti,
            )
            k3qm, k3vm, k3ql, k3vl = rhs(
                qm + 0.5 * dts * k2qm,
                vm + 0.5 * dts * k2vm,
                ql + 0.5 * dts * k2ql,
                vl + 0.5 * dts * k2vl,
                ti,
            )
            k4qm, k4vm, k4ql, k4vl = rhs(
                qm + dts * k3qm,
                vm + dts * k3vm,
                ql + dts * k3ql,
                vl + dts * k3vl,
                ti,
            )
            qm = qm + (dts / 6.0) * (k1qm + 2 * k2qm + 2 * k3qm + k4qm)
            vm = vm + (dts / 6.0) * (k1vm + 2 * k2vm + 2 * k3vm + k4vm)
            ql = ql + (dts / 6.0) * (k1ql + 2 * k2ql + 2 * k3ql + k4ql)
            vl = vl + (dts / 6.0) * (k1vl + 2 * k2vl + 2 * k3vl + k4vl)
        qms[kk], qls[kk] = qm, ql
    return SeriesElasticTelemetry(
        time=t,
        sample_rate_hz=sample_rate_hz,
        actuator_name="sea_0",
        tau_input=tau,
        q_motor=qms,
        q_load=qls,
    )


def test_validate_on_matched_telemetry_is_ok():
    """Matched params → near-zero residuals."""
    k, c, J_m, J_l = 100.0, 0.1, 1e-3, 1e-2
    tel = _synthetic_sea_telemetry(k=k, c=c, J_motor=J_m, J_load=J_l)
    report = SeriesElasticActuator(k=k, c=c, J_motor=J_m, J_load=J_l).validate(tel)
    assert report.ok, report.summary()


def test_validate_on_mismatched_spring_flags_warning():
    """A wrong spring stiffness produces a non-ok report."""
    tel = _synthetic_sea_telemetry(k=100.0, c=0.1, J_motor=1e-3, J_load=1e-2)
    report = SeriesElasticActuator(
        k=300.0,
        c=0.1,
        J_motor=1e-3,
        J_load=1e-2,  # 3× wrong k
    ).validate(tel)
    assert not report.ok
