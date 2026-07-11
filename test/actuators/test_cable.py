# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""CableActuator — asymmetric spring, validate residuals."""

from __future__ import annotations

import jax.numpy as jnp
import jaxonomy
import numpy as np
from jaxonomy import SimulatorOptions
from jaxonomy.library import Constant

from jaxterity.actuators import Actuator, CableActuator
from jaxterity.data import SeriesElasticTelemetry


def test_is_actuator_subclass():
    assert issubclass(CableActuator, Actuator)


def test_uncalibrated_defaults_are_positive():
    c = CableActuator.uncalibrated()
    assert float(c.k) > 0
    assert float(c.J_motor) > 0 and float(c.J_load) > 0


def test_parameter_spec_lists_six_leaves():
    assert tuple(CableActuator.uncalibrated().parameter_spec()) == (
        "k",
        "c",
        "J_motor",
        "J_load",
        "b_m",
        "b_l",
    )


def test_sysid_protocol_describes_cable_step():
    p = CableActuator.uncalibrated().sysid_protocol
    assert p.name == "cable-step"
    assert set(p.required_columns) == {"tau_input", "q_motor", "q_load"}


# --- asymmetric spring (the whole point) -----------------------------------


_RK4 = SimulatorOptions(
    math_backend="jax", ode_solver_method="rk4", max_minor_step_size=1e-4
)


def _simulate(tau: float, *, k=100.0, J_load=1e-2, t_end=0.5) -> tuple[float, float]:
    cable = CableActuator(k=k, J_load=J_load)
    src = Constant(jnp.asarray(tau), name="tau_src")
    b = jaxonomy.DiagramBuilder()
    b.add(src)
    b.add(cable)
    b.connect(src.output_ports[0], cable.input_ports[0])
    diagram = b.build()
    res = jaxonomy.simulate(
        diagram, diagram.create_context(), (0.0, t_end), options=_RK4
    )
    state = np.asarray(res.context[cable.system_id].continuous_state)
    return float(state[0]), float(state[2])  # q_motor, q_load


def test_positive_torque_loads_the_cable():
    """Positive tau_input pulls motor positive; cable becomes taut → load follows."""
    qm, ql = _simulate(1.0, t_end=0.1)
    assert qm > 0  # motor advanced
    assert ql > 0  # cable pulled load along
    assert qm > ql  # cable deflection is positive


def test_negative_torque_cable_goes_slack_load_does_not_reverse():
    """With negative tau_input the cable goes slack and the load can't be pushed.

    Initially both are at rest. tau_input is negative → motor moves negative.
    Deflection q_m - q_l becomes negative → softplus(k·neg) ≈ 0 → load gets
    near-zero spring force, so it stays approximately at rest while motor
    drifts away.
    """
    qm, ql = _simulate(-1.0, t_end=0.05)
    assert qm < 0  # motor moved negative
    # Load barely moved (within b_l = 0 friction-free drift only).
    assert abs(ql) < abs(qm) * 0.1


# --- validate --------------------------------------------------------------


def _synthetic_cable_telemetry(*, k, c, J_m, J_l, b_m=0.0, b_l=0.0):
    fs, dur = 500.0, 0.5
    n = int(dur * fs) + 1
    t = np.arange(n, dtype=np.float64) / fs
    dt = 1.0 / fs
    omega = np.sqrt(max(k / J_l, 1e-12))
    n_sub = max(1, int(np.ceil(dt / (0.1 * (2 * np.pi / omega)))))
    dts = dt / n_sub
    tau = 1.0 * np.where((t // 0.1) % 2 == 0, 1.0, -1.0)

    def softplus(x):
        return np.logaddexp(0.0, x)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def rhs(qm, vm, ql, vl, ti):
        defl = qm - ql
        tension = softplus(k * defl)
        gate = sigmoid(k * defl)
        ts = tension + gate * c * (vm - vl)
        return vm, (ti - ts - b_m * vm) / J_m, vl, (ts - b_l * vl) / J_l

    qms, qls = np.empty(n), np.empty(n)
    qm = vm = ql = vl = 0.0
    qms[0], qls[0] = 0.0, 0.0
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
                qm + dts * k3qm, vm + dts * k3vm, ql + dts * k3ql, vl + dts * k3vl, ti
            )
            qm = qm + (dts / 6.0) * (k1qm + 2 * k2qm + 2 * k3qm + k4qm)
            vm = vm + (dts / 6.0) * (k1vm + 2 * k2vm + 2 * k3vm + k4vm)
            ql = ql + (dts / 6.0) * (k1ql + 2 * k2ql + 2 * k3ql + k4ql)
            vl = vl + (dts / 6.0) * (k1vl + 2 * k2vl + 2 * k3vl + k4vl)
        qms[kk], qls[kk] = qm, ql
    return SeriesElasticTelemetry(
        time=t,
        sample_rate_hz=fs,
        actuator_name="c0",
        tau_input=tau,
        q_motor=qms,
        q_load=qls,
    )


def test_validate_on_matched_telemetry_is_ok():
    k, c, J_m, J_l = 100.0, 0.1, 1e-3, 1e-2
    tel = _synthetic_cable_telemetry(k=k, c=c, J_m=J_m, J_l=J_l)
    report = CableActuator(k=k, c=c, J_motor=J_m, J_load=J_l).validate(tel)
    assert report.ok, report.summary()
