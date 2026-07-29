# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Multi-node thermal network — analytics, steady state, network parity, fit.

Three references pin the model down: the closed-form two-node steady state,
the analytic step response (checked against a NumPy RK4 integration of the same
ODEs), and the network-wide conductance solve (checked against the compiled
acausal system). The fit test is the identifiability claim from the module
docstring, exercised: with both nodes measured and power in watts, all four
parameters come back.
"""

from __future__ import annotations

import jaxonomy
import numpy as np
import pytest

pytest.importorskip("sympy")

from jaxterity.data import ThermalTelemetry  # noqa: E402
from jaxterity.errors import CalibrationError, JaxterityError  # noqa: E402
from jaxterity.physics.thermal_network import (  # noqa: E402
    AmbientSpec,
    ConductionSpec,
    DriverSpec,
    ServoThermalSpec,
    ThermalNetwork,
    ThermalNetworkSpec,
    fit_two_node,
    steady_state_temperatures,
    two_node_response,
    two_node_steady_state,
    validate_two_node,
)
from jaxterity.testing import tolerances as tol  # noqa: E402

_T_AMB = 300.0
_K_TOL = tol.atol("motor_temperature", "bdf", "float64") * 1e5


def _servo(name="j0", **overrides) -> ServoThermalSpec:
    kwargs = dict(C_w=20.0, C_h=200.0, R_wh=0.8, R_ha=3.0)
    kwargs.update(overrides)
    return ServoThermalSpec(name, **kwargs)


def _spec(*servos, **overrides) -> ThermalNetworkSpec:
    kwargs = dict(servos=servos or (_servo(),), ambient=AmbientSpec(T_amb=_T_AMB))
    kwargs.update(overrides)
    return ThermalNetworkSpec(**kwargs)


# --- analytic two-node -----------------------------------------------------


def test_steady_state_stacks_the_two_drops():
    T_w, T_h = two_node_steady_state(4.0, T_amb=_T_AMB, R_wh=0.8, R_ha=3.0)
    assert float(T_h) == pytest.approx(_T_AMB + 4.0 * 3.0)
    assert float(T_w) == pytest.approx(_T_AMB + 4.0 * (3.0 + 0.8))


def test_winding_is_always_the_hotter_node_under_load():
    T_w, T_h = two_node_steady_state(2.5, T_amb=_T_AMB, R_wh=0.8, R_ha=3.0)
    assert float(T_w) > float(T_h) > _T_AMB


def test_response_starts_at_the_initial_state_and_ends_at_steady_state():
    kwargs = dict(
        T0_w=_T_AMB, T0_h=_T_AMB, T_amb=_T_AMB, C_w=20.0, C_h=200.0, R_wh=0.8, R_ha=3.0
    )
    T_w0, T_h0 = two_node_response(0.0, 4.0, **kwargs)
    assert float(T_w0) == pytest.approx(_T_AMB)
    assert float(T_h0) == pytest.approx(_T_AMB)
    T_w_inf, T_h_inf = two_node_response(1e5, 4.0, **kwargs)
    ss_w, ss_h = two_node_steady_state(4.0, T_amb=_T_AMB, R_wh=0.8, R_ha=3.0)
    assert float(T_w_inf) == pytest.approx(float(ss_w), abs=_K_TOL)
    assert float(T_h_inf) == pytest.approx(float(ss_h), abs=_K_TOL)


def test_winding_leads_the_housing_early_in_the_step():
    # The whole reason for the second node: the winding pole is ~C_w/C_h faster,
    # so a short burst shows up on the winding and barely moves the case.
    kwargs = dict(
        T0_w=_T_AMB, T0_h=_T_AMB, T_amb=_T_AMB, C_w=20.0, C_h=200.0, R_wh=0.8, R_ha=3.0
    )
    T_w, T_h = two_node_response(5.0, 8.0, **kwargs)
    assert float(T_w) - _T_AMB > 5.0 * (float(T_h) - _T_AMB)


def _rk4_two_node(t_end, power, *, C_w, C_h, R_wh, R_ha, steps=200_000):
    """Reference integration of the same ODEs (independent of the analytics)."""
    dt = t_end / steps
    T_w = T_h = _T_AMB
    for _ in range(steps):

        def rhs(w, h):
            through = (w - h) / R_wh
            return (power - through) / C_w, (through - (h - _T_AMB) / R_ha) / C_h

        k1 = rhs(T_w, T_h)
        k2 = rhs(T_w + 0.5 * dt * k1[0], T_h + 0.5 * dt * k1[1])
        k3 = rhs(T_w + 0.5 * dt * k2[0], T_h + 0.5 * dt * k2[1])
        k4 = rhs(T_w + dt * k3[0], T_h + dt * k3[1])
        T_w += (dt / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        T_h += (dt / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return T_w, T_h


def test_analytic_response_matches_a_numerical_integration():
    params = dict(C_w=20.0, C_h=200.0, R_wh=0.8, R_ha=3.0)
    ref_w, ref_h = _rk4_two_node(60.0, 6.0, steps=20_000, **params)
    T_w, T_h = two_node_response(
        60.0, 6.0, T0_w=_T_AMB, T0_h=_T_AMB, T_amb=_T_AMB, **params
    )
    assert float(T_w) == pytest.approx(ref_w, abs=_K_TOL)
    assert float(T_h) == pytest.approx(ref_h, abs=_K_TOL)


def test_response_is_differentiable_in_its_parameters():
    import jax

    grad = jax.grad(
        lambda R_ha: two_node_response(
            30.0,
            4.0,
            T0_w=_T_AMB,
            T0_h=_T_AMB,
            T_amb=_T_AMB,
            C_w=20.0,
            C_h=200.0,
            R_wh=0.8,
            R_ha=R_ha,
        )[0]
    )(3.0)
    # A worse path to ambient means a hotter winding.
    assert float(grad) > 0.0


# --- spec validation -------------------------------------------------------


def test_conduction_to_an_undeclared_servo_raises():
    with pytest.raises(JaxterityError, match="not declared"):
        _spec(_servo("a"), conduction=(ConductionSpec("a", "ghost", 5.0),))


def test_self_conduction_raises():
    with pytest.raises(JaxterityError, match="servo to itself"):
        ConductionSpec("a", "a", 5.0)


def test_celsius_ambient_is_rejected():
    with pytest.raises(JaxterityError, match="absolute temperature in kelvin"):
        AmbientSpec(T_amb=-5.0)


def test_nodes_list_winding_housing_and_driver():
    spec = _spec(_servo("a"), _servo("b", driver=DriverSpec(C_th=30.0, R_da=6.0)))
    assert spec.nodes == (
        "a.winding",
        "a.housing",
        "b.winding",
        "b.housing",
        "b.driver",
    )


# --- network steady state --------------------------------------------------


def test_network_steady_state_matches_the_isolated_two_node_formula():
    temps = steady_state_temperatures(_spec(), {"j0": 4.0})
    ss_w, ss_h = two_node_steady_state(4.0, T_amb=_T_AMB, R_wh=0.8, R_ha=3.0)
    assert temps["j0.winding"] == pytest.approx(float(ss_w))
    assert temps["j0.housing"] == pytest.approx(float(ss_h))


def test_a_hot_neighbour_raises_the_idle_servo_through_the_bracket():
    # The effect a per-servo lumped model structurally cannot show.
    spec = _spec(
        _servo("hot"), _servo("cold"), conduction=(ConductionSpec("hot", "cold", 4.0),)
    )
    coupled = steady_state_temperatures(spec, {"hot": 6.0})
    isolated = steady_state_temperatures(_spec(_servo("cold")), {})
    assert coupled["cold.housing"] > isolated["cold.housing"]
    # ... and the coupling cools the hot one, since it gained a second path out.
    assert (
        coupled["hot.housing"]
        < steady_state_temperatures(_spec(_servo("hot")), {"hot": 6.0})["hot.housing"]
    )


def test_a_coupled_driver_warms_its_own_servo_housing():
    tied = _spec(_servo("j0", driver=DriverSpec(C_th=30.0, R_da=6.0, R_dh=2.0)))
    isolated = _spec(_servo("j0", driver=DriverSpec(C_th=30.0, R_da=6.0)))
    powers = {"j0.driver": 3.0}
    assert (
        steady_state_temperatures(tied, powers)["j0.housing"]
        > steady_state_temperatures(isolated, powers)["j0.housing"]
    )
    # An isolated driver leaves the servo at ambient.
    assert steady_state_temperatures(isolated, powers)["j0.housing"] == pytest.approx(
        _T_AMB
    )


def test_steady_state_rejects_an_unknown_node():
    with pytest.raises(CalibrationError, match="Unknown thermal node"):
        steady_state_temperatures(_spec(), {"j0.rotor": 1.0})


# --- compiled network ------------------------------------------------------


def _simulate(network: ThermalNetwork, powers: dict[str, float], t_end: float):
    builder = jaxonomy.DiagramBuilder()
    system = builder.add(network.system)
    # Every node exposes a power input; the unexcited ones must still be tied
    # to zero (an acausal input port has no default source).
    for node, index in network.power_inputs.items():
        source = builder.add(
            jaxonomy.library.Constant(
                powers.get(node, 0.0), name=f"p_{node.replace('.', '_')}"
            )
        )
        builder.connect(source.output_ports[0], system.input_ports[index])
    diagram = builder.build()
    results = jaxonomy.simulate(
        diagram,
        diagram.create_context(),
        (0.0, t_end),
        recorded_signals={p.name: p for p in system.output_ports},
    )
    names = [p.name for p in system.output_ports]
    return {
        node: float(results.outputs[names[idx]][-1])
        for node, idx in network.temperature_outputs.items()
    }


@pytest.mark.slow
def test_compiled_network_reaches_the_conductance_solve():
    spec = _spec(
        _servo("a"),
        _servo("b", driver=DriverSpec(C_th=30.0, R_da=6.0, R_dh=2.0)),
        conduction=(ConductionSpec("a", "b", 4.0),),
    )
    network = ThermalNetwork.build(spec)
    powers = {"a.winding": 5.0, "b.winding": 2.0, "b.driver": 1.0}
    final = _simulate(network, powers, t_end=20_000.0)
    expected = steady_state_temperatures(spec, powers)
    for node, value in expected.items():
        assert final[node] == pytest.approx(value, abs=1e-3), node


@pytest.mark.slow
def test_compiled_network_transient_matches_the_analytic_response():
    spec = _spec()
    network = ThermalNetwork.build(spec)
    final = _simulate(network, {"j0.winding": 6.0}, t_end=60.0)
    T_w, T_h = two_node_response(
        60.0,
        6.0,
        T0_w=_T_AMB,
        T0_h=_T_AMB,
        T_amb=_T_AMB,
        C_w=20.0,
        C_h=200.0,
        R_wh=0.8,
        R_ha=3.0,
    )
    assert final["j0.winding"] == pytest.approx(float(T_w), abs=1e-3)
    assert final["j0.housing"] == pytest.approx(float(T_h), abs=1e-3)


def test_port_maps_cover_every_node():
    spec = _spec(_servo("a"), _servo("b", driver=DriverSpec(C_th=30.0, R_da=6.0)))
    network = ThermalNetwork.build(spec)
    assert set(network.power_inputs) == set(spec.nodes)
    assert set(network.temperature_outputs) == set(spec.nodes)


# --- fitting ---------------------------------------------------------------


def _synthetic_traces(servo: ServoThermalSpec, *, dt=0.5, n=600):
    """A power step held then released, sampled on both nodes."""
    power = np.where(np.arange(n) < n // 2, 6.0, 0.0)
    T_w = np.empty(n)
    T_h = np.empty(n)
    T_w[0] = T_h[0] = _T_AMB
    state = np.array([_T_AMB, _T_AMB])
    for i in range(1, n):

        def rhs(x, u):
            through = (x[0] - x[1]) / servo.R_wh
            return np.array(
                [
                    (u - through) / servo.C_w,
                    (through - (x[1] - _T_AMB) / servo.R_ha) / servo.C_h,
                ]
            )

        u = power[i - 1]
        k1 = rhs(state, u)
        k2 = rhs(state + 0.5 * dt * k1, u)
        k3 = rhs(state + 0.5 * dt * k2, u)
        k4 = rhs(state + dt * k3, u)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        T_w[i], T_h[i] = state
    time = np.arange(n) * dt
    winding = ThermalTelemetry(
        time=time,
        sample_rate_hz=1.0 / dt,
        servo_names=(servo.name,),
        temperature=T_w[:, None],
    )
    housing = ThermalTelemetry(
        time=time,
        sample_rate_hz=1.0 / dt,
        servo_names=(servo.name,),
        temperature=T_h[:, None],
    )
    return winding, housing, power


def test_fit_recovers_all_four_parameters_from_both_nodes():
    servo = _servo()
    winding, housing, power = _synthetic_traces(servo)
    fitted = fit_two_node(winding, housing, power, T_amb=_T_AMB)
    assert fitted["C_w"] == pytest.approx(servo.C_w, rel=2e-2)
    assert fitted["R_wh"] == pytest.approx(servo.R_wh, rel=2e-2)
    assert fitted["C_h"] == pytest.approx(servo.C_h, rel=2e-2)
    assert fitted["R_ha"] == pytest.approx(servo.R_ha, rel=2e-2)


def test_fit_rejects_a_trace_with_no_excitation():
    servo = _servo()
    winding, housing, _ = _synthetic_traces(servo, n=40)
    flat = np.zeros(40)
    # An unexcited trace sits at ambient: no gradient, no power => rank-deficient.
    telemetry = ThermalTelemetry(
        time=winding.time,
        sample_rate_hz=winding.sample_rate_hz,
        servo_names=("j0",),
        temperature=np.full((40, 1), _T_AMB),
    )
    with pytest.raises(CalibrationError, match="rank-deficient"):
        fit_two_node(telemetry, telemetry, flat, T_amb=_T_AMB)


def test_fit_rejects_mismatched_clocks():
    servo = _servo()
    winding, housing, power = _synthetic_traces(servo, n=40)
    shifted = ThermalTelemetry(
        time=housing.time + 1.0,
        sample_rate_hz=housing.sample_rate_hz,
        servo_names=("j0",),
        temperature=housing.temperature,
    )
    with pytest.raises(CalibrationError, match="same clock"):
        fit_two_node(winding, shifted, power, T_amb=_T_AMB)


def test_validate_passes_on_a_self_consistent_trace_and_flags_a_wrong_one():
    servo = _servo()
    winding, housing, power = _synthetic_traces(servo, n=200)
    assert validate_two_node(winding, housing, power, servo, T_amb=_T_AMB).ok
    import dataclasses

    wrong = dataclasses.replace(servo, R_ha=1.0)
    report = validate_two_node(winding, housing, power, wrong, T_amb=_T_AMB)
    assert not report.ok
    assert any("housing residual" in w for w in report.warnings)
