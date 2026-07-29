# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Actuation families — quasi-direct drive, stepper, pneumatic cylinder.

Each block is tested on the physics that made it worth a separate model: the
QDD's speed-collapsing torque envelope and its N²-reflected inertia, the
stepper's sinusoidal torque-angle curve and the pull-out cliff past its peak,
and the pneumatic cylinder's choked-flow fill and gas-spring behaviour. Forward
simulations run through a real Jaxonomy diagram so the blocks are exercised the
way a twin would use them.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import jaxonomy
import numpy as np
import pytest
from jaxonomy.library import Constant

from jaxterity.actuators import (
    PneumaticCylinder,
    QuasiDirectDrive,
    Stepper,
    chamber_pressure_rate,
    orifice_flow,
    pull_out_margin,
    qdd_torque_envelope,
    stepper_torque,
)
from jaxterity.actuators.pneumatic import R_AIR, T_REF
from jaxterity.data import (
    ActuatorStepTelemetry,
    PneumaticTelemetry,
    TorqueStepTelemetry,
)


def _simulate(block, inputs: dict[str, float], t_end: float, output: str):
    """Run one block with constant inputs; return the final value of ``output``."""
    builder = jaxonomy.DiagramBuilder()
    system = builder.add(block)
    names = [p.name for p in system.input_ports]
    for name, value in zip(names, [inputs[n] for n in names]):
        source = builder.add(Constant(value, name=f"c_{name}"))
        builder.connect(
            source.output_ports[0],
            system.input_ports[names.index(name)],
        )
    diagram = builder.build()
    port = {p.name: p for p in system.output_ports}[output]
    results = jaxonomy.simulate(
        diagram, diagram.create_context(), (0.0, t_end), recorded_signals={output: port}
    )
    return float(results.outputs[output][-1])


# --- quasi-direct drive ----------------------------------------------------


def test_envelope_is_current_limited_at_standstill():
    stall = qdd_torque_envelope(
        0.0, N=6.0, K_t=0.1, K_e=0.1, R=0.5, V_bus=24.0, i_max=30.0
    )
    assert float(stall) == pytest.approx(6.0 * 0.1 * 30.0)


def test_envelope_collapses_with_speed_and_never_goes_negative():
    kwargs = dict(N=6.0, K_t=0.1, K_e=0.1, R=0.5, V_bus=24.0, i_max=30.0)
    slow = float(qdd_torque_envelope(5.0, **kwargs))
    fast = float(qdd_torque_envelope(35.0, **kwargs))
    beyond = float(qdd_torque_envelope(100.0, **kwargs))
    assert slow >= fast > 0.0
    assert beyond == 0.0  # past the no-load speed there is nothing left


def test_no_load_speed_is_where_the_envelope_reaches_zero():
    qdd = QuasiDirectDrive(N=6.0, K_e=0.1, V_bus=24.0)
    speed = float(qdd.no_load_speed)
    assert float(
        qdd_torque_envelope(
            speed, N=6.0, K_t=0.1, K_e=0.1, R=0.5, V_bus=24.0, i_max=30.0
        )
    ) == pytest.approx(0.0, abs=1e-12)


def test_reflected_inertia_scales_with_the_square_of_the_ratio():
    low = QuasiDirectDrive(N=6.0, J_rotor=5e-5, J_load=0.0)
    high = QuasiDirectDrive(N=60.0, J_rotor=5e-5, J_load=0.0)
    assert float(high.reflected_inertia) == pytest.approx(
        100.0 * float(low.reflected_inertia)
    )
    # The QDD claim in one assertion: at 6:1 the rotor is a perturbation on a
    # 1e-2 kg m^2 leg link, at 60:1 it dominates it by an order of magnitude.
    link = 1.0e-2
    assert float(low.reflected_inertia) < link < float(high.reflected_inertia)


def test_total_inertia_includes_the_reflected_rotor():
    qdd = QuasiDirectDrive(N=6.0, J_rotor=5e-5, J_load=1e-3)
    assert float(qdd.total_inertia) == pytest.approx(1e-3 + 36 * 5e-5)


@pytest.mark.slow
def test_free_acceleration_follows_the_commanded_torque():
    # tau/J for one second, from rest, with no losses.
    qdd = QuasiDirectDrive(N=6.0, J_rotor=0.0, J_load=1e-2, b=0.0, F_c=0.0, i_max=30.0)
    velocity = _simulate(qdd, {"tau_ref": 0.5, "tau_ext": 0.0}, t_end=0.5, output="v")
    assert velocity == pytest.approx(0.5 / 1e-2 * 0.5, rel=1e-2)


@pytest.mark.slow
def test_a_command_beyond_the_envelope_is_saturated_not_delivered():
    qdd = QuasiDirectDrive(N=6.0, K_t=0.1, i_max=10.0, J_load=1e-2, J_rotor=0.0)
    delivered = _simulate(
        qdd, {"tau_ref": 1000.0, "tau_ext": 0.0}, t_end=1e-3, output="tau"
    )
    assert delivered <= 6.0 * 0.1 * 10.0 * 0.95 + 1e-9


def test_qdd_validate_flags_a_wrong_inertia():
    qdd = QuasiDirectDrive(N=6.0, J_rotor=0.0, J_load=1e-2, b=0.0, F_c=0.0)
    t = np.linspace(0.0, 0.5, 200)
    command = np.full_like(t, 0.5)
    position = 0.5 * (0.5 / 1e-2) * t**2  # the true response
    telemetry = TorqueStepTelemetry(
        time=t,
        sample_rate_hz=400.0,
        actuator_name="j0",
        tau_command=command,
        position=position,
    )
    assert qdd.validate(telemetry).ok
    heavy = QuasiDirectDrive(N=6.0, J_rotor=0.0, J_load=4e-2, b=0.0, F_c=0.0)
    assert not heavy.validate(telemetry).ok


def test_uncalibrated_qdd_has_no_assumed_loss():
    qdd = QuasiDirectDrive.uncalibrated()
    assert float(qdd.eta) == 1.0 and float(qdd.b) == 0.0 and float(qdd.F_c) == 0.0


# --- stepper ---------------------------------------------------------------


def test_torque_peaks_a_quarter_electrical_cycle_from_the_command():
    teeth = 50
    quarter = math.pi / (2 * teeth)
    peak = float(stepper_torque(0.0, 0.0, quarter, T_hold=2.0, rotor_teeth=teeth))
    assert peak == pytest.approx(2.0, rel=1e-9)
    # Past the peak the torque FALLS — the instability that defines pull-out.
    beyond = float(
        stepper_torque(0.0, 0.0, 1.8 * quarter, T_hold=2.0, rotor_teeth=teeth)
    )
    assert 0.0 < beyond < peak


def test_zero_error_produces_zero_torque():
    assert float(stepper_torque(0.13, 0.0, 0.13, T_hold=2.0)) == pytest.approx(0.0)


def test_pull_out_margin_hits_zero_at_the_torque_peak():
    teeth = 50
    quarter = math.pi / (2 * teeth)
    assert float(pull_out_margin(0.0, 0.0, rotor_teeth=teeth)) == pytest.approx(1.0)
    assert float(pull_out_margin(0.0, quarter, rotor_teeth=teeth)) == pytest.approx(
        0.0, abs=1e-12
    )
    assert float(pull_out_margin(0.0, 2 * quarter, rotor_teeth=teeth)) < 0.0


def test_detent_torque_acts_with_the_driver_unpowered():
    # T_hold = 0 (coils off): what is left is the cogging ripple.
    tau = float(
        stepper_torque(0.001, 0.0, 0.001, T_hold=0.0, T_detent=0.05, rotor_teeth=50)
    )
    assert tau != 0.0
    assert abs(tau) <= 0.05 + 1e-12


def test_detent_ripple_has_four_periods_per_electrical_cycle():
    teeth = 50
    # One electrical cycle, endpoint excluded so the wrap-around crossing is
    # not counted twice.
    thetas = np.linspace(0.0, 2 * np.pi / teeth, 400, endpoint=False)
    tau = np.asarray(
        stepper_torque(
            jnp.asarray(thetas),
            0.0,
            jnp.asarray(thetas),  # zero position error: detent only
            T_hold=1.0,
            T_detent=0.05,
            rotor_teeth=teeth,
        )
    )
    sign_changes = int(np.count_nonzero(np.diff(np.sign(tau)) != 0))
    assert sign_changes == 8  # four full periods


def test_holding_torque_droops_with_speed():
    kwargs = dict(T_hold=2.0, rotor_teeth=50, omega_corner=500.0)
    quarter = math.pi / (2 * 50)
    slow = float(stepper_torque(0.0, 0.0, quarter, **kwargs))
    fast = float(stepper_torque(0.0, 10.0, quarter, **kwargs))
    assert fast < slow
    # At the corner (electrical omega = omega_corner) it is T_hold/sqrt(2).
    at_corner = float(stepper_torque(0.0, 500.0 / 50, quarter, **kwargs))
    assert at_corner == pytest.approx(2.0 / math.sqrt(2.0), rel=1e-6)


def test_microstepping_quantizes_the_commanded_angle():
    microstep = 2 * math.pi / (200 * 16)
    exact = float(
        stepper_torque(0.0, 0.0, 1.4 * microstep, T_hold=2.0, microstep=microstep)
    )
    snapped = float(
        stepper_torque(0.0, 0.0, 1.0 * microstep, T_hold=2.0, microstep=microstep)
    )
    assert exact == pytest.approx(snapped)


def test_stepper_validate_reports_lost_synchronism_in_steps():
    motor = Stepper(T_hold=1.0, rotor_teeth=50)
    t = np.linspace(0.0, 1.0, 100)
    command = np.linspace(0.0, 0.5, 100)
    telemetry = ActuatorStepTelemetry(
        time=t,
        sample_rate_hz=100.0,
        actuator_name="s0",
        setpoint=command,
        position=command,
    )
    assert motor.validate(telemetry).ok
    slipped = ActuatorStepTelemetry(
        time=t,
        sample_rate_hz=100.0,
        actuator_name="s0",
        setpoint=command,
        position=command - 3 * motor.full_step_rad,
    )
    report = motor.validate(slipped)
    assert not report.ok
    assert "3.0 full step(s) behind" in report.warnings[0]


def test_full_step_of_a_1_8_degree_motor():
    assert math.degrees(Stepper(rotor_teeth=50).full_step_rad) == pytest.approx(1.8)


# --- pneumatic -------------------------------------------------------------


def test_choked_flow_is_proportional_to_upstream_pressure():
    m1 = float(orifice_flow(6e5, 1e5, conductance=1e-8))
    m2 = float(orifice_flow(12e5, 1e5, conductance=1e-8))
    assert m2 / m1 == pytest.approx(2.0, rel=1e-6)


def test_flow_reverses_sign_when_the_pressure_gradient_does():
    forward = float(orifice_flow(6e5, 1e5, conductance=1e-8))
    backward = float(orifice_flow(1e5, 6e5, conductance=1e-8))
    assert backward == pytest.approx(-forward)


def test_flow_vanishes_at_equal_pressures():
    assert float(orifice_flow(3e5, 3e5, conductance=1e-8)) == pytest.approx(0.0)


def test_subsonic_branch_reduces_flow_below_the_choked_value():
    choked = float(orifice_flow(6e5, 1e5, conductance=1e-8, b_crit=0.3))
    near_equal = float(orifice_flow(6e5, 5.5e5, conductance=1e-8, b_crit=0.3))
    assert 0.0 < near_equal < choked


def test_orifice_flow_is_differentiable_across_the_branch_switch():
    grad = jax.grad(lambda p: orifice_flow(6e5, p, conductance=1e-8, b_crit=0.3))
    for downstream in (1e5, 1.8e5, 2.0e5, 3e5, 5e5):
        assert np.isfinite(float(grad(downstream)))


def test_pressure_rate_follows_the_isothermal_gas_law():
    # No piston motion: dp/dt = R*T*mdot/V.
    rate = chamber_pressure_rate(2e5, 1e-4, 0.0, 1e-5, temperature=T_REF)
    assert float(rate) == pytest.approx(R_AIR * T_REF * 1e-5 / 1e-4)


def test_expanding_chamber_loses_pressure_with_no_flow():
    assert float(chamber_pressure_rate(2e5, 1e-4, 1e-5, 0.0)) < 0.0


@pytest.mark.slow
def test_blocked_fill_pressurizes_toward_supply():
    cylinder = PneumaticCylinder(
        conductance=1e-8, p_supply=6e5, p0_a=1.01325e5, p0_b=1.01325e5
    )
    partial = _simulate(
        cylinder, {"valve": 1.0, "x": 0.05, "v": 0.0}, t_end=0.05, output="p_a"
    )
    settled = _simulate(
        cylinder, {"valve": 1.0, "x": 0.05, "v": 0.0}, t_end=2.0, output="p_a"
    )
    # Mid-fill it is strictly between exhaust and supply; given time it settles
    # at supply and stops (no overshoot past the source).
    assert 1.02e5 < partial < settled
    assert settled == pytest.approx(6e5, rel=1e-4)


@pytest.mark.slow
def test_venting_a_charged_chamber_drops_it_toward_exhaust():
    cylinder = PneumaticCylinder(conductance=1e-7, p0_a=6e5, p0_b=1.01325e5)
    pressure = _simulate(
        cylinder, {"valve": -1.0, "x": 0.05, "v": 0.0}, t_end=2.0, output="p_a"
    )
    assert pressure < 6e5


def test_rod_side_asymmetry_makes_retract_force_smaller():
    cylinder = PneumaticCylinder(area_a=1.13e-3, area_b=9.0e-4)
    # Equal pressures on both sides still push the rod out: the annulus is
    # smaller. This is the asymmetry a "force = pressure * area" model loses.
    force = cylinder._force_out(
        0.0,
        type("S", (), {"continuous_state": jnp.asarray([3e5, 3e5])})(),
        0.0,
        0.05,
        0.0,
        area_a=1.13e-3,
        area_b=9.0e-4,
        f_friction=0.0,
    )
    assert float(force) == pytest.approx(3e5 * (1.13e-3 - 9.0e-4))


def test_pneumatic_validate_flags_a_wrong_conductance():
    # Synthesize a blocked-fill trace from the model itself, then score it with
    # a cylinder whose valve is ten times too small.
    truth = PneumaticCylinder(conductance=1e-8, p_supply=6e5)
    t = np.linspace(0.0, 1.0, 500)
    dt = float(t[1] - t[0])
    p = np.array([1.01325e5, 1.01325e5])
    trace = np.empty((t.size, 2))
    trace[0] = p
    params = {
        name: float(getattr(truth, name))
        for name in (
            "area_a",
            "area_b",
            "stroke",
            "dead_volume_a",
            "dead_volume_b",
            "conductance",
            "b_crit",
            "p_supply",
            "p_exhaust",
            "temperature",
        )
    }
    for i in range(1, t.size):
        v_a, v_b = truth._volumes(0.05, params)
        m_a, m_b = truth._flows(p[0], p[1], 1.0, params)
        p = p + dt * np.array(
            [
                float(chamber_pressure_rate(p[0], v_a, 0.0, m_a)),
                float(chamber_pressure_rate(p[1], v_b, 0.0, m_b)),
            ]
        )
        trace[i] = p
    telemetry = PneumaticTelemetry(
        time=t,
        sample_rate_hz=500.0,
        actuator_name="c0",
        valve_command=np.ones_like(t),
        pressure_a=trace[:, 0],
        pressure_b=trace[:, 1],
        position=np.full_like(t, 0.05),
    )
    assert truth.validate(telemetry).ok
    wrong = PneumaticCylinder(conductance=1e-9, p_supply=6e5)
    assert not wrong.validate(telemetry).ok


def test_uncalibrated_cylinder_has_no_assumed_friction():
    assert float(PneumaticCylinder.uncalibrated().f_friction) == 0.0
