# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Transmission family — positional backlash, gearbox, belt, lead screw.

Each block is checked on three things: the physics it exists to capture
(lost motion as a *position* effect; efficiency that differs by power
direction; belt compliance; screw non-backdrivability), the recovery of its
parameters from a synthetic trace by its fitter, and the honesty of the
fitter's refusals — a trace missing the quadrant or the encoder a parameter
needs must raise, not return a number.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxterity.actuators import (
    BeltDrive,
    Gearbox,
    LeadScrew,
    PositionalBacklash,
    backlash_torque,
    belt_force,
    belt_resonance_hz,
    deadzone,
    engagement,
    gearbox_torque,
    is_backdrivable,
    lost_motion,
    screw_efficiency,
    screw_force,
)
from jaxterity.actuators.backlash import _fit_backlash
from jaxterity.actuators.belt import _fit_belt
from jaxterity.actuators.gearbox import _fit_gearbox
from jaxterity.actuators.lead_screw import _fit_lead_screw
from jaxterity.data import (
    BacklashTelemetry,
    LinearTransmissionTelemetry,
    TransmissionTelemetry,
)
from jaxterity.errors import CalibrationError
from jaxterity.testing import tolerances as tol

_TAU_TOL = tol.atol("actuator_torque", "rk4", "float64") * 1e4


# --- positional backlash ---------------------------------------------------


def test_deadzone_is_flat_inside_the_gap_and_offset_outside():
    play = 0.02
    assert abs(float(deadzone(0.0, play))) < 1e-12
    # Well outside: x - play*sign(x).
    assert float(deadzone(1.0, play)) == pytest.approx(1.0 - play, abs=1e-9)
    assert float(deadzone(-1.0, play)) == pytest.approx(-1.0 + play, abs=1e-9)


def test_deadzone_with_zero_play_is_the_identity():
    assert float(deadzone(0.37, 0.0)) == pytest.approx(0.37)


def test_engagement_runs_from_free_to_loaded():
    play = 0.02
    assert float(engagement(0.0, play)) == pytest.approx(0.0)
    assert float(engagement(10 * play, play)) == pytest.approx(1.0, abs=1e-6)


def test_backlash_is_a_position_effect_not_a_speed_effect():
    # The claim that separates this block from the velocity-domain proxy: the
    # transmitted torque at a given deflection does not depend on how fast you
    # got there.
    slow = backlash_torque(0.05, 1e-4, play=0.02, k_mesh=100.0, d_mesh=0.0)
    fast = backlash_torque(0.05, 10.0, play=0.02, k_mesh=100.0, d_mesh=0.0)
    assert float(slow) == pytest.approx(float(fast))


def test_mesh_damping_only_acts_when_engaged():
    free = backlash_torque(0.0, 5.0, play=0.02, k_mesh=100.0, d_mesh=3.0)
    loaded = backlash_torque(0.5, 5.0, play=0.02, k_mesh=100.0, d_mesh=3.0)
    assert abs(float(free)) < 1e-9
    assert float(loaded) > 100.0 * (0.5 - 0.02)  # spring plus damping


def test_backlash_torque_is_differentiable_through_the_gap():
    grad = jax.grad(
        lambda d: backlash_torque(d, 0.0, play=0.02, k_mesh=100.0, d_mesh=0.0)
    )
    for deflection in (-0.05, -0.02, 0.0, 0.02, 0.05):
        assert np.isfinite(float(grad(deflection)))


def _backlash_trace(play=0.02, k_mesh=120.0, n=400):
    """A slow triangular reversal under a compliant load."""
    t = np.linspace(0.0, 20.0, n)
    theta_motor = 0.15 * np.sin(2 * np.pi * t / 20.0)
    # Joint clamped against a rigid stop: every bit of motor motion is either
    # taken up as slack or turned into mesh deflection, which is the cleanest
    # setting for reading the hysteresis loop.
    theta_joint = np.zeros_like(theta_motor)
    tau = np.asarray(
        backlash_torque(
            jnp.asarray(theta_motor - theta_joint),
            jnp.zeros_like(theta_motor),
            play=play,
            k_mesh=k_mesh,
            d_mesh=0.0,
        )
    )
    return BacklashTelemetry(
        time=t,
        sample_rate_hz=n / 20.0,
        transmission_name="j0",
        theta_motor=theta_motor,
        theta_joint=theta_joint,
        tau_joint=tau,
    )


def test_fit_recovers_play_and_stiffness():
    telemetry = _backlash_trace(play=0.02, k_mesh=120.0)
    fitted = _fit_backlash(telemetry)
    assert fitted["play"] == pytest.approx(0.02, rel=5e-2)
    assert fitted["k_mesh"] == pytest.approx(120.0, rel=5e-2)


def test_lost_motion_lands_within_a_factor_of_two_of_twice_the_play():
    # The documented claim for a stiffness-free cross-check: right order of
    # magnitude at any sane gate, not a precise play (that is _fit_backlash).
    play = 0.02
    telemetry = _backlash_trace(play=play)
    width = lost_motion(telemetry)
    assert play <= width <= 4 * play


def test_lost_motion_is_monotone_in_the_torque_gate():
    telemetry = _backlash_trace(play=0.02)
    widths = [
        lost_motion(telemetry, engaged_fraction=f) for f in (0.01, 0.05, 0.1, 0.2)
    ]
    assert widths == sorted(widths)


def test_fit_refuses_a_single_direction_trace():
    telemetry = _backlash_trace()
    one_sided = BacklashTelemetry(
        time=telemetry.time,
        sample_rate_hz=telemetry.sample_rate_hz,
        transmission_name="j0",
        theta_motor=np.abs(telemetry.theta_motor),
        theta_joint=telemetry.theta_joint,
        tau_joint=np.asarray(
            backlash_torque(
                jnp.asarray(np.abs(telemetry.theta_motor)),
                jnp.zeros_like(telemetry.theta_motor),
                play=0.02,
                k_mesh=120.0,
                d_mesh=0.0,
            )
        ),
    )
    with pytest.raises(CalibrationError, match="BOTH sides"):
        _fit_backlash(one_sided)


def test_lost_motion_refuses_a_never_unloaded_trace():
    telemetry = _backlash_trace()
    loaded = BacklashTelemetry(
        time=telemetry.time,
        sample_rate_hz=telemetry.sample_rate_hz,
        transmission_name="j0",
        theta_motor=telemetry.theta_motor + 10.0,
        theta_joint=telemetry.theta_joint,
        tau_joint=np.full_like(telemetry.tau_joint, 50.0),
    )
    with pytest.raises(CalibrationError, match="no unloaded samples"):
        lost_motion(loaded)


def test_backlash_block_validates_its_own_trace():
    telemetry = _backlash_trace(play=0.02, k_mesh=120.0)
    block = PositionalBacklash(play=0.02, k_mesh=120.0)
    assert block.validate(telemetry).ok
    wrong = PositionalBacklash(play=0.10, k_mesh=120.0)
    assert not wrong.validate(telemetry).ok


def test_uncalibrated_backlash_assumes_no_play():
    assert float(PositionalBacklash.uncalibrated().play) == 0.0


# --- gearbox ---------------------------------------------------------------


def test_forward_quadrant_applies_the_forward_efficiency():
    tau = gearbox_torque(1.0, 10.0, N=30.0, eta_f=0.8, eta_b=0.6, tau_loss=0.0, b=0.0)
    assert float(tau) == pytest.approx(30.0 * 0.8 * 1.0, rel=1e-6)


def test_regenerative_quadrant_costs_the_load_more_torque():
    # Back-driving a lossy stage: the joint must supply N*tau/eta_b, i.e. MORE
    # than the ideal N*tau, not less.
    tau = gearbox_torque(-1.0, 10.0, N=30.0, eta_f=0.8, eta_b=0.6, tau_loss=0.0, b=0.0)
    assert float(tau) == pytest.approx(-30.0 / 0.6, rel=1e-3)


def test_load_independent_loss_costs_torque_even_unloaded():
    tau = gearbox_torque(0.0, 10.0, N=30.0, eta_f=1.0, eta_b=1.0, tau_loss=0.2, b=0.0)
    assert float(tau) == pytest.approx(-30.0 * 0.2, rel=1e-3)


def _gearbox_trace(N=30.0, eta_f=0.8, eta_b=0.6, tau_loss=0.15, b=0.02, n=400):
    rng = np.random.default_rng(0)
    v_motor = np.concatenate(
        [rng.uniform(2.0, 20.0, n // 2), rng.uniform(2.0, 20.0, n // 2)]
    )
    tau_motor = np.concatenate(
        # Driving quadrant, then a back-driven (overhauling) segment.
        [rng.uniform(0.4, 3.0, n // 2), rng.uniform(-3.0, -0.4, n // 2)]
    )
    tau_joint = np.asarray(
        gearbox_torque(
            jnp.asarray(tau_motor),
            jnp.asarray(v_motor),
            N=N,
            eta_f=eta_f,
            eta_b=eta_b,
            tau_loss=tau_loss,
            b=b,
        )
    )
    return TransmissionTelemetry(
        time=np.arange(n) * 0.005,
        sample_rate_hz=200.0,
        transmission_name="g0",
        tau_motor=tau_motor,
        v_motor=v_motor,
        tau_joint=tau_joint,
        v_joint=v_motor / N,
    )


def test_fit_recovers_both_efficiencies_and_the_losses():
    telemetry = _gearbox_trace()
    fitted = _fit_gearbox(telemetry)
    assert fitted["N"] == pytest.approx(30.0, rel=1e-6)
    assert fitted["eta_f"] == pytest.approx(0.8, rel=1e-3)
    assert fitted["eta_b"] == pytest.approx(0.6, rel=1e-3)
    assert fitted["tau_loss"] == pytest.approx(0.15, rel=1e-2)
    assert fitted["b"] == pytest.approx(0.02, rel=1e-2)


def test_fit_refuses_a_single_quadrant_trace():
    full = _gearbox_trace()
    half = TransmissionTelemetry(
        time=full.time[:200],
        sample_rate_hz=200.0,
        transmission_name="g0",
        tau_motor=full.tau_motor[:200],
        v_motor=full.v_motor[:200],
        tau_joint=full.tau_joint[:200],
        v_joint=full.v_joint[:200],
    )
    with pytest.raises(CalibrationError, match="regenerative quadrant"):
        _fit_gearbox(half)


def test_gearbox_validates_its_own_trace():
    telemetry = _gearbox_trace()
    box = Gearbox(N=30.0, eta_f=0.8, eta_b=0.6, tau_loss=0.15, b=0.02)
    assert box.validate(telemetry).ok
    assert not Gearbox(N=30.0, eta_f=0.5, eta_b=0.6).validate(telemetry).ok


def test_uncalibrated_gearbox_is_lossless():
    box = Gearbox.uncalibrated()
    assert float(box.eta_f) == 1.0 and float(box.eta_b) == 1.0
    assert float(box.tau_loss) == 0.0


# --- belt ------------------------------------------------------------------


def test_belt_ratio_is_the_pulley_ratio():
    belt = BeltDrive(r_motor=0.01, r_load=0.03)
    assert float(belt.ratio) == pytest.approx(3.0)


def test_belt_torque_grows_with_stretch_and_saturates_at_slip():
    def tau_load(stretch_m):
        return 0.02 * float(
            belt_force(stretch_m, 0.0, k_belt=1e5, d_belt=0.0, f_slip=50.0)
        )

    assert tau_load(1e-5) < tau_load(1e-4)
    # Far past the slip force, the transmitted torque tops out at r_load*f_slip.
    assert tau_load(1.0) == pytest.approx(0.02 * 50.0, rel=1e-6)


def test_belt_resonance_matches_the_closed_form():
    assert float(belt_resonance_hz(2.0e5, 0.02, 4e-4)) == pytest.approx(
        0.02 * np.sqrt(2.0e5 / 4e-4) / (2 * np.pi)
    )


def test_stiffer_belt_means_higher_resonance():
    soft = float(belt_resonance_hz(1.0e5, 0.02, 4e-4))
    stiff = float(belt_resonance_hz(4.0e5, 0.02, 4e-4))
    assert stiff == pytest.approx(2.0 * soft, rel=1e-9)


def _belt_trace(ratio=2.0, n=200):
    v_load = np.linspace(0.5, 5.0, n)
    tau_motor = np.linspace(0.1, 1.0, n)
    return TransmissionTelemetry(
        time=np.arange(n) * 0.005,
        sample_rate_hz=200.0,
        transmission_name="b0",
        tau_motor=tau_motor,
        v_motor=v_load * ratio,
        tau_joint=tau_motor * ratio,
        v_joint=v_load,
    )


def test_fit_belt_recovers_the_ratio():
    fitted = _fit_belt(_belt_trace(ratio=2.5))
    assert fitted["ratio"] == pytest.approx(2.5, rel=1e-9)
    assert fitted["torque_ratio"] == pytest.approx(2.5, rel=1e-9)


def test_belt_validate_flags_a_wrong_pulley_pair():
    telemetry = _belt_trace(ratio=2.0)
    assert BeltDrive(r_motor=0.01, r_load=0.02).validate(telemetry).ok
    assert not BeltDrive(r_motor=0.01, r_load=0.05).validate(telemetry).ok


# --- lead screw ------------------------------------------------------------


def test_screw_kinematics_convert_revolutions_to_lead():
    screw = LeadScrew(lead=0.008)
    v_load = screw._v_load_out(0.0, None, 0.0, 2 * np.pi, lead=0.008)
    assert float(v_load) == pytest.approx(0.008, rel=1e-9)


def test_fine_pitch_screw_is_not_backdrivable():
    # tan(lambda) < mu: the classic self-locking condition.
    assert not is_backdrivable(lead=0.002, pitch_diameter=0.012, mu=0.15)
    assert is_backdrivable(lead=0.010, pitch_diameter=0.012, mu=0.15)


def test_self_locking_screw_reports_a_non_positive_backward_efficiency():
    efficiency = screw_efficiency(lead=0.002, pitch_diameter=0.012, mu=0.15)
    assert efficiency["eta_b"] <= 0.0
    assert 0.0 < efficiency["eta_f"] < 1.0


def test_ball_screw_is_efficient_in_both_directions():
    efficiency = screw_efficiency(lead=0.005, pitch_diameter=0.012, mu=0.01)
    assert efficiency["eta_f"] > 0.9 and efficiency["eta_b"] > 0.9


def test_from_geometry_floors_a_self_locking_screw():
    screw = LeadScrew.from_geometry(lead=0.002, pitch_diameter=0.012, mu=0.15)
    assert float(screw.eta_b) > 0.0
    # Back-driving it takes a huge load force: the eta_b -> 0+ limit.
    force = screw_force(-0.1, 5.0, lead=0.002, eta_f=0.4, eta_b=float(screw.eta_b))
    assert abs(float(force)) > 1e4


def _screw_trace(lead=0.008, eta_f=0.5, eta_b=0.3, tau_loss=0.05, n=400):
    rng = np.random.default_rng(1)
    v_motor = np.concatenate(
        [rng.uniform(1.0, 20.0, n // 2), rng.uniform(1.0, 20.0, n // 2)]
    )
    tau_motor = np.concatenate(
        [rng.uniform(0.2, 2.0, n // 2), rng.uniform(-2.0, -0.2, n // 2)]
    )
    force = np.asarray(
        screw_force(
            jnp.asarray(tau_motor),
            jnp.asarray(v_motor),
            lead=lead,
            eta_f=eta_f,
            eta_b=eta_b,
            tau_loss=tau_loss,
        )
    )
    return LinearTransmissionTelemetry(
        time=np.arange(n) * 0.005,
        sample_rate_hz=200.0,
        transmission_name="s0",
        tau_motor=tau_motor,
        v_motor=v_motor,
        force_load=force,
        v_load=v_motor * lead / (2 * np.pi),
    )


def test_fit_screw_recovers_both_efficiencies_with_a_known_lead():
    telemetry = _screw_trace()
    fitted = _fit_lead_screw(telemetry, lead=0.008)
    assert fitted["eta_f"] == pytest.approx(0.5, rel=1e-3)
    assert fitted["eta_b"] == pytest.approx(0.3, rel=1e-3)
    assert fitted["tau_loss"] == pytest.approx(0.05, rel=1e-2)


def test_fit_screw_refuses_a_missing_quadrant():
    full = _screw_trace()
    driving_only = LinearTransmissionTelemetry(
        time=full.time[:200],
        sample_rate_hz=200.0,
        transmission_name="s0",
        tau_motor=full.tau_motor[:200],
        v_motor=full.v_motor[:200],
        force_load=full.force_load[:200],
        v_load=full.v_load[:200],
    )
    with pytest.raises(CalibrationError, match="regenerative quadrant"):
        _fit_lead_screw(driving_only, lead=0.008)


def test_screw_validate_flags_a_wrong_lead():
    telemetry = _screw_trace()
    good = LeadScrew(lead=0.008, eta_f=0.5, eta_b=0.3, tau_loss=0.05)
    assert good.validate(telemetry).ok
    assert not LeadScrew(lead=0.004, eta_f=0.5, eta_b=0.3).validate(telemetry).ok
