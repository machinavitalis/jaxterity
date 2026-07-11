# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""HarmonicDrive — metadata, two-port relations, validate."""

from __future__ import annotations

import numpy as np
import pytest

from jaxterity.actuators import Actuator, HarmonicDrive
from jaxterity.data import TransmissionTelemetry

# --- metadata --------------------------------------------------------------


def test_is_actuator_subclass():
    assert issubclass(HarmonicDrive, Actuator)


def test_uncalibrated_constructs_with_neutral_defaults():
    hd = HarmonicDrive.uncalibrated()
    # N and backlash_v must be > 0; eta_f / eta_b in [0, 1].
    assert float(hd.N) > 0
    assert 0 <= float(hd.eta_f) <= 1
    assert 0 <= float(hd.eta_b) <= 1
    assert float(hd.backlash_v) > 0


def test_parameter_spec_lists_four_leaves():
    assert tuple(HarmonicDrive.uncalibrated().parameter_spec()) == (
        "N",
        "eta_f",
        "eta_b",
        "backlash_v",
    )


def test_sysid_protocol_describes_load_sweep():
    p = HarmonicDrive.uncalibrated().sysid_protocol
    assert p.name == "transmission-load-sweep"
    assert set(p.required_columns) == {"tau_motor", "v_motor", "tau_joint", "v_joint"}


# --- two-port relations ----------------------------------------------------


def test_kinematic_ratio_reduces_velocity():
    """v_joint = v_motor / N (no backlash, no efficiency in the velocity map)."""
    hd = HarmonicDrive(N=100.0, eta_f=0.9, eta_b=0.85, backlash_v=1e-3)
    pred = float(
        HarmonicDrive._torque_map(
            np.asarray(0.0),
            np.asarray(1.0),
            N=hd.N,
            eta_f=hd.eta_f,
            eta_b=hd.eta_b,
            backlash_v=hd.backlash_v,
            p_eps=hd._p_eps,
        )
    )
    # Zero motor torque → zero joint torque regardless of N or efficiency.
    assert pred == pytest.approx(0.0, abs=1e-9)


def test_torque_forward_efficiency_when_motor_drives_joint():
    """With positive τ_motor and positive v_motor (motor drives joint), η = eta_f."""
    hd = HarmonicDrive(N=100.0, eta_f=0.9, eta_b=0.5, backlash_v=1e-6)
    tau_m, v_m = 1.0, 1.0  # power into joint
    expected = 0.9 * 100.0 * 1.0  # eta_f · N · tau_motor (no backlash rolloff at v=1)
    pred = float(
        HarmonicDrive._torque_map(
            np.asarray(tau_m),
            np.asarray(v_m),
            N=hd.N,
            eta_f=hd.eta_f,
            eta_b=hd.eta_b,
            backlash_v=hd.backlash_v,
            p_eps=hd._p_eps,
        )
    )
    assert pred == pytest.approx(expected, rel=1e-3)


def test_torque_backward_efficiency_when_joint_drives_motor():
    """With τ_motor and v_motor opposite signs (joint drives motor), η = eta_b."""
    hd = HarmonicDrive(N=100.0, eta_f=0.9, eta_b=0.5, backlash_v=1e-6)
    tau_m, v_m = -1.0, 1.0  # negative power: joint pushes motor
    expected = 0.5 * 100.0 * (-1.0)  # eta_b · N · tau_motor
    pred = float(
        HarmonicDrive._torque_map(
            np.asarray(tau_m),
            np.asarray(v_m),
            N=hd.N,
            eta_f=hd.eta_f,
            eta_b=hd.eta_b,
            backlash_v=hd.backlash_v,
            p_eps=hd._p_eps,
        )
    )
    assert pred == pytest.approx(expected, rel=1e-3)


def test_backlash_attenuates_torque_near_zero_velocity():
    """At v_motor = 0 the tanh-smoothed backlash sends τ_joint → 0."""
    hd = HarmonicDrive(N=100.0, eta_f=0.9, eta_b=0.85, backlash_v=1.0)
    pred = float(
        HarmonicDrive._torque_map(
            np.asarray(1.0),
            np.asarray(0.0),  # full motor torque, no velocity
            N=hd.N,
            eta_f=hd.eta_f,
            eta_b=hd.eta_b,
            backlash_v=hd.backlash_v,
            p_eps=hd._p_eps,
        )
    )
    assert abs(pred) < 1e-9


# --- validate --------------------------------------------------------------


def _synthetic_transmission_telemetry(
    *,
    N: float,
    eta_f: float,
    eta_b: float,
    backlash_v: float = 1e-3,
    n: int = 200,
) -> TransmissionTelemetry:
    """Pure model: pick (τ_motor, v_motor) sweeps, compute (τ_joint, v_joint)."""
    rng = np.random.default_rng(0)
    t = np.arange(n, dtype=np.float64) / 200.0
    # Mix of motor-drives-joint and joint-drives-motor samples.
    tau_m = rng.uniform(-1.0, 1.0, n)
    v_m = rng.uniform(-10.0, 10.0, n)
    P_dir = tau_m * v_m
    eta_eff = 0.5 * (eta_f + eta_b) + 0.5 * (eta_f - eta_b) * np.tanh(P_dir / 1e-6)
    tau_j = eta_eff * N * tau_m * np.tanh(np.abs(v_m) / backlash_v)
    v_j = v_m / N
    return TransmissionTelemetry(
        time=t,
        sample_rate_hz=200.0,
        transmission_name="hd_0",
        tau_motor=tau_m,
        v_motor=v_m,
        tau_joint=tau_j,
        v_joint=v_j,
    )


def test_validate_on_matched_telemetry_is_ok():
    """Model on its own trace reports ok."""
    N, eta_f, eta_b = 100.0, 0.9, 0.85
    tel = _synthetic_transmission_telemetry(N=N, eta_f=eta_f, eta_b=eta_b)
    report = HarmonicDrive(N=N, eta_f=eta_f, eta_b=eta_b).validate(tel)
    assert report.ok, report.summary()


def test_validate_on_wrong_N_flags_warning():
    """A wrong gear ratio diverges both torque and velocity residuals."""
    tel = _synthetic_transmission_telemetry(N=100.0, eta_f=0.9, eta_b=0.85)
    report = HarmonicDrive(N=80.0, eta_f=0.9, eta_b=0.85).validate(tel)  # 20% off N
    assert not report.ok
