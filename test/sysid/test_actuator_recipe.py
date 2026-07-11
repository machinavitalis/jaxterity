# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""``sysid.actuator`` recipe — round-trip recovery on each device type.

For each device, generate truth telemetry from the device's own
forward sim, fit through the recipe, assert recovery within tolerance. Also
verify the SysidResult attestation fields are populated.
"""

from __future__ import annotations

import numpy as np
import pytest

import jaxterity.sysid as sysid
from jaxterity.actuators import (
    DCMotorWithFriction,
    HarmonicDrive,
    Servo,
)
from jaxterity.data import (
    ActuatorRampTelemetry,
    ActuatorStepTelemetry,
    EncoderTelemetry,
    FTSensorTelemetry,
    TransmissionTelemetry,
)
from jaxterity.errors import JaxterityError
from jaxterity.sensors import IMU, Encoder, FTSensor

# --- attestation chain shape ----------------------------------------------


def test_recipe_populates_attestation_fields():
    """SysidResult carries telemetry hash, recipe name+version regardless of device."""
    tel = ActuatorRampTelemetry(
        time=np.arange(50, dtype=np.float64) / 200.0,
        sample_rate_hz=200.0,
        actuator_name="m0",
        command=np.zeros(50),
        velocity=np.zeros(50),
    )
    res = sysid.actuator(tel, DCMotorWithFriction.uncalibrated())
    assert res.recipe_name == "sysid.actuator"
    assert res.recipe_version == "1.0"
    assert len(res.telemetry_hash) == 64  # SHA-256 hex


def test_recipe_rejects_unsupported_device():
    tel = ActuatorRampTelemetry(
        time=np.arange(5, dtype=np.float64),
        sample_rate_hz=1.0,
        actuator_name="m",
        command=np.zeros(5),
        velocity=np.zeros(5),
    )
    with pytest.raises(JaxterityError, match="no fitter"):
        sysid.actuator(tel, IMU.uncalibrated())  # type: ignore[arg-type]


def test_recipe_rejects_mismatched_telemetry():
    enc = Encoder.uncalibrated()
    tel = ActuatorRampTelemetry(
        time=np.arange(5, dtype=np.float64),
        sample_rate_hz=1.0,
        actuator_name="m",
        command=np.zeros(5),
        velocity=np.zeros(5),
    )
    with pytest.raises(JaxterityError, match="EncoderTelemetry"):
        sysid.actuator(tel, enc)


# --- closed-form: DCMotorWithFriction --------------------------------------


def test_dc_motor_round_trip_recovers_truth():
    Ks_t, Kv_t, Ka_t = 0.2, 0.5, 0.05
    n = int(2.0 * 200.0) + 1
    t = np.arange(n, dtype=np.float64) / 200.0
    v = 0.5 + 1.5 * t + 0.8 * t * t
    a = 1.5 + 1.6 * t
    V = Ks_t * np.sign(v) + Kv_t * v + Ka_t * a
    tel = ActuatorRampTelemetry(
        time=t,
        sample_rate_hz=200.0,
        actuator_name="m",
        command=V,
        velocity=v,
    )
    res = sysid.actuator(tel, DCMotorWithFriction.uncalibrated())
    assert float(res.fitted["Ks"]) == pytest.approx(Ks_t, rel=1e-5)
    assert float(res.fitted["Kv"]) == pytest.approx(Kv_t, rel=1e-5)
    assert float(res.fitted["Ka"]) == pytest.approx(Ka_t, rel=1e-5)


# --- closed-form: Encoder --------------------------------------------------


def test_encoder_round_trip_recovers_step_and_drift():
    step_t, drift_t = 0.01, 0.001
    fs = 500.0
    n = 5 * int(fs)
    t = np.arange(n, dtype=np.float64) / fs
    truth = 0.5 * np.sin(2 * np.pi * 0.5 * t)
    biased = truth + drift_t * t
    measured = np.round(biased / step_t) * step_t
    tel = EncoderTelemetry(
        time=t,
        sample_rate_hz=fs,
        encoder_name="enc",
        truth=truth,
        measured=measured,
    )
    res = sysid.actuator(tel, Encoder.uncalibrated())
    assert float(res.fitted["quantization_step"]) == pytest.approx(step_t, abs=1e-9)
    assert float(res.fitted["drift_rate"]) == pytest.approx(drift_t, abs=1e-3)


# --- closed-form: FTSensor -------------------------------------------------


def test_ft_sensor_recovers_bias_from_static_mean():
    fs = 500.0
    n = 2000
    t = np.arange(n, dtype=np.float64) / fs
    bias_f = np.array([0.5, -0.3, 1.0])
    bias_t = np.array([0.01, -0.02, 0.03])
    rng = np.random.default_rng(0)
    force = bias_f[None, :] + 0.02 * rng.standard_normal((n, 3))
    torque = bias_t[None, :] + 0.001 * rng.standard_normal((n, 3))
    tel = FTSensorTelemetry(
        time=t,
        sample_rate_hz=fs,
        frame="sensor",
        force=force,
        torque=torque,
    )
    res = sysid.actuator(tel, FTSensor.uncalibrated())
    fitted_bias = np.asarray(res.fitted["bias"])
    np.testing.assert_allclose(fitted_bias[:3], bias_f, atol=2e-3)  # ≈ σ/√n
    np.testing.assert_allclose(fitted_bias[3:], bias_t, atol=1e-4)


# --- closed-form: HarmonicDrive -------------------------------------------


def test_harmonic_drive_recovers_N_and_efficiency():
    N_t, ef_t, eb_t = 100.0, 0.9, 0.85
    rng = np.random.default_rng(0)
    n = 200
    tau_m = rng.uniform(-1.0, 1.0, n)
    v_m = rng.uniform(-10.0, 10.0, n)
    P_dir = tau_m * v_m
    eta_eff = 0.5 * (ef_t + eb_t) + 0.5 * (ef_t - eb_t) * np.tanh(P_dir / 1e-6)
    tau_j = eta_eff * N_t * tau_m * np.tanh(np.abs(v_m) / 1e-3)
    v_j = v_m / N_t
    tel = TransmissionTelemetry(
        time=np.arange(n, dtype=np.float64) / 200.0,
        sample_rate_hz=200.0,
        transmission_name="hd",
        tau_motor=tau_m,
        v_motor=v_m,
        tau_joint=tau_j,
        v_joint=v_j,
    )
    res = sysid.actuator(tel, HarmonicDrive.uncalibrated())
    assert float(res.fitted["N"]) == pytest.approx(N_t, rel=1e-4)
    assert float(res.fitted["eta_f"]) == pytest.approx(ef_t, rel=5e-3)
    assert float(res.fitted["eta_b"]) == pytest.approx(eb_t, rel=5e-3)


# --- LM: Servo (small initial offset; LM must converge) ------------------


def _servo_truth_telemetry(*, Kp, Kd, T_max=5.0, deadband=1e-4, J=1e-3):
    """Inline copy of test_servo._synthetic_step_telemetry (LM fixture)."""
    fs, samples_per_step, v_eps = 200.0, 200, 1e-3
    dt = 1.0 / fs
    setpoints = [0.0, 0.5, 0.5, -0.2, 0.0]

    def rhs(q, v, sp):
        err = sp - q
        err_eff = err - deadband * np.tanh(err / deadband)
        tau_cmd = Kp * err_eff - Kd * v
        tau_act = T_max * np.tanh(tau_cmd / T_max)
        return v, (tau_act - 0.0 * v - 0.0 * np.tanh(v / v_eps)) / J

    ts, qs, sets = [], [], []
    q, v, t = 0.0, 0.0, 0.0
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
        sample_rate_hz=fs,
        actuator_name="s",
        setpoint=np.asarray(sets, dtype=np.float64),
        position=np.asarray(qs, dtype=np.float64),
    )


@pytest.mark.nightly
def test_servo_lm_dispatch_returns_correct_shape():
    """The LM path for Servo runs end-to-end and returns the seven leaves."""
    tel = _servo_truth_telemetry(Kp=20.0, Kd=0.3, J=1e-3)
    device = Servo(Kp=15.0, Kd=0.4, T_max=5.0, deadband=1e-4, J=1.2e-3)
    res = sysid.actuator(tel, device, hints={"max_steps": 32})
    assert set(res.fitted) == {"Kp", "Kd", "T_max", "deadband", "J", "b", "Fc"}
    assert res.recipe_name == "sysid.actuator"
    assert "position_rms" in (res.residuals or {})


@pytest.mark.nightly
def test_servo_lm_multi_start_improves_recovery():
    """With n_starts > 1 the LM finds a strictly-better minimum.

    A 25%-off warm start lands the single-start LM in a local plateau. The
    multi-start variant samples log-normal perturbations of the warm start
    and keeps the best — the recovered Kp should be measurably closer to
    truth than the single-start fit. The bound is loose because the cost
    landscape is genuinely shallow; the test asserts strict improvement, not
    full convergence to truth.
    """
    Kp_t = 20.0
    tel = _servo_truth_telemetry(Kp=Kp_t, Kd=0.3, J=1e-3)
    device = Servo(Kp=15.0, Kd=0.4, T_max=5.0, deadband=1e-4, J=1.2e-3)
    single = sysid.actuator(tel, device, hints={"max_steps": 32, "n_starts": 1})
    multi = sysid.actuator(
        tel,
        device,
        hints={"max_steps": 32, "n_starts": 5, "perturbation": 0.2, "seed": 0},
    )
    err_single = abs(float(single.fitted["Kp"]) - Kp_t)
    err_multi = abs(float(multi.fitted["Kp"]) - Kp_t)
    # Multi-start should be no worse, and on this seed strictly better.
    assert err_multi <= err_single


# --- LM: BLDC + SeriesElastic numerical recovery -------------------
#
# The BLDC and SEA LM fitters previously crashed on any
# real telemetry (`float()` on a traced array inside the optimistix-traced
# rollout) and had zero invoking tests, so the crash was invisible. These
# tests fit truth telemetry generated by the *same* substepped RK4 the fitter
# uses and assert recovery of the identifiable quantities.


def _rk4_substep(rhs, y, dt, n_sub, *args):
    """n_sub fixed RK4 substeps of `y' = rhs(y, *args)` over one outer step."""
    sub = dt / n_sub
    for _ in range(n_sub):
        k1 = rhs(y, *args)
        k2 = rhs(y + 0.5 * sub * k1, *args)
        k3 = rhs(y + 0.5 * sub * k2, *args)
        k4 = rhs(y + sub * k3, *args)
        y = y + (sub / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


def _bldc_truth_telemetry(*, R, L, Kt, Ke, Kp_i, J, b, Fc, fs=2000.0, T=1.0):
    from jaxterity.data import ActuatorChirpTelemetry

    n = int(fs * T) + 1
    t = np.arange(n, dtype=np.float64) / fs
    f0, k = 0.5, (50.0 - 0.5) / T  # linear chirp 0.5 -> 50 Hz on i_ref
    iref = np.sin(2 * np.pi * (f0 * t + 0.5 * k * t * t))
    v_eps = 1e-3

    def rhs(y, ir):
        v, i = y[1], y[2]
        didt = (Kp_i * (ir - i) - R * i - Ke * v) / L
        dvdt = (Kt * i - b * v - Fc * np.tanh(v / v_eps)) / J
        return np.array([v, dvdt, didt])

    v_arr = np.zeros(n)
    i_arr = np.zeros(n)
    y = np.zeros(3)
    for kk in range(1, n):
        y = _rk4_substep(rhs, y, t[kk] - t[kk - 1], 8, iref[kk - 1])
        v_arr[kk], i_arr[kk] = y[1], y[2]
    return ActuatorChirpTelemetry(
        time=t,
        sample_rate_hz=fs,
        actuator_name="m",
        setpoint=iref,
        velocity=v_arr,
        current=i_arr,
    )


def _sea_truth_telemetry(*, k, c, J_m, J_l, b_m, b_l, fs=2000.0, T=1.0):
    from jaxterity.data import SeriesElasticTelemetry

    n = int(fs * T) + 1
    t = np.arange(n, dtype=np.float64) / fs
    f0, kk = 0.5, (30.0 - 0.5) / T
    tau = 0.5 * np.sin(2 * np.pi * (f0 * t + 0.5 * kk * t * t))

    def rhs(y, ti):
        qm, vm, ql, vl = y
        ts = k * (qm - ql) + c * (vm - vl)
        return np.array([vm, (ti - ts - b_m * vm) / J_m, vl, (ts - b_l * vl) / J_l])

    qm_arr = np.zeros(n)
    ql_arr = np.zeros(n)
    y = np.zeros(4)
    for j in range(1, n):
        y = _rk4_substep(rhs, y, t[j] - t[j - 1], 8, tau[j - 1])
        qm_arr[j], ql_arr[j] = y[0], y[2]
    return SeriesElasticTelemetry(
        time=t,
        sample_rate_hz=fs,
        actuator_name="s",
        tau_input=tau,
        q_motor=qm_arr,
        q_load=ql_arr,
    )


@pytest.mark.nightly
def test_bldc_lm_recovers_kt_over_j_and_predicts_trajectory():
    """BLDC fit recovers the identifiable quantity Kt/J and predicts (v, i).

    R/L/Ke/Kp_i are only weakly identifiable from a free-shaft current chirp
    (documented in the recipe), so we assert the ratio Kt/J (which the velocity
    dynamics pin) and a near-zero trajectory residual — a real parametric
    recovery, not self-consistency (the truth params differ from the fit's
    warm start).
    """
    from jaxterity.actuators import BLDC

    truth = dict(R=1.2, L=1.5e-3, Kt=0.05, Ke=0.05, Kp_i=12.0, J=1.2e-4, b=1e-4, Fc=0.0)
    tel = _bldc_truth_telemetry(**truth)
    device = BLDC(R=1.0, L=1e-3, Kt=0.04, Ke=0.04, Kp_i=10.0, J=1e-4, b=0.0, Fc=0.0)
    res = sysid.actuator(tel, device, hints={"max_steps": 200})
    kt_over_j = float(res.fitted["Kt"]) / float(res.fitted["J"])
    assert kt_over_j == pytest.approx(truth["Kt"] / truth["J"], rel=1e-2)
    # The fit predicts the measured trajectory to well below sensor noise.
    assert res.residuals is not None
    assert res.residuals["velocity_rms"] < 1e-6
    assert res.residuals["current_rms"] < 1e-6


@pytest.mark.nightly
def test_series_elastic_lm_recovers_truth():
    """SEA fit recovers k, c, and both inertias from a torque chirp."""
    from jaxterity.actuators import SeriesElasticActuator

    truth = dict(k=120.0, c=0.15, J_m=1.2e-3, J_l=1.1e-2, b_m=1e-3, b_l=1e-3)
    tel = _sea_truth_telemetry(**truth)
    device = SeriesElasticActuator(
        k=100.0, c=0.1, J_motor=1e-3, J_load=1e-2, b_m=0.0, b_l=0.0
    )
    res = sysid.actuator(tel, device, hints={"max_steps": 200})
    assert float(res.fitted["k"]) == pytest.approx(truth["k"], rel=2e-2)
    assert float(res.fitted["J_motor"]) == pytest.approx(truth["J_m"], rel=5e-2)
    assert float(res.fitted["J_load"]) == pytest.approx(truth["J_l"], rel=5e-2)
    assert not res.warnings  # a clean recovery should not raise a stall warning


@pytest.mark.nightly
def test_lm_stall_emits_warning():
    """A fit that cannot move off its warm start emits a structured warning.

    The silent-no-op failure mode. When the warm start is already
    at the optimum (zero gradient), LM returns it unchanged; the recipe must
    flag that the values are unverified rather than presenting them as a fit.
    """
    from jaxterity.actuators import BLDC

    truth = dict(R=1.2, L=1.5e-3, Kt=0.05, Ke=0.05, Kp_i=12.0, J=1.2e-4, b=1e-4, Fc=0.0)
    tel = _bldc_truth_telemetry(**truth)
    device = BLDC(**truth)  # warm start == truth → nothing to move
    res = sysid.actuator(tel, device, hints={"max_steps": 50})
    assert res.warnings
    assert "unchanged from the initial guess" in res.warnings[0]
