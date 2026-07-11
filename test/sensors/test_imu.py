# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""IMU — metadata, bias, misalignment, noise-density validate."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from jaxterity.data import IMUTelemetry
from jaxterity.sensors import IMU, Sensor
from jaxterity.sensors.imu import _rodrigues

# --- metadata --------------------------------------------------------------


def test_is_sensor_subclass():
    assert issubclass(IMU, Sensor)


def test_uncalibrated_constructs_with_zero_defaults():
    imu = IMU.uncalibrated()
    np.testing.assert_array_equal(np.asarray(imu.bias_accel), np.zeros(3))
    np.testing.assert_array_equal(np.asarray(imu.bias_gyro), np.zeros(3))
    np.testing.assert_array_equal(np.asarray(imu.misalignment), np.zeros(3))


def test_parameter_spec_lists_five_leaves():
    assert tuple(IMU.uncalibrated().parameter_spec()) == (
        "bias_accel",
        "bias_gyro",
        "noise_density_accel",
        "noise_density_gyro",
        "misalignment",
    )


def test_sysid_protocol_describes_turntable():
    p = IMU.uncalibrated().sysid_protocol
    assert p.name == "imu-turntable"
    assert set(p.required_columns) == {"accel", "gyro"}


# --- Rodrigues exp map -----------------------------------------------------


def test_rodrigues_identity_at_zero():
    R = np.asarray(_rodrigues(jnp.zeros(3)))
    np.testing.assert_allclose(R, np.eye(3), atol=1e-9)


def test_rodrigues_90deg_about_z():
    """[0, 0, π/2] should rotate (1, 0, 0) → (0, 1, 0)."""
    R = np.asarray(_rodrigues(jnp.array([0.0, 0.0, np.pi / 2])))
    np.testing.assert_allclose(
        R @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-7
    )


def test_rodrigues_small_angle_stable():
    """At very small angle the Taylor branch must produce a finite, near-identity R."""
    R = np.asarray(_rodrigues(jnp.array([1e-10, 0.0, 0.0])))
    assert np.all(np.isfinite(R))
    np.testing.assert_allclose(R, np.eye(3), atol=1e-8)


# --- forward measurement model --------------------------------------------


def test_zero_input_zero_bias_gives_zero_output():
    """Trivial: identity misalignment, zero bias, zero input → zero output."""
    imu = IMU.uncalibrated()
    R = np.asarray(_rodrigues(imu.misalignment))
    accel_in = np.zeros(3)
    out = R @ accel_in + np.asarray(imu.bias_accel)
    np.testing.assert_allclose(out, np.zeros(3), atol=1e-9)


def test_bias_appears_in_reading():
    imu = IMU(bias_accel=[0.1, -0.2, 0.05])
    R = np.asarray(_rodrigues(imu.misalignment))
    out = R @ np.zeros(3) + np.asarray(imu.bias_accel)
    np.testing.assert_allclose(out, [0.1, -0.2, 0.05], atol=1e-9)


def test_misalignment_rotates_truth():
    """A 90° z-axis misalignment maps body-frame (1, 0, 0) accel to (0, 1, 0)."""
    imu = IMU(misalignment=[0.0, 0.0, np.pi / 2])
    R = np.asarray(_rodrigues(imu.misalignment))
    out = R @ np.array([1.0, 0.0, 0.0]) + np.asarray(imu.bias_accel)
    np.testing.assert_allclose(out, [0.0, 1.0, 0.0], atol=1e-7)


# --- validate --------------------------------------------------------------


def _static_imu_telemetry(
    *,
    gravity: float = 9.81,
    bias_accel=(0.05, -0.1, 0.0),
    bias_gyro=(1e-3, 0.0, -2e-3),
    sigma_a: float = 0.02,
    sigma_g: float = 5e-4,
    n: int = 2000,
    sample_rate_hz: float = 200.0,
) -> IMUTelemetry:
    """Synthetic static-pose telemetry: gravity + bias + white noise."""
    rng = np.random.default_rng(0)
    t = np.arange(n, dtype=np.float64) / sample_rate_hz
    accel_truth = np.array([0.0, 0.0, gravity])
    accel = accel_truth + np.asarray(bias_accel) + sigma_a * rng.standard_normal((n, 3))
    gyro = np.asarray(bias_gyro) + sigma_g * rng.standard_normal((n, 3))
    return IMUTelemetry(
        time=t,
        sample_rate_hz=sample_rate_hz,
        frame="imu",
        accel=accel,
        gyro=gyro,
    )


def test_validate_with_matched_bias_and_noise_density_is_ok():
    """Provide bias and a noise_density compatible with the trace — report.ok."""
    bias_a = [0.05, -0.1, 0.0]
    bias_g = [1e-3, 0.0, -2e-3]
    sigma_a, sigma_g = 0.02, 5e-4
    sample_rate_hz = 200.0
    tel = _static_imu_telemetry(
        bias_accel=bias_a,
        bias_gyro=bias_g,
        sigma_a=sigma_a,
        sigma_g=sigma_g,
        sample_rate_hz=sample_rate_hz,
    )
    # noise_density = σ · √(2 / fs); pass the recovered density so the bound
    # (3 · density · √(fs/2)) = 3σ comfortably contains the per-axis RMS of σ.
    nd_a = sigma_a * np.sqrt(2.0 / sample_rate_hz)
    nd_g = sigma_g * np.sqrt(2.0 / sample_rate_hz)
    imu = IMU(
        bias_accel=bias_a,
        bias_gyro=bias_g,
        noise_density_accel=[nd_a] * 3,
        noise_density_gyro=[nd_g] * 3,
    )
    report = imu.validate(tel)
    assert report.ok, report.summary()


def test_validate_with_understated_noise_density_flags_warning():
    """If declared noise density is far too small, residual RMS exceeds the bound."""
    bias_a = [0.05, -0.1, 0.0]
    bias_g = [1e-3, 0.0, -2e-3]
    tel = _static_imu_telemetry(
        bias_accel=bias_a,
        bias_gyro=bias_g,
        sigma_a=0.02,
        sigma_g=5e-4,
    )
    imu = IMU(
        bias_accel=bias_a,
        bias_gyro=bias_g,
        noise_density_accel=[1e-6] * 3,  # claimed much quieter than reality
        noise_density_gyro=[1e-7] * 3,
    )
    report = imu.validate(tel)
    assert not report.ok


def test_validate_flags_wrong_accel_and_gyro_bias():
    """A wrong bias must FAIL validation.

    The previous implementation de-meaned every axis, which discarded the bias
    signal and let any wrong bias pass (the bug this replaces). Under the static
    assumption the truth is pinned — gyro truth is zero, accel magnitude is g —
    so a wrong bias is detectable from the means.
    """
    tel = _static_imu_telemetry(
        bias_accel=[0.05, -0.1, 0.0],
        bias_gyro=[1e-3, 0.0, -2e-3],
        sigma_a=0.02,
        sigma_g=5e-4,
    )
    sample_rate_hz = 200.0
    nd_a = 0.02 * np.sqrt(2.0 / sample_rate_hz)
    nd_g = 5e-4 * np.sqrt(2.0 / sample_rate_hz)
    imu = IMU(
        bias_accel=[10.0, 10.0, 10.0],  # very wrong
        bias_gyro=[10.0, 10.0, 10.0],
        noise_density_accel=[nd_a] * 3,
        noise_density_gyro=[nd_g] * 3,
    )
    report = imu.validate(tel)
    assert not report.ok
    summary = report.summary()
    assert "bias_accel" in summary and "bias_gyro" in summary


def test_validate_passes_correct_bias_under_nonzero_static_pose():
    """A small, correct bias on a realistic static pose still validates clean.

    Guards against the bias check being so tight it false-positives on the
    matched case (the gravity-magnitude tolerance must absorb local-g + noise).
    """
    bias_a = [0.05, -0.1, 0.0]
    bias_g = [1e-3, 0.0, -2e-3]
    sample_rate_hz = 200.0
    tel = _static_imu_telemetry(
        bias_accel=bias_a, bias_gyro=bias_g, sigma_a=0.02, sigma_g=5e-4
    )
    nd_a = 0.02 * np.sqrt(2.0 / sample_rate_hz)
    nd_g = 5e-4 * np.sqrt(2.0 / sample_rate_hz)
    imu = IMU(
        bias_accel=bias_a,
        bias_gyro=bias_g,
        noise_density_accel=[nd_a] * 3,
        noise_density_gyro=[nd_g] * 3,
    )
    assert imu.validate(tel).ok, imu.validate(tel).summary()
