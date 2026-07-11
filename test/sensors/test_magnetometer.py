# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Magnetometer — metadata, bias, validate."""

from __future__ import annotations

import numpy as np

from jaxterity.data import MagnetometerTelemetry
from jaxterity.sensors import Magnetometer, Sensor
from jaxterity.sensors.imu import _rodrigues


def test_is_sensor_subclass():
    assert issubclass(Magnetometer, Sensor)


def test_uncalibrated_defaults_are_zero():
    m = Magnetometer.uncalibrated()
    np.testing.assert_array_equal(np.asarray(m.bias), np.zeros(3))


def test_parameter_spec_lists_three_leaves():
    assert tuple(Magnetometer.uncalibrated().parameter_spec()) == (
        "bias",
        "noise_density",
        "misalignment",
    )


def test_sysid_protocol():
    p = Magnetometer.uncalibrated().sysid_protocol
    assert p.name == "mag-static-pose"


def test_misalignment_rotates_input():
    """90° z-axis misalignment maps body-frame x-axis field to y-axis reading."""
    m = Magnetometer(misalignment=[0.0, 0.0, np.pi / 2])
    R = np.asarray(_rodrigues(m.misalignment))
    out = R @ np.array([1.0, 0.0, 0.0]) + np.asarray(m.bias)
    np.testing.assert_allclose(out, [0.0, 1.0, 0.0], atol=1e-7)


def _static_mag_telemetry(
    *, mag_ambient=(0.0, 0.0, 5e-5), sigma=2e-6, n=2000, fs=200.0
):
    rng = np.random.default_rng(0)
    t = np.arange(n, dtype=np.float64) / fs
    mag = np.asarray(mag_ambient)[None, :] + sigma * rng.standard_normal((n, 3))
    return MagnetometerTelemetry(
        time=t,
        sample_rate_hz=fs,
        frame="mag",
        mag=mag,
    )


def test_validate_matched_noise_density_is_ok():
    sigma = 2e-6
    fs = 200.0
    tel = _static_mag_telemetry(sigma=sigma, fs=fs)
    nd = sigma * np.sqrt(2.0 / fs)
    assert Magnetometer(noise_density=[nd] * 3).validate(tel).ok


def test_validate_understated_noise_density_flags_warning():
    tel = _static_mag_telemetry()
    assert not Magnetometer(noise_density=[1e-12] * 3).validate(tel).ok
