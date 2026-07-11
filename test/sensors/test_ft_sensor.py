# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""FTSensor — metadata, coupling, bias, validate."""

from __future__ import annotations

import numpy as np

from jaxterity.data import FTSensorTelemetry
from jaxterity.sensors import FTSensor, Sensor


def test_is_sensor_subclass():
    assert issubclass(FTSensor, Sensor)


def test_uncalibrated_is_identity_with_zero_bias():
    s = FTSensor.uncalibrated()
    np.testing.assert_array_equal(np.asarray(s.bias), np.zeros(6))
    np.testing.assert_array_equal(np.asarray(s.coupling), np.eye(6))


def test_parameter_spec_lists_three_leaves():
    assert tuple(FTSensor.uncalibrated().parameter_spec()) == (
        "bias",
        "coupling",
        "noise_density",
    )


def test_sysid_protocol_describes_known_load_sweep():
    p = FTSensor.uncalibrated().sysid_protocol
    assert p.name == "ft-known-load-sweep"
    assert set(p.required_columns) == {"force", "torque"}


# --- forward measurement model --------------------------------------------


def test_identity_coupling_passes_input_through():
    s = FTSensor.uncalibrated()
    f_in = np.array([1.0, 2.0, 3.0])
    t_in = np.array([0.1, 0.2, 0.3])
    out = np.asarray(s.coupling) @ np.concatenate([f_in, t_in]) + np.asarray(s.bias)
    np.testing.assert_allclose(out[:3], f_in, atol=1e-9)
    np.testing.assert_allclose(out[3:], t_in, atol=1e-9)


def test_coupling_off_diagonal_leaks():
    """A 1% leak from F_x → F_y produces the documented mixing."""
    C = np.eye(6)
    C[1, 0] = 0.01  # F_y gets 1% of F_x
    s = FTSensor(coupling=C)
    f_in = np.array([100.0, 0.0, 0.0])
    t_in = np.zeros(3)
    out = np.asarray(s.coupling) @ np.concatenate([f_in, t_in]) + np.asarray(s.bias)
    assert out[0] == 100.0
    assert out[1] == 1.0  # 1% of 100


def test_bias_appears_in_reading():
    bias = np.array([0.5, -0.3, 1.0, 0.01, -0.02, 0.03])
    s = FTSensor(bias=bias)
    out = np.asarray(s.coupling) @ np.zeros(6) + np.asarray(s.bias)
    np.testing.assert_allclose(out, bias, atol=1e-9)


# --- validate --------------------------------------------------------------


def _static_ft_telemetry(*, sigma: float = 0.02, n: int = 2000, fs: float = 500.0):
    rng = np.random.default_rng(0)
    t = np.arange(n, dtype=np.float64) / fs
    force = np.array([0.0, 0.0, 9.8]) + sigma * rng.standard_normal((n, 3))
    torque = np.array([0.0, 0.5, 0.0]) + sigma * rng.standard_normal((n, 3))
    return FTSensorTelemetry(
        time=t,
        sample_rate_hz=fs,
        frame="sensor",
        force=force,
        torque=torque,
    )


def test_validate_matched_noise_density_is_ok():
    sigma = 0.02
    tel = _static_ft_telemetry(sigma=sigma)
    nd = sigma * np.sqrt(2.0 / 500.0)
    s = FTSensor(noise_density=[nd] * 6)
    assert s.validate(tel).ok


def test_validate_understated_noise_density_flags_warning():
    tel = _static_ft_telemetry(sigma=0.02)
    s = FTSensor(noise_density=[1e-8] * 6)  # claimed much quieter than reality
    assert not s.validate(tel).ok
