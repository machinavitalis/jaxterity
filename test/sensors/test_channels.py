# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Measurement channels — current sense, temperature, bus transport.

These blocks exist because their errors get absorbed into *physical*
parameters, so the tests check both halves: that each channel reproduces the
artifact (a comb of quantized levels, a lag, a delayed packet), and that the
artifact is recoverable — including the identifiability *floor* the coarse
temperature channel imposes, which is the whole point of modeling it instead
of fighting it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jaxonomy
import numpy as np
import pytest
from jaxonomy.library import Constant

from jaxterity.data import (
    BusLatencyTelemetry,
    CurrentSenseTelemetry,
    TemperatureSensorTelemetry,
)
from jaxterity.errors import CalibrationError
from jaxterity.sensors import (
    BusChannel,
    CurrentSense,
    TemperatureSensor,
    adc_step,
    phase_lag_deg,
    quantization_floor,
    quantization_step_estimate,
    quantize,
)
from jaxterity.sensors.bus_channel import _fit_bus_channel
from jaxterity.sensors.current_sense import _fit_current_sense
from jaxterity.sensors.temperature_sensor import _fit_temperature_channel

# --- current sense ---------------------------------------------------------


def test_adc_step_halves_with_each_extra_bit():
    assert float(adc_step(20.0, 12)) == pytest.approx(2 * 20.0 / 4096)
    assert float(adc_step(20.0, 13)) == pytest.approx(0.5 * float(adc_step(20.0, 12)))


def test_quantize_snaps_to_the_lattice_and_passes_through_at_zero_step():
    assert float(quantize(0.107, 0.05)) == pytest.approx(0.10)
    assert float(quantize(0.107, 0.0)) == pytest.approx(0.107)


def _current_trace(gain=1.03, offset=0.15, step=0.01, full_scale=20.0, n=500):
    truth = np.linspace(-25.0, 25.0, n)  # deliberately past full scale
    measured = np.asarray(
        quantize(jnp.clip(gain * truth + offset, -full_scale, full_scale), step)
    )
    return CurrentSenseTelemetry(
        time=np.arange(n) * 0.001,
        sample_rate_hz=1000.0,
        channel_name="j0",
        truth=truth,
        measured=measured,
    )


def test_fit_recovers_gain_and_offset_despite_clipping():
    telemetry = _current_trace(gain=1.03, offset=0.15)
    fitted = _fit_current_sense(telemetry)
    assert fitted["gain"] == pytest.approx(1.03, rel=1e-3)
    assert fitted["offset"] == pytest.approx(0.15, abs=0.01)


def test_quantization_step_is_read_off_the_lattice():
    # A slow sweep: consecutive readings differ by about one code, which is
    # what the estimator's documented precondition asks for.
    telemetry = _current_trace(step=0.01, n=5000)
    assert quantization_step_estimate(telemetry) == pytest.approx(0.01, rel=1e-6)


def test_fit_refuses_a_constant_reference():
    n = 20
    telemetry = CurrentSenseTelemetry(
        time=np.arange(n) * 0.001,
        sample_rate_hz=1000.0,
        channel_name="j0",
        truth=np.full(n, 2.0),
        measured=np.full(n, 2.1),
    )
    with pytest.raises(CalibrationError, match="varying reference current"):
        _fit_current_sense(telemetry)


def test_current_channel_validates_its_own_trace_and_flags_a_wrong_gain():
    telemetry = _current_trace(gain=1.03, offset=0.15, step=0.01)
    good = CurrentSense(gain=1.03, offset=0.15, quantization_step=0.01, full_scale=20.0)
    assert good.validate(telemetry).ok
    wrong = CurrentSense(gain=1.0, offset=0.0, quantization_step=0.01, full_scale=20.0)
    assert not wrong.validate(telemetry).ok


@pytest.mark.slow
def test_channel_bandwidth_lags_a_step():
    # A 10 Hz channel cannot follow a step within one time constant.
    channel = CurrentSense(bandwidth_hz=10.0, quantization_step=0.0, i0=0.0)
    builder = jaxonomy.DiagramBuilder()
    system = builder.add(channel)
    source = builder.add(Constant(5.0, name="step"))
    builder.connect(source.output_ports[0], system.input_ports[0])
    diagram = builder.build()
    port = system.output_ports[0]
    results = jaxonomy.simulate(
        diagram,
        diagram.create_context(),
        (0.0, 1.0 / (2 * np.pi * 10.0)),
        recorded_signals={"reading": port},
    )
    reading = float(results.outputs["reading"][-1])
    assert reading == pytest.approx(5.0 * (1 - np.exp(-1.0)), rel=1e-3)


def test_uncalibrated_current_channel_is_transparent():
    channel = CurrentSense.uncalibrated()
    assert float(channel.gain) == 1.0
    assert float(channel.offset) == 0.0
    assert float(channel.quantization_step) == 0.0


# --- temperature channel ---------------------------------------------------


def test_quantization_floor_scales_as_step_over_root_n():
    base = quantization_floor(1.0, 100, 8.0)
    assert quantization_floor(2.0, 100, 8.0) == pytest.approx(2 * base)
    assert quantization_floor(1.0, 400, 8.0) == pytest.approx(0.5 * base)
    assert quantization_floor(1.0, 100, 16.0) == pytest.approx(0.5 * base)


def test_quantization_floor_matches_the_uniform_error_formula():
    assert quantization_floor(1.0, 1, 1.0) == pytest.approx(1.0 / np.sqrt(12.0))


def test_quantization_floor_refuses_a_zero_sensitivity():
    # A parameter the experiment does not move has an infinite floor; saying
    # so is more useful than dividing by zero.
    with pytest.raises(CalibrationError, match="sensitivity > 0"):
        quantization_floor(1.0, 100, 0.0)


def test_one_degree_steps_put_a_real_floor_under_a_thermal_fit():
    # The identifiability wall, quantified: half an hour at 1 Hz with 8 K of
    # rise per K/W still cannot resolve R_th better than ~0.01 K/W.
    floor = quantization_floor(step=1.0, n_samples=1800, sensitivity=8.0)
    assert 5e-4 < floor < 2e-3
    # A finer sensor buys back an order of magnitude for free.
    assert quantization_floor(0.1, 1800, 8.0) == pytest.approx(0.1 * floor)


def _temperature_trace(offset=0.7, tau=25.0, step=1.0, n=600, dt=1.0):
    """A reference step, with the modeled channel's response to it."""
    t = np.arange(n) * dt
    truth = np.where(t < n * dt / 2, 300.0, 320.0)
    sensor = np.empty(n)
    state = 300.0
    sensor[0] = state
    for i in range(1, n):
        k1 = (truth[i - 1] - state) / tau
        k2 = (truth[i - 1] - (state + 0.5 * dt * k1)) / tau
        k3 = (truth[i - 1] - (state + 0.5 * dt * k2)) / tau
        k4 = (truth[i - 1] - (state + dt * k3)) / tau
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        sensor[i] = state
    measured = np.asarray(quantize(jnp.asarray(sensor + offset), step))
    return TemperatureSensorTelemetry(
        time=t,
        sample_rate_hz=1.0 / dt,
        channel_name="j0",
        truth=truth,
        measured=measured,
    )


def test_temperature_fit_recovers_lag_offset_and_step():
    telemetry = _temperature_trace(offset=0.7, tau=25.0, step=1.0)
    fitted = _fit_temperature_channel(telemetry)
    assert fitted["quantization_step"] == pytest.approx(1.0, rel=1e-9)
    # With 1 K quantization the lag is recoverable to a few percent, not more.
    assert fitted["tau_sensor"] == pytest.approx(25.0, rel=0.15)
    assert fitted["offset"] == pytest.approx(0.7, abs=1.0)


def test_temperature_fit_refuses_a_soak():
    n = 50
    telemetry = TemperatureSensorTelemetry(
        time=np.arange(n) * 1.0,
        sample_rate_hz=1.0,
        channel_name="j0",
        truth=np.full(n, 300.0),
        measured=np.full(n, 300.0),
    )
    with pytest.raises(CalibrationError, match="no transient"):
        _fit_temperature_channel(telemetry)


def test_temperature_fit_catches_swapped_channels():
    telemetry = _temperature_trace()
    swapped = TemperatureSensorTelemetry(
        time=telemetry.time,
        sample_rate_hz=telemetry.sample_rate_hz,
        channel_name="j0",
        truth=telemetry.measured,
        measured=telemetry.truth,
    )
    with pytest.raises(CalibrationError, match="non-positive lag rate"):
        _fit_temperature_channel(swapped)


def test_temperature_channel_validates_its_own_trace():
    telemetry = _temperature_trace(offset=0.7, tau=25.0, step=1.0)
    good = TemperatureSensor(offset=0.7, tau_sensor=25.0, quantization_step=1.0)
    assert good.validate(telemetry).ok
    slow = TemperatureSensor(offset=0.7, tau_sensor=200.0, quantization_step=1.0)
    assert not slow.validate(telemetry).ok


def test_uncalibrated_temperature_channel_is_transparent():
    channel = TemperatureSensor.uncalibrated()
    assert float(channel.offset) == 0.0
    assert float(channel.quantization_step) == 0.0


# --- bus channel -----------------------------------------------------------


def test_phase_lag_grows_with_frequency_and_delay():
    assert float(phase_lag_deg(0.005, 10.0)) == pytest.approx(18.0)
    assert float(phase_lag_deg(0.010, 10.0)) == pytest.approx(36.0)
    assert float(phase_lag_deg(0.005, 20.0)) == pytest.approx(36.0)


def test_latency_is_quantized_to_whole_bus_periods():
    assert BusChannel(dt=1e-3, latency_s=4.6e-3).latency_samples == 5
    assert BusChannel(dt=1e-3, latency_s=4.4e-3).latency_samples == 4


def test_buffer_is_sized_to_cover_latency_plus_jitter():
    channel = BusChannel(dt=1e-3, latency_s=5e-3, jitter_s=1e-3)
    assert channel.buffer_length >= 9  # 5 + 4 sigma, plus margin


@pytest.mark.slow
def test_a_ramp_arrives_delayed_by_the_declared_latency():
    dt, latency = 1e-3, 5e-3
    channel = BusChannel(dt=dt, latency_s=latency, initial_value=0.0)
    builder = jaxonomy.DiagramBuilder()
    system = builder.add(channel)
    ramp = builder.add(jaxonomy.library.Ramp(slope=1.0, start_time=0.0, name="ramp"))
    builder.connect(ramp.output_ports[0], system.input_ports[0])
    diagram = builder.build()
    results = jaxonomy.simulate(
        diagram,
        diagram.create_context(),
        (0.0, 0.05),
        recorded_signals={"received": system.output_ports[0]},
    )
    received = float(results.outputs["received"][-1])
    # A unit ramp delayed by 5 ms reads (t - 5 ms), up to one bus period of
    # sampling and the hold between updates.
    assert received == pytest.approx(0.05 - latency, abs=2 * dt)


@pytest.mark.slow
def test_the_channel_is_reproducible_under_a_fixed_key():
    def run(seed):
        channel = BusChannel(
            dt=1e-3,
            latency_s=3e-3,
            jitter_s=1e-3,
            drop_probability=0.2,
            key=jax.random.PRNGKey(seed),
        )
        builder = jaxonomy.DiagramBuilder()
        system = builder.add(channel)
        ramp = builder.add(
            jaxonomy.library.Ramp(slope=1.0, start_time=0.0, name="ramp")
        )
        builder.connect(ramp.output_ports[0], system.input_ports[0])
        diagram = builder.build()
        results = jaxonomy.simulate(
            diagram,
            diagram.create_context(),
            (0.0, 0.05),
            recorded_signals={"received": system.output_ports[0]},
        )
        return np.asarray(results.outputs["received"])

    np.testing.assert_array_equal(run(0), run(0))
    assert not np.array_equal(run(0), run(1))


def _latency_trace(latency_s=0.005, dt=0.001, n=2000, drop_every=0):
    t = np.arange(n) * dt
    rng = np.random.default_rng(0)
    sent = np.cumsum(rng.normal(size=n)) / 10.0
    shift = int(round(latency_s / dt))
    if shift:
        received = np.concatenate([np.zeros(shift), sent[:-shift]])
    else:
        received = sent.copy()
    if drop_every:
        for i in range(drop_every, n, drop_every):
            received[i] = received[i - 1]
    return BusLatencyTelemetry(
        time=t,
        sample_rate_hz=1.0 / dt,
        signal_name="goal_position",
        sent=sent,
        received=received,
    )


def test_fit_recovers_the_transport_latency():
    fitted = _fit_bus_channel(_latency_trace(latency_s=0.005))
    assert fitted["latency_s"] == pytest.approx(0.005, abs=1e-3)


def test_fit_reports_the_drop_rate_from_held_samples():
    fitted = _fit_bus_channel(_latency_trace(drop_every=10))
    assert fitted["drop_rate"] == pytest.approx(0.1, abs=0.03)


def test_fit_refuses_a_constant_signal():
    n = 50
    telemetry = BusLatencyTelemetry(
        time=np.arange(n) * 0.001,
        sample_rate_hz=1000.0,
        signal_name="goal_position",
        sent=np.ones(n),
        received=np.ones(n),
    )
    with pytest.raises(CalibrationError, match="varying sent signal"):
        _fit_bus_channel(telemetry)


def test_bus_validate_flags_an_undeclared_latency():
    telemetry = _latency_trace(latency_s=0.005)
    assert BusChannel(dt=1e-3, latency_s=0.005).validate(telemetry).ok
    report = BusChannel(dt=1e-3, latency_s=0.0).validate(telemetry)
    assert not report.ok
    assert "measured latency" in report.warnings[0]


def test_uncalibrated_bus_channel_is_transparent():
    channel = BusChannel.uncalibrated()
    assert float(channel.latency_s) == 0.0
    assert float(channel.drop_probability) == 0.0
