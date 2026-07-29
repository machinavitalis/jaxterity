# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Temperature channel: model the sensor instead of fighting it.

Bus servos report temperature in whole degrees. That single fact — a 1 K
quantization step — is the wall every servo-thermal identification runs into,
and it is usually met by trying to *beat* it: longer captures, tighter
tolerances, more elaborate filtering. It cannot be beaten, because it is not
noise that averages away arbitrarily; it is a hard floor set by the channel,
and the honest move is to model the channel and then compute what that floor
implies. :func:`quantization_floor` does exactly that: given the step, the
sample count, and how strongly the parameter moves the temperature, it returns
the smallest standard error any unbiased fit can achieve. A recipe that reports
a tighter uncertainty than that number is reporting an artifact.

Model — one state (the sensor's own thermal lag), stateless map on top::

    Ṫ_sensor = (T_true − T_sensor) / tau_sensor        (probe/case lag)
    reading  = quantize(T_sensor + offset, step)

Three identifiable parameters, and each is a distinct failure a thermal fit
would otherwise absorb into *its* parameters:

* ``offset`` (K) — a miscalibrated probe looks like a wrong ambient.
* ``tau_sensor`` (s) — the probe's own lag looks like extra thermal mass. On a
  servo whose winding time constant is seconds and whose case sensor lags by
  tens of seconds, this is not a small correction.
* ``quantization_step`` (K) — the staircase looks like measurement noise with
  a suspiciously periodic structure, and biases any derivative-based fit.

Quantization is a hard :func:`jnp.round`, as in
:class:`~jaxterity.sensors.encoder.Encoder` and
:class:`~jaxterity.sensors.current_sense.CurrentSense`: smoothing it would
erase the artifact being identified.

Units
-----
K, s.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..data.schema import TemperatureSensorTelemetry, ValidationReport
from ..errors import CalibrationError
from ..parameter import ParameterSpec
from ..sysid.protocol import SysidProtocol
from ._base import Sensor
from .current_sense import quantize

#: Variance of a uniform quantization error is ``step²/12``; this is its
#: standard deviation coefficient, ``1/√12``.
_UNIFORM_SIGMA = 1.0 / np.sqrt(12.0)

_SYSID_PROTOCOL = SysidProtocol(
    name="temperature-channel-step",
    excitation="step",
    required_columns=("truth", "measured"),
    duration_s=1800.0,
    sample_rate_hz=1.0,
    setup_notes=(
        "Bond a fast reference probe (fine-gauge thermocouple, or an RTD with "
        "a known time constant) to the same surface the sensor under test "
        "reads. (1) SOAK at ambient: the mean residual is the offset and the "
        "residual's distinct levels give the quantization step. (2) STEP: "
        "move the assembly to a hotter still-air environment (or drive the "
        "servo hard, then stop) and record the whole rise AND decay — the lag "
        "between the reference and the sensor is tau_sensor. Sample at least "
        "10x faster than the expected lag; the default 1 Hz suits a case "
        "sensor with a tens-of-seconds constant. Half an hour is not "
        "excessive: with 1 K steps, resolution comes from sample count."
    ),
)


def quantization_floor(step: float, n_samples: int, sensitivity: float) -> float:
    """The smallest parameter standard error a quantized channel allows.

    A uniform quantizer with step ``q`` contributes measurement error of
    standard deviation ``q/√12``. For an unbiased least-squares estimate of a
    parameter ``θ`` from ``n`` samples whose sensitivity is
    ``s = RMS(∂T/∂θ)``, the standard error cannot beat

        σ_θ ≥ (q/√12) / (s·√n)

    This is the number to quote when a thermal fit's uncertainty looks
    suspiciously good, and the number to plan a capture against: halving
    ``σ_θ`` costs 4× the samples, and no amount of sampling helps if ``s`` is
    small (which is the real content of "this parameter is not identifiable
    from this experiment").

    Args
    ----
    step : float
        Quantization step of the channel, K. Must be > 0.
    n_samples : int
        Number of (independent) samples in the fit. Must be >= 1.
    sensitivity : float
        RMS sensitivity of the *measured* quantity to the parameter, in K per
        parameter unit. Must be > 0.

    Returns
    -------
    float
        Standard-error floor in the parameter's own unit.

    Raises
    ------
    CalibrationError
        If any argument is non-positive.

    Units
    -----
    K, count, K/[θ] → [θ].

    Examples
    --------
    >>> # 1 K steps, 1800 samples, 8 K of rise per unit of K/W thermal
    >>> # resistance: under a milli-K/W is the best a fit can claim.
    >>> round(quantization_floor(1.0, 1800, 8.0), 5)
    0.00085
    """
    if step <= 0 or n_samples < 1 or sensitivity <= 0:
        raise CalibrationError(
            f"quantization_floor needs step > 0, n_samples >= 1 and "
            f"sensitivity > 0; got {step}, {n_samples}, {sensitivity}. A "
            "zero sensitivity means the experiment carries no information "
            "about this parameter at all — the floor is then infinite, which "
            "is a statement about the experiment, not a number to divide by."
        )
    return float(_UNIFORM_SIGMA * step / (sensitivity * np.sqrt(n_samples)))


class TemperatureSensor(Sensor):
    """Temperature channel with offset, thermal lag, and quantization.

    See the module docstring for the model and why the quantizer is hard.
    Input port ``truth`` is the true surface temperature (K); output port
    ``reading`` is what the drive reports (K). State is the sensor's own
    filtered temperature.

    Args
    ----
    offset : float or jax.Array
        Additive calibration offset, K.
    tau_sensor : float or jax.Array
        Sensor thermal time constant, s. Must be > 0.
    quantization_step : float or jax.Array
        Reading resolution, K (1.0 for a servo reporting whole degrees). 0
        disables quantization.
    noise_density : float or jax.Array
        White-noise density, K/√Hz. Sysid metadata; not in the forward model.
    T0 : float
        Initial sensor temperature, K.
    name : str
        Block name.

    Examples
    --------
    >>> probe = TemperatureSensor(quantization_step=1.0, tau_sensor=30.0)
    >>> tuple(probe.parameter_spec())
    ('offset', 'tau_sensor', 'quantization_step')
    """

    sysid_protocol = _SYSID_PROTOCOL

    def __init__(
        self,
        *,
        offset=0.0,
        tau_sensor=1.0,
        quantization_step=0.0,
        noise_density=0.0,
        T0: float = 298.15,
        name: str = "temperature_sensor",
    ):
        super().__init__(name=name)
        for n, x in (
            ("offset", offset),
            ("tau_sensor", tau_sensor),
            ("quantization_step", quantization_step),
            ("noise_density", noise_density),
        ):
            self.declare_dynamic_parameter(n, jnp.asarray(x, dtype=float))
        self.declare_input_port(name="truth")
        self.declare_continuous_state(
            default_value=jnp.asarray(T0, dtype=float), ode=self._ode
        )
        self.declare_output_port(self._read, name="reading", requires_inputs=False)

    @property
    def offset(self):
        return self.dynamic_parameters["offset"].get()

    @property
    def tau_sensor(self):
        return self.dynamic_parameters["tau_sensor"].get()

    @property
    def quantization_step(self):
        return self.dynamic_parameters["quantization_step"].get()

    @property
    def noise_density(self):
        return self.dynamic_parameters["noise_density"].get()

    @classmethod
    def uncalibrated(cls, *, name: str = "temperature_sensor") -> "TemperatureSensor":
        """Return an instance with identifiability-neutral defaults.

        Zero offset, no quantization, and a 1 s lag (fast enough to be
        transparent next to any servo's thermal constants, non-zero so the ODE
        stays well defined).
        """
        return cls(name=name)

    def _ode(self, time, state, *inputs, **params):
        return (inputs[0] - state.continuous_state) / params["tau_sensor"]

    def _read(self, time, state, *inputs, **params):
        return quantize(
            state.continuous_state + params["offset"], params["quantization_step"]
        )

    def parameter_spec(self) -> ParameterSpec:
        return ParameterSpec(("offset", "tau_sensor", "quantization_step"))

    def validate(self, telemetry: TemperatureSensorTelemetry) -> ValidationReport:
        """Forward-simulate the channel against a reference-probe trace.

        Integrates the lag ODE with a fixed-step NumPy RK4 (zero-order hold on
        the reference, matching a real recording), applies the offset and the
        quantizer, and reports the RMS residual. The bound includes the
        quantization contribution ``step/√12`` — a channel cannot track a
        reference more closely than its own resolution, and a report that
        ignored that would fail every correctly-modeled coarse sensor.

        Args
        ----
        telemetry : TemperatureSensorTelemetry
            Reference vs measured temperature.

        Returns
        -------
        ValidationReport
            ``ok`` iff the RMS residual is within the widened bound.

        Units
        -----
        K.
        """
        from ..testing import tolerances as tol

        t = np.asarray(telemetry.time, dtype=np.float64)
        truth = np.asarray(telemetry.truth, dtype=np.float64)
        measured = np.asarray(telemetry.measured, dtype=np.float64)
        tau = float(self.tau_sensor)
        offset = float(self.offset)
        step = float(self.quantization_step)

        simulated = np.empty_like(measured)
        state = float(truth[0])
        simulated[0] = float(quantize(jnp.asarray(state + offset), step))
        for i in range(1, t.size):
            dt = float(t[i] - t[i - 1])
            u = float(truth[i - 1])
            k1 = (u - state) / tau
            k2 = (u - (state + 0.5 * dt * k1)) / tau
            k3 = (u - (state + 0.5 * dt * k2)) / tau
            k4 = (u - (state + dt * k3)) / tau
            state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            simulated[i] = float(quantize(jnp.asarray(state + offset), step))

        residual = simulated - measured
        rms = float(np.sqrt(np.mean(residual * residual)))
        report = ValidationReport()
        bound = (
            tol.atol("motor_temperature", "rk4", "float64") * 1e4
            + _UNIFORM_SIGMA * step
        )
        if rms > bound:
            report.warnings.append(
                f"temperature-channel residual RMS={rms:.3g} K exceeds bound "
                f"{bound:.3g} K — the offset, the sensor lag, or the "
                "quantization step may not match this channel."
            )
        return report


def _fit_temperature_channel(
    telemetry: TemperatureSensorTelemetry,
) -> dict[str, float]:
    """Recover the channel's offset, lag, and quantization step from a trace.

    Three independent, closed-form estimates, each from the part of the trace
    that carries its information:

    * ``quantization_step`` — the smallest positive gap between distinct
      readings (model-free, exactly as for the current channel).
    * ``tau_sensor`` — regressed from the lag ODE in its identifiable form,
      ``Ṫ_meas = (T_true − T_meas + offset)/tau``, using central differences.
      This needs a *transient*: a soak identifies nothing about the lag.
    * ``offset`` — the mean residual over the quasi-static part of the trace
      (where the lag contributes nothing), so it is not confounded with the
      lag's own steady-state error.

    Private to ``temperature_sensor.py`` (to make the recovery test runnable);
    the formal ``sysid`` recipe home is later work.

    Args
    ----
    telemetry : TemperatureSensorTelemetry
        Reference vs measured temperature covering a step.

    Returns
    -------
    dict[str, float]
        ``{"offset", "tau_sensor", "quantization_step"}`` in K, s, K.

    Raises
    ------
    CalibrationError
        If the trace has no transient (the lag is then unidentifiable) or
        fewer than 5 samples.

    Units
    -----
    K, s.
    """
    t = np.asarray(telemetry.time, dtype=np.float64)
    truth = np.asarray(telemetry.truth, dtype=np.float64)
    measured = np.asarray(telemetry.measured, dtype=np.float64)
    if t.size < 5:
        raise CalibrationError(
            f"_fit_temperature_channel needs at least 5 samples, got {t.size}."
        )

    levels = np.unique(measured)
    gaps = np.diff(levels)
    positive = gaps[gaps > 0]
    step = float(np.min(positive)) if positive.size else 0.0

    derivative = (measured[2:] - measured[:-2]) / (t[2:] - t[:-2])
    gradient = (truth - measured)[1:-1]
    if float(np.ptp(gradient)) <= 0.0 or not np.any(np.abs(derivative) > 0):
        raise CalibrationError(
            "_fit_temperature_channel found no transient: the reading never "
            "moves relative to the reference, so the sensor lag is not "
            "identifiable from this trace. Fix: record a temperature STEP "
            "(rise and decay), not a soak."
        )
    slope = float(np.sum(gradient * derivative) / np.sum(gradient * gradient))
    if slope <= 0:
        raise CalibrationError(
            f"_fit_temperature_channel regressed a non-positive lag rate "
            f"({slope:.3g} 1/s); the reading moves *against* the reference. "
            "This is the signature of swapped truth/measured channels or a "
            "sign error at the loader boundary."
        )
    tau_sensor = 1.0 / slope

    # Offset from the quasi-static part: where the reference is barely moving,
    # the lag contributes nothing and the residual is the offset alone.
    truth_rate = np.abs(np.gradient(truth, t))
    quiet = truth_rate <= np.quantile(truth_rate, 0.25)
    offset = float(np.mean((measured - truth)[quiet]))
    return {"offset": offset, "tau_sensor": tau_sensor, "quantization_step": step}


__all__ = ["TemperatureSensor", "quantization_floor"]
