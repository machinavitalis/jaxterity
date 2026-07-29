# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Current-sense channel: the path between a real current and a reported one.

Current is the cheapest torque estimate a robot has, and on a bus servo it is
the *only* one. Every model that reads torque from current — contact detection,
the ``k_i`` map feeding :mod:`jaxterity.physics.power_tree`, thermal power in
:mod:`jaxterity.physics.thermal_network` — is really reading this channel, and
inherits its errors. Four of them matter enough to model:

* **Gain error**, which is indistinguishable from a wrong torque constant. A
  ±3 % shunt tolerance is a ±3 % error in every torque you thought you
  measured.
* **Offset**, which turns into a phantom standing torque (and, squared into a
  thermal power proxy, a phantom heat source that never turns off).
* **Bandwidth.** A shunt-plus-amplifier chain rolls off at tens of kHz, a Hall
  sensor lower; the current a controller sees during an impact is not the
  current that flowed. Modeled as one pole, so the channel has *state*.
* **Quantization and saturation.** The ADC has finite bits and finite range;
  a clipped current reads as a torque ceiling that is not the motor's.

Model — one state (the filtered current), stateless output map on top::

    i̇_filt = 2π·f_c · (i_true − i_filt)                  (single-pole channel)
    reading = quantize( clip(gain·i_filt + offset, ±full_scale), step )
    step    = 2·full_scale / 2^bits

Quantization is a hard :func:`jnp.round`, deliberately — the same choice
:class:`~jaxterity.sensors.encoder.Encoder` makes and for the same reason: a
smoothed quantizer would mis-model the very artifact a calibration is trying to
see. The reading is therefore not differentiable at code boundaries; the fits
that use it are residual-statistics fits, not gradient descent on the reading.

Noise is metadata, not forward model (again matching ``Encoder`` and ``IMU``):
``noise_density`` is carried so a recipe can weight residuals and compute the
identifiability floor, but the block is deterministic.

Units
-----
A, Hz, A/√Hz; ``gain`` is dimensionless.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..data.schema import CurrentSenseTelemetry, ValidationReport
from ..errors import CalibrationError
from ..parameter import ParameterSpec
from ..sysid.protocol import SysidProtocol
from ._base import Sensor

_SYSID_PROTOCOL = SysidProtocol(
    name="current-sense-reference-sweep",
    excitation="ramp",
    required_columns=("truth", "measured"),
    duration_s=20.0,
    sample_rate_hz=2000.0,
    setup_notes=(
        "Put a reference instrument (calibrated shunt or clamp meter) in "
        "series with the sensor under test. (1) ZERO: record with the bus "
        "de-energized — the mean reading is the offset, and it must be taken "
        "first because every later estimate rides on it. (2) SLOW RAMP over "
        "the full range in both polarities: the slope is the gain, the "
        "flat spots at each end are the saturation limits, and the histogram "
        "of (measured - truth) has a comb whose spacing is the quantization "
        "step. (3) STEP: a fast load step identifies the bandwidth from the "
        "reading's rise time — this segment needs a reference instrument "
        "faster than the sensor under test, or it measures the reference."
    ),
)


def quantize(x, step):
    """Round to the nearest multiple of ``step`` (``step <= 0`` passes through).

    Args
    ----
    x : float or jax.Array
        The value to quantize.
    step : float or jax.Array
        Quantization step in the same unit as ``x``. Non-positive disables it.

    Returns
    -------
    jax.Array
        The quantized value.

    Units
    -----
    Unit-agnostic: output carries the unit of ``x``.
    """
    return jnp.where(step > 0, jnp.round(x / jnp.where(step > 0, step, 1.0)) * step, x)


def adc_step(full_scale, bits: int):
    """Quantization step of a bipolar ADC: ``2·full_scale / 2**bits``.

    Args
    ----
    full_scale : float or jax.Array
        Positive full-scale reading (the range is ``±full_scale``).
    bits : int
        ADC resolution in bits.

    Returns
    -------
    jax.Array
        Least-significant-bit size, in the unit of ``full_scale``.

    Units
    -----
    Same unit as ``full_scale``.

    Examples
    --------
    >>> float(adc_step(20.0, 12))  # +/-20 A on 12 bits
    0.009765625
    """
    return 2.0 * full_scale / float(2**bits)


class CurrentSense(Sensor):
    """Current-sense channel with gain, offset, bandwidth, quantization, clip.

    See the module docstring for the model and the deliberate hard quantizer.
    Input port ``truth`` is the actual current (A); output port ``reading`` is
    what the drive reports (A). State is the single-pole filtered current.

    Args
    ----
    gain : float or jax.Array
        Channel gain, dimensionless (1 = perfect).
    offset : float or jax.Array
        Additive offset referred to the output, A.
    bandwidth_hz : float or jax.Array
        Single-pole cutoff, Hz. Must be > 0.
    full_scale : float or jax.Array
        Positive full-scale current, A; readings clip at ``±full_scale``.
    quantization_step : float or jax.Array
        Reading resolution, A. 0 disables quantization; use :func:`adc_step`
        to derive it from bits and range.
    noise_density : float or jax.Array
        White-noise density, A/√Hz. Sysid metadata; not in the forward model.
    i0 : float
        Initial filtered current, A.
    name : str
        Block name.

    Examples
    --------
    >>> channel = CurrentSense(full_scale=20.0, quantization_step=adc_step(20.0, 12))
    >>> tuple(channel.parameter_spec())
    ('gain', 'offset', 'bandwidth_hz', 'full_scale', 'quantization_step')
    """

    sysid_protocol = _SYSID_PROTOCOL

    def __init__(
        self,
        *,
        gain=1.0,
        offset=0.0,
        bandwidth_hz=10_000.0,
        full_scale=20.0,
        quantization_step=0.0,
        noise_density=0.0,
        i0: float = 0.0,
        name: str = "current_sense",
    ):
        super().__init__(name=name)
        for n, x in (
            ("gain", gain),
            ("offset", offset),
            ("bandwidth_hz", bandwidth_hz),
            ("full_scale", full_scale),
            ("quantization_step", quantization_step),
            ("noise_density", noise_density),
        ):
            self.declare_dynamic_parameter(n, jnp.asarray(x, dtype=float))
        self.declare_input_port(name="truth")
        self.declare_continuous_state(
            default_value=jnp.asarray(i0, dtype=float), ode=self._ode
        )
        self.declare_output_port(self._read, name="reading", requires_inputs=False)

    @property
    def gain(self):
        return self.dynamic_parameters["gain"].get()

    @property
    def offset(self):
        return self.dynamic_parameters["offset"].get()

    @property
    def bandwidth_hz(self):
        return self.dynamic_parameters["bandwidth_hz"].get()

    @property
    def full_scale(self):
        return self.dynamic_parameters["full_scale"].get()

    @property
    def quantization_step(self):
        return self.dynamic_parameters["quantization_step"].get()

    @property
    def noise_density(self):
        return self.dynamic_parameters["noise_density"].get()

    @classmethod
    def uncalibrated(cls, *, name: str = "current_sense") -> "CurrentSense":
        """Return an instance with identifiability-neutral defaults.

        Unit gain, zero offset, no quantization — a transparent channel, so a
        fit can only add the error it measures. The bandwidth and full scale
        keep hardware-plausible values (zero would make the channel
        degenerate, not neutral).
        """
        return cls(name=name)

    def _ode(self, time, state, *inputs, **params):
        return (
            2.0 * jnp.pi * params["bandwidth_hz"] * (inputs[0] - state.continuous_state)
        )

    def _read(self, time, state, *inputs, **params):
        scaled = params["gain"] * state.continuous_state + params["offset"]
        clipped = jnp.clip(scaled, -params["full_scale"], params["full_scale"])
        return quantize(clipped, params["quantization_step"])

    def parameter_spec(self) -> ParameterSpec:
        return ParameterSpec(
            ("gain", "offset", "bandwidth_hz", "full_scale", "quantization_step")
        )

    def validate(self, telemetry: CurrentSenseTelemetry) -> ValidationReport:
        """Score the *static* channel map against a reference sweep.

        Applies gain, offset, clipping and quantization to the reference
        channel and compares to the reading. The bandwidth is deliberately not
        exercised here: a slow reference sweep (the protocol's segment 2)
        contains no information about it, and pretending otherwise would let a
        wrong pole hide inside a passing report. Use the step segment and a
        forward simulation for that.

        Args
        ----
        telemetry : CurrentSenseTelemetry
            Reference vs measured current.

        Returns
        -------
        ValidationReport
            ``ok`` iff the RMS residual is within the ``sensor_reading`` policy
            bound, widened by the quantization floor (a channel cannot be more
            accurate than half an LSB).

        Units
        -----
        A.
        """
        from ..testing import tolerances as tol

        truth = np.asarray(telemetry.truth, dtype=np.float64)
        measured = np.asarray(telemetry.measured, dtype=np.float64)
        predicted = np.asarray(
            quantize(
                jnp.clip(
                    float(self.gain) * jnp.asarray(truth) + float(self.offset),
                    -float(self.full_scale),
                    float(self.full_scale),
                ),
                float(self.quantization_step),
            )
        )
        residual = predicted - measured
        rms = float(np.sqrt(np.mean(residual * residual)))
        report = ValidationReport()
        bound = tol.atol("sensor_reading", "rk4", "float64") * 1e4 + float(
            self.quantization_step
        )
        if rms > bound:
            report.warnings.append(
                f"current-sense residual RMS={rms:.3g} A exceeds bound "
                f"{bound:.3g} A — gain, offset, or the quantization step may "
                "not match this channel."
            )
        return report


def _fit_current_sense(telemetry: CurrentSenseTelemetry) -> dict[str, float]:
    """Recover gain and offset from a reference sweep (unsaturated samples).

    ``measured = gain·truth + offset`` is linear, so OLS over the samples that
    are not clipped returns both. Saturated samples are excluded by dropping
    the extremes of the measured range: including them would bend the fitted
    gain toward zero. The quantization step is *not* fitted from this
    regression — it is read off the spacing of the residual comb, which
    :func:`quantization_step_estimate` does.

    Private to ``current_sense.py`` (to make the recovery test runnable); the
    formal ``sysid`` recipe home is later work.

    Args
    ----
    telemetry : CurrentSenseTelemetry
        Reference vs measured current.

    Returns
    -------
    dict[str, float]
        ``{"gain": ..., "offset": ...}``.

    Raises
    ------
    CalibrationError
        If the reference current never varies (gain and offset then trade off
        exactly).

    Units
    -----
    A.
    """
    truth = np.asarray(telemetry.truth, dtype=np.float64)
    measured = np.asarray(telemetry.measured, dtype=np.float64)
    if float(np.ptp(truth)) <= 0.0:
        raise CalibrationError(
            "_fit_current_sense needs a varying reference current to separate "
            "gain from offset; this trace holds a constant "
            f"{float(truth[0]):.3g} A. Fix: record a ramp over the full range."
        )
    limit = 0.999 * float(np.max(np.abs(measured)))
    unsaturated = np.abs(measured) < limit
    if np.count_nonzero(unsaturated) < 2:
        unsaturated = np.ones_like(measured, dtype=bool)
    design = np.stack([truth[unsaturated], np.ones_like(truth[unsaturated])], axis=1)
    coefficients, *_ = np.linalg.lstsq(design, measured[unsaturated], rcond=None)
    return {"gain": float(coefficients[0]), "offset": float(coefficients[1])}


def quantization_step_estimate(telemetry: CurrentSenseTelemetry) -> float:
    """Estimate the reading's quantization step from its distinct levels.

    A quantized channel takes values on a lattice; the smallest non-zero gap
    between sorted distinct readings is that lattice's spacing. Model-free, and
    a good sanity check against the ADC bits the datasheet claims — drives
    routinely report fewer effective bits than they advertise.

    The sweep must be slow enough that consecutive readings differ by at most
    one code. A trace that jumps ten codes per sample never *lands* on adjacent
    levels, and this returns a multiple of the true step — so ramp slowly, or
    treat a suspiciously round answer as a sampling artifact.

    Args
    ----
    telemetry : CurrentSenseTelemetry
        A trace whose measured channel sweeps over many codes.

    Returns
    -------
    float
        Estimated step, A (0.0 if the reading appears continuous).

    Units
    -----
    A.
    """
    measured = np.unique(np.asarray(telemetry.measured, dtype=np.float64))
    if measured.size < 2:
        return 0.0
    gaps = np.diff(measured)
    positive = gaps[gaps > 0]
    return float(np.min(positive)) if positive.size else 0.0


__all__ = ["CurrentSense", "adc_step", "quantization_step_estimate", "quantize"]
