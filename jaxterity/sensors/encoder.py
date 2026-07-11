# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Joint-position encoder with quantization and drift.

``Encoder`` models the position-sensor family the SO-100, Crazyflie's IMU
attitude pipe, and most robots use: a continuous joint position is quantized
to a discrete count, with a slow bias drift on top. Three identifiable
parameters:

* ``quantization_step`` (rad): the encoder count resolution. 12-bit absolute
  encoder over 2π → ~1.5e-3 rad; 14-bit → ~4e-4 rad.
* ``drift_rate`` (rad/s): linear-in-time bias drift. Real encoders drift on
  thermal time scales; the linear model is a first-order approximation that
  fits over the duration of a single experiment.
* ``noise_density`` (rad/√Hz): white-noise density on the reading. Sysid
  metadata — not in forward model; see :class:`~jaxterity.sensors.imu.IMU`
  for the rationale.

Model — stateless::

    out = quantize(in + drift_rate · t, step)

Quantization is :func:`jnp.round` to the nearest step. It is *not*
differentiable at step boundaries — the encoder is fundamentally a hard
quantizer and a smooth approximation would mis-model the artifact a sysid
recipe is trying to identify. The non-differentiability surfaces only if a
caller tries to gradient-trace the encoder reading; the sysid recipe
uses residual-magnitude statistics (not gradient descent on the readings).

Scope — what this model does *not* include:

* **Latency.** A real encoder lags reality by a fixed delay. Modeling that
  requires a discrete-time delay block (Jaxonomy's ``UnitDelay`` or similar)
  composed externally; the stateless Encoder doesn't carry the buffer.
  Deferred.
* **Incremental vs absolute distinction.** This is an *absolute* encoder
  model. An incremental encoder additionally has a startup-undefined zero
  offset and accumulates count errors on missed pulses. Add when the zoo
  needs it.
* **Magnetic encoder hysteresis.** AS5048-class magnetic encoders show a
  small hysteresis (~1 LSB) that the model collapses into ``noise_density``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..data.schema import EncoderTelemetry, ValidationReport
from ..parameter import ParameterSpec
from ..sysid.protocol import SysidProtocol
from ._base import Sensor

_SYSID_PROTOCOL = SysidProtocol(
    name="encoder-reference-sweep",
    excitation="ramp",
    required_columns=("truth", "measured"),
    duration_s=10.0,
    sample_rate_hz=500.0,
    setup_notes=(
        "Drive the joint through a slow sinusoidal sweep covering its full "
        "range, recording the encoder under test against a higher-resolution "
        "reference (e.g. an instrumented fixture). quantization_step is the "
        "common difference in the (measured − truth) histogram; drift_rate "
        "is the linear-in-time slope of the residual; noise_density is the "
        "PSD of the high-frequency residual."
    ),
)


def _quantize(x, step):
    """Round x to the nearest multiple of step. ``step <= 0`` passes through."""
    return jnp.where(step > 0, jnp.round(x / step) * step, x)


class Encoder(Sensor):
    """Absolute joint-position encoder with quantization and linear drift.

    Args
    ----
    quantization_step : float or jax.Array
        Encoder resolution, rad. Set to 0 to disable quantization.
    drift_rate : float or jax.Array
        Linear drift in the bias, rad/s.
    noise_density : float or jax.Array
        White-noise density, rad/√Hz. Sysid metadata; not in forward model.
    name : str
        Block name.
    """

    sysid_protocol = _SYSID_PROTOCOL

    def __init__(
        self,
        *,
        quantization_step=0.0,
        drift_rate=0.0,
        noise_density=0.0,
        name: str = "encoder",
    ):
        super().__init__(name=name)
        for n, x in (
            ("quantization_step", quantization_step),
            ("drift_rate", drift_rate),
            ("noise_density", noise_density),
        ):
            self.declare_dynamic_parameter(n, jnp.asarray(x, dtype=float))
        self.declare_input_port(name="truth")
        self.declare_output_port(self._read, name="reading", requires_inputs=True)

    @property
    def quantization_step(self):
        return self.dynamic_parameters["quantization_step"].get()

    @property
    def drift_rate(self):
        return self.dynamic_parameters["drift_rate"].get()

    @property
    def noise_density(self):
        return self.dynamic_parameters["noise_density"].get()

    @classmethod
    def uncalibrated(cls, *, name: str = "encoder") -> "Encoder":
        """Return an instance that passes through perfectly (all params zero)."""
        return cls(name=name)

    def _read(self, time, state, *inputs, **params):
        return _quantize(
            inputs[0] + params["drift_rate"] * time,
            params["quantization_step"],
        )

    def parameter_spec(self) -> ParameterSpec:
        return ParameterSpec(("quantization_step", "drift_rate", "noise_density"))

    def validate(self, telemetry: EncoderTelemetry) -> ValidationReport:
        """Residuals of (measured − model(truth)) against the noise floor.

        Applies the model to ``telemetry.truth`` and compares to
        ``telemetry.measured``. Per-sample residuals decompose into
        quantization noise (bounded by step/2) + drift (linear-in-time) + the
        identified noise process. A warning is raised if the RMS residual
        exceeds the larger of (step/√3, 3 · noise_density · √(fs/2)).
        """
        from ..testing import tolerances as tol

        t = np.asarray(telemetry.time, dtype=np.float64)
        truth = np.asarray(telemetry.truth, dtype=np.float64)
        meas = np.asarray(telemetry.measured, dtype=np.float64)
        step = float(self.quantization_step)
        drift = float(self.drift_rate)
        nd = float(self.noise_density)
        fs = telemetry.sample_rate_hz
        biased = truth + drift * t
        pred = np.round(biased / step) * step if step > 0 else biased
        residual = meas - pred
        rms = float(np.sqrt(np.mean(residual * residual)))
        # Uniform quantization noise variance: step² / 12 → RMS = step/√12.
        # Use the looser step/2 + 3·noise·√(fs/2) + tolerance floor.
        bound = max(
            step / np.sqrt(12.0) * 3.0,
            3.0 * nd * np.sqrt(fs / 2.0),
            tol.atol("joint_position", "rk4", "float64") * 1e3,
        )
        report = ValidationReport()
        if rms > bound:
            report.warnings.append(
                f"encoder residual RMS={rms:.3g} rad exceeds bound "
                f"{bound:.3g} rad — quantization_step, drift_rate, or "
                "noise_density may not match this trace."
            )
        return report


def make_latent_encoder(
    *,
    latency_samples: int,
    sample_dt: float,
    initial_reading: float = 0.0,
    encoder: "Encoder | None" = None,
    name: str = "latent_encoder",
):
    """Compose an :class:`Encoder` with N ``UnitDelay`` blocks.

    Returns a ``jaxonomy.Diagram`` with one input port (``truth``) and one
    output port (``reading``). Internally: truth → Encoder → UnitDelay×N → out.
    The composed system reports the encoder's reading lagged by
    ``latency_samples · sample_dt`` seconds — the standard discrete-time
    transport-delay model.

    Use this rather than a stateful ``Encoder`` subclass: keeping the encoder
    stateless preserves LeafSystem composability (the encoder is reusable
    standalone and inside other diagrams), and the delay block is the
    canonical Jaxonomy primitive for exactly this pattern.

    Args
    ----
    latency_samples : int
        Number of sample periods of delay. ``1`` means one-step lag.
    sample_dt : float
        The period at which the upstream system samples the encoder, in seconds.
    initial_reading : float
        Value the delay buffer holds before the first sample arrives.
    encoder : Encoder or None
        Optional pre-built encoder (e.g. with a custom quantization step). If
        ``None``, an uncalibrated :class:`Encoder` is constructed.
    name : str
        Name for the wrapping diagram.

    Raises
    ------
    JaxterityError
        If ``latency_samples < 0``.

    Examples
    --------
    >>> # 5 ms latency at 200 Hz sampling
    >>> diagram = make_latent_encoder(latency_samples=1, sample_dt=1/200)
    """
    import jax.numpy as jnp
    import jaxonomy  # local: keeps the module import-light when this helper is unused
    from jaxonomy.library import UnitDelay

    from ..errors import JaxterityError

    if latency_samples < 0:
        raise JaxterityError(
            f"make_latent_encoder: latency_samples must be ≥ 0, got {latency_samples}."
        )
    enc = encoder if encoder is not None else Encoder()
    b = jaxonomy.DiagramBuilder()
    b.add(enc)
    prev = enc.output_ports[0]
    for i in range(latency_samples):
        d = UnitDelay(
            dt=sample_dt,
            initial_state=jnp.asarray(initial_reading, dtype=float),
            name=f"delay_{i}",
        )
        b.add(d)
        b.connect(prev, d.input_ports[0])
        prev = d.output_ports[0]
    b.export_input(enc.input_ports[0], name="truth")
    b.export_output(prev, name="reading")
    return b.build(name=name)


__all__ = ["Encoder", "make_latent_encoder"]
