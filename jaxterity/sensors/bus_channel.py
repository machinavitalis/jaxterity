# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Bus transport channel: latency, jitter, and dropped packets.

Between a controller's decision and the joint that acts on it sits a serial
bus, and between the joint's state and the estimator that reads it sits the
same bus going the other way. On a daisy-chained servo bus the round trip is
milliseconds and it is *not* constant: the reply lands one bus period late, or
two, or not at all. Three consequences, each of which shows up as a wrong
*physical* parameter if the channel is not modeled:

* **Latency looks like phase lag**, and a fit that cannot see the lag pays for
  it in damping. :func:`phase_lag_deg` converts a latency into the phase error
  it produces at a given excitation frequency — at 5 ms and 10 Hz that is 18°,
  which a chirp-based identification will happily absorb as viscous friction it
  then reports as a physical property of the joint.
* **Jitter looks like process noise**, inflating the residual floor of every
  fit run on the channel and making a good model look mediocre.
* **Drops look like a stuck sensor**, because a consumer holds its last value:
  the trace goes flat for a sample or two, exactly like a quantized encoder
  that has not ticked.

Model — a packet channel on a fixed bus period ``dt``::

    every dt:  push u into a ring buffer
               k = clip(round((latency + jitter_k)/dt), 0, buffer−1)
               delivered = buffer[k]    unless dropped, in which case
               delivered = the previously delivered value  (consumer holds)
    output  =  delivered, held between updates

Both the jitter and the drop are drawn from an explicit PRNG key folded with
the update index (``jax.random.fold_in``), so the channel is stochastic but
*reproducible*: the same key and the same clock give the same packet history,
run after run and under ``jit``/``vmap`` (PATTERNS.md §5.5). There is no global
RNG anywhere in it.

Relationship to Jaxonomy. For a *fixed, smooth, interpolating* delay, use
Jaxonomy's ``TransportDelay`` (or ``VariableTransportDelay``) — this block does
not replace it and does not try to be differentiable in the delay. What it adds
is the packet-level behaviour a bus actually has: quantization of the delay to
the bus period, jitter, loss, and last-value hold.

Units
-----
s, Hz; the payload itself is unit-agnostic (one scalar channel).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from jaxonomy.framework.dependency_graph import DependencyTicket

from ..data.schema import BusLatencyTelemetry, ValidationReport
from ..errors import CalibrationError
from ..parameter import ParameterSpec
from ..sysid.protocol import SysidProtocol
from ._base import Sensor

_SYSID_PROTOCOL = SysidProtocol(
    name="bus-loopback-timing",
    excitation="chirp",
    required_columns=("sent", "received"),
    duration_s=60.0,
    sample_rate_hz=1000.0,
    setup_notes=(
        "Log the commanded signal at the sender and the same signal as it "
        "comes back from the device, on ONE clock — a host timestamp on the "
        "outgoing write and the device's echo of it. A step or a chirp both "
        "work; a chirp is better because it also exposes how the lag varies "
        "with frequency (it should not: a transport delay is frequency-flat, "
        "and any frequency dependence means you are looking at a filter, not "
        "a channel). Sample at least 10x the bus period. Run the capture with "
        "the bus at its production packet load — latency measured on an idle "
        "bus is not the latency your controller will see."
    ),
)


def phase_lag_deg(latency_s, frequency_hz):
    """Phase lag a pure transport delay produces at a frequency, in degrees.

    ``360·f·τ`` — linear in both, and the reason an unmodeled latency is not a
    small effect: it grows without bound with frequency, so the highest tone in
    a chirp is the one that misleads a fit the most.

    Args
    ----
    latency_s : float or jax.Array
        Transport delay, s.
    frequency_hz : float or jax.Array
        Excitation frequency, Hz.

    Returns
    -------
    jax.Array
        Phase lag, degrees.

    Units
    -----
    s, Hz → degrees.

    Examples
    --------
    >>> float(phase_lag_deg(0.005, 10.0))
    18.0
    """
    return 360.0 * frequency_hz * latency_s


class BusChannel(Sensor):
    """Packet transport with quantized latency, jitter, and drops.

    See the module docstring for the model and the PRNG contract. Input port
    ``sent`` is the payload at the sender; output port ``received`` is what the
    consumer sees (zero-order held between bus updates).

    Args
    ----
    dt : float
        Bus period, s. Must be > 0. Static structure (it sets the buffer
        length), not an identifiable leaf.
    latency_s : float or jax.Array
        Mean transport delay, s. Quantized to whole bus periods at delivery —
        which is itself a real effect, not an approximation.
    jitter_s : float or jax.Array
        Standard deviation of the per-packet delay, s.
    drop_probability : float or jax.Array
        Probability that a packet is lost and the consumer holds its previous
        value, in [0, 1).
    key : jax.Array | None
        PRNG key seeding the jitter/drop draws. Defaults to
        ``jax.random.PRNGKey(0)`` — explicit and reproducible, never a global
        RNG. Static (not a leaf).
    buffer_length : int | None
        Ring-buffer depth in packets. Defaults to
        ``ceil((latency + 4·jitter)/dt) + 2``, and delays beyond it are
        clipped rather than silently wrapped.
    initial_value : float
        Value the consumer holds before the first packet arrives.
    name : str
        Block name.

    Examples
    --------
    >>> channel = BusChannel(dt=1e-3, latency_s=5e-3)
    >>> channel.latency_samples
    5
    """

    sysid_protocol = _SYSID_PROTOCOL

    def __init__(
        self,
        *,
        dt: float = 1e-3,
        latency_s=0.0,
        jitter_s=0.0,
        drop_probability=0.0,
        key=None,
        buffer_length: int | None = None,
        initial_value: float = 0.0,
        name: str = "bus_channel",
    ):
        super().__init__(name=name)
        if dt <= 0:
            raise CalibrationError(
                f"BusChannel needs a positive bus period dt, got {dt}. dt is "
                "the packet clock; the latency is quantized to it."
            )
        self._dt = float(dt)
        self._key = jax.random.PRNGKey(0) if key is None else key
        if buffer_length is None:
            span = float(latency_s) + 4.0 * float(jitter_s)
            buffer_length = int(math.ceil(span / self._dt)) + 2
        self._buffer_length = max(int(buffer_length), 2)
        for n, x in (
            ("latency_s", latency_s),
            ("jitter_s", jitter_s),
            ("drop_probability", drop_probability),
        ):
            self.declare_dynamic_parameter(n, jnp.asarray(x, dtype=float))
        self.declare_input_port(name="sent")
        # State layout: [ring buffer (newest first), last delivered value].
        self.declare_discrete_state(
            default_value=jnp.full(self._buffer_length + 1, float(initial_value))
        )
        self.declare_periodic_update(self._update, period=self._dt, offset=0.0)
        self._output_index = self.declare_output_port(
            self._received,
            name="received",
            requires_inputs=False,
            prerequisites_of_calc=[DependencyTicket.xd],
            default_value=jnp.asarray(float(initial_value)),
        )

    @property
    def latency_s(self):
        return self.dynamic_parameters["latency_s"].get()

    @property
    def jitter_s(self):
        return self.dynamic_parameters["jitter_s"].get()

    @property
    def drop_probability(self):
        return self.dynamic_parameters["drop_probability"].get()

    @property
    def dt(self) -> float:
        """Bus period, s (static structure, not an identifiable leaf)."""
        return self._dt

    @property
    def buffer_length(self) -> int:
        """Ring-buffer depth in packets (static)."""
        return self._buffer_length

    @property
    def latency_samples(self) -> int:
        """Mean latency in whole bus periods (what actually gets delivered)."""
        return int(round(float(self.latency_s) / self._dt))

    @classmethod
    def uncalibrated(cls, *, dt: float = 1e-3, name: str = "bus_channel"):
        """Return an instance with identifiability-neutral defaults.

        A transparent channel: no latency, no jitter, no drops — so a fit can
        only add the impairment it measures.
        """
        return cls(dt=dt, name=name)

    def _update(self, time, state, *inputs, **params):
        buffer = state.discrete_state[: self._buffer_length]
        last = state.discrete_state[self._buffer_length]
        pushed = jnp.roll(buffer, 1).at[0].set(inputs[0])

        # Deterministic per-step randomness: fold the update index into the
        # block's key, so the packet history is reproducible under jit/vmap.
        step = jnp.round(time / self._dt).astype(jnp.int32)
        key = jax.random.fold_in(self._key, step)
        jitter_key, drop_key = jax.random.split(key)
        jitter = params["jitter_s"] * jax.random.normal(jitter_key)
        delay = jnp.clip(
            jnp.round((params["latency_s"] + jitter) / self._dt),
            0,
            self._buffer_length - 1,
        ).astype(jnp.int32)
        arriving = pushed[delay]
        dropped = jax.random.uniform(drop_key) < params["drop_probability"]
        delivered = jnp.where(dropped, last, arriving)
        return jnp.concatenate([pushed, delivered[None]])

    def _received(self, time, state, *inputs, **params):
        return state.discrete_state[self._buffer_length]

    def parameter_spec(self) -> ParameterSpec:
        return ParameterSpec(("latency_s", "jitter_s", "drop_probability"))

    def validate(self, telemetry: BusLatencyTelemetry) -> ValidationReport:
        """Compare the channel's declared timing to what a paired trace shows.

        Estimates the trace's latency, jitter and drop rate with
        :func:`_fit_bus_channel` and warns when any of them disagrees with the
        block's parameters by more than half a bus period (latency), a factor
        of two (jitter), or one percentage point (drops). Absolute thresholds
        rather than a tolerance-policy lookup: none of these quantities has a
        solver-dependent error, and half a bus period is the resolution the
        estimate genuinely has.

        Args
        ----
        telemetry : BusLatencyTelemetry
            The same signal at both ends of the transport.

        Returns
        -------
        ValidationReport
            ``ok`` iff all three agree within those thresholds.

        Units
        -----
        s.
        """
        measured = _fit_bus_channel(telemetry)
        report = ValidationReport()
        if abs(measured["latency_s"] - float(self.latency_s)) > 0.5 * self._dt:
            report.warnings.append(
                f"measured latency {measured['latency_s'] * 1e3:.3g} ms differs "
                f"from the declared {float(self.latency_s) * 1e3:.3g} ms by "
                f"more than half a bus period ({self._dt * 1e3:.3g} ms)."
            )
        declared_jitter = float(self.jitter_s)
        if measured["jitter_s"] > max(2.0 * declared_jitter, 0.5 * self._dt):
            report.warnings.append(
                f"measured jitter {measured['jitter_s'] * 1e3:.3g} ms exceeds "
                f"twice the declared {declared_jitter * 1e3:.3g} ms — the bus "
                "may be more loaded than when this channel was characterized."
            )
        if measured["drop_rate"] > float(self.drop_probability) + 0.01:
            report.warnings.append(
                f"measured drop rate {measured['drop_rate']:.2%} exceeds the "
                f"declared {float(self.drop_probability):.2%} by more than a "
                "percentage point."
            )
        return report


def _fit_bus_channel(telemetry: BusLatencyTelemetry) -> dict[str, float]:
    """Estimate latency, jitter, and drop rate from a paired send/receive trace.

    * **Latency** — the lag that maximizes the cross-correlation of the two
      (mean-removed) channels. Frequency-flat by construction, which is the
      point: a delay is a delay, and a lag that varies with frequency means the
      trace is showing a filter instead.
    * **Jitter** — from the residual left after shifting by the estimated
      latency. A timing error ``δ`` on a signal with slew rate ``ṡ`` shows up
      as an amplitude error ``δ·ṡ``, so ``jitter ≈ RMS(residual)/RMS(ṡ)``.
      A perfectly regular channel leaves no residual and reports zero, and a
      signal that barely moves reports (correctly) that it cannot see timing
      at all. Note this conflates jitter with any *other* residual — added
      noise, a nonlinearity — so read it as an upper bound on timing spread.
    * **Drop rate** — the fraction of samples where the received channel holds
      *exactly* its previous value while the sent channel moved. That is the
      observable signature of a lost packet under a last-value hold.

    Private to ``bus_channel.py`` (to make the recovery test runnable); the
    formal ``sysid`` recipe home is later work.

    Args
    ----
    telemetry : BusLatencyTelemetry
        The same signal at both ends of the transport.

    Returns
    -------
    dict[str, float]
        ``{"latency_s", "jitter_s", "drop_rate"}`` in s, s, and fraction.

    Raises
    ------
    CalibrationError
        If either channel is constant (no feature to correlate) or the trace
        is shorter than 4 samples.

    Units
    -----
    s.
    """
    t = np.asarray(telemetry.time, dtype=np.float64)
    sent = np.asarray(telemetry.sent, dtype=np.float64)
    received = np.asarray(telemetry.received, dtype=np.float64)
    if t.size < 4:
        raise CalibrationError(
            f"_fit_bus_channel needs at least 4 samples, got {t.size}."
        )
    if float(np.ptp(sent)) <= 0.0:
        raise CalibrationError(
            "_fit_bus_channel needs a varying sent signal to correlate "
            "against; this trace holds a constant value, which is delayed by "
            "any latency at all and therefore identifies none."
        )
    dt = float(np.median(np.diff(t)))

    a = sent - sent.mean()
    b = received - received.mean()
    correlation = np.correlate(b, a, mode="full")
    lags = np.arange(-a.size + 1, a.size)
    positive = lags >= 0
    peak_index = int(np.argmax(correlation[positive]))
    latency = float(lags[positive][peak_index] * dt)

    # Jitter from the aligned residual, converted through the slew rate.
    shift = int(round(latency / dt))
    aligned = received[shift:] if shift else received
    reference = sent[: sent.size - shift] if shift else sent
    slew = np.gradient(reference, dt)
    slew_rms = float(np.sqrt(np.mean(slew * slew)))
    residual = aligned - reference
    residual_rms = float(np.sqrt(np.mean(residual * residual)))
    jitter = residual_rms / slew_rms if slew_rms > 0 else 0.0

    moved = np.abs(np.diff(sent)) > 0
    held = np.diff(received) == 0.0
    drop_rate = float(np.mean(held[moved])) if np.any(moved) else 0.0
    return {"latency_s": latency, "jitter_s": jitter, "drop_rate": drop_rate}


__all__ = ["BusChannel", "phase_lag_deg"]
