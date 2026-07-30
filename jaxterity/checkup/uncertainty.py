# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Uncertainty that survives a re-fit on part of the same capture.

:func:`jaxterity.checkup.probe.fit_stability` measured something awkward on the
committed LK-0001 captures: sub-window re-fits agree with *each other* to ~3 %
while landing systematically **outside** the full capture's 95 % Wald interval
(``shoulder_pan`` 0.4150 [0.396, 0.434] against sub-window values 0.454 / 0.486 /
0.478). Nothing is wrong with the least-squares covariance — it answers "given
this much data and this noise model, how well is the parameter pinned?". It does
not answer "how much does the answer depend on *how much* data I took", and on
these captures that second term is the larger one.

Parameters-with-intervals is the differentiator against every learned-surrogate
competitor, so an interval a trivial re-fit escapes is worse than no interval: it
is a quantified claim that is wrong. This module reports a **total** uncertainty
with both components kept visible:

    total_half = sqrt(wald_half² + bias_half²)

``wald_half`` is the within-capture interval; ``bias_half`` is the largest
departure of a shorter-window re-fit from the full-capture estimate. The
combination is deliberately a quadrature sum of a *statistical* and a
*systematic* term, and because ``total_half >= bias_half`` the reported interval
covers every sub-window re-fit by construction — which is the acceptance test
this exists to pass.

Why not a bootstrap
-------------------
The obvious move is a block bootstrap over excitation cycles: resample cycles
with replacement, re-fit, take the spread. It is wrong here. The fit is a
*dynamical* one — it reads velocity and acceleration out of a continuous
trajectory — so stitching resampled blocks together fabricates discontinuities at
every join, and the fit sees those step changes as enormous accelerations. The
resulting spread measures the splicing, not the parameter.

What is sound on a dynamical capture is **contiguous** subsampling: re-fit on
unbroken sub-intervals of several lengths and look at how the estimate moves with
length (:func:`length_ladder`). That keeps every re-fit physically valid, and
length dependence is exactly the effect the Wald interval misses. So the ladder
is the estimator here, and the bias term is its spread, not a resampling
distribution.

Units
-----
Half-widths carry the channel's units (N·m·s/rad for viscous damping, N·m for
Coulomb friction). Fractions are dimensionless.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from ..errors import CalibrationError
from .probe import StabilityOfFit, fit_stability
from .report import Gate

#: Share of the total *variance* above which the systematic (length) term is
#: called dominant. Half, which is exactly where the two components are equal —
#: and the reason the share is defined on variances rather than half-widths: the
#: total is a quadrature sum, so equal components split it 0.71/0.71 in amplitude
#: but 0.5/0.5 in variance. Only the variance form makes "half" mean half.
BIAS_DOMINANT_FRACTION = 0.5

#: Total relative half-width past which a channel is called poorly determined.
#: Matches the existing checkup confidence gate's 25 %, so adding the systematic
#: term tightens the same gate rather than introducing a second standard.
CONFIDENCE_RELATIVE_LIMIT = 0.25

#: Default contiguous-window lengths for :func:`length_ladder`, as fractions of
#: the capture. Spans a 2.5x range so length dependence has room to show, and
#: stops at 1.0 because the full capture is the estimate being judged.
DEFAULT_LADDER = (0.4, 0.6, 0.8, 1.0)


@dataclasses.dataclass(frozen=True)
class TotalUncertainty:
    """A channel's statistical and systematic uncertainty, kept separate.

    Attributes
    ----------
    channel : str
        ``"<joint>.damping"`` / ``"<joint>.friction"``.
    value : float
        The full-capture estimate.
    wald_half : float
        Half-width of the fit's own 95 % interval — the within-capture term. NaN
        when the fit reported no interval.
    bias_half : float
        Largest absolute departure of a shorter-window re-fit from ``value`` —
        the capture-length term.
    n_windows : int
        How many shorter-window re-fits the bias term is built from.
    """

    channel: str
    value: float
    wald_half: float
    bias_half: float
    n_windows: int

    @property
    def total_half(self) -> float:
        """Combined half-width, ``sqrt(wald² + bias²)``.

        A missing Wald term (NaN) is treated as zero here, so the total falls
        back to the systematic term alone rather than poisoning the interval:
        "we only know the length dependence" is still a usable statement, and
        :attr:`wald_half` stays NaN so the omission is visible.
        """
        wald = 0.0 if not np.isfinite(self.wald_half) else self.wald_half
        bias = 0.0 if not np.isfinite(self.bias_half) else self.bias_half
        return float(np.hypot(wald, bias))

    @property
    def ci95(self) -> tuple[float, float]:
        """``value ± total_half`` — the interval to report."""
        half = self.total_half
        return (self.value - half, self.value + half)

    @property
    def relative(self) -> float:
        """Total half-width relative to the value (NaN at a zero value)."""
        if self.value == 0:
            return float("nan")
        return self.total_half / abs(self.value)

    @property
    def bias_fraction(self) -> float:
        """Share of the total *variance* contributed by the systematic term.

        ``bias² / (wald² + bias²)``, so 0.5 means the two components are equal
        and 1.0 means the reported interval is *entirely* capture-length
        dependence — the least-squares covariance having told you nothing about
        the real uncertainty. Defined on variances, not half-widths, because the
        total is a quadrature sum: as an amplitude ratio, equal components would
        read 0.71 and "half the uncertainty is systematic" would have no
        expressible threshold.
        """
        wald = 0.0 if not np.isfinite(self.wald_half) else self.wald_half
        bias = 0.0 if not np.isfinite(self.bias_half) else self.bias_half
        # From the components, not from total_half**2: squaring the hypot back
        # costs an epsilon that puts equal components at 0.49999999999999994 and
        # so on the wrong side of the "half" threshold.
        denominator = wald**2 + bias**2
        if denominator == 0:
            return float("nan")
        return float(bias**2 / denominator)

    @property
    def bias_dominates(self) -> bool:
        """Whether the systematic term is at least half the total."""
        fraction = self.bias_fraction
        return bool(np.isfinite(fraction) and fraction >= BIAS_DOMINANT_FRACTION)

    def covers(self, value: float) -> bool:
        """Whether *value* falls inside the reported interval."""
        lo, hi = self.ci95
        return bool(lo <= value <= hi)

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON form, with the derived quantities materialized.

        The derived fields are written out rather than left to the reader: a
        stored interval whose two components are only recoverable by rerunning
        the arithmetic invites someone to quote ``wald_half`` as *the* interval,
        which is the bug this module exists to fix.
        """
        d = dataclasses.asdict(self)
        d["total_half"] = self.total_half
        d["ci95"] = list(self.ci95)
        d["relative"] = self.relative
        d["bias_fraction"] = self.bias_fraction
        d["bias_dominates"] = self.bias_dominates
        return d


def total_uncertainty(stability: StabilityOfFit) -> TotalUncertainty:
    """Combine a channel's Wald interval with its sub-window bias.

    Args
    ----
    stability : StabilityOfFit
        One channel's ladder, from :func:`jaxterity.checkup.probe.fit_stability`
        or :func:`length_ladder`.

    Returns
    -------
    TotalUncertainty
        With both components visible and an interval that covers every window in
        *stability* by construction.

    Raises
    ------
    CalibrationError
        If the ladder carries no sub-window values, so there is no systematic
        term to report — returning the Wald interval alone under a name promising
        a total would be the original error wearing a new label.
    """
    if not stability.window_values:
        raise CalibrationError(
            f"total_uncertainty: {stability.channel} has no sub-window re-fits, "
            "so the capture-length term is unmeasured. Run fit_stability (or "
            "length_ladder) on the capture first — the Wald interval alone is "
            "what this function exists to correct, so it will not be dressed up "
            "as a total."
        )
    wald = (
        (stability.ci95[1] - stability.ci95[0]) / 2.0
        if stability.ci95 is not None
        else float("nan")
    )
    bias = max(abs(v - stability.full) for v in stability.window_values)
    return TotalUncertainty(
        channel=stability.channel,
        value=float(stability.full),
        wald_half=float(wald),
        bias_half=float(bias),
        n_windows=len(stability.window_values),
    )


def length_ladder(
    robot,
    capture: Any,
    *,
    fractions: tuple[float, ...] = DEFAULT_LADDER,
) -> tuple[StabilityOfFit, ...]:
    """Re-fit on contiguous prefixes of several lengths, per channel.

    Where :func:`jaxterity.checkup.probe.fit_stability` slides a *fixed*-length
    window along the capture (answering "does the answer depend on *which* part
    I used?"), this varies the length (answering "does it depend on *how much* I
    used?"). The second question is the one the Wald interval cannot see, so this
    is the estimator behind the systematic term.

    Every window is a contiguous prefix — see the module docstring on why
    resampled blocks are not an option for a dynamical fit.

    Args
    ----
    robot : Robot
        Model the fit runs against.
    capture : CommandedCapture
        A parsed capture.
    fractions : tuple[float, ...]
        Prefix lengths as fractions of the capture. Values of 1.0 are dropped
        from the *window* set (the full capture is the estimate, not a window).

    Returns
    -------
    tuple[StabilityOfFit, ...]
        One per channel, with ``window_values`` in ascending length order and
        ``full`` / ``ci95`` from the whole capture.

    Raises
    ------
    CalibrationError
        If fewer than two usable fractions remain, or any fraction is outside
        (0, 1].
    """
    if any(not 0.0 < f <= 1.0 for f in fractions):
        raise CalibrationError(
            f"length_ladder: fractions must lie in (0, 1], got {fractions}."
        )
    lengths = tuple(sorted({f for f in fractions if f < 1.0}))
    if len(lengths) < 2:
        raise CalibrationError(
            "length_ladder: needs at least two prefix lengths below the full "
            f"capture to see length dependence, got {lengths} from {fractions}."
        )

    # One sliding-window pass per length, each with a single window anchored at
    # the start, reusing fit_stability's slicing and validation.
    per_channel: dict[str, list[float]] = {}
    full: dict[str, float] = {}
    intervals: dict[str, tuple[float, float] | None] = {}
    for fraction in lengths:
        for entry in fit_stability(robot, capture, windows=2, window_fraction=fraction):
            per_channel.setdefault(entry.channel, [])
            # windows=2 gives a prefix and a suffix window of this length; keep
            # the prefix, which is the "I stopped the capture early" case.
            if entry.window_values:
                per_channel[entry.channel].append(entry.window_values[0])
            full[entry.channel] = entry.full
            intervals[entry.channel] = entry.ci95

    return tuple(
        StabilityOfFit(
            channel=channel,
            full=full[channel],
            window_values=tuple(values),
            ci95=intervals[channel],
        )
        for channel, values in sorted(per_channel.items())
    )


def uncertainty_gates(
    uncertainties: tuple[TotalUncertainty, ...] | list[TotalUncertainty],
) -> list[Gate]:
    """Confidence gates over total uncertainties.

    Two gates per channel, and the split matters:

    * ``confidence`` — **warn** when the *total* relative half-width exceeds
      :data:`CONFIDENCE_RELATIVE_LIMIT`. Same threshold as the existing checkup
      confidence gate, now applied to an interval that includes the length term,
      so channels that only looked well determined stop looking that way.
    * ``uncertainty_composition`` — **warn** when the systematic term is at least
      :data:`BIAS_DOMINANT_FRACTION` of the total. Separate from ``confidence``
      on purpose: a channel can be perfectly *tight* and still have essentially
      all of its uncertainty come from capture length, and that reader needs to
      know their number will move if they shorten the probe, even though nothing
      is out of tolerance.

    Neither is a **fail**. Both describe how well something is known rather than
    whether the robot is healthy, and failing a unit for the epistemics of its
    own measurement would be a category error.

    Args
    ----
    uncertainties : sequence of TotalUncertainty
        Channels to judge.

    Returns
    -------
    list[Gate]
        In the checkup's own :class:`~jaxterity.checkup.report.Gate` form.
    """
    gates: list[Gate] = []
    for item in uncertainties:
        channel = item.channel
        relative = item.relative
        if np.isfinite(relative) and relative > CONFIDENCE_RELATIVE_LIMIT:
            gates.append(
                Gate(
                    gate="confidence",
                    scope=channel,
                    level="warn",
                    detail=(
                        f"total half-width is {relative:.0%} of the value "
                        f"({item.total_half:.4g} on {item.value:.4g}), of which "
                        f"{item.bias_fraction:.0%} is capture-length dependence "
                        f"rather than fit noise."
                    ),
                    remedy="Capture longer, or excite this channel harder — a "
                    "tighter Wald interval alone will not close this.",
                )
            )
        if item.bias_dominates:
            gates.append(
                Gate(
                    gate="uncertainty_composition",
                    scope=channel,
                    level="warn",
                    detail=(
                        f"{item.bias_fraction:.0%} of the reported interval is "
                        f"capture-length dependence (bias {item.bias_half:.4g} vs "
                        f"Wald {item.wald_half:.4g}), measured over "
                        f"{item.n_windows} shorter re-fits."
                    ),
                    remedy="Quote the total interval, not the least-squares one, "
                    "and do not compare this channel against a fit from a "
                    "different capture length.",
                )
            )
    return gates


def uncertainty_summary(
    uncertainties: tuple[TotalUncertainty, ...] | list[TotalUncertainty],
) -> str:
    """Terminal-friendly table: value, both components, and their split."""
    lines = [
        f"  {'channel':28s} {'value':>10} {'wald':>10} {'bias':>10} "
        f"{'total':>10} {'bias%':>6}"
    ]
    for item in uncertainties:
        lines.append(
            f"  {item.channel:28s} {item.value:10.4g} {item.wald_half:10.4g} "
            f"{item.bias_half:10.4g} {item.total_half:10.4g} "
            f"{item.bias_fraction:5.0%}"
        )
    return "\n".join(lines)


__all__ = [
    "BIAS_DOMINANT_FRACTION",
    "CONFIDENCE_RELATIVE_LIMIT",
    "DEFAULT_LADDER",
    "TotalUncertainty",
    "length_ladder",
    "total_uncertainty",
    "uncertainty_gates",
    "uncertainty_summary",
]
