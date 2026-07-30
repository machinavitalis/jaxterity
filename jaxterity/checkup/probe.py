# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Score a checkup probe before it moves, and measure how stable its fit is.

Two questions stand between the current ~2.5-minute checkup session and a
shorter one, and both have to be answered with numbers rather than intuition.

**"Can this planned segment identify anything?"** — answerable *before any
motion*, because the plan fully determines the commanded trajectory. The
closed-loop friction fit (:func:`~jaxterity.calibrate.identify_closed_loop`) is
a four-column linear regression whose two physical columns are functions of the
commanded velocity alone: ``−q̇`` for viscous damping and ``−tanh(q̇/ε)`` for
Coulomb friction. :func:`segment_identifiability` builds exactly those columns
from the planned waveform and reports their conditioning and per-channel
marginal variance. A segment that cannot separate damping from Coulomb friction
says so here, for free, instead of after 16 seconds of arm motion and a fit.

**"Would a shorter segment give the same answer?"** — measured, and the answer on
the committed real captures is **no**, in a specific and instructive way.
:func:`fit_stability` re-fits a capture over sliding sub-windows. The sub-window
estimates agree closely *with each other* and sit systematically **outside** the
full capture's own 95 % Wald interval:

* ``shoulder_pan`` damping: full capture 0.4150 N·m·s/rad, CI [0.396, 0.434];
  10 s sub-windows give 0.454, 0.486, 0.478 — a consistent +10 to +17 %, every
  one of them past the interval's upper edge.
* ``elbow_flex`` damping: full 0.0674, CI [0.060, 0.075]; sub-windows 0.0775,
  0.0813, 0.0814 — again all high, all outside.

So this is not noise: the sub-windows' mutual spread (~3 %) is *smaller* than the
interval, while their offset from the full fit is larger than it. The estimate
depends on how much capture it saw, and the least-squares covariance — which
assumes iid residuals — does not model that dependence at all. Warm-up,
position-dependent friction and the low-frequency content only a full capture
carries are all candidates; whole-cycle bookkeeping is not (12 s is three whole
cycles of the 0.25 Hz tone and is among the farthest out).

The consequence for probe compression, stated plainly: **it is blocked on this
measurement, not on the code.** The plan's acceptance criterion for 0.1 — "same
fit values within CI on the committed captures" — cannot be met by shortening,
because a shortened capture of *this* unit does not reproduce its own long-capture
answer. What has to happen first is either a model that explains the
length-dependence (thermal state, position-dependent friction) or an accepted
reproducibility band wider than the Wald interval. :attr:`StabilityOfFit.bias`
and :attr:`StabilityOfFit.windows_outside_ci` are the numbers to re-check once
either lands.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from ..errors import CalibrationError
from ..robot import Robot
from .stability import MULTISINE_KIND, schroeder_multisine

#: Coulomb smoothing velocity used when building the design columns — must match
#: :func:`jaxterity.calibrate.identify_closed_loop`'s ``eps`` so the score is
#: about the same regression the fit runs.
COULOMB_EPS = 1e-2

#: Conditioning ceiling for the ``identifiable`` verdict, matching the active
#: lane's default so the two reports read on one scale.
COND_MAX = 1.0e6


@dataclasses.dataclass(frozen=True)
class SegmentScore:
    """What a planned excitation can identify, scored before it runs.

    Attributes
    ----------
    joint : str
        The joint the segment excites.
    duration_s : float
        Planned duration, s.
    condition_number : float
        Condition number of the column-normalized design matrix. Large means
        the channels trade off against each other — redesign rather than
        fit harder.
    relative_variance : dict[str, float]
        Per-channel normalized marginal variance ``[(XᵀX)⁻¹]_ii`` (scale-free).
        Bigger is worse: it is the factor by which sensor noise is amplified
        into that channel's estimate.
    information : dict[str, float]
        Per-channel Fisher information (the diagonal of ``XᵀX`` for the
        column-normalized design) — what a longer or livelier segment buys.
    identifiable : bool
        Whether the conditioning is inside :data:`COND_MAX`.
    """

    joint: str
    duration_s: float
    condition_number: float
    relative_variance: dict[str, float]
    information: dict[str, float]
    identifiable: bool

    @property
    def weakest_channel(self) -> str:
        """The channel this segment pins down worst (largest marginal variance)."""
        return max(self.relative_variance, key=lambda k: self.relative_variance[k])

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON form."""
        return dataclasses.asdict(self)


def commanded_velocity(
    amp_rad: float,
    *,
    duration_s: float,
    rate_hz: float,
    kind: str = MULTISINE_KIND,
) -> tuple[np.ndarray, np.ndarray]:
    """The commanded position and velocity of one planned segment.

    Differentiates the commanded waveform analytically-in-effect (central
    differences on a grid at the *command* rate), because that is the signal the
    servo is asked to follow and the only thing available before the capture
    exists.

    Args
    ----
    amp_rad : float
        Excitation amplitude, rad.
    duration_s : float
        Segment duration, s.
    rate_hz : float
        Command rate, Hz.
    kind : str
        Waveform version (see
        :func:`jaxterity.checkup.stability.schroeder_multisine`).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(q, qd)`` about a zero center, rad and rad/s.

    Raises
    ------
    CalibrationError
        If the segment would carry fewer than 20 samples — the fit's own
        minimum, so a shorter one cannot be scored honestly.

    Units
    -----
    rad, s, Hz.
    """
    if duration_s <= 0 or rate_hz <= 0:
        raise CalibrationError(
            f"commanded_velocity: duration_s and rate_hz must be positive, got "
            f"{duration_s} and {rate_hz}."
        )
    t = np.arange(0.0, duration_s, 1.0 / rate_hz)
    if t.size < 20:
        raise CalibrationError(
            f"commanded_velocity: {t.size} sample(s) at {rate_hz:g} Hz over "
            f"{duration_s:g} s — identify_closed_loop needs at least 20 to fit "
            "and hold out a split, so a shorter segment cannot be scored."
        )
    q = amp_rad * np.asarray(schroeder_multisine(t, duration_s, kind=kind))
    qd = np.gradient(q, t)
    return q, qd


def segment_identifiability(
    joint: str,
    amp_rad: float,
    *,
    duration_s: float,
    rate_hz: float,
    kind: str = MULTISINE_KIND,
    eps: float = COULOMB_EPS,
    cond_max: float = COND_MAX,
) -> SegmentScore:
    """Score a planned segment's ability to separate damping from friction.

    Builds the two velocity-dependent columns of the closed-loop fit's design
    matrix — ``−q̇`` and ``−tanh(q̇/ε)`` — plus its constant offset column, from
    the *commanded* waveform, and reports their conditioning and per-channel
    marginal variance. The measurement columns the fit also carries (the servo's
    load proxy) are deliberately excluded: they are not functions of the plan, so
    including a guess about them would dress an assumption up as a score.

    What a good score means: the segment's *geometry* can separate the channels.
    What it does not mean: that the fit will be precise — that depends on the
    residual structure, which :func:`fit_stability` measures and which the
    fit's own confidence interval understates (see the module docstring).

    Args
    ----
    joint : str
        Joint name (carried into the score for reporting).
    amp_rad : float
        Excitation amplitude, rad.
    duration_s : float
        Segment duration, s.
    rate_hz : float
        Command rate, Hz.
    kind : str
        Waveform version.
    eps : float
        Coulomb smoothing velocity, rad/s — match the fit's.
    cond_max : float
        Conditioning ceiling for the verdict.

    Returns
    -------
    SegmentScore
        Conditioning, per-channel relative variance and information.

    Raises
    ------
    CalibrationError
        If the segment is too short to score (see :func:`commanded_velocity`).

    Units
    -----
    rad, s, Hz, rad/s.

    Examples
    --------
    >>> score = segment_identifiability(
    ...     "elbow_flex", 0.3, duration_s=16.0, rate_hz=80.0
    ... )
    >>> score.identifiable
    True
    """
    _q, qd = commanded_velocity(
        amp_rad, duration_s=duration_s, rate_hz=rate_hz, kind=kind
    )
    columns = {
        "offset": np.ones_like(qd),
        "damping": -qd,
        "friction": -np.tanh(qd / eps),
    }
    design = np.column_stack([columns[k] for k in ("offset", "damping", "friction")])
    norms = np.linalg.norm(design, axis=0)
    if np.any(norms <= 0):
        raise CalibrationError(
            f"segment_identifiability: the {joint!r} segment has a degenerate "
            "design column (no commanded motion at all), so nothing is "
            "identifiable from it. Fix: a non-zero amplitude."
        )
    scaled = design / norms
    gram = scaled.T @ scaled
    singular = np.linalg.svd(scaled, compute_uv=False)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else float("inf")
    inverse = np.linalg.pinv(gram)
    names = ("offset", "damping", "friction")
    return SegmentScore(
        joint=joint,
        duration_s=float(duration_s),
        condition_number=condition,
        relative_variance={
            n: float(inverse[i, i]) for i, n in enumerate(names) if n != "offset"
        },
        information={
            n: float(gram[i, i]) for i, n in enumerate(names) if n != "offset"
        },
        identifiable=bool(np.isfinite(condition) and condition <= cond_max),
    )


@dataclasses.dataclass(frozen=True)
class StabilityOfFit:
    """How much a channel's estimate moves across sub-windows of one capture.

    Attributes
    ----------
    channel : str
        ``"<joint>.damping"`` / ``"<joint>.friction"``.
    full : float
        The full-capture estimate.
    window_values : tuple[float, ...]
        Estimates from each sub-window (in window order).
    ci95 : tuple[float, float] | None
        The full fit's own 95 % Wald interval, when it reported one.
    """

    channel: str
    full: float
    window_values: tuple[float, ...]
    ci95: tuple[float, float] | None = None

    @property
    def spread(self) -> float:
        """Half-range *among* the sub-windows, relative to the full fit.

        ``(max − min) / (2·|full|)`` — how much the sub-windows disagree with
        each other. On the committed captures this is small; the interesting
        quantity is :attr:`bias`.
        """
        if not self.window_values or self.full == 0:
            return float("nan")
        return (max(self.window_values) - min(self.window_values)) / (
            2.0 * abs(self.full)
        )

    @property
    def bias(self) -> float:
        """Largest sub-window departure *from the full fit*, relative.

        ``max|window − full| / |full|``. This is the length-dependence a
        shortened probe would inherit, and the number that blocks compression:
        it is systematic, so averaging more windows does not remove it.
        """
        if not self.window_values or self.full == 0:
            return float("nan")
        return max(abs(v - self.full) for v in self.window_values) / abs(self.full)

    @property
    def ci_relative(self) -> float | None:
        """The reported CI's half-width relative to the full fit, if any."""
        if self.ci95 is None or self.full == 0:
            return None
        return (self.ci95[1] - self.ci95[0]) / (2.0 * abs(self.full))

    @property
    def windows_outside_ci(self) -> int:
        """How many sub-window estimates fall outside the full fit's interval.

        Any non-zero count means the interval does not cover the estimate's
        dependence on capture length — so "the shorter probe agrees within CI"
        is not an acceptance test this capture can pass, whatever the shorter
        probe does.
        """
        if self.ci95 is None:
            return 0
        lo, hi = self.ci95
        return sum(1 for v in self.window_values if v < lo or v > hi)


def fit_stability(
    robot: Robot,
    capture: Any,
    *,
    windows: int = 3,
    window_fraction: float = 0.625,
) -> tuple[StabilityOfFit, ...]:
    """Re-fit a capture over sliding sub-windows and report the spread.

    The measurement behind this module's central claim. Splits the capture into
    ``windows`` overlapping sub-windows of ``window_fraction`` of its length,
    re-runs the closed-loop fit on each, and reports each channel's full-capture
    value, its sub-window values, and the fit's own interval — so a caller can
    see directly whether the interval covers the spread.

    Args
    ----
    robot : Robot
        The model the fit runs against.
    capture : CommandedCapture
        A parsed capture (:func:`jaxterity.calibrate.load_commanded_capture`).
    windows : int
        Number of sub-windows (>= 2).
    window_fraction : float
        Each sub-window's length as a fraction of the capture (in (0, 1)).

    Returns
    -------
    tuple[StabilityOfFit, ...]
        One entry per channel the full fit identified.

    Raises
    ------
    CalibrationError
        If ``windows`` < 2, ``window_fraction`` is outside (0, 1), or a
        sub-window comes out too short for the fit.

    Units
    -----
    Channel units follow the fit (N·m·s/rad, N·m).
    """
    import dataclasses as _dataclasses

    from ..calibrate import identify_closed_loop

    if windows < 2:
        raise CalibrationError(
            f"fit_stability: needs at least 2 windows to have a spread, got {windows}."
        )
    if not 0.0 < window_fraction < 1.0:
        raise CalibrationError(
            f"fit_stability: window_fraction must lie in (0, 1), got {window_fraction}."
        )
    full_result = identify_closed_loop(robot, capture)
    intervals = full_result.interval(0.95) or {}

    n = capture.n_samples
    span = int(window_fraction * n)
    if span < 20:
        raise CalibrationError(
            f"fit_stability: sub-windows of {span} sample(s) are shorter than "
            "the fit's 20-sample minimum. Use a longer capture or a larger "
            "window_fraction."
        )
    starts = np.linspace(0, n - span, windows).astype(int)

    per_channel: dict[str, list[float]] = {k: [] for k in full_result.fitted}
    for start in starts:
        sliced = _dataclasses.replace(
            capture,
            t=capture.t[start : start + span] - capture.t[start],
            q=capture.q[start : start + span],
            load_proxy=capture.load_proxy[start : start + span],
        )
        result = identify_closed_loop(robot, sliced)
        for channel in per_channel:
            if channel in result.fitted:
                per_channel[channel].append(float(result.fitted[channel]))

    return tuple(
        StabilityOfFit(
            channel=channel,
            full=float(full_result.fitted[channel]),
            window_values=tuple(values),
            ci95=(
                (float(intervals[channel][0]), float(intervals[channel][1]))
                if channel in intervals
                else None
            ),
        )
        for channel, values in sorted(per_channel.items())
    )


__all__ = [
    "COND_MAX",
    "COULOMB_EPS",
    "SegmentScore",
    "StabilityOfFit",
    "commanded_velocity",
    "fit_stability",
    "segment_identifiability",
]
