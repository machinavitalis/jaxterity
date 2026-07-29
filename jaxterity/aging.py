# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Aging priors: how a fitted parameter is expected to move with use.

:mod:`jaxterity.drift` answers "what does the plant look like at time *t*" — it
takes a schedule you hand it and applies it to the MJX model. It does not say
where the schedule comes from. This module does: a small family of *growth
models* over the two parameters that actually age on a robot, fitted from a
unit's own history of windowed re-fits and projected forward.

Two mechanisms, one equation. Both are

    value(u) = initial + rate · u^exponent

over a **usage clock** ``u`` — operating hours, cumulative joint travel in
radians, or cycles, whichever the fleet logs. The difference is the exponent,
and it is a physical claim, not a curve-fitting preference:

* **Friction growth** (:func:`friction_growth_prior`) uses ``exponent = 1``.
  Adhesive/abrasive wear removes material in proportion to sliding distance
  (Archard), and the friction torque a worn gearset shows tracks that removal
  roughly linearly over the mid-life plateau. This is the classic slow drift
  and the fault-precursor channel — ``robot.<joint>.friction`` in
  :func:`jaxterity.twin.g1_drift_parameters`.
* **Backlash growth** (:func:`backlash_growth_prior`) uses ``exponent = 0.5``.
  Tooth-flank wear opens the mesh, but the contact pressure falls as the flank
  wears in, so lost motion grows sub-linearly — the square root is the standard
  mild-wear form. The parameter is ``play`` on
  :class:`~jaxterity.actuators.backlash.PositionalBacklash`, which is what a
  dual-encoder re-fit measures.

What this is *not*: a life model. A real component's late life accelerates —
wear debris makes more wear, a dry bearing heats and dries faster — and no
power law captures the knee. These models are for the plateau, and the honest
use is (a) projecting a *trend* between re-fits, and (b) flagging when the
measured history stops matching the projection, which is the interesting
signal. :meth:`GrowthFit.residual_ratio` is that flag.

The link to the rest of the stack runs both ways: a growth model is *fitted*
from a sequence of windowed sysid results (each one a
:class:`~jaxterity.twin.DriftParameter` path and a value at a known usage), and
it is *replayed* as a :class:`~jaxterity.drift.DriftChannel` so the same curve
that summarizes the history can drive a simulated plant forward.

Units
-----
The usage clock ``u`` and the parameter carry whatever units the caller uses —
this module is unit-agnostic and unit-preserving. State them in
:attr:`AgingPrior.usage_unit` so a report can print them.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .drift import DriftChannel
from .errors import CalibrationError, JaxterityError

#: Exponents encoding the two wear mechanisms (see the module docstring).
FRICTION_EXPONENT = 1.0
BACKLASH_EXPONENT = 0.5

#: Deterministic exponent grid used when :func:`fit_growth` is asked to choose
#: one. Coarse on purpose: an exponent fitted finely from a handful of noisy
#: re-fits is over-fitting, and the mechanism claim (1 vs ½) is what matters.
_EXPONENT_GRID: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


@dataclasses.dataclass(frozen=True)
class PowerLawGrowth:
    """``value(u) = initial + rate·u^exponent`` — the one growth curve.

    Args
    ----
    initial : float
        Value at zero usage, in the parameter's own unit.
    rate : float
        Growth coefficient, in parameter-unit per ``usage^exponent``. Must be
        >= 0: a parameter that spontaneously improves with use is a fit
        artifact, and this class refuses to represent one so the artifact
        surfaces at construction instead of in a projection.
    exponent : float
        Wear exponent (1 for friction, ½ for backlash — see the module
        docstring). Must be > 0.

    Raises
    ------
    JaxterityError
        On a negative rate or a non-positive exponent.

    Examples
    --------
    >>> growth = PowerLawGrowth(initial=0.1, rate=2e-4, exponent=1.0)
    >>> round(growth.value(1000.0), 4)
    0.3
    """

    initial: float
    rate: float
    exponent: float = 1.0

    def __post_init__(self) -> None:
        if self.rate < 0:
            raise JaxterityError(
                f"PowerLawGrowth.rate must be >= 0, got {self.rate}. A "
                "parameter that decreases with use is not wear; if a fit "
                "produced this, the history is telling you something else "
                "changed (a re-lubrication, a swapped part, a re-zeroed "
                "sensor) and the segment should be split there."
            )
        if self.exponent <= 0:
            raise JaxterityError(
                f"PowerLawGrowth.exponent must be > 0, got {self.exponent}."
            )

    def value(self, usage):
        """The parameter's value at a usage level (array-friendly).

        Args
        ----
        usage : float or numpy.ndarray
            Usage clock reading(s), >= 0.

        Returns
        -------
        float or numpy.ndarray
            Projected parameter value(s), in the parameter's unit.

        Units
        -----
        Unit-preserving.
        """
        u = np.maximum(np.asarray(usage, dtype=float), 0.0)
        result = self.initial + self.rate * u**self.exponent
        return float(result) if np.ndim(usage) == 0 else result

    def usage_at(self, threshold: float) -> float:
        """Usage at which the parameter first reaches ``threshold``.

        Args
        ----
        threshold : float
            Value of interest (a maintenance limit, a fault trip point).

        Returns
        -------
        float
            The usage level, or ``inf`` if the curve never gets there (a zero
            growth rate, or a threshold already below the initial value gives
            0.0).

        Raises
        ------
        CalibrationError
            If ``threshold`` is below the initial value *and* the rate is
            positive — the curve is increasing, so it passed that point before
            the clock started, and the caller has mixed up a limit with a
            floor.

        Units
        -----
        Parameter unit in, usage unit out.
        """
        if threshold <= self.initial:
            if self.rate == 0.0:
                return 0.0
            raise CalibrationError(
                f"usage_at: threshold {threshold:g} is at or below the initial "
                f"value {self.initial:g} of an increasing curve, so it was "
                "already crossed before this clock started. Fix: pass a limit "
                "above the as-new value."
            )
        if self.rate == 0.0:
            return float("inf")
        return float(((threshold - self.initial) / self.rate) ** (1.0 / self.exponent))

    def remaining_life(self, current_usage: float, threshold: float) -> float:
        """Usage left before ``threshold``, from ``current_usage``.

        Args
        ----
        current_usage : float
            Where the unit is now on the usage clock.
        threshold : float
            The limit.

        Returns
        -------
        float
            Remaining usage (0.0 if already past, ``inf`` if never reached).

        Units
        -----
        Usage unit.
        """
        return max(self.usage_at(threshold) - float(current_usage), 0.0)


def friction_growth_prior(
    initial: float, rate: float = 2.0e-4, exponent: float = FRICTION_EXPONENT
) -> PowerLawGrowth:
    """A friction-growth prior: linear in usage (Archard wear).

    The default ``rate`` is **representative, not measured** — it doubles a
    0.1 N·m friction loss over 500 usage units, which is the right order for a
    geared leg joint on an hours clock. Fit it from the unit's own history
    with :func:`fit_growth` before trusting a projection.

    Args
    ----
    initial : float
        As-new friction, N·m.
    rate : float
        Growth per usage unit, N·m per usage.
    exponent : float
        Wear exponent; leave at 1 unless a fitted history says otherwise.

    Returns
    -------
    PowerLawGrowth
        The prior.

    Units
    -----
    N·m, N·m per usage unit.
    """
    return PowerLawGrowth(initial=initial, rate=rate, exponent=exponent)


def backlash_growth_prior(
    initial: float, rate: float = 1.0e-5, exponent: float = BACKLASH_EXPONENT
) -> PowerLawGrowth:
    """A backlash-growth prior: square-root in usage (mild flank wear).

    As with :func:`friction_growth_prior` the default ``rate`` is
    representative, not measured: it opens about 0.3 mrad of extra play over
    1000 usage units. The parameter it projects is ``play`` on
    :class:`~jaxterity.actuators.backlash.PositionalBacklash`, so a dual-encoder
    re-fit is what feeds :func:`fit_growth`.

    Args
    ----
    initial : float
        As-new backlash half-width, rad.
    rate : float
        Growth coefficient, rad per usage^½.
    exponent : float
        Wear exponent; leave at ½ unless a fitted history says otherwise.

    Returns
    -------
    PowerLawGrowth
        The prior.

    Units
    -----
    rad, rad per usage^½.
    """
    return PowerLawGrowth(initial=initial, rate=rate, exponent=exponent)


@dataclasses.dataclass(frozen=True)
class GrowthFit:
    """A growth model fitted to a unit's history, with its diagnostics.

    Attributes
    ----------
    model : PowerLawGrowth
        The fitted curve.
    rmse : float
        Root-mean-square residual, in the parameter's unit.
    r_squared : float
        Fraction of the history's variance the curve explains. Low values on a
        *rising* history usually mean the exponent is wrong; low values on a
        flat one just mean nothing has aged yet.
    n_points : int
        How many re-fits went in. Three is the minimum and is not enough to
        trust an extrapolation — it is enough to notice a trend.
    """

    model: PowerLawGrowth
    rmse: float
    r_squared: float
    n_points: int

    def residual_ratio(self, usage: float, measured: float) -> float:
        """How far a new measurement sits from the curve, in RMSE units.

        The flag the module docstring promises: a value beyond ~3 means the
        newest re-fit does not belong to the trend the history established —
        the knee, a fault, or something that was serviced. Returns ``inf``
        when the fit has no spread to compare against.

        Args
        ----
        usage : float
            Usage clock at the new measurement.
        measured : float
            The newly fitted parameter value.

        Returns
        -------
        float
            ``|measured − model.value(usage)| / rmse``.

        Units
        -----
        Dimensionless.
        """
        if self.rmse <= 0:
            return float("inf")
        return abs(measured - self.model.value(usage)) / self.rmse


def fit_growth(
    usage, values, *, exponent: float | None = None, initial: float | None = None
) -> GrowthFit:
    """Fit a growth curve to a history of re-fitted parameter values.

    The model is linear in ``(initial, rate)`` once the exponent is fixed, so
    each candidate exponent is one ordinary least squares on ``[1, u^p]``. With
    ``exponent=None`` the function tries a fixed, coarse grid
    (:data:`_EXPONENT_GRID`) and keeps the best — deterministic, and coarse on
    purpose (see the module docstring). Pass ``exponent=1.0`` or ``0.5`` to
    hold the mechanism claim fixed, which is usually the right call: two or
    three noisy re-fits cannot distinguish exponents, and letting them try
    produces a confident-looking curve built on nothing.

    Args
    ----
    usage : array-like, shape ``(n,)``
        Usage-clock reading at each re-fit, >= 0.
    values : array-like, shape ``(n,)``
        The re-fitted parameter value at each point.
    exponent : float | None
        Fix the wear exponent, or ``None`` to select from the grid.
    initial : float | None
        Fix the as-new value (e.g. from the factory calibration) instead of
        fitting it. Doing so buys a real degree of freedom back, which matters
        when the history is short.

    Returns
    -------
    GrowthFit
        The fitted curve and its diagnostics.

    Raises
    ------
    CalibrationError
        If fewer than 3 points are supplied (with a fixed ``initial``, 2), the
        two arrays disagree in length, the usage never varies, or the fitted
        rate comes out negative (which :class:`PowerLawGrowth` refuses — see
        its error for what that means).

    Units
    -----
    Unit-preserving.

    Examples
    --------
    >>> u = [0.0, 100.0, 200.0, 300.0]
    >>> v = [0.10, 0.12, 0.14, 0.16]
    >>> fit = fit_growth(u, v, exponent=1.0)
    >>> round(fit.model.rate, 6)
    0.0002
    """
    u = np.asarray(usage, dtype=float)
    y = np.asarray(values, dtype=float)
    if u.shape != y.shape:
        raise CalibrationError(
            f"fit_growth got {u.size} usage point(s) and {y.size} value(s); "
            "each re-fit contributes exactly one of each."
        )
    minimum = 2 if initial is not None else 3
    if u.size < minimum:
        raise CalibrationError(
            f"fit_growth needs at least {minimum} history point(s), got "
            f"{u.size}. A growth curve through fewer points is not a trend, "
            "it is an interpolation."
        )
    if float(np.ptp(u)) <= 0.0:
        raise CalibrationError(
            "fit_growth needs the usage clock to advance between re-fits; "
            f"every point in this history sits at {float(u[0]):g}. Fix: record "
            "the operating hours / travel / cycles alongside each fit."
        )

    candidates = (exponent,) if exponent is not None else _EXPONENT_GRID
    best: tuple[float, PowerLawGrowth] | None = None
    for p in candidates:
        basis = u**p
        if initial is None:
            design = np.stack([np.ones_like(u), basis], axis=1)
            solution, *_ = np.linalg.lstsq(design, y, rcond=None)
            offset, rate = float(solution[0]), float(solution[1])
        else:
            offset = float(initial)
            denominator = float(np.sum(basis * basis))
            rate = float(np.sum(basis * (y - offset)) / denominator)
        predicted = offset + rate * basis
        sse = float(np.sum((predicted - y) ** 2))
        if best is None or sse < best[0]:
            curve = PowerLawGrowth(initial=offset, rate=max(rate, 0.0), exponent=p)
            best = (sse, curve)

    assert best is not None  # candidates is non-empty
    sse, model = best
    if model.rate == 0.0 and float(np.ptp(y)) > 0.0:
        raise CalibrationError(
            "fit_growth: the best fit has a non-positive growth rate, i.e. the "
            "history trends DOWN. Wear does not undo itself — split the "
            "history at whatever was serviced, re-zeroed, or replaced, and fit "
            "the segments separately."
        )
    residuals = model.value(u) - y
    rmse = float(np.sqrt(np.mean(residuals**2)))
    total = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - sse / total if total > 0 else 1.0
    return GrowthFit(
        model=model, rmse=rmse, r_squared=float(r_squared), n_points=int(u.size)
    )


@dataclasses.dataclass(frozen=True)
class AgingPrior:
    """A growth model attached to one declared drift-parameter path.

    The bridge to :func:`jaxterity.twin.g1_drift_parameters`: ``path`` is one
    of that schema's canonical keys, so a projection lands exactly where the
    online re-fit reads.

    Args
    ----
    path : str
        Canonical parameter path, e.g. ``"robot.left_knee_joint.friction"``.
    model : PowerLawGrowth
        The growth curve for that parameter.
    threshold : float | None
        Optional maintenance/fault limit in the parameter's unit, used by
        :meth:`remaining_life`.
    usage_unit : str
        What the usage clock counts (``"hours"``, ``"rad"``, ``"cycles"``) —
        carried so a report can print it, since the math is unit-agnostic.

    Examples
    --------
    >>> prior = AgingPrior(
    ...     path="robot.left_knee_joint.friction",
    ...     model=friction_growth_prior(0.1),
    ...     threshold=0.5,
    ... )
    >>> round(prior.remaining_life(0.0))
    2000
    """

    path: str
    model: PowerLawGrowth
    threshold: float | None = None
    usage_unit: str = "hours"

    def remaining_life(self, current_usage: float) -> float:
        """Usage left before this parameter reaches its threshold.

        Raises
        ------
        JaxterityError
            If no ``threshold`` was declared — there is no life to report
            without a limit, and returning ``inf`` would read as "fine".
        """
        if self.threshold is None:
            raise JaxterityError(
                f"AgingPrior({self.path!r}).remaining_life needs a declared "
                "threshold: 'how long until it is bad' is meaningless without "
                "'what counts as bad'."
            )
        return self.model.remaining_life(current_usage, self.threshold)


def project(priors: tuple[AgingPrior, ...], usage: float) -> dict[str, float]:
    """Project every prior forward to a usage level.

    The output is keyed by drift-parameter path, so it lines up directly with
    :meth:`jaxterity.twin.Twin.identifiable_params` and with whatever the
    windowed re-fit produced — which is what makes a comparison meaningful.

    Args
    ----
    priors : tuple[AgingPrior, ...]
        The models.
    usage : float
        Usage-clock reading to project to.

    Returns
    -------
    dict[str, float]
        Path → projected value.

    Units
    -----
    Unit-preserving per parameter.
    """
    return {prior.path: prior.model.value(usage) for prior in priors}


def to_drift_channel(
    model: PowerLawGrowth,
    *,
    field: str,
    index: int,
    times: tuple[float, ...],
    usage_per_second: float,
    mode: str = "set",
) -> DriftChannel:
    """Sample a growth curve into a :class:`~jaxterity.drift.DriftChannel`.

    ``DriftSchedule`` interpolates piecewise-linearly between breakpoints in
    *simulated seconds*, so a curved growth model has to be sampled onto a
    breakpoint grid. This does that conversion explicitly — the schedule stays
    the dumb, deterministic data structure it is, and the curve stays here
    where it can be fitted and reasoned about.

    Args
    ----
    model : PowerLawGrowth
        The curve to sample.
    field : str
        MJX field the channel targets (``"dof_frictionloss"`` etc. — see
        :class:`~jaxterity.drift.DriftChannel`).
    index : int
        DOF or actuator index.
    times : tuple[float, ...]
        Strictly increasing simulated-time breakpoints, s. Use enough of them
        to resolve the curvature: a ``½``-exponent curve sampled at two points
        is a straight line and loses exactly the shape you fitted.
    usage_per_second : float
        Conversion from simulated seconds to the model's usage clock (e.g.
        ``1/3600`` for an hours clock). Must be > 0.
    mode : str
        ``"set"`` (absolute values) or ``"scale"``.

    Returns
    -------
    DriftChannel
        The sampled channel, ready for a
        :class:`~jaxterity.drift.DriftSchedule`.

    Raises
    ------
    CalibrationError
        If ``usage_per_second`` is not positive or fewer than 2 breakpoints
        are given.

    Units
    -----
    s in; the channel carries the parameter's own unit.
    """
    if usage_per_second <= 0:
        raise CalibrationError(
            f"to_drift_channel needs usage_per_second > 0, got "
            f"{usage_per_second}. It converts simulated seconds into the "
            "model's usage clock; a zero rate means the plant never ages and "
            "the channel would be constant."
        )
    if len(times) < 2:
        raise CalibrationError(
            f"to_drift_channel needs at least 2 breakpoints, got {len(times)}."
        )
    values = tuple(float(model.value(t * usage_per_second)) for t in times)
    return DriftChannel(
        field=field,
        index=index,
        times=tuple(float(t) for t in times),
        values=values,
        mode=mode,
    )


__all__ = [
    "BACKLASH_EXPONENT",
    "FRICTION_EXPONENT",
    "AgingPrior",
    "GrowthFit",
    "PowerLawGrowth",
    "backlash_growth_prior",
    "fit_growth",
    "friction_growth_prior",
    "project",
    "to_drift_channel",
]
