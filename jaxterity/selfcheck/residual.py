# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Drift anomaly monitor over the recalibration stream.

The windowed re-fit (:func:`jaxterity.calibrate.refit_drift_parameters`)
produces a *stream* of commits: fitted drift parameters + a window residual
RMS, one per recalibration epoch. This module watches that stream and flags
when a unit stops looking healthy — **days before mechanical failure**, while
the body-only baseline's first signal is the failure itself:

- **Statistical anomaly (z-score).** Each scalar channel of each fitted leaf
  is z-scored against this unit's own recalibration history: a slow gearbox
  drift shows up as a parameter trending out of its own past distribution.
- **Manifold anomaly (Mahalanobis).** A :class:`HealthyManifold` — a Gaussian
  ellipsoid fitted over parameter vectors observed across many healthy units
  (e.g. a calibrated-fleet sample) — flags a unit whose parameters leave the
  healthy region even if its own history drifted slowly enough to keep
  z-scores low (the boiled-frog case).
- **Residual anomaly.** The window prediction-residual RMS exceeding a
  threshold means the model class itself no longer explains the telemetry
  (something changed that the declared drift parameters cannot absorb).

All decision math is plain NumPy on a handful of scalars per epoch (≪ any
control budget, same policy as the safety monitor). The monitor holds
per-unit mutable history; everything it consumes comes from the re-fit
stream, so it adds zero instrumentation cost.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Mapping, NamedTuple, Sequence

import numpy as np

from ..errors import JaxterityError

#: Standard-normal quantile for the default manifold confidence (99.9 %).
_Z_999 = 3.0902


def _chi2_quantile(p_z: float, dof: int) -> float:
    """Wilson–Hilferty approximation of the χ² quantile (no SciPy dependency).

    Args
    ----
    p_z : float
        The standard-normal quantile of the target probability (e.g. 3.09
        for 99.9 %).
    dof : int
        Degrees of freedom (the parameter-vector dimension).

    Returns
    -------
    float
        The approximate χ²(dof) quantile — a squared-Mahalanobis threshold.
    """
    k = float(dof)
    return k * (1.0 - 2.0 / (9.0 * k) + p_z * math.sqrt(2.0 / (9.0 * k))) ** 3


def _flatten(
    fitted: Mapping[str, Any], labels: Mapping[str, tuple[str, ...]] | None = None
) -> dict[str, float]:
    """Flatten fitted leaves to scalar channels.

    A scalar leaf keeps its path; a vector leaf becomes one channel per
    element, suffixed ``[label]`` (the per-path labels if given — e.g. servo
    names for ``offbody.K_t_scale`` — else the index).
    """
    channels: dict[str, float] = {}
    for path, value in fitted.items():
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            channels[path] = float(arr)
            continue
        if arr.ndim != 1:
            raise JaxterityError(
                f"ResidualMonitor: fitted leaf {path!r} has shape {arr.shape}; "
                "only scalars and 1-D vectors are monitorable channels."
            )
        names = (labels or {}).get(path)
        if names is not None and len(names) != arr.shape[0]:
            raise JaxterityError(
                f"ResidualMonitor: {len(names)} labels for {path!r} but the "
                f"leaf has {arr.shape[0]} elements."
            )
        for i, v in enumerate(arr):
            suffix = names[i] if names is not None else str(i)
            channels[f"{path}[{suffix}]"] = float(v)
    return channels


@dataclasses.dataclass(frozen=True)
class HealthyManifold:
    """A Gaussian ellipsoid over healthy parameter vectors.

    Fit it once over the fitted-parameter vectors of many healthy units
    (manufacturing-variance samples, a calibrated fleet) with
    :meth:`from_samples`; then :meth:`mahalanobis` measures how far a unit's
    current parameters sit from the healthy population, and
    :meth:`contains` applies the χ² threshold.

    Args
    ----
    keys : tuple[str, ...]
        The flattened channel names, in vector order.
    mean : numpy.ndarray, shape ``(d,)``
        Population mean.
    covariance_inverse : numpy.ndarray, shape ``(d, d)``
        Inverse of the (regularized) population covariance.
    threshold : float
        Squared-Mahalanobis flag level. :meth:`from_samples` defaults it to
        the χ²(d) 99.9 % quantile (Wilson–Hilferty).
    """

    keys: tuple[str, ...]
    mean: np.ndarray
    covariance_inverse: np.ndarray
    threshold: float

    @classmethod
    def from_samples(
        cls,
        samples: Sequence[Mapping[str, Any]],
        *,
        labels: Mapping[str, tuple[str, ...]] | None = None,
        threshold: float | None = None,
        shrinkage: float = 1.0,
        regularization: float = 1e-9,
    ) -> "HealthyManifold":
        """Fit the ellipsoid from healthy units' fitted parameters.

        Args
        ----
        samples : list[Mapping[str, array-like]]
            One fitted-parameter mapping per healthy unit (the
            ``sysid_result.fitted`` dicts of their calibrations, or a
            sampled fleet's declared parameters).
        labels : Mapping[str, tuple[str, ...]] | None
            Optional per-path element labels for vector leaves.
        threshold : float | None
            Squared-Mahalanobis flag level; default χ²(d) at 99.9 %.
        shrinkage : float
            Covariance shrinkage toward its diagonal, in ``[0, 1]``.
            **Default 1.0 — independent channels.** A realistic fleet has
            far fewer units than channels (30 units, 70 channels), where a
            full sample covariance is rank-deficient and its inverse blows
            up in the null directions; per-channel variances are estimable
            from a handful of units. Lower it only when units ≫ channels
            and cross-channel structure matters.
        regularization : float
            Diagonal jitter added before inversion. Channels that are
            exactly constant across units are additionally floored so a
            constant channel contributes zero distance instead of NaN.

        Returns
        -------
        HealthyManifold

        Raises
        ------
        JaxterityError
            With fewer than 3 samples, or inconsistent channels.
        """
        rows = [_flatten(s, labels) for s in samples]
        if len(rows) < 3:
            raise JaxterityError(
                f"HealthyManifold.from_samples needs ≥ 3 healthy units to "
                f"estimate a covariance, got {len(rows)}."
            )
        if not 0.0 <= shrinkage <= 1.0:
            raise JaxterityError(
                f"HealthyManifold.from_samples: shrinkage must be in [0, 1], "
                f"got {shrinkage}."
            )
        keys = tuple(sorted(rows[0]))
        for r in rows[1:]:
            if tuple(sorted(r)) != keys:
                raise JaxterityError(
                    "HealthyManifold.from_samples: units disagree on channels "
                    "— every sample must carry the same fitted leaves."
                )
        matrix = np.asarray([[r[k] for k in keys] for r in rows], dtype=float)
        mean = matrix.mean(axis=0)
        centred = matrix - mean[None, :]
        cov = centred.T @ centred / (matrix.shape[0] - 1)
        variances = np.diag(cov).copy()
        # Floor exactly-constant channels (zero variance) so they contribute
        # zero distance for identical values rather than dividing by ~0.
        variances = np.maximum(variances, 1e-12 * max(float(variances.max()), 1e-12))
        cov = (1.0 - shrinkage) * cov + shrinkage * np.diag(variances)
        cov += regularization * np.eye(len(keys))
        return cls(
            keys=keys,
            mean=mean,
            covariance_inverse=np.linalg.inv(cov),
            threshold=(
                _chi2_quantile(_Z_999, len(keys)) if threshold is None else threshold
            ),
        )

    def _vector(self, fitted, labels=None) -> np.ndarray:
        channels = _flatten(fitted, labels)
        missing = sorted(set(self.keys) - set(channels))
        if missing:
            raise JaxterityError(
                f"HealthyManifold: the fitted parameters are missing channels "
                f"{missing[:5]}{'…' if len(missing) > 5 else ''} the manifold "
                "was built over."
            )
        return np.asarray([channels[k] for k in self.keys], dtype=float)

    def mahalanobis(self, fitted, labels=None) -> float:
        """Squared Mahalanobis distance of a unit's parameters to the mean."""
        d = self._vector(fitted, labels) - self.mean
        return float(d @ self.covariance_inverse @ d)

    def contains(self, fitted, labels=None) -> bool:
        """True iff the parameters sit inside the healthy ellipsoid."""
        return self.mahalanobis(fitted, labels) <= self.threshold


class AnomalyReport(NamedTuple):
    """One epoch's verdict from the :class:`ResidualMonitor`.

    Fields
    ------
    ok : bool
        True iff nothing flagged.
    flags : tuple[str, ...]
        Human-readable findings, e.g.
        ``"robot.right_knee_joint.friction: z=4.2 over its own history
        (threshold 3.0)"``.
    z_scores : dict[str, float]
        Per-channel z-score vs this unit's recalibration history (empty
        until the history is long enough).
    mahalanobis : float | None
        Squared distance to the healthy manifold (``None`` without one).
    residual_rms : float | None
        The window prediction-residual RMS passed in, echoed back.
    """

    ok: bool
    flags: tuple[str, ...]
    z_scores: dict[str, float]
    mahalanobis: float | None
    residual_rms: float | None


class ResidualMonitor:
    """Watch one unit's recalibration stream for drift anomalies.

    Feed it each re-fit commit via :meth:`observe`; it returns an
    :class:`AnomalyReport` combining the three checks in the module
    docstring. Stateful per unit (it owns the rolling history); construction
    is cheap and everything is plain NumPy.

    Args
    ----
    z_threshold : float
        |z| flag level for per-channel drift vs the unit's own history.
        Default 4.0 — the check runs per channel per epoch, so a 3σ
        threshold false-alarms within days on a healthy many-channel unit
        (multiple testing); 4σ keeps the healthy-stream false-positive rate
        negligible while a real fault crosses it days before failure.
    min_history : int
        Epochs of history required before z-scoring engages (a z-score
        against two points is noise).
    manifold : HealthyManifold | None
        Optional population check (built from healthy-unit samples).
    residual_rms_threshold : float | None
        Optional flag level for the window prediction-residual RMS.
    labels : Mapping[str, tuple[str, ...]] | None
        Optional per-path element labels for vector leaves (e.g.
        ``{"offbody.K_t_scale": twin.servo_names}``) so flags read as joints,
        not indices.
    std_floor : Mapping[str, float] | float | None
        Per-leaf-path (or scalar, applied to all) **floor on the history
        standard deviation** used in the z-score, in the channel's own
        units. Without it, a channel the re-fit holds nearly constant
        (e.g. prior-anchored, weakly-excited parameters) has an
        artificially tiny history variance, and harmless estimate jitter
        z-scores as an anomaly. Set it to the smallest change you would
        *care* about — a natural choice is a percent or two of the drift
        schema's bound range. ``None`` keeps a purely numerical floor.
    """

    def __init__(
        self,
        *,
        z_threshold: float = 4.0,
        min_history: int = 5,
        manifold: HealthyManifold | None = None,
        residual_rms_threshold: float | None = None,
        labels: Mapping[str, tuple[str, ...]] | None = None,
        std_floor: Mapping[str, float] | float | None = None,
    ):
        if z_threshold <= 0:
            raise JaxterityError(
                f"ResidualMonitor: z_threshold must be > 0, got {z_threshold}."
            )
        if min_history < 3:
            raise JaxterityError(
                f"ResidualMonitor: min_history must be ≥ 3 (a z-score needs a "
                f"std estimate), got {min_history}."
            )
        self.z_threshold = float(z_threshold)
        self.min_history = int(min_history)
        self.manifold = manifold
        self.residual_rms_threshold = residual_rms_threshold
        self.labels = labels
        self.std_floor = std_floor
        self._history: list[dict[str, float]] = []
        self._times: list[float] = []

    def _floor_for(self, channel: str) -> float:
        """The configured std floor for a flattened channel name."""
        if self.std_floor is None:
            return 0.0
        if not isinstance(self.std_floor, Mapping):
            return float(self.std_floor)
        # Vector channels are flattened as "path[label]" — match the path.
        path = channel.partition("[")[0]
        return float(self.std_floor.get(path, 0.0))

    @property
    def n_epochs(self) -> int:
        """Recalibration epochs observed so far."""
        return len(self._history)

    def observe(
        self,
        t: float,
        fitted: Mapping[str, Any],
        *,
        residual_rms: float | None = None,
    ) -> AnomalyReport:
        """Ingest one re-fit commit and return the anomaly verdict.

        Args
        ----
        t : float
            The commit's (simulated) time, s. Must increase monotonically.
        fitted : Mapping[str, array-like]
            The commit's fitted drift parameters
            (``WindowedFitResult.sysid_result.fitted``).
        residual_rms : float | None
            The window prediction-residual RMS (rad/s²), if available.

        Returns
        -------
        AnomalyReport
        """
        if self._times and t <= self._times[-1]:
            raise JaxterityError(
                f"ResidualMonitor.observe: time went backwards ({t} after "
                f"{self._times[-1]}); commits must arrive in order."
            )
        channels = _flatten(fitted, self.labels)
        if self._history and set(channels) != set(self._history[-1]):
            raise JaxterityError(
                "ResidualMonitor.observe: the fitted channels changed between "
                "commits; a unit's drift schema is fixed."
            )

        flags: list[str] = []
        z_scores: dict[str, float] = {}
        if len(self._history) >= self.min_history:
            for key, value in channels.items():
                past = np.asarray([h[key] for h in self._history], dtype=float)
                std = float(past.std(ddof=1))
                # Floor the std: the configured physical floor (see
                # ``std_floor``), plus a numerical floor so a perfectly-flat
                # history doesn't turn rounding dust into an infinite z.
                scale = max(
                    std,
                    self._floor_for(key),
                    1e-9,
                    1e-6 * max(abs(float(past.mean())), 1.0),
                )
                z = (value - float(past.mean())) / scale
                z_scores[key] = z
                if abs(z) > self.z_threshold:
                    flags.append(
                        f"{key}: z={z:+.1f} over its own recalibration history "
                        f"(threshold {self.z_threshold:.1f})"
                    )

        mahalanobis = None
        if self.manifold is not None:
            mahalanobis = self.manifold.mahalanobis(fitted, self.labels)
            if mahalanobis > self.manifold.threshold:
                flags.append(
                    f"parameters outside the healthy manifold "
                    f"(d²={mahalanobis:.1f} > {self.manifold.threshold:.1f})"
                )

        if (
            self.residual_rms_threshold is not None
            and residual_rms is not None
            and residual_rms > self.residual_rms_threshold
        ):
            flags.append(
                f"prediction residual RMS {residual_rms:.3g} exceeds "
                f"{self.residual_rms_threshold:.3g} — the declared drift "
                "parameters no longer explain the telemetry"
            )

        self._history.append(channels)
        self._times.append(float(t))
        return AnomalyReport(
            ok=not flags,
            flags=tuple(flags),
            z_scores=z_scores,
            mahalanobis=mahalanobis,
            residual_rms=residual_rms,
        )


__all__ = ["ResidualMonitor", "AnomalyReport", "HealthyManifold"]
