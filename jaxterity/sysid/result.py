# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""The structured result of a system-identification recipe (ADR-010).

Every sysid recipe returns a :class:`SysidResult` — never a bare tuple
(CONTEXT.md invariant 4). It carries the fitted parameters, their covariance,
residual statistics, the telemetry hash, the recipe name + version, and a
structured warning list, and applies itself to a robot via :meth:`apply_to`,
contributing the recipe version and telemetry hash to the attestation chain
(ADR-008). :class:`SysidRecipe` is the type protocol every recipe satisfies.
"""

from __future__ import annotations

from statistics import NormalDist
from typing import Any, Protocol, runtime_checkable

import equinox as eqx
import numpy as np

from ..robot import Robot
from .protocol import SysidProtocol


class SysidResult(eqx.Module):
    """The outcome of a system-identification fit.

    Args
    ----
    fitted : dict[str, jax.Array]
        Identified parameter values, keyed by ``"<component>.<field>"`` (the
        :meth:`Robot.with_parameters` convention).
    telemetry_hash : str
        Content hash of the telemetry used in the fit (provenance).
    recipe_name, recipe_version : str
        Identity of the recipe that produced this result.
    covariance : dict[str, float] | None
        Optional per-parameter **marginal variance** (the diagonal of the local
        fit covariance, keyed like ``fitted``), produced by the estimators
        (:mod:`jaxterity.sysid.uncertainty`). ``None`` when uncertainty was not
        requested. Read it through :meth:`std_error` / :meth:`interval`.
    residuals : dict[str, float]
        Per-quantity residual statistics (e.g. RMS, max).
    warnings : tuple[str, ...]
        Structured non-fatal warnings (e.g. weak identifiability).
    """

    fitted: dict[str, Any]
    covariance: dict[str, Any] | None = None
    residuals: dict[str, float] | None = None
    telemetry_hash: str = eqx.field(static=True, default="")
    recipe_name: str = eqx.field(static=True, default="")
    recipe_version: str = eqx.field(static=True, default="")
    warnings: tuple[str, ...] = eqx.field(static=True, default=())

    def apply_to(self, robot: Robot) -> Robot:
        """Return a calibrated robot with the fitted parameters and provenance.

        Binds :attr:`fitted` via :meth:`Robot.with_parameters`, then records the
        recipe name/version and telemetry hash via
        :meth:`Robot.with_provenance` (so the attestation handle reflects the
        fit). The input robot is unchanged.
        """
        calibrated = robot.with_parameters(self.fitted)
        return calibrated.with_provenance(
            (self.recipe_name, self.recipe_version, self.telemetry_hash)
        )

    def std_error(self) -> dict[str, float] | None:
        """Per-parameter standard error (√ marginal variance), or ``None``.

        Returns
        -------
        dict[str, float] or None
            ``{param: standard_error}`` derived from :attr:`covariance`, in each
            parameter's own units; ``None`` if no covariance was estimated.
        """
        if self.covariance is None:
            return None
        return {
            k: float(np.sqrt(max(float(v), 0.0))) for k, v in self.covariance.items()
        }

    def interval(self, coverage: float = 0.95) -> dict[str, tuple[float, float]] | None:
        """Per-parameter confidence interval at the stated coverage, or ``None``.

        A Wald (normal-approximation) interval ``θ̂ ± z·SE`` where ``z`` is the
        two-sided normal quantile for ``coverage`` (e.g. 1.96 at 0.95). Local —
        as honest as the underlying Laplace/bootstrap covariance.

        Args
        ----
        coverage : float
            Target coverage in ``(0, 1)`` (default 0.95).

        Returns
        -------
        dict[str, tuple[float, float]] or None
            ``{param: (lower, upper)}`` for each fitted parameter; ``None`` if no
            covariance was estimated.

        Raises
        ------
        ValueError
            If ``coverage`` is not in ``(0, 1)``.
        """
        if not 0.0 < coverage < 1.0:
            raise ValueError(
                f"interval coverage must be in (0, 1); got {coverage!r}. "
                "Use e.g. 0.95 for a 95% interval."
            )
        se = self.std_error()
        if se is None:
            return None
        z = NormalDist().inv_cdf(0.5 + coverage / 2.0)
        out: dict[str, tuple[float, float]] = {}
        for k, point in self.fitted.items():
            arr = np.asarray(point)
            if arr.ndim == 0:
                center = float(arr)
                half = z * se.get(k, 0.0)
                out[k] = (center - half, center + half)
            else:
                # Vector leaf (e.g. a payload CoM): one interval per component,
                # keyed ``"<param>[i]"`` — the covariance convention of the
                # recipes that fit vector leaves.
                for i, center in enumerate(arr.ravel()):
                    half = z * se.get(f"{k}[{i}]", 0.0)
                    out[f"{k}[{i}]"] = (float(center) - half, float(center) + half)
        return out


@runtime_checkable
class SysidRecipe(Protocol):
    """The type every system-identification recipe satisfies.

    A recipe declares its identity and the data-collection protocol it expects,
    and is callable on telemetry + a robot, returning a :class:`SysidResult`.
    """

    recipe_name: str
    recipe_version: str
    protocol: SysidProtocol

    def __call__(
        self, telemetry: Any, robot: Robot, hints: dict | None = None
    ) -> SysidResult: ...


__all__ = ["SysidResult", "SysidRecipe"]
