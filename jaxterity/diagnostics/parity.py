# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Sim-to-real parity report.

``ParityReport`` is the structured diff between a simulated and a real (or
synthetic-truth) trajectory. It collects per-quantity RMS and peak divergence
plus a list of suggested parameter perturbations (sensitivity-based) — what a
maintainer needs to read at a glance to decide whether a calibrated model is
close enough to ship, and if not, which knob to nudge.

The current implementation ships the residual statistics. The
sensitivity-based suggested-perturbations field is populated only when callers
supply a parameter-sensitivity dict (otherwise it's empty); computing it
end-to-end from a Robot requires the forward-mode
:func:`~jaxterity.sysid.parameter_jacobian` and lives on the caller side for
now. The Rerun visualization (CONTEXT.md mentions it) is deferred —
``ParityReport`` is data-only and can be passed straight to a Rerun adapter
in a follow-up.
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class ParityReport:
    """Per-quantity sim-to-real divergence summary.

    Fields
    ------
    quantities : tuple[str, ...]
        Names of the quantities compared (e.g. ``("q", "qd")``).
    rms : dict[str, float]
        RMS divergence per quantity, in the quantity's units.
    peak : dict[str, float]
        Max absolute divergence per quantity.
    n_samples : int
        Length of the compared trajectories.
    suggested_perturbations : dict[str, float]
        Per-parameter sensitivity-weighted nudge magnitudes (empty unless the
        caller supplied a sensitivity dict; see :func:`parity_report`).

    Methods
    -------
    summary() -> str
        One-screen text summary suitable for printing.
    ok(rms_threshold) -> bool
        True iff every quantity's RMS is below ``rms_threshold``.
    """

    quantities: tuple[str, ...]
    rms: dict[str, float]
    peak: dict[str, float]
    n_samples: int
    suggested_perturbations: dict[str, float] = dataclasses.field(default_factory=dict)

    def ok(self, rms_threshold: float | dict[str, float]) -> bool:
        """True iff every quantity's RMS is at or below the threshold."""
        if isinstance(rms_threshold, dict):
            return all(self.rms[q] <= rms_threshold[q] for q in self.quantities)
        return all(v <= rms_threshold for v in self.rms.values())

    def summary(self) -> str:
        """Multi-line text summary."""
        lines = [f"ParityReport over {self.n_samples} samples:"]
        for q in self.quantities:
            lines.append(f"  {q:12s} RMS={self.rms[q]:.3g}  peak={self.peak[q]:.3g}")
        if self.suggested_perturbations:
            lines.append("  suggested perturbations:")
            for p, mag in self.suggested_perturbations.items():
                lines.append(f"    {p:24s} ±{mag:.3g}")
        return "\n".join(lines)


def parity_report(
    sim_trajectory: dict[str, np.ndarray],
    real_trajectory: dict[str, np.ndarray],
    *,
    sensitivities: dict[str, dict[str, np.ndarray]] | None = None,
) -> ParityReport:
    """Compare a simulated trajectory to a real (or truth) trajectory.

    Both trajectories are dicts of ``{quantity_name: array}`` with matching
    keys and matching leading-axis lengths.

    Args
    ----
    sim_trajectory : dict[str, ndarray]
        Simulator output. Each value's shape is ``(n, ...)`` — the first axis
        is time, the rest is the quantity's natural shape.
    real_trajectory : dict[str, ndarray]
        Real (or synthetic-truth) data, same keys and shapes as ``sim_trajectory``.
    sensitivities : dict[str, dict[str, ndarray]] or None
        Optional per-parameter sensitivity of each quantity (rows: parameter
        names, cols: quantity dimensions). If provided, the suggested
        perturbations are computed by projecting the residual onto each
        parameter's sensitivity direction (cheap pseudo-inverse step).

    Returns
    -------
    ParityReport

    Raises
    ------
    ValueError
        If ``sim_trajectory`` and ``real_trajectory`` have mismatched keys or
        per-quantity shapes.
    """
    if set(sim_trajectory) != set(real_trajectory):
        raise ValueError(
            "parity_report: sim and real trajectories must share keys; "
            f"got sim={sorted(sim_trajectory)}, real={sorted(real_trajectory)}."
        )
    quantities = tuple(sorted(sim_trajectory))
    rms: dict[str, float] = {}
    peak: dict[str, float] = {}
    n_samples = next(iter(sim_trajectory.values())).shape[0]
    for q in quantities:
        s = np.asarray(sim_trajectory[q], dtype=np.float64)
        r = np.asarray(real_trajectory[q], dtype=np.float64)
        if s.shape != r.shape:
            raise ValueError(
                f"parity_report: '{q}' shape mismatch: sim {s.shape} vs real {r.shape}."
            )
        diff = (s - r).reshape(s.shape[0], -1)
        rms[q] = float(np.sqrt(np.mean(diff * diff)))
        peak[q] = float(np.max(np.abs(diff)))
    suggested: dict[str, float] = {}
    if sensitivities:
        # Per-parameter nudge magnitude: project the residual onto the
        # sensitivity direction (pseudo-inverse step on the flattened residual).
        for param, by_quantity in sensitivities.items():
            num = 0.0
            den = 0.0
            for q, J in by_quantity.items():
                if q not in sim_trajectory:
                    continue
                diff = (
                    np.asarray(real_trajectory[q]) - np.asarray(sim_trajectory[q])
                ).flatten()
                Jf = np.asarray(J).flatten()
                num += float(np.dot(Jf, diff))
                den += float(np.dot(Jf, Jf))
            if den > 0:
                suggested[param] = abs(num / den)
    return ParityReport(
        quantities=quantities,
        rms=rms,
        peak=peak,
        n_samples=n_samples,
        suggested_perturbations=suggested,
    )


__all__ = ["ParityReport", "parity_report"]
