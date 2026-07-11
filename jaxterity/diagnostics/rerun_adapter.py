# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Rerun.io adapter for ``ParityReport`` and sim/real trajectories.

Rerun (https://rerun.io) is the optional visualization sink jaxterity targets
for time-series + structured-state debugging. This adapter logs a
:class:`~jaxterity.diagnostics.parity.ParityReport` alongside the sim and real
trajectories that produced it, so the maintainer can scroll the residual
visually instead of grepping log lines.

The ``rerun-sdk`` package is an optional dependency (see ``pyproject.toml``'s
``rerun`` extra). If it isn't installed, the adapter raises a typed
:class:`~jaxterity.errors.JaxterityError` with installation instructions —
callers can ``try/except`` to make Rerun strictly optional.
"""

from __future__ import annotations

import numpy as np

from ..errors import JaxterityError
from .parity import ParityReport


def _require_rerun():
    try:
        import rerun as rr

        return rr
    except ImportError as e:
        raise JaxterityError(
            'rerun-sdk is not installed. Install with `pip install "jaxterity[rerun]"` '
            "(or `pip install rerun-sdk>=0.20`) to use the rerun_adapter."
        ) from e


def log_parity_report(
    sim_trajectory: dict[str, np.ndarray],
    real_trajectory: dict[str, np.ndarray],
    report: ParityReport,
    *,
    times: np.ndarray | None = None,
    namespace: str = "parity",
    recording_id: str | None = None,
) -> None:
    """Stream a sim-vs-real comparison to a Rerun recording.

    For each quantity in ``report.quantities``:

    * Logs the sim trace under ``{namespace}/{quantity}/sim`` as a scalar
      time series (one entity per dimension if the quantity is multi-axis).
    * Logs the real trace under ``{namespace}/{quantity}/real``.
    * Logs the per-timestep residual under ``{namespace}/{quantity}/residual``.

    Per-quantity RMS and peak are also logged as run-level text via
    ``rr.TextLog`` so the report summary is visible in the timeline panel.

    Args
    ----
    sim_trajectory, real_trajectory : dict[str, ndarray]
        Same shape contract as :func:`~jaxterity.diagnostics.parity.parity_report`.
    report : ParityReport
        The pre-computed parity report. Logged alongside the trajectories so
        the recording is self-explanatory.
    times : numpy.ndarray, shape ``(n,)``
        Optional time vector; defaults to ``arange(n)``.
    namespace : str
        Root entity path. Useful for distinguishing multiple comparisons in
        one recording.
    recording_id : str or None
        Passed through to ``rerun.init``. Pass a stable id to append to an
        existing recording across multiple log calls.

    Raises
    ------
    JaxterityError
        If ``rerun-sdk`` is not installed (with install instructions).
    """
    rr = _require_rerun()
    rr.init(f"jaxterity-parity-{namespace}", recording_id=recording_id, spawn=False)
    n = next(iter(sim_trajectory.values())).shape[0]
    if times is None:
        times = np.arange(n, dtype=np.float64)
    for q in report.quantities:
        s = np.asarray(sim_trajectory[q], dtype=np.float64).reshape(n, -1)
        r = np.asarray(real_trajectory[q], dtype=np.float64).reshape(n, -1)
        for i in range(n):
            rr.set_time("time", duration=float(times[i]))
            for d in range(s.shape[1]):
                suffix = f".{d}" if s.shape[1] > 1 else ""
                rr.log(f"{namespace}/{q}/sim{suffix}", rr.Scalars(float(s[i, d])))
                rr.log(f"{namespace}/{q}/real{suffix}", rr.Scalars(float(r[i, d])))
                rr.log(
                    f"{namespace}/{q}/residual{suffix}",
                    rr.Scalars(float(s[i, d] - r[i, d])),
                )
    rr.log(
        f"{namespace}/_report",
        rr.TextLog(report.summary(), level=rr.TextLogLevel.INFO),
    )


__all__ = ["log_parity_report"]
