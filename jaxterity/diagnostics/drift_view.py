# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Local single-unit drift view.

Renders one unit's drift over its re-fit history: how its fitted off-body
parameters and their divergence from nominal evolve as it ages. The data-prep
layer (:func:`prepare_drift_view`) is pure NumPy and reproducible from the stored
:class:`~jaxterity.twin.TwinHistory` records; :func:`log_drift_view` streams the
prepared series to Rerun (consistent with the ``rerun_robot`` adapter,
``rerun-sdk`` kept lazy).

Honest scope: a **single-unit, local/offline** view — driven from stored records,
no hosted service. The fleet view / hosted dashboard is the commercial layer,
out of scope here.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np

from .rerun_robot import _require_rerun

if TYPE_CHECKING:
    # Only for annotations — importing at runtime would cycle (twin → diagnostics
    # → drift_view → twin). The functions duck-type the history's records.
    from ..twin.aging import TwinHistory


@dataclasses.dataclass(frozen=True)
class DriftView:
    """Prepared per-unit drift series (the render-ready, reproducible layer).

    Fields
    ------
    serial : str
        The unit's serial.
    times : numpy.ndarray, shape ``(n_updates,)``
        Per-update timestamp (or update index when no timestamps were recorded).
    keys : tuple[str, ...]
        The fitted leaves tracked (e.g. ``"offbody.R_th"``, ``"offbody.k_pwr"``).
    values : dict[str, numpy.ndarray]
        ``key -> (n_updates, n_servos)`` fitted value per update.
    nominal : dict[str, numpy.ndarray]
        ``key -> (n_servos,)`` the baseline each key's divergence is measured from.
    std : dict[str, numpy.ndarray]
        ``key -> (n_updates, n_servos)`` per-update standard error (√ the
        variance), or zeros when no uncertainty was recorded.
    """

    serial: str
    times: np.ndarray
    keys: tuple[str, ...]
    values: dict[str, np.ndarray]
    nominal: dict[str, np.ndarray]
    std: dict[str, np.ndarray]

    def divergence(self, key: str) -> np.ndarray:
        """Per-servo divergence from nominal for ``key``, ``(n_updates, n_servos)``."""
        return self.values[key] - self.nominal[key][None, :]

    def divergence_norm(self, key: str) -> np.ndarray:
        """Per-update divergence (L2 over servos) for ``key``, ``(n_updates,)``."""
        return np.linalg.norm(self.divergence(key), axis=1)


def prepare_drift_view(history: TwinHistory, *, nominal=None) -> DriftView:
    """Build a :class:`DriftView` from a unit's re-fit history.

    Pure and reproducible from the stored records. The divergence baseline
    defaults to the **first** update's fitted values (the as-shipped fit); pass
    ``nominal`` to measure drift from a different reference.

    Args
    ----
    history : TwinHistory
        The unit's update history.
    nominal : dict[str, array-like] or None
        Per-key baseline (``key -> (n_servos,)``). Defaults to the first update.

    Returns
    -------
    DriftView
        The prepared, render-ready series.

    Raises
    ------
    ValueError
        If ``history`` has no updates.
    """
    if not history.updates:
        raise ValueError(
            "prepare_drift_view needs a non-empty TwinHistory (no re-fits to "
            "show). Run track_thermal_drift over at least one window first."
        )
    keys = tuple(k for k, _ in history.updates[0].fitted)
    values: dict[str, np.ndarray] = {}
    std: dict[str, np.ndarray] = {}
    times = None
    for key in keys:
        t, v = history.parameter_series(key)
        times = t
        values[key] = v
        # √ variance per update, if the update carried uncertainty.
        rows = []
        for u in history.updates:
            unc = dict(u.uncertainty)
            if key in unc:
                rows.append(np.sqrt(np.asarray(unc[key], dtype=float)))
            else:
                rows.append(np.zeros(v.shape[1]))
        std[key] = np.stack(rows, axis=0)
    nominal_map: dict[str, np.ndarray] = {}
    supplied = dict(nominal) if nominal is not None else {}
    for key in keys:
        if key in supplied:
            nominal_map[key] = np.asarray(supplied[key], dtype=float)
        else:
            nominal_map[key] = values[key][0].copy()
    return DriftView(
        serial=history.serial,
        times=np.asarray(times),
        keys=keys,
        values=values,
        nominal=nominal_map,
        std=std,
    )


def log_drift_view(
    history: TwinHistory,
    *,
    nominal=None,
    namespace: str | None = None,
    spawn: bool = False,
    save_path=None,
    recording_id: str | None = None,
) -> None:
    """Stream a unit's drift series to Rerun.

    Logs, per fitted leaf and servo, the value and its divergence from nominal as
    scalar timeseries, plus a per-leaf scalar divergence norm — over the unit's
    re-fit timeline. Local/offline: ``spawn`` a viewer or write a ``.rrd`` via
    ``save_path``.

    Args
    ----
    history : TwinHistory
        The unit's update history.
    nominal : dict[str, array-like] or None
        Divergence baseline (see :func:`prepare_drift_view`).
    namespace : str or None
        Rerun entity-path root. Defaults to ``f"drift/{serial}"``.
    spawn : bool
        Spawn the Rerun viewer (live). Default ``False``.
    save_path : str or pathlib.Path, optional
        Write the recording to this ``.rrd`` instead of spawning.
    recording_id : str or None
        Explicit Rerun recording id.

    Raises
    ------
    ImportError
        If ``rerun-sdk`` is not installed.
    """
    rr = _require_rerun()
    view = prepare_drift_view(history, nominal=nominal)
    ns = namespace or f"drift/{view.serial}"
    rr.init(
        "jaxterity_drift",
        recording_id=recording_id or f"drift-{view.serial}",
        spawn=spawn,
    )
    if save_path is not None:
        rr.save(str(save_path))
    times = view.times
    for i in range(len(times)):
        rr.set_time("update", duration=float(times[i]))
        for key in view.keys:
            short = key.split(".")[-1]
            vals = view.values[key][i]
            div = view.divergence(key)[i]
            for j in range(vals.shape[0]):
                rr.log(f"{ns}/{short}/servo_{j}/value", rr.Scalars(float(vals[j])))
                rr.log(f"{ns}/{short}/servo_{j}/divergence", rr.Scalars(float(div[j])))
            rr.log(
                f"{ns}/{short}/divergence_norm",
                rr.Scalars(float(view.divergence_norm(key)[i])),
            )


__all__ = ["DriftView", "prepare_drift_view", "log_drift_view"]
