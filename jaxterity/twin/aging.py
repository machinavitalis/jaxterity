# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Twin currency: keep a `Twin` tracking a unit as it ages.

A `Twin` is fit once at calibration; a real unit drifts — servo bushings wear,
thermal paths change, friction climbs. This module keeps the twin *current* by
re-fitting its off-body leaves from rolling windows of recent telemetry, so the
twin a self-check and capability estimate read is the unit as it
is *now*, not as it left the bench.

Each re-fit:

- recovers the off-body thermal leaves from the window (via
  :func:`jaxterity.calibrate.calibrate_twin_thermal`), **carrying the fit
  uncertainty** onto the twin;
- advances the unit's attestation handle (a re-fit is a new unit version) and
  **links the prior handle**, so the version history is an explicit chain; and
- appends a :class:`TwinUpdate` to a :class:`TwinHistory` — the stored record
  the local drift view renders.

Honest scope: this is a **windowed re-fit** (each window fit independently from a
flat start), not a recursive estimator (EKF/UKF) that propagates a running state
+ covariance. The recursive-filter version is later work; the windowed form is
what a unit with periodic recalibration telemetry needs, and it composes — a
caller can feed overlapping windows for a smoother track.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ..calibrate.thermal import calibrate_twin_thermal
from ..data.schema import ThermalTelemetry
from .twin import Twin


@dataclasses.dataclass(frozen=True)
class TwinUpdate:
    """One re-fit in a twin's currency history.

    Fields
    ------
    index : int
        Position in the update stream (0-based).
    handle : str
        The twin's attestation handle *after* this re-fit.
    prior_handle : str
        The handle *before* this re-fit — the link to the predecessor version.
    fitted : tuple[tuple[str, tuple[float, ...]], ...]
        The off-body leaves this update set, keyed by canonical path (e.g.
        ``"offbody.R_th"``), each a per-servo tuple.
    uncertainty : tuple[tuple[str, tuple[float, ...]], ...]
        The per-leaf marginal variance attached by the fit, same keying.
    telemetry_hash : str
        Content hash of the window the re-fit consumed (provenance).
    timestamp : float | None
        Caller-supplied wall/elapsed time for this update (s), or ``None``. Used
        only to order/space the drift view; never affects the handle.
    """

    index: int
    handle: str
    prior_handle: str
    fitted: tuple[tuple[str, tuple[float, ...]], ...]
    uncertainty: tuple[tuple[str, tuple[float, ...]], ...]
    telemetry_hash: str
    timestamp: float | None = None

    def value(self, key: str) -> np.ndarray:
        """The fitted per-servo values for ``key`` (e.g. ``"offbody.R_th"``)."""
        for k, vals in self.fitted:
            if k == key:
                return np.asarray(vals, dtype=float)
        have = [k for k, _ in self.fitted]
        raise KeyError(f"TwinUpdate has no fitted leaf {key!r}; have {have}.")


@dataclasses.dataclass(frozen=True)
class TwinHistory:
    """The ordered re-fit history of one unit (drives the drift view).

    Fields
    ------
    serial : str
        The unit's serial (constant across the history).
    initial_handle : str
        The twin's handle before any re-fit.
    updates : tuple[TwinUpdate, ...]
        The re-fits in order.
    """

    serial: str
    initial_handle: str
    updates: tuple[TwinUpdate, ...] = ()

    @classmethod
    def empty(cls, twin: Twin) -> "TwinHistory":
        """An empty history rooted at ``twin``'s current handle."""
        return cls(
            serial=twin.serial, initial_handle=twin.attestation_handle, updates=()
        )

    def append(self, update: TwinUpdate) -> "TwinHistory":
        """Return a new history with ``update`` appended."""
        return dataclasses.replace(self, updates=(*self.updates, update))

    @property
    def handles(self) -> tuple[str, ...]:
        """Every handle in order: the initial one then each re-fit's."""
        return (self.initial_handle, *(u.handle for u in self.updates))

    @property
    def is_linked_chain(self) -> bool:
        """True iff every update's ``prior_handle`` is its predecessor's handle.

        The integrity check the provenance-chain test asserts: the version
        history is an unbroken links-to-predecessor chain rooted at
        :attr:`initial_handle`.
        """
        prev = self.initial_handle
        for u in self.updates:
            if u.prior_handle != prev:
                return False
            prev = u.handle
        return True

    def parameter_series(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        """Per-update timeseries of a fitted leaf, for the drift view.

        Args
        ----
        key : str
            The leaf path (e.g. ``"offbody.R_th"``).

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            ``(timestamps, values)`` — ``timestamps`` shape ``(n_updates,)``
            (``index`` when no timestamps were supplied), ``values`` shape
            ``(n_updates, n_servos)``.
        """
        if not self.updates:
            return np.zeros(0), np.zeros((0, 0))
        values = np.stack([u.value(key) for u in self.updates], axis=0)
        times = np.array(
            [
                float(u.timestamp) if u.timestamp is not None else float(u.index)
                for u in self.updates
            ]
        )
        return times, values


def update_twin_thermal(
    twin: Twin,
    telemetry: ThermalTelemetry,
    tau,
    *,
    index: int = 0,
    C_th=None,
    recipe_version: str = "1.0",
    timestamp: float | None = None,
) -> tuple[Twin, TwinUpdate]:
    """Re-fit a twin's off-body thermal from one window, returning the update.

    A single windowed re-fit: recovers ``R_th`` / ``k_pwr`` from ``telemetry``
    (carrying the uncertainty), advances + links the attestation handle,
    and packages the move as a :class:`TwinUpdate`.

    Args
    ----
    twin : Twin
        The current twin (the version to advance from).
    telemetry : ThermalTelemetry
        The recalibration window for this unit.
    tau : numpy.ndarray, shape ``(n, n_servos)``, units N·m
        Driving torque aligned with the window.
    index : int
        The update's position in the stream.
    C_th : array-like or None
        Known thermal capacitance(s); defaults to the twin's current ``C_th``.
    recipe_version : str
        Provenance recipe version.
    timestamp : float or None
        Optional wall/elapsed time tag for the drift view.

    Returns
    -------
    tuple[Twin, TwinUpdate]
        The re-fit twin (new handle) and the update record linking it to the
        prior version.
    """
    prior_handle = twin.attestation_handle
    refit = calibrate_twin_thermal(
        twin, telemetry, tau, C_th=C_th, recipe_version=recipe_version
    )
    fitted = (
        ("offbody.R_th", tuple(float(v) for v in np.asarray(refit.R_th))),
        ("offbody.k_pwr", tuple(float(v) for v in np.asarray(refit.k_pwr))),
    )
    update = TwinUpdate(
        index=index,
        handle=refit.attestation_handle,
        prior_handle=prior_handle,
        fitted=fitted,
        uncertainty=refit.param_uncertainty,
        telemetry_hash=telemetry.hash().hex(),
        timestamp=timestamp,
    )
    return refit, update


def track_thermal_drift(
    twin: Twin,
    windows,
    *,
    C_th=None,
    recipe_version: str = "1.0",
) -> tuple[Twin, TwinHistory]:
    """Track a unit across a stream of recalibration windows.

    Folds :func:`update_twin_thermal` over a sequence of windows, returning the
    current twin and the full :class:`TwinHistory`. As the unit ages (its true
    thermal leaves drifting), the tracked twin follows; the history is an
    unbroken handle chain.

    Args
    ----
    twin : Twin
        The starting (freshly-calibrated) twin.
    windows : sequence of (ThermalTelemetry, tau) or (ThermalTelemetry, tau, timestamp)
        The rolling recalibration windows in time order.
    C_th : array-like or None
        Known thermal capacitance(s); defaults to the twin's current ``C_th``.
    recipe_version : str
        Provenance recipe version.

    Returns
    -------
    tuple[Twin, TwinHistory]
        The current (latest) twin and its update history.
    """
    history = TwinHistory.empty(twin)
    current = twin
    for i, window in enumerate(windows):
        if len(window) == 3:
            telemetry, tau, timestamp = window
        else:
            telemetry, tau = window
            timestamp = None
        current, update = update_twin_thermal(
            current,
            telemetry,
            tau,
            index=i,
            C_th=C_th,
            recipe_version=recipe_version,
            timestamp=timestamp,
        )
        history = history.append(update)
    return current, history


__all__ = [
    "TwinUpdate",
    "TwinHistory",
    "update_twin_thermal",
    "track_thermal_drift",
]
