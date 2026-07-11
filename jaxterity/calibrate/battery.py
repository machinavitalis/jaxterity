# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Fit a LeKiwi battery's Thévenin sag from a real aggregate-load trace.

A single servo barely moves the shared bus voltage; several drawing current at
once do. ``scripts/lekiwi_load_capture.py`` commands a synchronized multisine on
a set of arm servos and records, each tick, every servo's PWM-duty ``load`` plus
the **shared bus voltage**. This module fits the Thévenin model

    V_bus = V_oc - R_int * I_total ,    I_total = i_stall * Σ_servo |load| / 1000

by linear least squares (``V_bus`` regressed on the aggregate current proxy). The
per-servo duty fraction ``|load|/1000`` times the servo stall current ``i_stall``
estimates each servo's current draw; ``i_stall`` sets the absolute current scale,
so ``R_int`` is identified up to that scale (the *bus-voltage prediction* from the
commanded load — what the brownout veto consumes — is what the fit validates).

The fitted ``V_oc``/``R_int`` build a :func:`jaxterity.twin.lekiwi_twin` whose
predicted bus voltage the :class:`~jaxterity.selfcheck.SafetyMonitor` checks
against a floor — the LeKiwi brownout veto.

Current-frame caveat (honest): this fit measures current as ``i_stall · duty``,
but the twin's :func:`jaxterity.physics.battery.bus_voltage` measures it as
``k_i · |tau|`` (a torque→current gain). The fitted ``R_int`` is therefore in the
duty-current frame; applied in the twin's torque-current frame the *absolute*
predicted brownout voltage is approximate (the two frames agree only up to the
``i_stall``/``k_i`` ratio). The brownout veto demo uses it as a *scenario*
(aged-pack ``R_int``), not a calibrated absolute voltage prediction — see
``examples/selfcheck_demo_lekiwi.py``. Unifying the two current models is a follow-up.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

import numpy as np

from ..errors import CalibrationError
from ..sysid import SysidResult

#: STS3215 stall current at ~12 V (A) — sets the absolute current scale.
_DEFAULT_I_STALL = 2.7


@dataclasses.dataclass(frozen=True)
class BusLoadCapture:
    """Parsed aggregate multi-servo load trace (``scripts/lekiwi_load_capture.py``).

    Fields
    ------
    servo_ids : tuple[int, ...]
        The commanded servos.
    t : np.ndarray
        Sample times, s.
    aggregate_duty : np.ndarray
        Σ over servos of ``|load|/1000`` (dimensionless duty sum), shape ``(n,)``.
    bus_voltage : np.ndarray
        Shared bus voltage, V, shape ``(n,)``.
    source_hash : str
        BLAKE2b hash of the raw capture bytes.
    """

    servo_ids: tuple[int, ...]
    t: np.ndarray
    aggregate_duty: np.ndarray
    bus_voltage: np.ndarray
    source_hash: str

    @property
    def n_samples(self) -> int:
        return int(self.t.shape[0])


def load_bus_load_trace(source: str | dict) -> BusLoadCapture:
    """Load an aggregate-load capture JSON (schema ``lekiwi-load-capture-v1``)."""
    if isinstance(source, dict):
        cap = source
        raw = json.dumps(cap, sort_keys=True).encode()
    else:
        with open(source, "rb") as f:
            raw = f.read()
        cap = json.loads(raw)
    if cap.get("schema") != "lekiwi-load-capture-v1":
        raise CalibrationError(
            f"load_bus_load_trace: unrecognized schema {cap.get('schema')!r}; "
            "expected 'lekiwi-load-capture-v1' (scripts/lekiwi_load_capture.py)."
        )
    if cap.get("aborted"):
        raise CalibrationError("load_bus_load_trace: capture aborted (guard tripped).")
    ids = tuple(int(s) for s in cap["servo_ids"])
    loads = cap["load_raw"]
    agg = sum(np.abs(np.asarray(loads[str(s)], float)) / 1000.0 for s in ids)
    return BusLoadCapture(
        servo_ids=ids,
        t=np.asarray(cap["t"], float),
        aggregate_duty=np.asarray(agg, float),
        bus_voltage=np.asarray(cap["volt_dV"], float) / 10.0,
        source_hash=hashlib.blake2b(raw, digest_size=16).hexdigest(),
    )


def fit_battery_sag(
    capture: BusLoadCapture, *, i_stall: float = _DEFAULT_I_STALL
) -> SysidResult:
    """Fit Thévenin ``V_oc``/``R_int`` from an aggregate-load trace (linear LS).

    Regresses ``V_bus`` on the aggregate current ``I_total = i_stall * aggregate_duty``
    to recover ``V_oc`` (intercept) and ``R_int`` (negative slope). Reports the fit
    ``R^2``, the observed sag, and the peak current in the residuals.

    Args
    ----
    capture : BusLoadCapture
        The parsed aggregate-load trace (:func:`load_bus_load_trace`).
    i_stall : float
        Per-servo stall current, A, setting the absolute current scale.

    Returns
    -------
    SysidResult
        ``fitted`` = ``{"V_oc": V, "R_int": Ω}``; ``residuals`` carry ``r_squared``,
        the observed ``sag_v``, and ``peak_current_a``.

    Raises
    ------
    CalibrationError
        If the trace is too short or shows no load variation to fit against.
    """
    if capture.n_samples < 20:
        raise CalibrationError(
            f"fit_battery_sag: trace has only {capture.n_samples} samples; need >= 20."
        )
    i_total = float(i_stall) * capture.aggregate_duty
    v = capture.bus_voltage
    if float(np.ptp(i_total)) < 1e-6:
        raise CalibrationError(
            "fit_battery_sag: no current variation in the trace; cannot identify "
            "R_int (was anything actually commanded to move?)."
        )
    design = np.column_stack([np.ones_like(i_total), -i_total])
    coef, *_ = np.linalg.lstsq(design, v, rcond=None)
    v_oc, r_int = float(coef[0]), float(coef[1])
    resid = v - design @ coef
    ss_tot = float(np.sum((v - v.mean()) ** 2))
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else 0.0
    return SysidResult(
        fitted={"V_oc": np.asarray(v_oc), "R_int": np.asarray(r_int)},
        covariance=None,
        residuals={
            "r_squared": r2,
            "sag_v": float(v.max() - v.min()),
            "peak_current_a": float(i_total.max()),
            "i_stall_a": float(i_stall),
        },
        telemetry_hash=capture.source_hash,
        recipe_name="battery.thevenin",
        recipe_version="1.0",
    )


__all__ = ["BusLoadCapture", "load_bus_load_trace", "fit_battery_sag"]
