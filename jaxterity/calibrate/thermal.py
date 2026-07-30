# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Per-unit off-body thermal identification.

`so101_twin` / `lekiwi_twin` ship *representative* servo-thermal defaults. This
module fits them to a specific unit from a temperature trace, closing the
"placeholder thermal" gap: :func:`fit_servo_thermal` recovers per-servo ``R_th``
and ``k_pwr`` from a :class:`~jaxterity.data.schema.ThermalTelemetry` trace plus
the driving torque, and :func:`calibrate_twin_thermal` applies the fit to a
:class:`~jaxterity.twin.Twin` (recording provenance, so the unit's handle moves).

Identifiability (see :mod:`jaxterity.physics.thermal`): a temperature+torque
trace identifies only ``k_pwr/C_th`` and ``R_th·C_th`` per servo, so ``C_th`` is
held **known** (the lumped thermal mass, from geometry/datasheet) and ``R_th`` /
``k_pwr`` are fitted. The per-servo regression is the lumped ODE in linear form:

    Ṫ_j = (k_pwr_j/C_th_j)·tau_j²  −  (1/(R_th_j·C_th_j))·(T_j − T_amb)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..data.schema import ThermalTelemetry
from ..errors import CalibrationError

#: Schema of the on-device static-hold thermal capture
#: (``scripts/lekiwi_thermal_capture.py``).
THERMAL_CAPTURE_SCHEMA = "lekiwi-thermal-capture-v1"

#: Kelvin at 0 °C — the register reports whole degrees Celsius.
_KELVIN_OFFSET = 273.15

#: Rolling-median window and jump threshold (°C) for the Feetech
#: present-temperature register's occasional wild single-sample readings. Same
#: treatment ``scripts/fit_twin_defaults.py`` applies, kept here so the library
#: path and the script path clean the channel identically.
_DESPIKE_WINDOW = 7
_DESPIKE_JUMP_C = 5.0


def fit_servo_thermal(
    telemetry: ThermalTelemetry,
    tau,
    *,
    C_th,
    T_amb: float,
) -> dict[str, np.ndarray]:
    """Recover per-servo ``R_th`` and ``k_pwr`` from a temperature trace.

    Linear-least-squares fit of the lumped servo-thermal ODE per servo column,
    with ``C_th`` supplied (the trace fixes only the two combinations
    ``k_pwr/C_th`` and ``R_th·C_th`` — see the module docstring).

    Args
    ----
    telemetry : ThermalTelemetry
        Per-servo temperature trace, shape ``(n, n_servos)`` (K).
    tau : numpy.ndarray, shape ``(n, n_servos)``, units N·m
        Per-servo driving torque, aligned with the telemetry timestamps.
    C_th : float or array-like, shape ``(n_servos,)``
        Known lumped thermal capacitance(s), J/K.
    T_amb : float
        Ambient temperature, K.

    Returns
    -------
    dict[str, numpy.ndarray]
        ``{"R_th": (n_servos,), "k_pwr": (n_servos,)}`` plus their per-servo
        marginal variances ``{"R_th_var": (n_servos,), "k_pwr_var":
        (n_servos,)}`` — the Laplace covariance of the linear fit, propagated
        from the regression coefficients. All float64.

    Raises
    ------
    ValueError
        If any ``C_th`` is not strictly positive, or shapes disagree.
    """
    t = np.asarray(telemetry.time, dtype=np.float64)
    T = np.asarray(telemetry.temperature, dtype=np.float64)
    tau_arr = np.asarray(tau, dtype=np.float64)
    if T.ndim != 2 or tau_arr.shape != T.shape:
        raise ValueError(
            f"fit_servo_thermal: temperature {T.shape} and tau {tau_arr.shape} "
            "must be the same 2-D (n, n_servos) shape."
        )
    n_servos = T.shape[1]
    C = np.broadcast_to(np.asarray(C_th, dtype=np.float64), (n_servos,))
    if not np.all(C > 0):
        raise ValueError(f"fit_servo_thermal needs C_th > 0 (J/K); got {C_th!r}.")

    R_th = np.empty(n_servos)
    k_pwr = np.empty(n_servos)
    R_th_var = np.empty(n_servos)
    k_pwr_var = np.empty(n_servos)
    for j in range(n_servos):
        # central difference for Ṫ; boundary samples dropped.
        Tdot = (T[2:, j] - T[:-2, j]) / (t[2:] - t[:-2])
        T_c = T[1:-1, j]
        tau_c = tau_arr[1:-1, j]
        X = np.stack([tau_c * tau_c, -(T_c - T_amb)], axis=1)
        coef, *_ = np.linalg.lstsq(X, Tdot, rcond=None)
        a, b = float(coef[0]), float(coef[1])
        k_pwr[j] = a * C[j]
        R_th[j] = 1.0 / (b * C[j])
        # Laplace covariance of the linear coefficients, propagated to the
        # physical leaves. a = k_pwr/C, b = 1/(R_th·C):
        #   var(k_pwr) = C²·var(a);  var(R_th) = var(b)/(b⁴·C²)  (delta method).
        resid = Tdot - X @ coef
        dof = max(X.shape[0] - X.shape[1], 1)
        sigma2 = float(resid @ resid) / dof
        cov = sigma2 * np.linalg.pinv(X.T @ X)
        k_pwr_var[j] = C[j] * C[j] * float(cov[0, 0])
        R_th_var[j] = float(cov[1, 1]) / (b**4 * C[j] * C[j])
    return {
        "R_th": R_th,
        "k_pwr": k_pwr,
        "R_th_var": R_th_var,
        "k_pwr_var": k_pwr_var,
    }


def calibrate_twin_thermal(
    twin,
    telemetry: ThermalTelemetry,
    tau,
    *,
    C_th=None,
    recipe_version: str = "1.0",
    telemetry_hash: str | None = None,
):
    """Fit a twin's off-body thermal from telemetry and return the updated twin.

    Recovers per-servo ``R_th`` / ``k_pwr`` (holding ``C_th`` known — defaults to
    the twin's current ``C_th``) and applies them via
    :meth:`~jaxterity.twin.Twin.with_thermal`, appending a ``thermal.sysid``
    provenance record so the unit's attestation handle reflects the fit.

    Args
    ----
    twin : jaxterity.twin.Twin
        The unit whose thermal model to calibrate.
    telemetry : ThermalTelemetry
        Per-servo temperature trace for this unit.
    tau : numpy.ndarray, shape ``(n, n_servos)``, units N·m
        The driving torque aligned with the trace.
    C_th : array-like or None
        Known thermal capacitance(s); defaults to the twin's current ``C_th``.
    recipe_version : str
        Provenance recipe version.
    telemetry_hash : str or None
        Provenance telemetry hash; defaults to the trace's content hash.

    Returns
    -------
    jaxterity.twin.Twin
        A twin with the fitted thermal leaves + provenance (new handle).
    """
    import numpy as _np

    C = _np.asarray(twin.C_th) if C_th is None else C_th
    fit = fit_servo_thermal(telemetry, tau, C_th=C, T_amb=twin.T_amb)
    record = (
        "thermal.sysid",
        recipe_version,
        telemetry_hash if telemetry_hash is not None else telemetry.hash().hex(),
    )
    # Carry the fit uncertainty into the twin (a derived quantity — does
    # not change the attestation handle).
    param_uncertainty = (
        ("offbody.R_th", tuple(float(v) for v in fit["R_th_var"])),
        ("offbody.k_pwr", tuple(float(v) for v in fit["k_pwr_var"])),
    )
    return twin.with_thermal(
        R_th=fit["R_th"],
        k_pwr=fit["k_pwr"],
        provenance_record=record,
        param_uncertainty=param_uncertainty,
    )


def _despike(temp: np.ndarray) -> np.ndarray:
    """Replace wild single-sample temperature readings with the local median.

    The Feetech present-temperature register intermittently returns nonsense (a
    32 °C servo reporting 78), so any statistic with a ``max`` in it reports a
    temperature the hardware never reached — that is how a +1 °C rise once got
    written down as +8 °C.

    This only rescues *isolated* bad samples. Under sustained drive the register
    garbles often enough that bad samples land adjacent and the local median is
    itself contaminated; there, no filter helps and the answer is to report a
    median over the whole segment plus the fraction discarded (see
    :mod:`jaxterity.checkup.wheels`). Prefer medians to maxima either way.
    """
    out = temp.astype(float).copy()
    n = out.size
    for i in range(n):
        lo, hi = max(0, i - _DESPIKE_WINDOW // 2), min(n, i + _DESPIKE_WINDOW // 2 + 1)
        median = float(np.median(temp[lo:hi]))
        if abs(float(temp[i]) - median) > _DESPIKE_JUMP_C:
            out[i] = median
    return out


def load_thermal_capture(
    source: str | Path | dict,
    *,
    phases: tuple[str, ...] = ("baseline", "hold", "cool"),
) -> tuple[ThermalTelemetry, np.ndarray, dict]:
    """Load a static-hold thermal capture into the fitter's inputs.

    Reads ``scripts/lekiwi_thermal_capture.py``'s output and returns the pieces
    :func:`fit_servo_thermal` needs: the per-servo temperature trace in **kelvin**
    (SI at the boundary, per the unit contract), the per-servo driving torque, and
    the capture's metadata.

    The torque column is the servo's own load-duty proxy scaled by bus voltage —
    ``(load/1000) · V_bus`` — the same frame
    :func:`~jaxterity.calibrate.identify_closed_loop` uses. That means the fitted
    ``k_pwr`` is *in that frame*, not in true N·m: the duty proxy is proportional
    to torque with an unmeasured constant, so ``k_pwr`` absorbs it. Stated here
    because it is the same caveat the battery fit carries, and it is the reason a
    per-unit ``k_i`` measurement (a current-frame capture) is still open.

    The temperature register reports whole degrees and occasionally returns a
    wild single sample; both are handled — the wild samples by a rolling median,
    the quantization by *not pretending it away* (see
    :func:`jaxterity.sensors.temperature_sensor.quantization_floor` for the
    standard error it implies).

    Args
    ----
    source : str | Path | dict
        Capture path or already-parsed document.
    phases : tuple[str, ...]
        Which logged phases to keep. The default keeps all three; pass
        ``("hold",)`` to fit the rise alone, or ``("cool",)`` for the decay,
        which is how the cooling constant is separated from the heating gain.

    Returns
    -------
    tuple[ThermalTelemetry, numpy.ndarray, dict]
        ``(telemetry, tau, meta)`` — temperature ``(n, n_servos)`` in K, torque
        proxy ``(n, n_servos)``, and the capture's metadata (schema, servo ids,
        joint names, phase spans, pose).

    Raises
    ------
    CalibrationError
        If the schema is unrecognized, a requested phase is absent, or the
        capture carries no usable samples.

    Units
    -----
    K, and the duty-proxy torque frame described above.
    """
    doc = source if isinstance(source, dict) else json.loads(Path(source).read_text())
    if doc.get("schema") != THERMAL_CAPTURE_SCHEMA:
        raise CalibrationError(
            f"load_thermal_capture: unrecognized schema {doc.get('schema')!r}; "
            f"expected {THERMAL_CAPTURE_SCHEMA!r}. The short "
            "commanded-excitation captures are a different schema and carry no "
            "usable thermal signal — use a static-hold capture."
        )
    servo_ids = [int(s) for s in doc["servo_ids"]]
    phase_column = np.asarray(doc["phase"], dtype=object)
    keep = np.isin(phase_column, list(phases))
    if not keep.any():
        raise CalibrationError(
            f"load_thermal_capture: no samples in phase(s) {list(phases)}; the "
            f"capture logged {sorted(set(phase_column))}."
        )
    time = np.asarray(doc["t"], dtype=np.float64)[keep]
    time = time - time[0]

    temperature = np.empty((time.size, len(servo_ids)))
    torque = np.empty((time.size, len(servo_ids)))
    for column, sid in enumerate(servo_ids):
        key = str(sid)
        raw_temp = np.asarray(
            [np.nan if v is None else float(v) for v in doc["temp_C"][key]]
        )[keep]
        temperature[:, column] = _despike(raw_temp) + _KELVIN_OFFSET
        load = np.asarray(
            [0.0 if v is None else float(v) for v in doc["load_raw"][key]]
        )[keep]
        volts = np.asarray(
            [0.0 if v is None else float(v) for v in doc["volt_dV"][key]]
        )[keep]
        torque[:, column] = (load / 1000.0) * (volts / 10.0)

    telemetry = ThermalTelemetry(
        time=time,
        sample_rate_hz=float(doc.get("achieved_rate_hz") or 1.0),
        servo_names=tuple(doc["joints"].get(str(sid), str(sid)) for sid in servo_ids),
        temperature=temperature,
    )
    meta = {
        "schema": doc["schema"],
        "servo_ids": servo_ids,
        "joints": dict(doc.get("joints", {})),
        "excitation": dict(doc.get("excitation", {})),
        "plan": dict(doc.get("plan", {})),
        "pose_ticks": dict(doc.get("pose_ticks", {})),
        "phases": list(doc.get("phase_spans", doc.get("phases", []))),
        "dry_run": bool(doc.get("dry_run", False)),
        "aborted": bool(doc.get("aborted", False)),
    }
    return telemetry, torque, meta


__all__ = [
    "THERMAL_CAPTURE_SCHEMA",
    "fit_servo_thermal",
    "calibrate_twin_thermal",
    "load_thermal_capture",
]
