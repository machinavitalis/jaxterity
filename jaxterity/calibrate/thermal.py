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

import numpy as np

from ..data.schema import ThermalTelemetry


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


__all__ = ["fit_servo_thermal", "calibrate_twin_thermal"]
