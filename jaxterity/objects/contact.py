# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Contact/impact identification: restitution from bounces, stiffness from a
squeeze.

Two probe modalities recover an object's *contact* parameters — the ones a
grasp cannot feel and a camera cannot see:

- **Toss/bounce** (:func:`identify_restitution`): the coefficient of
  restitution from observed impact times of a vertical bounce. The parameter
  Jacobian comes from Jaxonomy's saltation lane
  (:func:`jaxonomy.multi_event_time_gradient` — forward sensitivities
  propagated through each reset map), so the fit differentiates *through the
  impacts* rather than smoothing them away.
- **Squeeze** (:func:`identify_squeeze`): contact stiffness and
  Hunt–Crossley-style damping from a force–deformation trace,
  ``F = k·x + c·x·ẋ`` (force vanishes at zero deformation, loading/unloading
  hysteresis carries the damping information). Linear in ``(k, c)``.

Both fits carry the lane's honesty guards (T-219/C-119): an identifiability
report on the probe's sensitivity matrix, physical bounds (``0 ≤ e ≤ 1``,
``k, c ≥ 0``) with pinned fits named, and per-parameter uncertainty with
unresolved parameters flagged. A quasi-static squeeze (no rate variation)
cannot resolve damping and says so; a single bounce cannot separate
restitution from the drop height and says so.

Honest scope: this is the **smooth/event-based path**. Gradients through
MJX's contact complementarity solver remain out of scope (KNOWN_GAPS —
contact-capable models are rejected by the implicit reverse-mode wrap); the
squeeze model is a 1-D lumped contact law, not a full FEM/mesh model.

Frame
-----
Bounce states are 1-D vertical (height above the surface, upward positive);
squeeze deformation is the 1-D compression depth, positive into the object.

Units
-----
SI: time s, height m, deformation m, force N, stiffness N/m,
damping N·s/m², restitution dimensionless.
"""

from __future__ import annotations

import hashlib

import jax.numpy as jnp
import numpy as np

from ..errors import CalibrationError
from ..sysid import SysidResult
from ..sysid.recipes import _DEFAULT_COND_MAX, identifiability_from_jacobian

#: Gravity default (m/s²) — overridable for non-Earth or tilted-rig probes.
_G_DEFAULT = 9.81


def predict_bounce_times(
    e: float, h0: float, n: int, *, g: float = _G_DEFAULT
) -> np.ndarray:
    """Analytic impact times of a vertical drop-and-bounce.

    Drop from rest at height ``h0``; each impact scales the rebound speed by
    the restitution ``e``. The first impact is at ``sqrt(2 h0 / g)`` and each
    subsequent flight lasts ``2 e^k v_1 / g``.

    Args
    ----
    e : float
        Coefficient of restitution (0..1).
    h0 : float
        Drop height, m.
    n : int
        Number of impacts to predict.
    g : float
        Gravity, m/s².

    Returns
    -------
    numpy.ndarray, shape ``(n,)``
        Impact times, s, strictly increasing.

    Units
    -----
    s, m, m/s².
    """
    t1 = float(np.sqrt(2.0 * h0 / g))
    out = [t1]
    v = float(np.sqrt(2.0 * g * h0)) * e
    for _ in range(n - 1):
        out.append(out[-1] + 2.0 * v / g)
        v *= e
    return np.asarray(out)


def _saltation_time_jacobian(theta: np.ndarray, times: np.ndarray, g: float):
    """d(impact times)/d(e, h0) via the saltation lane (multi-event forward
    sensitivities through the restitution reset map)."""
    from jaxonomy import multi_event_time_gradient

    def guard(_t, x, _p):
        return x[0]

    def rhs(_t, x, _p):
        return jnp.stack([x[1], jnp.asarray(-g)])

    def reset(_t_e, x, p):
        return jnp.stack([x[0], -p["e"] * x[1]])

    def x0(p):
        return jnp.stack([p["h0"], jnp.asarray(0.0)])

    params = {"e": jnp.asarray(float(theta[0])), "h0": jnp.asarray(float(theta[1]))}
    grads = multi_event_time_gradient(guard, rhs, reset, x0, jnp.asarray(times), params)
    return np.column_stack(
        [np.asarray(grads["e"], dtype=float), np.asarray(grads["h0"], dtype=float)]
    )


def identify_restitution(
    impact_times,
    *,
    h0: float | None = None,
    g: float = _G_DEFAULT,
    e0: float = 0.5,
    iterations: int = 12,
    lam: float = 1e-9,
    cond_max: float = _DEFAULT_COND_MAX,
    warn_rel_std: float = 1.0,
) -> SysidResult:
    """Identify the coefficient of restitution (and, if unknown, the drop
    height) from observed impact times of a vertical toss/bounce.

    A Gauss–Newton fit of the analytic bounce-time model to the observed
    impact times, with the parameter Jacobian computed by
    :func:`jaxonomy.multi_event_time_gradient` — the saltation lane that
    propagates forward sensitivities through every restitution reset, i.e.
    the gradient is taken *through the impacts*. Estimates are projected onto
    ``0 ≤ e ≤ 1`` and ``h0 > 0`` each iterate; a pinned fit is named.

    Args
    ----
    impact_times : array-like, shape ``(n,)``
        Observed impact instants, s, strictly increasing (e.g. from audio or
        the camera's contact frames).
    h0 : float or None
        Known drop height, m. ``None`` (default) fits it jointly with ``e`` —
        which needs at least two impacts (one impact alone cannot separate
        them, and the fit says so).
    g : float
        Gravity, m/s².
    e0 : float
        Initial restitution estimate.
    iterations : int
        Gauss–Newton iterations.
    lam : float
        Levenberg damping.
    cond_max : float
        Conditioning ceiling for the identifiability warning.
    warn_rel_std : float
        Relative standard-error threshold for the unresolved warning.

    Returns
    -------
    SysidResult
        ``fitted["object.restitution"]`` (and ``fitted["object.drop_height"]``
        when fitted), per-parameter covariance, residual RMS, and the honesty
        warnings.

    Raises
    ------
    CalibrationError
        If fewer impact times than parameters are supplied, or the times are
        not strictly increasing/positive.

    Units
    -----
    s, m; restitution dimensionless.
    """
    times = np.asarray(impact_times, dtype=float).ravel()
    fit_h0 = h0 is None
    n_params = 2 if fit_h0 else 1
    names = (
        ("object.restitution", "object.drop_height")
        if fit_h0
        else ("object.restitution",)
    )
    if times.size < n_params:
        raise CalibrationError(
            f"identify_restitution: {times.size} impact time(s) cannot pin "
            f"{n_params} parameter(s) ({', '.join(names)}). Record more bounces"
            + (" or pass a known drop height h0." if fit_h0 else ".")
        )
    if times.size and (np.any(times <= 0.0) or np.any(np.diff(times) <= 0.0)):
        raise CalibrationError(
            "identify_restitution: impact times must be positive and strictly "
            f"increasing; got {times!r}. Check the event extraction."
        )

    # Initial h0 from the first impact when unknown (exact if t1 is clean).
    h = float(h0) if h0 is not None else 0.5 * g * float(times[0]) ** 2
    theta = np.array([float(e0), h], dtype=float)
    lower = np.array([0.0, 1e-6])
    upper = np.array([1.0, np.inf])

    hit_lower = np.zeros(2, dtype=bool)
    hit_upper = np.zeros(2, dtype=bool)
    jac_used: np.ndarray | None = None
    residual = np.zeros_like(times)
    for _ in range(iterations):
        predicted = predict_bounce_times(theta[0], theta[1], times.size, g=g)
        residual = predicted - times
        jac_full = _saltation_time_jacobian(theta, predicted, g)
        jac = jac_full if fit_h0 else jac_full[:, :1]
        if jac_used is None:
            jac_used = jac
        h_mat = jac.T @ jac
        damped = h_mat + lam * np.diag(np.diag(h_mat) + 1e-12)
        step = -np.linalg.solve(damped, jac.T @ residual)
        raw = theta[:n_params] + step
        hit_lower[:n_params] = raw < lower[:n_params]
        hit_upper[:n_params] = raw > upper[:n_params]
        theta[:n_params] = np.clip(raw, lower[:n_params], upper[:n_params])

    assert jac_used is not None  # iterations >= 1
    notes: list[str] = []
    report = identifiability_from_jacobian(jac_used, names, cond_max=cond_max)
    if not report.identifiable:
        notes.append(
            f"identify_restitution: the bounce record is not identifiable — "
            f"rank {report.rank} of {len(names)} parameters, condition number "
            f"{report.condition_number:.3g} > {cond_max:.3g}. Record more "
            "bounces or pass a known drop height."
        )
    elif report.weak_params:
        notes.append(
            f"identify_restitution: weakly identified: "
            f"{', '.join(report.weak_params)} (condition number "
            f"{report.condition_number:.3g}). Treat them as loosely constrained."
        )
    for i, name in enumerate(names):
        if hit_lower[i] or hit_upper[i]:
            bound = "0" if hit_lower[i] else ("1" if i == 0 else "its bound")
            notes.append(
                f"identify_restitution: {name} fit hit bound {bound} (the "
                "unconstrained step wanted an unphysical value); the record "
                "likely cannot identify it."
            )

    # Gauss–Newton covariance at the optimum: sigma² (JᵀJ)⁻¹.
    predicted = predict_bounce_times(theta[0], theta[1], times.size, g=g)
    residual = predicted - times
    dof = max(times.size - n_params, 1)
    sigma2 = float(residual @ residual) / dof
    jac_full = _saltation_time_jacobian(theta, predicted, g)
    jac = jac_full if fit_h0 else jac_full[:, :1]
    try:
        cov_mat = sigma2 * np.linalg.pinv(jac.T @ jac)
    except np.linalg.LinAlgError:  # pragma: no cover - pinv is robust
        cov_mat = np.full((n_params, n_params), np.inf)
    covariance = {name: float(cov_mat[i, i]) for i, name in enumerate(names)}
    for name, value in zip(names, theta[:n_params]):
        std = float(np.sqrt(max(covariance[name], 0.0)))
        if std > warn_rel_std * max(abs(float(value)), float(np.finfo(float).tiny)):
            notes.append(
                f"identify_restitution: {name} is unresolved — standard error "
                f"{std:.3g} exceeds {warn_rel_std:.0%} of the fitted value "
                f"{abs(float(value)):.3g}. Record more bounces or cleaner "
                "impact times."
            )

    fitted = {"object.restitution": jnp.asarray(float(theta[0]))}
    if fit_h0:
        fitted["object.drop_height"] = jnp.asarray(float(theta[1]))
    rms = float(np.sqrt(np.mean(residual**2)))
    digest = hashlib.blake2b(times.tobytes(), digest_size=16)
    return SysidResult(
        fitted=fitted,
        covariance=covariance,
        residuals={"impact_time_rms": rms},
        telemetry_hash=digest.hexdigest(),
        recipe_name="objects.restitution",
        recipe_version="1.0",
        warnings=tuple(notes),
    )


def identify_squeeze(
    deformation,
    rate,
    force,
    *,
    cond_max: float = _DEFAULT_COND_MAX,
    warn_rel_std: float = 1.0,
) -> SysidResult:
    """Identify contact stiffness and damping from a squeeze's
    force–deformation trace.

    Fits the Hunt–Crossley-style lumped contact law ``F = k·x + c·x·ẋ``
    (force vanishes at zero deformation; the damping term scales with
    deformation, so loading/unloading hysteresis carries ``c``). Linear in
    ``(k, c)`` — solved by least squares with an explicit conditioning
    report: a quasi-static squeeze (ẋ ≈ 0 throughout) leaves the ``c`` column
    ~zero and is flagged, not silently fitted. Negative solutions are
    projected to 0 and named (an unphysical best fit is an identifiability
    failure, not a fit).

    Args
    ----
    deformation : array-like, shape ``(n,)``
        Compression depth ``x``, m, ≥ 0 (contact engaged).
    rate : array-like, shape ``(n,)``
        Compression rate ``ẋ``, m/s (positive loading, negative unloading).
    force : array-like, shape ``(n,)``
        Measured contact force, N.
    cond_max : float
        Conditioning ceiling for the identifiability warning.
    warn_rel_std : float
        Relative standard-error threshold for the unresolved warning.

    Returns
    -------
    SysidResult
        ``fitted["object.stiffness"]`` (N/m) and ``fitted["object.damping"]``
        (N·s/m²), per-parameter covariance, residual RMS, and the honesty
        warnings.

    Raises
    ------
    CalibrationError
        If the traces disagree in length or hold fewer than 3 samples.

    Units
    -----
    m, m/s, N; stiffness N/m, damping N·s/m².
    """
    x = np.asarray(deformation, dtype=float).ravel()
    xd = np.asarray(rate, dtype=float).ravel()
    f = np.asarray(force, dtype=float).ravel()
    if not (x.size == xd.size == f.size):
        raise CalibrationError(
            "identify_squeeze: deformation, rate, and force must be the same "
            f"length; got {x.size}/{xd.size}/{f.size}."
        )
    if x.size < 3:
        raise CalibrationError(
            f"identify_squeeze: {x.size} sample(s) cannot support a 2-parameter "
            "fit with a residual estimate. Capture a longer squeeze."
        )

    names = ("object.stiffness", "object.damping")
    design = np.column_stack([x, x * xd])
    theta, *_ = np.linalg.lstsq(design, f, rcond=None)

    notes: list[str] = []
    report = identifiability_from_jacobian(design, names, cond_max=cond_max)
    if not report.identifiable:
        notes.append(
            f"identify_squeeze: the squeeze is not identifiable — rank "
            f"{report.rank} of 2 parameters, condition number "
            f"{report.condition_number:.3g} > {cond_max:.3g}. A quasi-static "
            "squeeze carries no damping information; squeeze faster, and "
            "capture both loading and unloading."
        )
    elif report.weak_params:
        notes.append(
            f"identify_squeeze: weakly identified: "
            f"{', '.join(report.weak_params)} (condition number "
            f"{report.condition_number:.3g}). Treat them as loosely constrained."
        )

    # Physical non-negativity: project and refit the free parameter.
    pinned = [name for name, value in zip(names, theta) if value < 0.0]
    if pinned:
        theta = np.maximum(theta, 0.0)
        free = [i for i, name in enumerate(names) if name not in pinned]
        if free:
            sub = design[:, free]
            theta_free, *_ = np.linalg.lstsq(sub, f, rcond=None)
            theta[free] = np.maximum(theta_free, 0.0)
        for name in pinned:
            notes.append(
                f"identify_squeeze: {name} fit hit lower bound 0 (the "
                "unconstrained solution was negative, unphysical); the squeeze "
                "likely cannot identify it."
            )

    residual = design @ theta - f
    dof = max(f.size - 2, 1)
    sigma2 = float(residual @ residual) / dof
    cov_mat = sigma2 * np.linalg.pinv(design.T @ design)
    covariance = {name: float(cov_mat[i, i]) for i, name in enumerate(names)}
    for name, value in zip(names, theta):
        std = float(np.sqrt(max(covariance[name], 0.0)))
        if std > warn_rel_std * max(abs(float(value)), float(np.finfo(float).tiny)):
            notes.append(
                f"identify_squeeze: {name} is unresolved — standard error "
                f"{std:.3g} exceeds {warn_rel_std:.0%} of the fitted value "
                f"{abs(float(value)):.3g}. Squeeze with more rate variation or "
                "lower-noise force sensing."
            )

    digest = hashlib.blake2b(np.concatenate([x, xd, f]).tobytes(), digest_size=16)
    return SysidResult(
        fitted={
            "object.stiffness": jnp.asarray(float(theta[0])),
            "object.damping": jnp.asarray(float(theta[1])),
        },
        covariance=covariance,
        residuals={"force_rms": float(np.sqrt(np.mean(residual**2)))},
        telemetry_hash=digest.hexdigest(),
        recipe_name="objects.squeeze",
        recipe_version="1.0",
        warnings=tuple(notes),
    )


__all__ = [
    "predict_bounce_times",
    "identify_restitution",
    "identify_squeeze",
]
