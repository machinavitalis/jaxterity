# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Local fit-uncertainty estimators for system identification.

A :class:`~jaxterity.sysid.SysidResult` reports *which* parameters were
identified; this module reports *how well*. Two local estimators turn a
converged least-squares fit into a per-parameter covariance:

- :func:`laplace_covariance` — the Gauss-Newton / Laplace approximation at the
  optimum: ``Σ ≈ σ̂²·(JᵀJ)⁻¹`` with ``σ̂² = RSS / (n_obs − n_params)``. One
  forward-mode Jacobian — the fit's own, when the caller passes it — plus a
  solve; deterministic; assumes the residual is locally well-approximated by a
  Gaussian around the optimum.
- :func:`bootstrap_covariance` — a residual bootstrap: resample the fit
  residuals, refit, and take the empirical covariance over the refits. Heavier
  (``n_boot`` refits) but makes no Gaussian-residual assumption; deterministic
  under a seed.

Both are **local** approximations, not a full Bayesian posterior (the honest
scope here). They populate ``SysidResult.covariance`` and, carried through
the twin, set the residual thresholds and the capability envelope
on a principled rather than hand-tuned footing.

Each works off a plain ``residual_fn(theta) -> residual_vector`` closure, so the
same estimators serve the forward-mode active fit, the robot recipes, and the
linear thermal fit.
"""

from __future__ import annotations

from typing import Callable, Sequence

import jax
import numpy as np


def laplace_covariance(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    theta_hat,
    *,
    n_obs: int | None = None,
    jac=None,
) -> np.ndarray:
    """Gauss-Newton / Laplace parameter covariance at a least-squares optimum.

    Forms the residual Jacobian ``J = ∂r/∂θ`` at ``theta_hat`` by forward-mode
    autodiff (so it traverses MJX's constraint solver, ADR-017), estimates the
    residual variance ``σ̂² = RSS / (n_obs − n_params)``, and returns
    ``Σ = σ̂²·(JᵀJ)⁻¹``. A pseudo-inverse is used so a rank-deficient Gram
    (a non-identifiable excitation) yields a finite — if large — covariance
    rather than raising.

    Args
    ----
    residual_fn : Callable[[array], array]
        Maps a parameter vector to the fit residual ``model(θ) − data``. Must be
        ``jax.jacfwd``-differentiable.
    theta_hat : array-like, shape ``(n_params,)``
        The converged parameter estimate.
    n_obs : int or None
        Number of independent observations (residual length). Defaults to the
        residual vector's length. Used for the ``n_obs − n_params`` correction.
    jac : array-like, shape ``(n_obs, n_params)``, or None
        The residual Jacobian at ``theta_hat``, when the caller already holds
        it. A Gauss-Newton fit does: the data is constant, so ``∂r/∂θ`` *is* the
        model Jacobian the fit converged on, and passing it skips a second trace
        and compile of the same graph — which for a stiff solver is the dominant
        cost of the whole estimate. Default ``None``: differentiate
        ``residual_fn`` here.

    Returns
    -------
    numpy.ndarray, shape ``(n_params, n_params)``
        The local parameter covariance ``Σ`` (float64).

    Units
    -----
    Each diagonal entry is the variance of the corresponding parameter, in that
    parameter's units squared.
    """
    theta = np.asarray(theta_hat, dtype=np.float64)
    r = np.asarray(residual_fn(theta), dtype=np.float64).ravel()
    if jac is None:
        jac = jax.jacfwd(residual_fn)(theta)
    jac = np.asarray(jac, dtype=np.float64).reshape(r.size, theta.size)
    n = int(n_obs if n_obs is not None else r.size)
    p = int(theta.size)
    dof = max(n - p, 1)
    sigma2 = float(r @ r) / dof
    gram = jac.T @ jac
    return sigma2 * np.linalg.pinv(gram)


def bootstrap_covariance(
    predict_hat,
    residual_hat,
    refit: Callable[[np.ndarray], np.ndarray],
    *,
    n_boot: int = 64,
    seed: int = 0,
) -> np.ndarray:
    """Residual-bootstrap parameter covariance over ``n_boot`` refits.

    Draws bootstrap datasets ``y* = ŷ + e*`` where ``ŷ`` is the model prediction
    at the optimum and ``e*`` is sampled with replacement from the (centered)
    fit residuals, refits each, and returns the empirical covariance of the
    refit parameters. Makes no Gaussian-residual assumption; deterministic for a
    fixed ``seed``.

    Args
    ----
    predict_hat : array-like, shape ``(n_obs,)``
        The model prediction at the optimum, ``model(θ̂)``.
    residual_hat : array-like, shape ``(n_obs,)``
        The fit residuals ``data − model(θ̂)``.
    refit : Callable[[array], array]
        Refits a bootstrap dataset, mapping a target vector ``y*`` to a parameter
        estimate ``θ*``.
    n_boot : int
        Number of bootstrap refits. Default 64.
    seed : int
        PRNG seed for the resampling (determinism).

    Returns
    -------
    numpy.ndarray, shape ``(n_params, n_params)``
        The empirical parameter covariance over the refits (float64, ``ddof=1``).
    """
    yhat = np.asarray(predict_hat, dtype=np.float64).ravel()
    resid = np.asarray(residual_hat, dtype=np.float64).ravel()
    resid = resid - resid.mean()  # center so e* has zero mean
    rng = np.random.default_rng(int(seed))
    thetas: list[np.ndarray] = []
    for _ in range(int(n_boot)):
        e_star = rng.choice(resid, size=resid.size, replace=True)
        theta_star = np.asarray(refit(yhat + e_star), dtype=np.float64).ravel()
        thetas.append(theta_star)
    stacked = np.stack(thetas, axis=0)
    p = stacked.shape[1]
    return np.cov(stacked, rowvar=False, ddof=1).reshape(p, p)


def per_parameter_variance(
    covariance: np.ndarray, names: Sequence[str]
) -> dict[str, float]:
    """Marginal per-parameter variance dict from a covariance matrix.

    Args
    ----
    covariance : numpy.ndarray, shape ``(n_params, n_params)``
        A parameter covariance (e.g. from :func:`laplace_covariance`).
    names : sequence of str
        Parameter names in the covariance's row/column order.

    Returns
    -------
    dict[str, float]
        ``{name: variance}`` — the diagonal of ``covariance``, clipped at 0 to
        guard tiny negative round-off.
    """
    diag = np.diag(np.atleast_2d(np.asarray(covariance, dtype=np.float64)))
    return {name: float(max(float(v), 0.0)) for name, v in zip(names, diag)}


__all__ = [
    "laplace_covariance",
    "bootstrap_covariance",
    "per_parameter_variance",
]
