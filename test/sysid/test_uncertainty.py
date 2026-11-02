# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Local fit-uncertainty estimators.

Validated on a cheap analytic *linear* least-squares model, for which the
Laplace covariance ``σ̂²(JᵀJ)⁻¹`` is exact and the OLS sampling covariance is
known in closed form — so the estimators, the interval/coverage contract, the
monotone widening with noise, and Laplace≈bootstrap agreement can all be checked
deterministically without invoking MJX. The MJX-backed integration lives in
``test/calibrate/test_fit_uncertainty.py``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaxterity.sysid import SysidResult
from jaxterity.sysid.uncertainty import (
    bootstrap_covariance,
    laplace_covariance,
    per_parameter_variance,
)

# A fixed, well-conditioned design matrix (12 obs, 3 params) and a planted truth.
_A = np.array(
    [
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 2.0, 0.0],
        [1.0, 0.0, 2.0],
        [1.0, 2.0, 1.0],
        [1.0, 1.0, 2.0],
        [1.0, 2.0, 2.0],
        [1.0, 3.0, 1.0],
        [1.0, 1.0, 3.0],
        [1.0, 3.0, 3.0],
    ]
)
_THETA_TRUE = np.array([0.5, -1.2, 2.0])
_NAMES = ("intercept", "slope_x", "slope_y")


def _ols(y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(_A, y, rcond=None)[0]


def _make_data(sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return _A @ _THETA_TRUE + rng.normal(0.0, sigma, size=_A.shape[0])


def test_laplace_matches_closed_form_ols_covariance():
    """On a linear model the Laplace covariance is the exact OLS covariance."""
    sigma = 0.1
    y = _make_data(sigma, seed=0)
    theta_hat = _ols(y)
    y_j = jnp.asarray(y)
    a_j = jnp.asarray(_A)

    def residual_fn(theta):
        return a_j @ theta - y_j

    cov = laplace_covariance(residual_fn, theta_hat, n_obs=_A.shape[0])
    # Closed form: σ̂²·(AᵀA)⁻¹ with σ̂² = RSS/(n-p).
    resid = _A @ theta_hat - y
    sigma2 = float(resid @ resid) / (_A.shape[0] - _A.shape[1])
    expected = sigma2 * np.linalg.inv(_A.T @ _A)
    np.testing.assert_allclose(cov, expected, rtol=1e-6, atol=1e-9)


def test_a_supplied_jacobian_matches_differentiating_the_residual():
    """``jac=`` must be the matrix ``jacfwd`` would have built, not an estimate.

    A caller passes the Jacobian its own Gauss-Newton fit already compiled. The
    data is constant, so ``∂r/∂θ`` *is* that model Jacobian — and if the two
    paths ever diverged, every interval from a fit that reuses its Jacobian
    would be quietly wrong.
    """
    y = _make_data(0.1, seed=0)
    theta_hat = _ols(y)
    a_j, y_j = jnp.asarray(_A), jnp.asarray(y)

    def residual_fn(theta):
        return a_j @ theta - y_j

    differentiated = laplace_covariance(residual_fn, theta_hat, n_obs=_A.shape[0])
    supplied = laplace_covariance(residual_fn, theta_hat, n_obs=_A.shape[0], jac=_A)
    np.testing.assert_allclose(supplied, differentiated, rtol=1e-12, atol=1e-15)


def test_a_wrongly_shaped_jacobian_is_rejected():
    """A mismatched ``jac=`` raises rather than yielding a garbage covariance."""
    y = _make_data(0.1, seed=0)
    theta_hat = _ols(y)
    a_j, y_j = jnp.asarray(_A), jnp.asarray(y)
    with pytest.raises(ValueError):
        laplace_covariance(
            lambda th: a_j @ th - y_j,
            theta_hat,
            n_obs=_A.shape[0],
            jac=_A[:, :2],
        )


def test_per_parameter_variance_is_the_diagonal():
    cov = np.array([[4.0, 0.5], [0.5, 9.0]])
    var = per_parameter_variance(cov, ("a", "b"))
    assert var == {"a": 4.0, "b": 9.0}


def test_intervals_widen_monotonically_with_noise():
    """Higher telemetry noise ⟹ wider reported standard errors, every param."""
    prev = None
    for sigma in (0.02, 0.1, 0.5):
        y = _make_data(sigma, seed=1)
        theta_hat = _ols(y)
        a_j, y_j = jnp.asarray(_A), jnp.asarray(y)
        cov = laplace_covariance(
            lambda th: a_j @ th - y_j, theta_hat, n_obs=_A.shape[0]
        )
        se = np.sqrt(np.diag(cov))
        if prev is not None:
            assert np.all(se > prev), f"std error did not grow at sigma={sigma}"
        prev = se


def test_wald_interval_coverage_is_near_nominal():
    """Across many noise realizations the 95% interval covers truth ≈95% of the time."""
    n_trials = 400
    coverage_hits = np.zeros(3, dtype=int)
    for trial in range(n_trials):
        y = _make_data(sigma=0.2, seed=1000 + trial)
        theta_hat = _ols(y)
        a_j, y_j = jnp.asarray(_A), jnp.asarray(y)
        cov = laplace_covariance(
            lambda th: a_j @ th - y_j, theta_hat, n_obs=_A.shape[0]
        )
        var = per_parameter_variance(cov, _NAMES)
        result = SysidResult(
            fitted={n: jnp.asarray(float(v)) for n, v in zip(_NAMES, theta_hat)},
            covariance=var,
        )
        intervals = result.interval(0.95)
        for i, name in enumerate(_NAMES):
            lo, hi = intervals[name]
            if lo <= _THETA_TRUE[i] <= hi:
                coverage_hits[i] += 1
    empirical = coverage_hits / n_trials
    # Wald coverage on an exact-Gaussian linear model sits at ~0.95; allow a
    # sampling band around nominal for 400 trials.
    assert np.all(empirical > 0.90), f"under-coverage: {empirical}"
    assert np.all(empirical < 0.99), f"over-coverage: {empirical}"


def test_bootstrap_agrees_with_laplace_on_a_linear_model():
    """Residual bootstrap recovers the Laplace covariance scale on a linear fit."""
    sigma = 0.15
    y = _make_data(sigma, seed=7)
    theta_hat = _ols(y)
    a_j, y_j = jnp.asarray(_A), jnp.asarray(y)
    laplace = laplace_covariance(
        lambda th: a_j @ th - y_j, theta_hat, n_obs=_A.shape[0]
    )
    predict_hat = _A @ theta_hat
    residual_hat = y - predict_hat
    boot = bootstrap_covariance(predict_hat, residual_hat, _ols, n_boot=400, seed=3)
    # Standard errors agree to ~25% (bootstrap sampling noise at n_boot=400).
    se_l = np.sqrt(np.diag(laplace))
    se_b = np.sqrt(np.diag(boot))
    np.testing.assert_allclose(se_b, se_l, rtol=0.25)


def test_interval_rejects_bad_coverage():
    result = SysidResult(fitted={"a": jnp.asarray(1.0)}, covariance={"a": 0.01})
    for bad in (0.0, 1.0, -0.1, 1.5):
        try:
            result.interval(bad)
        except ValueError:
            continue
        raise AssertionError(f"interval({bad}) should have raised")


def test_std_error_and_interval_none_without_covariance():
    result = SysidResult(fitted={"a": jnp.asarray(1.0)})
    assert result.std_error() is None
    assert result.interval() is None
