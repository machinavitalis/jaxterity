# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Contact/impact identification — restitution through the saltation lane,
stiffness/damping from a squeeze, honesty gates throughout.

No MJX here: the bounce fit differentiates through jaxonomy's
``multi_event_time_gradient`` (impacts as hybrid events), the squeeze is a
closed-form contact law. Fast enough to run un-marked.
"""

from __future__ import annotations

import numpy as np
import pytest

from jaxterity.errors import CalibrationError
from jaxterity.objects import (
    identify_restitution,
    identify_squeeze,
    predict_bounce_times,
)
from jaxterity.objects.contact import _saltation_time_jacobian

_E_TRUE = 0.72
_H0_TRUE = 0.9


# --- restitution (toss/bounce) ----------------------------------------------


def test_saltation_jacobian_matches_finite_differences():
    """d(impact time)/d(e, h0) through the reset maps matches central FD."""
    times = predict_bounce_times(_E_TRUE, _H0_TRUE, 4)
    jac = _saltation_time_jacobian(np.array([_E_TRUE, _H0_TRUE]), times, 9.81)
    h = 1e-5
    for j, (de, dh) in enumerate([(h, 0.0), (0.0, h)]):
        tp = predict_bounce_times(_E_TRUE + de, _H0_TRUE + dh, 4)
        tm = predict_bounce_times(_E_TRUE - de, _H0_TRUE - dh, 4)
        fd = (tp - tm) / (2 * h)
        for k in range(4):
            if abs(fd[k]) < 1e-12:
                assert abs(jac[k, j]) < 1e-8  # first bounce is e-insensitive
            else:
                assert jac[k, j] == pytest.approx(fd[k], rel=5e-2)


def test_recovers_planted_restitution_and_height():
    """Clean impact times → exact recovery of (e, h0), zero warnings."""
    times = predict_bounce_times(_E_TRUE, _H0_TRUE, 5)
    result = identify_restitution(times)
    assert result.recipe_name == "objects.restitution"
    assert float(result.fitted["object.restitution"]) == pytest.approx(
        _E_TRUE, abs=1e-4
    )
    assert float(result.fitted["object.drop_height"]) == pytest.approx(
        _H0_TRUE, abs=1e-4
    )
    assert result.warnings == ()


def test_recovers_restitution_with_known_height():
    times = predict_bounce_times(_E_TRUE, _H0_TRUE, 3)
    result = identify_restitution(times, h0=_H0_TRUE)
    assert float(result.fitted["object.restitution"]) == pytest.approx(
        _E_TRUE, abs=1e-4
    )
    assert "object.drop_height" not in result.fitted


def test_single_impact_cannot_pin_two_parameters():
    """One impact + unknown height: refused with a fix, not silently fitted."""
    times = predict_bounce_times(_E_TRUE, _H0_TRUE, 1)
    with pytest.raises(CalibrationError, match="more bounces"):
        identify_restitution(times)


def test_noisy_impacts_surface_uncertainty():
    """Noise inflates the standard error, and the unresolved flag fires at
    the caller's threshold — clean data stays quiet at the same threshold."""
    times = predict_bounce_times(_E_TRUE, _H0_TRUE, 5)
    clean = identify_restitution(times, warn_rel_std=0.05)
    assert clean.warnings == ()  # exact data: tiny std, no flag

    rng = np.random.default_rng(11)
    noisy = np.sort(np.abs(times + rng.normal(0.0, 0.2, size=times.shape))) + 1e-3
    result = identify_restitution(noisy, warn_rel_std=0.05)
    assert result.covariance is not None
    se = result.std_error()
    assert se is not None and max(se.values()) > max(
        identify_restitution(times).std_error().values()
    )
    assert any("unresolved" in w for w in result.warnings)


def test_unsorted_impact_times_are_refused():
    with pytest.raises(CalibrationError, match="increasing"):
        identify_restitution([0.5, 0.4, 0.6])


def test_restitution_is_bounded():
    """Impact times implying e > 1 pin at the bound and are named."""
    # Flights that *grow* over time imply e > 1 (energy gain — unphysical).
    times = np.array([0.4, 1.4, 3.0, 5.2])
    result = identify_restitution(times, h0=0.9 * 9.81 * 0.4**2 / 2)
    assert 0.0 <= float(result.fitted["object.restitution"]) <= 1.0
    assert any("bound" in w for w in result.warnings)


# --- squeeze (stiffness + damping) -------------------------------------------


def _squeeze_trace(k: float, c: float, n: int = 200):
    """A loading/unloading squeeze: x follows a half-sine, F = k x + c x xd."""
    t = np.linspace(0.0, 1.0, n)
    x = 0.01 * np.sin(np.pi * t)  # 0 → 1 cm → 0
    xd = 0.01 * np.pi * np.cos(np.pi * t)
    f = k * x + c * x * xd
    return x, xd, f


def test_recovers_planted_stiffness_and_damping():
    x, xd, f = _squeeze_trace(k=800.0, c=300.0)
    result = identify_squeeze(x, xd, f)
    assert result.recipe_name == "objects.squeeze"
    assert float(result.fitted["object.stiffness"]) == pytest.approx(800.0, rel=1e-6)
    assert float(result.fitted["object.damping"]) == pytest.approx(300.0, rel=1e-6)
    assert result.warnings == ()


def test_quasi_static_squeeze_cannot_resolve_damping():
    """No rate variation → the damping column vanishes; the fit says so."""
    x = np.linspace(0.0, 0.01, 100)
    xd = np.zeros_like(x)  # infinitely slow squeeze
    f = 800.0 * x
    result = identify_squeeze(x, xd, f)
    assert result.warnings != ()
    assert any("identifiab" in w or "weakly" in w for w in result.warnings)


def test_negative_solution_is_projected_and_named():
    """A trace whose best linear fit wants c < 0 pins at 0 and warns."""
    x, xd, f = _squeeze_trace(k=800.0, c=300.0)
    f_anti = 800.0 * x - 300.0 * x * xd  # energy-creating hysteresis
    result = identify_squeeze(x, xd, f_anti)
    assert float(result.fitted["object.damping"]) == 0.0
    assert any("object.damping" in w and "bound" in w for w in result.warnings)


def test_length_mismatch_is_refused():
    with pytest.raises(CalibrationError, match="length"):
        identify_squeeze([0.0, 0.01], [0.0], [0.0, 1.0])
