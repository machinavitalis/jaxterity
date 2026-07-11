# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""ParityReport — RMS/peak per quantity, suggested perturbations."""

from __future__ import annotations

import numpy as np
import pytest

from jaxterity.diagnostics import ParityReport, parity_report


def test_identical_trajectories_have_zero_residual():
    rng = np.random.default_rng(0)
    sim = {"q": rng.standard_normal((20, 3)), "qd": rng.standard_normal((20, 3))}
    real = {k: v.copy() for k, v in sim.items()}
    rep = parity_report(sim, real)
    assert rep.quantities == ("q", "qd")
    assert rep.n_samples == 20
    assert rep.rms["q"] == 0.0
    assert rep.peak["q"] == 0.0
    assert rep.ok(rms_threshold=1e-9)


def test_perturbed_trajectory_reports_nontrivial_rms():
    rng = np.random.default_rng(1)
    sim = {"q": np.zeros((50, 2))}
    real = {"q": 0.1 * rng.standard_normal((50, 2))}
    rep = parity_report(sim, real)
    assert rep.rms["q"] > 0.05
    assert rep.peak["q"] > 0.05
    assert not rep.ok(rms_threshold=1e-3)


def test_key_mismatch_raises():
    with pytest.raises(ValueError, match="share keys"):
        parity_report({"q": np.zeros(5)}, {"qd": np.zeros(5)})


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        parity_report({"q": np.zeros((5, 2))}, {"q": np.zeros((5, 3))})


def test_suggested_perturbations_from_sensitivity():
    """A scalar parameter with non-zero sensitivity gets a perturbation suggestion."""
    sim = {"q": np.zeros((10, 1))}
    real = {"q": np.ones((10, 1)) * 0.5}  # constant offset of 0.5
    # Sensitivity: dq/dp = 1 everywhere — to close the gap we need to push p by 0.5.
    sens = {"p": {"q": np.ones((10, 1))}}
    rep = parity_report(sim, real, sensitivities=sens)
    assert "p" in rep.suggested_perturbations
    assert rep.suggested_perturbations["p"] == pytest.approx(0.5, abs=1e-9)


def test_per_quantity_rms_threshold():
    rep = ParityReport(
        quantities=("q", "qd"),
        rms={"q": 0.001, "qd": 0.05},
        peak={"q": 0.002, "qd": 0.1},
        n_samples=10,
    )
    assert rep.ok({"q": 0.01, "qd": 0.1})
    assert not rep.ok({"q": 0.01, "qd": 0.01})


def test_summary_is_multi_line_and_includes_each_quantity():
    rep = ParityReport(
        quantities=("q", "qd"),
        rms={"q": 1.2e-3, "qd": 3.4e-4},
        peak={"q": 2.1e-3, "qd": 5.6e-4},
        n_samples=42,
    )
    text = rep.summary()
    assert "ParityReport over 42 samples" in text
    assert "q " in text and "qd " in text
