# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Drift anomaly monitor.

Synthetic recalibration streams with known anomalies: a slow friction ramp
must trip the z-score on exactly the ramped channel; a unit leaving the
healthy population must trip the manifold check; an unexplained-residual
spike must trip the residual flag — and a healthy stream must stay quiet.
"""

from __future__ import annotations

import numpy as np
import pytest

from jaxterity.errors import JaxterityError
from jaxterity.selfcheck import AnomalyReport, HealthyManifold, ResidualMonitor

_RNG = np.random.default_rng(20260611)

#: A miniature drift schema: two scalar frictions + a 3-servo K_t vector.
_LABELS = {"offbody.K_t_scale": ("hip", "knee", "ankle")}


def _healthy_fit(rng) -> dict:
    """One healthy unit's fitted parameters (small manufacturing variance)."""
    return {
        "robot.knee.friction": 0.30 + 0.01 * rng.standard_normal(),
        "robot.hip.friction": 0.28 + 0.01 * rng.standard_normal(),
        "offbody.K_t_scale": 1.0 + 0.02 * rng.standard_normal(3),
    }


def test_healthy_stream_stays_quiet():
    monitor = ResidualMonitor(min_history=5, labels=_LABELS)
    for k in range(20):
        report = monitor.observe(float(k), _healthy_fit(_RNG))
        assert isinstance(report, AnomalyReport)
        assert report.ok, report.flags
    assert monitor.n_epochs == 20


def test_zscore_flags_exactly_the_ramped_channel():
    """A slow knee-friction ramp trips its z-score; other channels stay quiet."""
    monitor = ResidualMonitor(min_history=5, z_threshold=3.0, labels=_LABELS)
    rng = np.random.default_rng(7)
    for k in range(12):
        monitor.observe(float(k), _healthy_fit(rng))
    # The fault: knee friction jumps far outside its own history's spread.
    drifted = _healthy_fit(rng)
    drifted["robot.knee.friction"] = 0.45
    report = monitor.observe(12.0, drifted)
    assert not report.ok
    assert len(report.flags) == 1, report.flags
    assert "robot.knee.friction" in report.flags[0]
    assert report.z_scores["robot.knee.friction"] > 3.0
    assert abs(report.z_scores["robot.hip.friction"]) < 3.0


def test_zscore_needs_history_first():
    """No z verdicts until the unit has a history to be compared against."""
    monitor = ResidualMonitor(min_history=5)
    for k in range(5):
        report = monitor.observe(float(k), {"robot.knee.friction": 0.3 + 0.1 * k})
        assert report.ok and not report.z_scores


def test_manifold_flags_population_outlier():
    """A unit outside the healthy fleet's ellipsoid is flagged even when its
    own history drifted too slowly for a z-score (the boiled-frog case)."""
    rng = np.random.default_rng(11)
    manifold = HealthyManifold.from_samples(
        [_healthy_fit(rng) for _ in range(60)], labels=_LABELS
    )
    # A 99.9 % ellipsoid: check *coverage* over fresh healthy draws (any
    # single draw can legitimately be a tail point).
    fresh = [_healthy_fit(rng) for _ in range(20)]
    covered = sum(manifold.contains(s, _LABELS) for s in fresh)
    assert covered >= 18, covered

    sick = dict(fresh[0])
    sick["robot.knee.friction"] = 0.9  # far outside the fleet distribution
    assert not manifold.contains(sick, _LABELS)

    monitor = ResidualMonitor(manifold=manifold, labels=_LABELS)
    report = monitor.observe(0.0, sick)
    assert not report.ok
    assert any("healthy manifold" in flag for flag in report.flags)
    assert report.mahalanobis is not None and report.mahalanobis > manifold.threshold


def test_residual_threshold_flags_unexplained_windows():
    monitor = ResidualMonitor(residual_rms_threshold=5.0)
    quiet = monitor.observe(0.0, {"robot.knee.friction": 0.3}, residual_rms=1.2)
    assert quiet.ok
    loud = monitor.observe(1.0, {"robot.knee.friction": 0.3}, residual_rms=40.0)
    assert not loud.ok
    assert any("residual" in flag for flag in loud.flags)


def test_monitor_validation_surface():
    monitor = ResidualMonitor()
    monitor.observe(1.0, {"a": 0.1})
    with pytest.raises(JaxterityError, match="backwards"):
        monitor.observe(0.5, {"a": 0.1})
    with pytest.raises(JaxterityError, match="channels changed"):
        monitor.observe(2.0, {"b": 0.1})
    with pytest.raises(JaxterityError, match="≥ 3 healthy units"):
        HealthyManifold.from_samples([{"a": 0.1}, {"a": 0.2}])
    with pytest.raises(JaxterityError, match="missing channels"):
        manifold = HealthyManifold.from_samples(
            [{"a": 0.1 * k, "b": 0.2} for k in range(5)]
        )
        manifold.mahalanobis({"a": 0.1})


def test_std_floor_suppresses_anchored_channel_jitter():
    """A nearly-constant (prior-anchored) channel must not flag on jitter.

    Found live in the drift-dashboard harness: a weakly-excited parameter
    the re-fit holds essentially constant has a tiny history std, so a
    harmless 0.01 estimate wobble z-scores past any threshold. The physical
    std floor (a fraction of the schema bound range) keeps it quiet while
    a real fault still dwarfs the floor.
    """
    floors = {"robot.knee.friction": 0.02, "offbody.K_t_scale": 0.01}
    quiet = ResidualMonitor(min_history=5, std_floor=floors, labels=_LABELS)
    noisy = ResidualMonitor(min_history=5, labels=_LABELS)  # no floor
    rng = np.random.default_rng(5)
    for k in range(10):
        fit = {
            "robot.knee.friction": 0.300,  # anchored: bit-constant history
            "robot.hip.friction": 0.28 + 0.01 * rng.standard_normal(),
            "offbody.K_t_scale": np.ones(3),
        }
        quiet.observe(float(k), fit)
        noisy.observe(float(k), fit)
    wobble = {
        "robot.knee.friction": 0.301,  # 1e-3 jitter — far below the floor
        "robot.hip.friction": 0.28,
        "offbody.K_t_scale": np.ones(3),
    }
    assert quiet.observe(10.0, wobble).ok
    assert not noisy.observe(10.0, wobble).ok  # the failure mode, unfloored
    # A real fault still dwarfs the floor: +0.3 N·m ⇒ z = 15 ≫ 4.
    fault = dict(wobble)
    fault["robot.knee.friction"] = 0.6
    report = quiet.observe(11.0, fault)
    assert not report.ok and "knee" in report.flags[0]


def test_b3_storyline_ramp_is_caught_before_seizure():
    """The B3 arc end-to-end on synthetic commits: a gearbox friction ramp
    starting at epoch 10 is flagged by epoch ~14, well before the epoch-18
    seizure value — the monitor's whole reason to exist."""
    rng = np.random.default_rng(3)
    monitor = ResidualMonitor(min_history=5, z_threshold=3.0, labels=_LABELS)
    first_flag = None
    for epoch in range(18):
        fit = _healthy_fit(rng)
        if epoch >= 10:  # the failing gearbox: +0.04 N·m per epoch
            fit["robot.knee.friction"] += 0.04 * (epoch - 10)
        report = monitor.observe(float(epoch), fit)
        if not report.ok and first_flag is None:
            first_flag = epoch
    assert first_flag is not None and 11 <= first_flag <= 15, first_flag
