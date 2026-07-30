# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Designing the excitation's shape, subject to the tip-over floor.

The finding that shaped this module is that shape is **not** free in margin
terms. True-peak normalization preserves how far the joint travels, which makes
it tempting to conclude a reshaped spectrum inherits the flat one's tip-over
vetting. ZMP is a function of acceleration, and acceleration scales as f² per
tone, so moving energy upward in frequency raises the inertial load steeply at
constant travel. The unconstrained design measured here walks straight through
the floor.
"""

from __future__ import annotations

import numpy as np
import pytest

from jaxterity import zoo
from jaxterity.calibrate.active import dof_joint_names
from jaxterity.checkup.probe import (
    design_spectrum,
    segment_identifiability,
    worst_absolute_variance,
)
from jaxterity.checkup.stability import (
    LEKIWI_BASE,
    MULTISINE_HZ,
    MULTISINE_KIND_V1,
    multisine_trajectory,
    schroeder_multisine,
    zmp_margins,
)
from jaxterity.errors import CalibrationError

_HOME = {
    "shoulder_pan": 1.916,
    "shoulder_lift": -0.8311,
    "elbow_flex": -1.5417,
    "wrist_flex": 0.1449,
    "wrist_roll": -0.5704,
}
_AMP_RAD = 0.3
_SEGMENT = {"duration_s": 16.0, "rate_hz": 80.0}


@pytest.fixture(scope="module")
def robot():
    return zoo.load("so100")


@pytest.fixture(scope="module")
def margin_of(robot):
    """Weights → worst ZMP margin of the shoulder-lift excitation, m."""
    dof = dof_joint_names(robot)
    pose = np.array([_HOME.get(n, 0.0) for n in dof])
    index = dof.index("shoulder_lift")

    def evaluate(weights):
        q_t, dt = multisine_trajectory(
            pose,
            index,
            float(pose[index]),
            _AMP_RAD,
            duration_s=_SEGMENT["duration_s"],
            weights=weights,
        )
        return float(zmp_margins(robot, q_t, dt, dof, LEKIWI_BASE).min())

    return evaluate


# --- the waveform carries weights -----------------------------------------


def test_a_weighted_waveform_is_still_peak_normalized():
    t = np.linspace(0.0, 16.0, 4001)
    for weights in (None, (0.2, 0.5, 1.0, 1.0, 0.3), (0.1, 0.1, 0.1, 0.1, 1.0)):
        wave = schroeder_multisine(t, 16.0, weights=weights)
        assert float(np.max(np.abs(wave))) == pytest.approx(1.0, abs=1e-4)


def test_weights_must_match_the_tone_count():
    t = np.linspace(0.0, 16.0, 101)
    with pytest.raises(CalibrationError, match="one weight per tone"):
        schroeder_multisine(t, 16.0, weights=(1.0, 1.0))


def test_the_legacy_waveform_refuses_weights():
    # Its sqrt(2n) normalizer assumes equal tones, so a weighted v1 waveform
    # would command a swing nothing vetted.
    t = np.linspace(0.0, 16.0, 101)
    with pytest.raises(CalibrationError, match="not defined for"):
        schroeder_multisine(t, 16.0, kind=MULTISINE_KIND_V1, weights=(1.0,) * 5)


def test_a_flat_weight_vector_reproduces_the_unweighted_waveform():
    t = np.linspace(0.0, 16.0, 501)
    assert schroeder_multisine(t, 16.0, weights=(1.0,) * len(MULTISINE_HZ)) == (
        pytest.approx(schroeder_multisine(t, 16.0))
    )


# --- the objective ---------------------------------------------------------


def test_worst_absolute_variance_penalizes_exciting_less():
    """Why the objective is absolute variance and not conditioning.

    A slow, low-frequency tone set separates the two friction columns nicely and
    pins neither. Scale-free measures reward it; the absolute variance does not.
    """
    low = (0.1, 0.2, 0.4, 0.6, 0.8)
    fast = segment_identifiability("j", _AMP_RAD, tones=MULTISINE_HZ, **_SEGMENT)
    slow = segment_identifiability("j", _AMP_RAD, tones=low, **_SEGMENT)
    # The slow set looks comparable on conditioning...
    assert slow.condition_number < fast.condition_number * 1.5
    # ...and is far worse at actually pinning the parameters.
    assert worst_absolute_variance(slow) > 3.0 * worst_absolute_variance(fast)


def test_worst_absolute_variance_is_infinite_when_a_channel_is_dead():
    score = segment_identifiability("j", _AMP_RAD, **_SEGMENT)
    dead = type(score)(
        joint=score.joint,
        duration_s=score.duration_s,
        condition_number=score.condition_number,
        relative_variance=score.relative_variance,
        information={"damping": 0.0, "friction": 1.0},
        identifiable=score.identifiable,
    )
    assert worst_absolute_variance(dead) == float("inf")


# --- the constraint is the point -------------------------------------------


def test_an_unconstrained_design_walks_through_the_tip_over_floor(margin_of):
    """The measurement that made the constraint non-optional.

    Peak-to-peak barely moves, so displacement-based reasoning says this is
    safe. It is not: the design pushes energy to the high tones and ZMP is about
    acceleration.
    """
    weights, _score = design_spectrum("shoulder_lift", _AMP_RAD, **_SEGMENT)
    assert margin_of(None) >= LEKIWI_BASE.margin_min_m  # flat is fine
    assert margin_of(tuple(weights)) < LEKIWI_BASE.margin_min_m


def test_a_constrained_design_stays_above_the_floor_and_still_wins(margin_of):
    weights, designed = design_spectrum(
        "shoulder_lift",
        _AMP_RAD,
        margin=margin_of,
        margin_floor=LEKIWI_BASE.margin_min_m,
        **_SEGMENT,
    )
    assert margin_of(tuple(weights)) >= LEKIWI_BASE.margin_min_m
    flat = segment_identifiability("j", _AMP_RAD, **_SEGMENT)
    gain = worst_absolute_variance(flat) / worst_absolute_variance(designed)
    assert gain > 1.2, f"constrained design bought only {gain:.2f}x"


def test_the_constraint_costs_some_of_the_unconstrained_gain(margin_of):
    # Honest accounting: the unconstrained optimum is better *and* unusable.
    flat = worst_absolute_variance(segment_identifiability("j", _AMP_RAD, **_SEGMENT))
    _w1, free = design_spectrum("shoulder_lift", _AMP_RAD, **_SEGMENT)
    _w2, bound = design_spectrum(
        "shoulder_lift",
        _AMP_RAD,
        margin=margin_of,
        margin_floor=LEKIWI_BASE.margin_min_m,
        **_SEGMENT,
    )
    assert flat / worst_absolute_variance(free) > flat / worst_absolute_variance(bound)


def test_a_flat_spectrum_that_already_tips_is_refused_rather_than_reshaped():
    # Shape cannot rescue a motion that is simply too big, and pretending it can
    # would return a design whose margin was never above the floor.
    with pytest.raises(CalibrationError, match="no shape to search from"):
        design_spectrum(
            "shoulder_lift",
            _AMP_RAD,
            margin=lambda w: 0.001,
            margin_floor=0.02,
            **_SEGMENT,
        )


# --- the search itself -----------------------------------------------------


def test_the_design_never_scores_worse_than_flat():
    flat = segment_identifiability("j", _AMP_RAD, **_SEGMENT)
    _weights, designed = design_spectrum("shoulder_lift", _AMP_RAD, **_SEGMENT)
    assert worst_absolute_variance(designed) <= worst_absolute_variance(flat)


def test_the_search_is_deterministic():
    first = design_spectrum("shoulder_lift", _AMP_RAD, **_SEGMENT)[0]
    second = design_spectrum("shoulder_lift", _AMP_RAD, **_SEGMENT)[0]
    assert first == second


def test_weights_are_normalized_to_a_unit_maximum():
    weights, _score = design_spectrum("shoulder_lift", _AMP_RAD, **_SEGMENT)
    assert max(weights) == pytest.approx(1.0)
    assert all(w > 0 for w in weights)


def test_one_weight_per_tone_is_returned():
    tones = (0.3, 0.7, 1.4)
    weights, _score = design_spectrum(
        "shoulder_lift", _AMP_RAD, tones=tones, **_SEGMENT
    )
    assert len(weights) == len(tones)


def test_shape_design_refuses_the_legacy_waveform():
    with pytest.raises(CalibrationError, match="true-peak waveform"):
        design_spectrum("shoulder_lift", _AMP_RAD, kind=MULTISINE_KIND_V1, **_SEGMENT)
