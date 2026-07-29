# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Aging priors — growth curves, history fits, projections, drift bridge.

The load-bearing claims are that the two mechanisms carry different exponents
(linear friction wear, square-root backlash wear), that a fitted history
recovers the curve it was generated from, and that the module refuses the two
things a naive fit would happily do: extrapolate from a history that trends
down, and report a remaining life without a declared limit.
"""

from __future__ import annotations

import numpy as np
import pytest

from jaxterity.aging import (
    BACKLASH_EXPONENT,
    FRICTION_EXPONENT,
    AgingPrior,
    PowerLawGrowth,
    backlash_growth_prior,
    fit_growth,
    friction_growth_prior,
    project,
    to_drift_channel,
)
from jaxterity.errors import CalibrationError, JaxterityError

# --- the curve -------------------------------------------------------------


def test_value_starts_at_the_initial_and_rises():
    growth = PowerLawGrowth(initial=0.1, rate=2e-4, exponent=1.0)
    assert growth.value(0.0) == pytest.approx(0.1)
    assert growth.value(1000.0) == pytest.approx(0.3)


def test_value_is_array_friendly():
    growth = PowerLawGrowth(initial=0.1, rate=2e-4)
    out = growth.value(np.array([0.0, 500.0, 1000.0]))
    np.testing.assert_allclose(out, [0.1, 0.2, 0.3])


def test_negative_rate_is_refused_with_the_reason():
    with pytest.raises(JaxterityError, match="not wear"):
        PowerLawGrowth(initial=0.1, rate=-1e-4)


def test_the_two_mechanisms_carry_different_exponents():
    assert friction_growth_prior(0.1).exponent == FRICTION_EXPONENT == 1.0
    assert backlash_growth_prior(1e-3).exponent == BACKLASH_EXPONENT == 0.5


def test_square_root_growth_decelerates_and_linear_does_not():
    backlash = backlash_growth_prior(0.0, rate=1e-3)
    friction = friction_growth_prior(0.0, rate=1e-3)
    first = backlash.value(100.0) - backlash.value(0.0)
    second = backlash.value(200.0) - backlash.value(100.0)
    assert second < first  # mild wear slows down
    assert friction.value(200.0) - friction.value(100.0) == pytest.approx(
        friction.value(100.0) - friction.value(0.0)
    )


def test_usage_at_inverts_the_curve():
    growth = PowerLawGrowth(initial=0.1, rate=2e-4, exponent=1.0)
    usage = growth.usage_at(0.5)
    assert usage == pytest.approx(2000.0)
    assert growth.value(usage) == pytest.approx(0.5)


def test_usage_at_inverts_a_square_root_curve_too():
    growth = PowerLawGrowth(initial=0.0, rate=1e-3, exponent=0.5)
    usage = growth.usage_at(0.01)
    assert growth.value(usage) == pytest.approx(0.01)


def test_a_static_parameter_never_reaches_the_threshold():
    assert PowerLawGrowth(initial=0.1, rate=0.0).usage_at(0.5) == float("inf")


def test_a_threshold_below_the_initial_value_is_an_error():
    with pytest.raises(CalibrationError, match="already crossed"):
        PowerLawGrowth(initial=0.5, rate=1e-4).usage_at(0.2)


def test_remaining_life_counts_down_and_floors_at_zero():
    growth = PowerLawGrowth(initial=0.1, rate=2e-4, exponent=1.0)
    assert growth.remaining_life(0.0, 0.5) == pytest.approx(2000.0)
    assert growth.remaining_life(1500.0, 0.5) == pytest.approx(500.0)
    assert growth.remaining_life(5000.0, 0.5) == 0.0


# --- fitting a history -----------------------------------------------------


def _history(model: PowerLawGrowth, usages, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    u = np.asarray(usages, dtype=float)
    return u, model.value(u) + rng.normal(scale=noise, size=u.size)


def test_fit_recovers_a_linear_friction_history():
    truth = friction_growth_prior(0.1, rate=2e-4)
    u, v = _history(truth, [0.0, 250.0, 500.0, 750.0, 1000.0])
    fit = fit_growth(u, v, exponent=1.0)
    assert fit.model.initial == pytest.approx(0.1, abs=1e-9)
    assert fit.model.rate == pytest.approx(2e-4, rel=1e-6)
    assert fit.r_squared == pytest.approx(1.0)
    assert fit.n_points == 5


def test_fit_recovers_a_square_root_backlash_history():
    truth = backlash_growth_prior(1e-3, rate=1e-5)
    u, v = _history(truth, [0.0, 100.0, 400.0, 900.0, 1600.0])
    fit = fit_growth(u, v, exponent=0.5)
    assert fit.model.rate == pytest.approx(1e-5, rel=1e-6)


def test_fit_selects_the_exponent_from_the_grid_when_asked():
    truth = backlash_growth_prior(1e-3, rate=1e-5)
    u, v = _history(truth, [0.0, 100.0, 400.0, 900.0, 1600.0, 2500.0])
    fit = fit_growth(u, v)
    assert fit.model.exponent == pytest.approx(0.5)


def test_fixing_the_initial_value_buys_back_a_degree_of_freedom():
    truth = friction_growth_prior(0.1, rate=2e-4)
    u, v = _history(truth, [200.0, 600.0])  # only two points
    fit = fit_growth(u, v, exponent=1.0, initial=0.1)
    assert fit.model.rate == pytest.approx(2e-4, rel=1e-6)


def test_fit_refuses_a_history_that_trends_down():
    with pytest.raises(CalibrationError, match="trends DOWN"):
        fit_growth([0.0, 100.0, 200.0], [0.3, 0.2, 0.1], exponent=1.0)


def test_fit_refuses_a_history_with_a_frozen_clock():
    with pytest.raises(CalibrationError, match="usage clock to advance"):
        fit_growth([10.0, 10.0, 10.0], [0.1, 0.2, 0.3], exponent=1.0)


def test_fit_refuses_too_short_a_history():
    with pytest.raises(CalibrationError, match="at least 3 history point"):
        fit_growth([0.0, 100.0], [0.1, 0.2], exponent=1.0)


def test_fit_requires_matching_lengths():
    with pytest.raises(CalibrationError, match="exactly one of each"):
        fit_growth([0.0, 100.0, 200.0], [0.1, 0.2], exponent=1.0)


def test_residual_ratio_flags_a_point_off_the_trend():
    truth = friction_growth_prior(0.1, rate=2e-4)
    u, v = _history(truth, np.linspace(0.0, 1000.0, 9), noise=1e-3, seed=1)
    fit = fit_growth(u, v, exponent=1.0)
    on_trend = fit.residual_ratio(1200.0, truth.value(1200.0))
    knee = fit.residual_ratio(1200.0, truth.value(1200.0) + 0.05)
    assert on_trend < 3.0 < knee


# --- priors, projection, drift bridge --------------------------------------


def test_project_keys_by_drift_parameter_path():
    priors = (
        AgingPrior("robot.left_knee_joint.friction", friction_growth_prior(0.1)),
        AgingPrior("robot.right_knee_joint.friction", friction_growth_prior(0.12)),
    )
    projected = project(priors, 1000.0)
    assert set(projected) == {
        "robot.left_knee_joint.friction",
        "robot.right_knee_joint.friction",
    }
    assert (
        projected["robot.right_knee_joint.friction"]
        > projected["robot.left_knee_joint.friction"]
    )


def test_projected_paths_exist_in_the_declared_drift_schema():
    pytest.importorskip("mujoco")
    from jaxterity.twin import g1_drift_parameters

    declared = {p.path for p in g1_drift_parameters()}
    prior = AgingPrior("robot.left_knee_joint.friction", friction_growth_prior(0.1))
    assert prior.path in declared


def test_remaining_life_without_a_threshold_is_an_error():
    prior = AgingPrior("robot.left_knee_joint.friction", friction_growth_prior(0.1))
    with pytest.raises(JaxterityError, match="what counts as bad"):
        prior.remaining_life(0.0)


def test_remaining_life_uses_the_declared_threshold():
    prior = AgingPrior(
        "robot.left_knee_joint.friction",
        friction_growth_prior(0.1, rate=2e-4),
        threshold=0.5,
    )
    assert prior.remaining_life(500.0) == pytest.approx(1500.0)


def test_drift_channel_samples_the_curve_onto_breakpoints():
    day = 86400.0
    channel = to_drift_channel(
        friction_growth_prior(0.1, rate=2e-4),
        field="dof_frictionloss",
        index=3,
        times=(0.0, 10 * day, 20 * day),
        usage_per_second=1.0 / 3600.0,  # an operating-hours clock
    )
    assert channel.field == "dof_frictionloss"
    assert channel.values[0] == pytest.approx(0.1)
    assert channel.values[2] > channel.values[1] > channel.values[0]
    # And it is a valid schedule input.
    from jaxterity.drift import DriftSchedule

    schedule = DriftSchedule((channel,))
    assert float(schedule.channels[0].value(10 * day)) == pytest.approx(
        channel.values[1]
    )


def test_drift_channel_refuses_a_frozen_usage_clock():
    with pytest.raises(CalibrationError, match="usage_per_second > 0"):
        to_drift_channel(
            friction_growth_prior(0.1),
            field="dof_frictionloss",
            index=0,
            times=(0.0, 1.0),
            usage_per_second=0.0,
        )
