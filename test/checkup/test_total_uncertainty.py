# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Capture-length-honest uncertainty.

The acceptance criterion is one assertion, on the real capture that exposed the
problem: the reported interval must contain its own sub-window re-fits, which the
least-squares Wald interval demonstrably does not
(:func:`test_the_total_interval_covers_the_refits_the_wald_interval_missed`).
"""

from __future__ import annotations

import json
import math
import pathlib

import pytest

from jaxterity import zoo
from jaxterity.calibrate import load_commanded_capture
from jaxterity.checkup import (
    BIAS_DOMINANT_FRACTION,
    TotalUncertainty,
    fit_stability,
    length_ladder,
    total_uncertainty,
    uncertainty_gates,
    uncertainty_summary,
)
from jaxterity.checkup.probe import StabilityOfFit
from jaxterity.errors import CalibrationError

_DATA = pathlib.Path(__file__).parents[1] / "calibrate" / "data" / "lekiwi"
_CAL = str(_DATA / "calibration_final.json")


@pytest.fixture(scope="module")
def robot():
    return zoo.load("so100")


@pytest.fixture(scope="module")
def capture(robot):
    doc = json.loads((_DATA / "commanded_shoulder_pan.json").read_text())
    return load_commanded_capture(doc, robot, calibration=_CAL)


@pytest.fixture(scope="module")
def stability(robot, capture):
    return fit_stability(robot, capture)


# --- the acceptance criterion ---------------------------------------------


def test_the_total_interval_covers_the_refits_the_wald_interval_missed(stability):
    """T-236's acceptance criterion, on the capture that motivated it."""
    damping = next(s for s in stability if s.channel.endswith(".damping"))
    # Precondition: the Wald interval really does fail here. If this ever stops
    # being true the rest of the test proves nothing, so assert it explicitly.
    assert damping.windows_outside_ci > 0, (
        "the Wald interval now covers its sub-windows; this fixture no longer "
        "exercises the problem T-236 exists to fix"
    )
    total = total_uncertainty(damping)
    for value in damping.window_values:
        assert total.covers(value), f"{value} outside {total.ci95}"


def test_every_channel_of_the_real_capture_passes_the_criterion(stability):
    for entry in stability:
        total = total_uncertainty(entry)
        assert all(total.covers(v) for v in entry.window_values)


def test_the_real_dampings_uncertainty_is_mostly_length_not_noise(stability):
    # The finding: for shoulder_pan.damping the systematic term is several times
    # the statistical one, so quoting the Wald interval understates by ~4x.
    damping = next(s for s in stability if s.channel.endswith(".damping"))
    total = total_uncertainty(damping)
    assert total.bias_half > 3.0 * total.wald_half
    assert total.bias_fraction > 0.9  # variance share
    assert total.bias_dominates


# --- the combination ------------------------------------------------------


def _synthetic(value=1.0, ci_half=0.1, windows=(1.2,)):
    return StabilityOfFit(
        channel="j.damping",
        full=value,
        window_values=windows,
        ci95=(value - ci_half, value + ci_half),
    )


def test_the_total_is_the_quadrature_sum_of_both_components():
    total = total_uncertainty(_synthetic(value=1.0, ci_half=0.3, windows=(1.4,)))
    assert total.wald_half == pytest.approx(0.3)
    assert total.bias_half == pytest.approx(0.4)
    assert total.total_half == pytest.approx(0.5)  # 3-4-5
    assert total.ci95 == pytest.approx((0.5, 1.5))
    assert total.relative == pytest.approx(0.5)
    assert total.bias_fraction == pytest.approx(0.64)  # variance share: .4²/.5²


def test_the_bias_term_is_the_largest_departure_not_the_average():
    # A single badly-off window must set the term: averaging would let one real
    # length dependence hide behind several agreeing windows.
    total = total_uncertainty(
        _synthetic(value=1.0, ci_half=0.0, windows=(1.01, 1.02, 1.5))
    )
    assert total.bias_half == pytest.approx(0.5)


def test_a_missing_wald_interval_falls_back_to_the_systematic_term_alone():
    entry = StabilityOfFit(
        channel="j.damping", full=1.0, window_values=(1.2,), ci95=None
    )
    total = total_uncertainty(entry)
    assert math.isnan(total.wald_half)  # the omission stays visible
    assert total.total_half == pytest.approx(0.2)
    assert total.bias_fraction == pytest.approx(1.0)
    assert total.covers(1.2)


def test_the_interval_always_covers_every_window_by_construction():
    # total_half >= bias_half >= |window - full| for every window, so this is a
    # structural guarantee rather than a property of any particular capture.
    for windows in ((1.5,), (0.1, 1.9), (1.0, 1.0), (2.0, 0.5, 1.3)):
        total = total_uncertainty(_synthetic(value=1.0, ci_half=0.05, windows=windows))
        assert all(total.covers(v) for v in windows), windows


def test_a_ladder_with_no_windows_is_refused_rather_than_dressed_up():
    entry = StabilityOfFit(
        channel="j.damping", full=1.0, window_values=(), ci95=(0.9, 1.1)
    )
    with pytest.raises(CalibrationError, match="no sub-window re-fits"):
        total_uncertainty(entry)


def test_zero_value_gives_nan_relative_not_a_division_error():
    total = total_uncertainty(_synthetic(value=0.0, ci_half=0.1, windows=(0.2,)))
    assert math.isnan(total.relative)
    assert total.total_half > 0


def test_to_dict_materializes_the_derived_fields_and_round_trips():
    total = total_uncertainty(_synthetic(value=1.0, ci_half=0.3, windows=(1.4,)))
    d = total.to_dict()
    assert json.loads(json.dumps(d)) == d
    # The point of materializing: a reader must not have to recompute the total
    # or be able to mistake wald_half for the interval.
    assert d["total_half"] == pytest.approx(0.5)
    assert d["ci95"] == [pytest.approx(0.5), pytest.approx(1.5)]
    assert d["bias_dominates"] is True


# --- the length ladder ----------------------------------------------------


def test_the_ladder_refits_on_several_contiguous_lengths(robot, capture):
    ladder = length_ladder(robot, capture, fractions=(0.4, 0.6, 0.8, 1.0))
    assert ladder
    for entry in ladder:
        assert len(entry.window_values) == 3  # 0.4, 0.6, 0.8 — 1.0 is `full`
        assert entry.ci95 is not None


def test_the_ladder_shows_real_length_dependence_on_the_real_capture(robot, capture):
    # If the estimate did not depend on capture length there would be nothing to
    # correct; on this capture friction moves by more than an order of magnitude.
    ladder = length_ladder(robot, capture)
    friction = next(e for e in ladder if e.channel.endswith(".friction"))
    values = friction.window_values
    assert max(values) > 5.0 * min(values), values


def test_the_ladder_needs_two_lengths_below_the_full_capture(robot, capture):
    with pytest.raises(CalibrationError, match="at least two prefix lengths"):
        length_ladder(robot, capture, fractions=(0.8, 1.0))


def test_the_ladder_refuses_fractions_outside_the_unit_interval(robot, capture):
    for bad in ((0.0, 0.5), (0.5, 1.5), (-0.2, 0.5)):
        with pytest.raises(CalibrationError, match=r"must lie in \(0, 1\]"):
            length_ladder(robot, capture, fractions=bad)


def test_duplicate_fractions_collapse(robot, capture):
    ladder = length_ladder(robot, capture, fractions=(0.4, 0.4, 0.6, 0.6))
    assert all(len(e.window_values) == 2 for e in ladder)


# --- the gates ------------------------------------------------------------


def _total(value, wald, bias):
    return TotalUncertainty(
        channel="j.damping", value=value, wald_half=wald, bias_half=bias, n_windows=3
    )


def test_a_tight_well_composed_channel_raises_no_gate():
    assert uncertainty_gates([_total(1.0, 0.02, 0.01)]) == []


def test_a_wide_total_warns_on_confidence():
    gates = uncertainty_gates([_total(1.0, 0.4, 0.05)])
    assert [g.gate for g in gates] == ["confidence"]
    assert gates[0].level == "warn"
    assert "capture-length" in gates[0].detail


def test_a_bias_dominated_channel_warns_even_when_it_is_tight():
    # The case the composition gate exists for: nothing is out of tolerance, but
    # the number will move if the probe is shortened.
    gates = uncertainty_gates([_total(1.0, 0.001, 0.02)])
    assert [g.gate for g in gates] == ["uncertainty_composition"]
    assert gates[0].level == "warn"
    assert "do not compare" in gates[0].remedy


def test_a_wide_and_bias_dominated_channel_raises_both():
    gates = uncertainty_gates([_total(1.0, 0.05, 0.5)])
    assert sorted(g.gate for g in gates) == ["confidence", "uncertainty_composition"]


def test_no_gate_is_ever_a_failure():
    # These describe how well something is known, not whether it is healthy.
    gates = uncertainty_gates([_total(1.0, 5.0, 5.0), _total(1.0, 0.0, 1.0)])
    assert gates and all(g.level == "warn" for g in gates)


def test_equal_components_split_the_variance_exactly_in_half():
    # The reason bias_fraction is a variance share: as an amplitude ratio equal
    # components would read 0.71 and the "half" threshold could not be stated.
    equal = _total(1.0, 1.0, 1.0)
    assert equal.bias_fraction == pytest.approx(0.5)
    assert equal.bias_dominates  # equal shares count as dominant
    just_under = _total(1.0, 1.0, 0.99)
    assert just_under.bias_fraction < BIAS_DOMINANT_FRACTION
    assert not just_under.bias_dominates


def test_the_real_capture_raises_gates_naming_its_channels(stability):
    gates = uncertainty_gates([total_uncertainty(s) for s in stability])
    assert gates
    assert all(g.scope.startswith("shoulder_pan.") for g in gates)


def test_summary_renders_a_row_per_channel(stability):
    text = uncertainty_summary([total_uncertainty(s) for s in stability])
    assert text.count("\n") == len(stability)
    assert "bias%" in text.splitlines()[0]
    for entry in stability:
        assert entry.channel in text
