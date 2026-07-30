# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Designing the capture amplitude instead of derating it.

The acceptance criterion is that the optimized plan beats the derated one on
information per second while respecting the same tip-over floor. It does, with
one honest qualification the tests pin explicitly: the *viscous* channel gains
(amplitude², so 300 → 317 ticks is +11.7 %) and the *Coulomb* channel does not,
because ``tanh(q̇/ε)`` has already saturated. That is a property of the waveform
family, not of the optimizer.
"""

from __future__ import annotations

import pathlib

import pytest

from jaxterity import zoo
from jaxterity.checkup import build_capture_plan, plan_information
from jaxterity.checkup.plan import _MIN_AMP_TICKS, _largest_safe_amplitude
from jaxterity.checkup.stability import LEKIWI_BASE

_CALIB = (
    pathlib.Path(__file__).parents[1]
    / "calibrate"
    / "data"
    / "lekiwi"
    / "calibration_final.json"
)
_HOME = {
    "shoulder_pan": 1.916,
    "shoulder_lift": -0.8311,
    "elbow_flex": -1.5417,
    "wrist_flex": 0.1449,
    "wrist_roll": -0.5704,
}
#: A request large enough that the derate ladder actually fires on a joint. At
#: the default 200 the joint entries are band-limited rather than derated, so the
#: optimizer has nothing to recover and the comparison would be vacuous.
_BIG_REQUEST = 400


@pytest.fixture(scope="module")
def robot():
    return zoo.load("so100")


def _plan(robot, **kwargs):
    return build_capture_plan(
        robot,
        str(_CALIB),
        serial="LK-TEST",
        base=LEKIWI_BASE,
        pose_rad=_HOME,
        **kwargs,
    )


@pytest.fixture(scope="module")
def derated(robot):
    return _plan(robot, amp_ticks=_BIG_REQUEST)


@pytest.fixture(scope="module")
def optimized(robot):
    return _plan(robot, amp_ticks=_BIG_REQUEST, optimize=True)


# --- the acceptance criterion ---------------------------------------------


def test_the_optimized_plan_respects_the_same_floor(optimized):
    assert all(
        e.min_margin_m >= LEKIWI_BASE.margin_min_m
        for e in optimized.entries
        if e.feasible
    )


def test_the_optimized_plan_never_loses_information_on_any_channel(derated, optimized):
    before = plan_information(derated, str(_CALIB))
    after = plan_information(optimized, str(_CALIB))
    assert set(before) == set(after)
    for joint, channels in before.items():
        for channel, value in channels.items():
            assert after[joint][channel] >= value - 1e-9, (joint, channel)


def test_the_optimizer_recovers_amplitude_the_derate_ladder_gave_away(
    derated, optimized
):
    pairs = {
        e.joints: (e.amp_ticks, o.amp_ticks)
        for e, o in zip(derated.entries, optimized.entries)
        if e.kind == "joint"
    }
    improved = {j: p for j, p in pairs.items() if p[1] > p[0]}
    assert improved, f"no entry improved; ladder never fired: {pairs}"
    assert ("shoulder_lift",) in improved
    before, after = improved[("shoulder_lift",)]
    assert (before, after) == (300, 317)


def test_the_recovered_amplitude_buys_viscous_information_as_amplitude_squared(
    derated, optimized
):
    before = plan_information(derated, str(_CALIB))["shoulder_lift"]
    after = plan_information(optimized, str(_CALIB))["shoulder_lift"]
    expected = (317 / 300) ** 2
    assert after["damping"] / before["damping"] == pytest.approx(expected, rel=0.01)


def test_the_coulomb_channel_gains_nothing_because_it_has_saturated(derated, optimized):
    # The honest limit of amplitude-only optimization, and the reason
    # plan_information reports per channel instead of collapsing to one number.
    before = plan_information(derated, str(_CALIB))["shoulder_lift"]
    after = plan_information(optimized, str(_CALIB))["shoulder_lift"]
    assert after["friction"] / before["friction"] == pytest.approx(1.0, abs=0.01)


def test_the_load_trace_is_optimized_too(derated, optimized):
    before = next(e for e in derated.entries if e.kind == "load")
    after = next(e for e in optimized.entries if e.kind == "load")
    assert after.amp_ticks > before.amp_ticks
    assert after.min_margin_m >= LEKIWI_BASE.margin_min_m


def test_optimizing_spends_margin_the_ladder_left_unused(derated, optimized):
    # Stated plainly because it is the trade, not a regression: approaching the
    # floor consumes margin. "Equal-or-better margin" is true of the *floor*
    # being respected, and false of the margin number itself.
    before = next(e for e in derated.entries if e.kind == "load")
    after = next(e for e in optimized.entries if e.kind == "load")
    assert after.min_margin_m < before.min_margin_m
    assert after.min_margin_m >= LEKIWI_BASE.margin_min_m


# --- reproducibility of the default path ----------------------------------


def test_the_default_path_is_unchanged_by_the_new_option(robot):
    # optimize=False must reproduce the historical plan bit for bit, or every
    # committed plan and its recorded margins silently change meaning.
    a = _plan(robot, amp_ticks=200)
    b = _plan(robot, amp_ticks=200, optimize=False)
    assert a.to_dict() == b.to_dict()


def test_at_the_default_request_the_joints_are_band_limited_not_derated(robot):
    # Documents why _BIG_REQUEST exists: at 200 the optimizer changes only the
    # load trace, because no joint entry was derated in the first place.
    a = _plan(robot, amp_ticks=200)
    b = _plan(robot, amp_ticks=200, optimize=True)
    joint_changes = [
        (e.joints, e.amp_ticks, o.amp_ticks)
        for e, o in zip(a.entries, b.entries)
        if e.kind == "joint" and e.amp_ticks != o.amp_ticks
    ]
    assert joint_changes == []


# --- the bisection itself -------------------------------------------------


def test_bisection_finds_the_largest_amplitude_under_a_known_threshold():
    # margin(amp) = 1 - amp/1000, floor 0.5 -> largest safe amp is 500.
    calls = []

    def evaluate(amp):
        calls.append(amp)
        return 1.0 - amp / 1000.0

    amp, margin = _largest_safe_amplitude(evaluate, 900, 0.5)
    assert amp == 500
    assert margin == pytest.approx(0.5)
    # And it is cheap: bisecting 60..900 to one tick takes ~log2(840) probes.
    assert len(calls) < 15


def test_bisection_returns_the_request_untouched_when_it_is_already_safe():
    probes = []

    def evaluate(amp):
        probes.append(amp)
        return 1.0

    amp, margin = _largest_safe_amplitude(evaluate, 250, 0.5)
    assert (amp, margin) == (250, 1.0)
    assert probes == [250], "a safe request should cost exactly one simulation"


def test_bisection_reports_infeasible_when_even_the_minimum_tips():
    amp, margin = _largest_safe_amplitude(lambda a: 0.0, 900, 0.5)
    assert amp < _MIN_AMP_TICKS  # so the caller's feasibility test fails
    assert margin == 0.0


def test_bisection_handles_a_band_narrower_than_the_minimum_amplitude():
    amp, margin = _largest_safe_amplitude(lambda a: 1.0, 30, 0.5)
    assert amp == 30
    assert margin == 1.0


def test_the_returned_margin_is_simulated_at_the_returned_amplitude():
    # Not interpolated: a caller gates on this number, so it has to be the real
    # simulated margin of the amplitude actually shipped.
    seen = {}

    def evaluate(amp):
        value = 1.0 - amp / 1000.0
        seen[amp] = value
        return value

    amp, margin = _largest_safe_amplitude(evaluate, 900, 0.5)
    assert seen[amp] == margin


# --- plan_information -----------------------------------------------------


def test_information_is_reported_per_channel_per_joint(derated):
    scores = plan_information(derated, str(_CALIB))
    assert scores
    for channels in scores.values():
        assert set(channels) == {"damping", "friction"}
        assert all(v > 0 for v in channels.values())


def test_information_is_per_second_so_a_longer_capture_is_not_a_better_design(
    robot,
):
    short = _plan(robot, amp_ticks=200, duration_s=8.0)
    long_ = _plan(robot, amp_ticks=200, duration_s=16.0)
    a = plan_information(short, str(_CALIB))["wrist_roll"]["damping"]
    b = plan_information(long_, str(_CALIB))["wrist_roll"]["damping"]
    assert b == pytest.approx(a, rel=0.05)


def test_infeasible_entries_are_omitted_rather_than_scored_zero(derated):
    scores = plan_information(derated, str(_CALIB))
    for entry in derated.entries:
        if entry.kind == "joint" and not entry.feasible:
            assert entry.joints[0] not in scores
