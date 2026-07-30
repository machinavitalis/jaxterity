# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Pre-motion probe scoring, and the fit-stability finding that blocks 0.1.

Two groups. The scorer's tests are ordinary properties: a planned segment's
identifiability is computable before any motion, more excitation buys
information, and a degenerate segment is refused. The stability tests carry the
measured result that changes the compression plan — on the committed **real**
LeKiwi captures, sub-window re-fits agree closely with each other but sit
systematically *outside* the full capture's own 95 % interval, so the estimate
depends on how much capture it saw and "the shorter probe agrees within CI"
cannot be the acceptance test.
"""

from __future__ import annotations

import json
import pathlib

import pytest

pytest.importorskip("mujoco")

from jaxterity import zoo  # noqa: E402
from jaxterity.calibrate import load_commanded_capture  # noqa: E402
from jaxterity.checkup.probe import (  # noqa: E402
    commanded_velocity,
    fit_stability,
    segment_identifiability,
)
from jaxterity.checkup.stability import MULTISINE_KIND_V1  # noqa: E402
from jaxterity.errors import CalibrationError  # noqa: E402

_DATA = pathlib.Path(__file__).parents[1] / "calibrate" / "data" / "lekiwi"
_CAL = str(_DATA / "calibration_final.json")


@pytest.fixture(scope="module")
def robot():
    return zoo.load("so100")


def _capture(robot, joint):
    doc = json.loads((_DATA / f"commanded_{joint}.json").read_text())
    return load_commanded_capture(doc, robot, calibration=_CAL)


# --- pre-motion scoring ----------------------------------------------------


def test_a_planned_segment_is_identifiable_before_any_motion():
    score = segment_identifiability("elbow_flex", 0.3, duration_s=16.0, rate_hz=80.0)
    assert score.identifiable
    assert set(score.relative_variance) == {"damping", "friction"}
    assert score.condition_number > 1.0


def test_more_amplitude_buys_damping_information():
    small = segment_identifiability("j", 0.05, duration_s=16.0, rate_hz=80.0)
    large = segment_identifiability("j", 0.50, duration_s=16.0, rate_hz=80.0)
    # Viscous information goes as amplitude squared: 10x the amplitude is ~100x.
    ratio = large.information["damping"] / small.information["damping"]
    assert ratio == pytest.approx(100.0, rel=0.05), ratio
    # The Coulomb column saturates, so it gains far less — which is the reason
    # the two channels stop separating as amplitude grows.
    assert large.information["friction"] < 10.0 * small.information["friction"]
    assert large.relative_variance["friction"] != small.relative_variance["friction"]
    assert large.identifiable and small.identifiable


def test_information_responds_to_amplitude_at_all():
    # Regression guard. This field was the diagonal of a column-normalized Gram
    # matrix, hence identically 1.0, so it could not tell a 60-tick excitation
    # from a 200-tick one and no test noticed.
    values = [
        segment_identifiability("j", a, duration_s=16.0, rate_hz=80.0).information[
            "damping"
        ]
        for a in (0.1, 0.2, 0.4)
    ]
    assert len(set(values)) == 3, values
    assert values == sorted(values)


def test_a_longer_segment_never_identifies_less():
    short = segment_identifiability("j", 0.3, duration_s=8.0, rate_hz=80.0)
    long_ = segment_identifiability("j", 0.3, duration_s=16.0, rate_hz=80.0)
    # Twice the samples at the same amplitude: strictly more information, and
    # roughly twice as much. The old form asserted `>= short * 0.5` on two
    # values that were both exactly 1.0, which no change could have broken.
    assert long_.information["damping"] > short.information["damping"]
    assert long_.information["damping"] == pytest.approx(
        2.0 * short.information["damping"], rel=0.2
    )
    assert long_.identifiable


def test_the_weakest_channel_is_named():
    score = segment_identifiability("j", 0.3, duration_s=16.0, rate_hz=80.0)
    assert score.weakest_channel in {"damping", "friction"}


def test_a_zero_amplitude_segment_is_refused():
    with pytest.raises(CalibrationError, match="degenerate design column"):
        segment_identifiability("j", 0.0, duration_s=16.0, rate_hz=80.0)


def test_a_segment_too_short_to_fit_is_refused():
    with pytest.raises(CalibrationError, match="at least 20"):
        segment_identifiability("j", 0.3, duration_s=0.1, rate_hz=80.0)


def test_the_waveform_version_reaches_the_score():
    v2 = segment_identifiability("j", 0.3, duration_s=16.0, rate_hz=80.0)
    v1 = segment_identifiability(
        "j", 0.3, duration_s=16.0, rate_hz=80.0, kind=MULTISINE_KIND_V1
    )
    # The legacy waveform swings 52% wider for the same amplitude, so its
    # velocity columns differ — the score must see that, not average over it.
    assert v1.information["damping"] != v2.information["damping"]


def test_commanded_velocity_is_zero_mean_and_bounded():
    q, qd = commanded_velocity(0.3, duration_s=16.0, rate_hz=80.0)
    assert abs(float(q.mean())) < 0.05
    assert float(abs(q).max()) <= 0.3 + 1e-9  # v2 never exceeds the amplitude
    assert float(abs(qd).max()) > 0.0


def test_scores_round_trip_to_json():
    score = segment_identifiability("j", 0.3, duration_s=16.0, rate_hz=80.0)
    assert json.loads(json.dumps(score.to_dict()))["joint"] == "j"


# --- the fit-stability finding ---------------------------------------------


@pytest.mark.parametrize("joint", ["shoulder_pan", "elbow_flex"])
def test_sub_window_fits_land_outside_the_full_captures_interval(robot, joint):
    """The measured result that blocks duration compression.

    Re-fitting sub-windows of the *same* real capture moves damping by 10-20%
    and every sub-window lands past the full fit's 95% interval. The interval
    does not model the estimate's dependence on capture length, so no shortened
    probe can be accepted on "agrees within CI" — whatever the probe does.
    """
    stability = fit_stability(robot, _capture(robot, joint))
    damping = next(s for s in stability if s.channel.endswith(".damping"))
    assert damping.ci95 is not None
    assert damping.windows_outside_ci == len(damping.window_values), (
        f"{joint}: {damping.windows_outside_ci}/{len(damping.window_values)} "
        f"windows outside CI (bias {damping.bias:.3f}) — if this ever stops "
        "holding, duration compression can be re-attempted against the CI."
    )


def test_the_bias_is_systematic_rather_than_scatter(robot):
    """Why averaging more windows would not rescue a shortened probe."""
    stability = fit_stability(robot, _capture(robot, "shoulder_pan"))
    damping = next(s for s in stability if s.channel.endswith(".damping"))
    # The windows agree with each other far better than they agree with the
    # full capture: a systematic offset, not noise.
    assert damping.spread < damping.bias
    # Order of magnitude of the docstring's numbers (not a golden value).
    assert 0.05 < damping.bias < 0.40
    assert damping.ci_relative is not None and damping.ci_relative < 0.10


def test_stability_reports_every_identified_channel(robot):
    stability = fit_stability(robot, _capture(robot, "shoulder_pan"))
    assert stability  # at least the damping channel
    for entry in stability:
        assert entry.channel.startswith("shoulder_pan.")
        assert len(entry.window_values) >= 2


def test_stability_refuses_a_single_window(robot):
    with pytest.raises(CalibrationError, match="at least 2 windows"):
        fit_stability(robot, _capture(robot, "shoulder_pan"), windows=1)


def test_stability_refuses_an_impossible_window_fraction(robot):
    with pytest.raises(CalibrationError, match="must lie in"):
        fit_stability(robot, _capture(robot, "shoulder_pan"), window_fraction=1.5)


def test_stability_refuses_windows_shorter_than_the_fit_minimum(robot):
    with pytest.raises(CalibrationError, match="20-sample minimum"):
        fit_stability(robot, _capture(robot, "shoulder_pan"), window_fraction=0.01)
