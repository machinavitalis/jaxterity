# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Lost motion from a reversal sweep.

There is no hardware capture here yet: the LeKiwi was powered down before this
probe existed, so the per-joint number on LK-0001 is still owed. What is tested
is that a *known* lost motion is recovered from a synthetic sweep, that the
protocol requirements are enforced by name, and that nothing here claims to have
measured gear play — which a single-encoder servo cannot support.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from jaxterity.checkup import (
    BACKLASH_CAPTURE_SCHEMA,
    backlash_summary,
    fit_reversal,
    load_backlash_capture,
    reversal_lost_motion,
)
from jaxterity.errors import CalibrationError

_TICKS_PER_RAD = 651.9
_CAL = {
    "calibrations": [
        {"servo_id": 1, "joint": "shoulder_pan", "ticks_per_rad": _TICKS_PER_RAD},
        {"servo_id": 2, "joint": "shoulder_lift", "ticks_per_rad": _TICKS_PER_RAD},
    ]
}


def _sweep(
    *,
    servo_ids=(1,),
    lost_motion_ticks=30.0,
    span=250,
    n=400,
    directions=("forward", "reverse"),
    noise=0.0,
    seed=7,
    stiffness=4.0,
    load_amp=40.0,
):
    """A synthetic reversal capture with a known hysteresis width.

    Built from the model the analysis assumes, so both halves are checkable:

    * ``play = lost_motion_ticks / 2`` is the dead zone, entered with the sign of
      travel — so the measured position trails the goal going up and leads it
      coming back.
    * The load varies along the sweep (standing in for a gravity torque that
      changes with angle) and relates to the *engaged* part of the error by
      ``load = k·(error − play·sign(error))``.

    The load term is what makes this non-degenerate. An earlier version set
    ``load = k·error`` with no dead zone, which makes ``play = 0`` the *correct*
    fit — the loop was in position-versus-goal only, and the fit reads
    load-versus-error. ``load_amp`` is kept below ``k·play`` so the error never
    changes sign within a branch, which would break the sign convention the
    piecewise-linear model is written in.
    """
    rng = np.random.default_rng(seed)
    play = lost_motion_ticks / 2.0
    rows = {
        k: [] for k in ("t", "servo_id", "phase", "goal_ticks", "pos_ticks", "load_raw")
    }
    clock = 0.0
    for sid in servo_ids:
        centre = 2048
        for direction in directions:
            goals = np.linspace(centre - span, centre + span, n)
            if direction == "reverse":
                goals = goals[::-1]
            sign = 1.0 if direction == "forward" else -1.0
            for goal in goals:
                load = load_amp * (goal - centre) / span
                # error = the dead zone, entered with the direction of travel,
                # plus the elastic deflection the load implies.
                error = sign * play + (load / stiffness if stiffness else 0.0)
                position = goal - error
                if noise:
                    position += rng.normal(scale=noise)
                rows["t"].append(clock)
                rows["servo_id"].append(sid)
                rows["phase"].append(direction)
                rows["goal_ticks"].append(float(goal))
                rows["pos_ticks"].append(float(position))
                rows["load_raw"].append(float(load))
                clock += 0.02
    return {
        "schema": BACKLASH_CAPTURE_SCHEMA,
        "servo_ids": list(servo_ids),
        "joints": {
            str(s): {1: "shoulder_pan", 2: "shoulder_lift"}.get(s, "")
            for s in servo_ids
        },
        "excitation": {"kind": "slow_reversal_sweep", "ticks_per_s": 40.0},
        **rows,
    }


# --- the measurement ------------------------------------------------------


def test_a_known_lost_motion_is_recovered():
    (result,) = reversal_lost_motion(_sweep(lost_motion_ticks=30.0), calibration=_CAL)
    assert result.lost_motion_ticks == pytest.approx(30.0, abs=0.5)
    assert result.lost_motion_rad == pytest.approx(30.0 / _TICKS_PER_RAD, rel=0.02)
    assert result.joint == "shoulder_pan"


def test_the_recovered_width_tracks_the_true_width():
    measured = [
        reversal_lost_motion(_sweep(lost_motion_ticks=w), calibration=_CAL)[
            0
        ].lost_motion_ticks
        for w in (10.0, 20.0, 40.0, 80.0)
    ]
    assert measured == pytest.approx([10.0, 20.0, 40.0, 80.0], abs=0.5)


def test_a_joint_with_no_lost_motion_reads_essentially_zero():
    (result,) = reversal_lost_motion(
        _sweep(lost_motion_ticks=0.0, load_amp=0.0), calibration=_CAL
    )
    assert result.lost_motion_ticks == pytest.approx(0.0, abs=0.5)


def test_noise_widens_the_interval_without_moving_the_value():
    clean = reversal_lost_motion(_sweep(noise=0.0), calibration=_CAL)[0]
    noisy = reversal_lost_motion(_sweep(noise=4.0), calibration=_CAL)[0]
    assert noisy.lost_motion_ticks == pytest.approx(clean.lost_motion_ticks, rel=0.15)
    clean_half = clean.ci95_rad[1] - clean.ci95_rad[0]
    noisy_half = noisy.ci95_rad[1] - noisy.ci95_rad[0]
    assert noisy_half > clean_half


def test_the_interval_brackets_the_value():
    (result,) = reversal_lost_motion(_sweep(noise=3.0), calibration=_CAL)
    lo, hi = result.ci95_rad
    assert lo <= result.lost_motion_rad <= hi
    assert 0.0 < result.relative_ci < 1.0


def test_every_swept_joint_is_reported():
    results = reversal_lost_motion(_sweep(servo_ids=(1, 2)), calibration=_CAL)
    assert [r.joint for r in results] == ["shoulder_pan", "shoulder_lift"]
    assert [r.servo_id for r in results] == [1, 2]


def test_without_a_calibration_the_tick_width_still_reports():
    # A width in ticks is a real measurement; only the unit conversion is missing.
    (result,) = reversal_lost_motion(_sweep())
    assert result.lost_motion_ticks == pytest.approx(30.0, abs=0.5)
    assert np.isnan(result.lost_motion_rad)


def test_the_sweep_rate_is_carried_through_so_quasi_static_is_checkable():
    (result,) = reversal_lost_motion(_sweep(), calibration=_CAL)
    assert result.sweep_rate_ticks_per_s > 0


# --- the protocol requirements --------------------------------------------


def test_a_single_direction_sweep_is_refused_by_joint_name():
    """The requirement that matters most: one branch has no loop."""
    with pytest.raises(CalibrationError) as exc:
        reversal_lost_motion(_sweep(directions=("forward",)), calibration=_CAL)
    message = str(exc.value)
    assert "shoulder_pan" in message and "servo 1" in message
    assert "reverse" in message
    assert "no hysteresis loop" in message


def test_a_reverse_only_sweep_is_refused_too():
    with pytest.raises(CalibrationError, match="forward"):
        reversal_lost_motion(_sweep(directions=("reverse",)), calibration=_CAL)


def test_a_branch_too_short_to_median_is_refused():
    with pytest.raises(CalibrationError, match="after trimming"):
        reversal_lost_motion(_sweep(n=10), calibration=_CAL)


def test_non_overlapping_sweeps_are_refused_rather_than_extrapolated():
    doc = _sweep()
    goals = np.asarray(doc["goal_ticks"], dtype=float)
    phase = np.asarray(doc["phase"], dtype=object)
    # Shift the reverse branch clear of the forward one: any "gap" between them
    # would then be extrapolation, not a loop width.
    goals[phase == "reverse"] += 10_000
    doc["goal_ticks"] = goals.tolist()
    doc["pos_ticks"] = (
        np.asarray(doc["pos_ticks"], float) + (phase == "reverse") * 10_000
    ).tolist()
    with pytest.raises(CalibrationError, match="overlap over only"):
        reversal_lost_motion(doc, calibration=_CAL)


def test_the_loader_rejects_a_foreign_schema():
    with pytest.raises(CalibrationError, match="unrecognized schema"):
        load_backlash_capture({"schema": "nope", "t": [0.0]})


def test_the_loader_rejects_an_empty_capture():
    with pytest.raises(CalibrationError, match="no samples"):
        load_backlash_capture({"schema": BACKLASH_CAPTURE_SCHEMA, "t": []})


# --- the model-dependent decomposition ------------------------------------


def test_the_fit_and_the_model_free_width_agree_within_a_factor_of_two():
    # The whole reason both exist: the stiffness-free width is the cross-check on
    # the fitted dead zone. Order-of-magnitude agreement is the bar; the fit sees
    # a smoothed loop the model treats as hard-edged.
    doc = _sweep(lost_motion_ticks=30.0, stiffness=4.0)
    (width,) = reversal_lost_motion(doc, calibration=_CAL)
    (fit,) = fit_reversal(doc, calibration=_CAL)
    ratio = fit.play_rad / (width.lost_motion_rad / 2.0)
    assert 0.5 < ratio < 2.0, (
        f"fit {fit.play_rad} vs half-width {width.lost_motion_rad / 2}"
    )


def test_the_fit_recovers_the_stiffness_sign_and_scale():
    doc = _sweep(stiffness=4.0)
    (fit,) = fit_reversal(doc, calibration=_CAL)
    # 4 load units per tick of error, reported per radian.
    assert fit.k_mesh_proxy == pytest.approx(4.0 * _TICKS_PER_RAD, rel=0.05)
    assert fit.r_squared > 0.9


def test_a_capture_with_no_load_channel_is_refused_with_the_alternative_named():
    doc = _sweep()
    doc["load_raw"] = []
    with pytest.raises(CalibrationError, match="reversal_lost_motion"):
        fit_reversal(doc, calibration=_CAL)


def test_a_sweep_that_never_loaded_the_mesh_is_refused():
    doc = _sweep(load_amp=0.0)
    with pytest.raises(CalibrationError, match="never loaded the mesh"):
        fit_reversal(doc, calibration=_CAL)


# --- reporting ------------------------------------------------------------


def test_the_summary_renders_a_row_per_joint_with_its_interval():
    results = reversal_lost_motion(_sweep(servo_ids=(1, 2)), calibration=_CAL)
    text = backlash_summary(results)
    assert text.count("\n") == len(results)
    assert "ci95 mrad" in text.splitlines()[0]
    for item in results:
        assert item.joint in text


def test_results_round_trip_to_json():
    (result,) = reversal_lost_motion(_sweep(), calibration=_CAL)
    d = result.to_dict()
    assert json.loads(json.dumps(d)) == d
    assert d["ci95_rad"][0] <= d["lost_motion_rad"] <= d["ci95_rad"][1]


def test_nothing_in_the_result_is_named_backlash_or_play():
    """The naming *is* the honesty claim, so it is asserted.

    One encoder cannot separate gear play from the servo's position-loop deadband
    or from compliance, so the measured quantity is total lost motion. Only the
    explicitly model-dependent ``ReversalFit`` carries a ``play`` field, and its
    docstring says it is still not gear play.
    """
    (result,) = reversal_lost_motion(_sweep(), calibration=_CAL)
    fields = set(result.to_dict())
    assert not any("backlash" in f or "play" in f for f in fields), fields
    assert "lost_motion_rad" in fields
