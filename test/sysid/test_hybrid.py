# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Parametric-first hybrid, with the residual norm as the health metric.

Two things are tested: that the metric responds to *injected* unmodelled
structure and not to noise, and that the comparison is fair to the physics. The
second is the one that bit — omitting the load-proxy-to-torque calibration made
the parametric model look 6x worse than it is and drove the absorbed fraction
from 14 % to 86 %.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np
import pytest

from jaxterity import zoo
from jaxterity.calibrate import (
    axis_inertia_feature,
    identify_closed_loop,
    load_commanded_capture,
)
from jaxterity.calibrate.types import JointCalibration
from jaxterity.errors import CalibrationError
from jaxterity.sysid import (
    ABSORBED_WARN_FRACTION,
    ModelCompleteness,
    fit_residual,
    model_completeness,
    residual_gates,
)

_DATA = pathlib.Path(__file__).parents[1] / "calibrate" / "data" / "lekiwi"
_CAL = str(_DATA / "calibration_final.json")
_QUANTUM = 1.0 / 651.9
_INERTIA = 0.015
_DAMPING = 0.4
_FRICTION = 0.015


@dataclasses.dataclass
class _Capture:
    """The minimal capture surface the hybrid fit reads."""

    t: np.ndarray
    q: np.ndarray
    load_proxy: np.ndarray


#: Strength of the injected unmodelled stiffness, N·m/rad. Chosen so the effect
#: clears the encoder quantum by a wide margin: at 0.6 the extra acceleration moves
#: the one-step prediction by only ~0.3 mrad against a 1.53 mrad quantum, so the
#: metric would correctly report "at the instrument floor" and the test would be
#: asserting that a sub-quantum effect is visible.
_EXTRA_STIFFNESS = 40.0


def _synthetic(*, extra=None, noise=0.0, n=1200, dt=0.0125, seed=3):
    """A capture whose motion obeys the parametric model, plus optional extras.

    Integrated forward with the same one-step rule the fit inverts, so with
    ``extra=None`` and no noise the parametric model explains the motion exactly
    and there is nothing for a residual to absorb.
    """
    rng = np.random.default_rng(seed)
    time = np.arange(n) * dt
    q = np.zeros(n)
    qd = np.zeros(n)
    load = 0.4 * np.sin(2 * np.pi * 0.7 * time) + 0.2 * np.sin(2 * np.pi * 1.9 * time)
    for i in range(n - 1):
        tau = load[i] - _DAMPING * qd[i] - _FRICTION * np.tanh(qd[i] / 0.01)
        if extra is not None:
            tau = tau + extra(q[i], qd[i])
        accel = tau / _INERTIA
        q[i + 1] = q[i] + qd[i] * dt + 0.5 * accel * dt**2
        qd[i + 1] = (q[i + 1] - q[i]) / dt
    measured = q + rng.normal(scale=noise, size=n) if noise else q
    return _Capture(t=time, q=measured, load_proxy=load)


def _score(capture, **kwargs):
    return model_completeness(
        "j",
        capture,
        inertia=_INERTIA,
        damping=_DAMPING,
        friction=_FRICTION,
        quantum_rad=_QUANTUM,
        **kwargs,
    )


def _fit(capture, **kwargs):
    return fit_residual(
        "j",
        capture,
        inertia=_INERTIA,
        damping=_DAMPING,
        friction=_FRICTION,
        quantum_rad=_QUANTUM,
        **kwargs,
    )


# --- the metric responds to structure, not to noise -----------------------


def test_a_capture_the_physics_explains_leaves_nothing_to_absorb():
    score = _score(_synthetic())
    assert score.parametric_rms_rad < _QUANTUM
    assert score.at_instrument_floor
    assert score.absorbed_fraction == pytest.approx(0.0, abs=1e-9)


def test_injected_unmodelled_structure_shows_up_as_parametric_error():
    # A position-dependent stiffness the single-joint model does not carry.
    clean = _score(_synthetic())
    warped = _score(_synthetic(extra=lambda q, qd: -_EXTRA_STIFFNESS * q))
    assert warped.parametric_rms_rad > 20.0 * clean.parametric_rms_rad
    assert not warped.at_instrument_floor


def test_the_residual_absorbs_injected_structure_and_the_gate_fires():
    _model, score = _fit(_synthetic(extra=lambda q, qd: -_EXTRA_STIFFNESS * q))
    assert score.absorbed_fraction > ABSORBED_WARN_FRACTION
    (gate,) = residual_gates([score])
    assert gate.gate == "model_completeness"
    assert gate.level == "warn"
    # The gate has to say what it means, because the intuition runs the wrong way.
    assert "Do not read this as a better model" in gate.remedy


def test_noise_alone_does_not_produce_a_completeness_warning():
    # The metric must not fire on measurement noise: a residual cannot predict
    # white noise, so the absorbed fraction should stay small even though the
    # parametric error is well above the quantum.
    _model, score = _fit(_synthetic(noise=3.0 * _QUANTUM))
    assert not score.at_instrument_floor
    assert score.absorbed_fraction < ABSORBED_WARN_FRACTION
    assert residual_gates([score]) == []


def test_the_velocity_feature_cannot_contain_the_sample_being_predicted():
    """Regression guard on the leak that inflated this metric.

    ``numpy.gradient`` is a central difference, so ``q̇[i]`` is built from
    ``q[i+1]`` — the very target. A residual handed that reconstructs part of
    its own label, and the absorbed fraction then measures the differencing
    scheme.

    Asserted structurally rather than statistically. The obvious statistical
    version — "a leak-free fit cannot absorb white noise" — is not true: where
    the position is noise the parametric one-step extrapolation is itself bad,
    so a residual improves on it substantially with no leakage at all. What is
    checkable exactly is that the velocity is a *backward* difference and that
    perturbing the target leaves every feature untouched.
    """
    from jaxterity.sysid.hybrid import _prepare

    rng = np.random.default_rng(11)
    n = 60
    dt = 0.0125
    raw = np.cumsum(rng.normal(size=n))
    capture = _Capture(t=np.arange(n) * dt, q=raw.copy(), load_proxy=np.zeros(n))
    q_now, qd_now, _load, q_next, step = _prepare(capture, _INERTIA)

    # Backward difference: the velocity at each state uses that state and the one
    # before it — never the target.
    q_prev = raw[: n - 2]
    assert qd_now == pytest.approx((q_now - q_prev) / dt)
    assert q_next == pytest.approx(raw[2:])
    assert step == pytest.approx(dt)

    # And the decisive check: move every target sample and no feature changes.
    perturbed = raw.copy()
    perturbed[2:] += 1000.0
    shifted = _Capture(t=capture.t, q=perturbed, load_proxy=np.zeros(n))
    _q2, qd2, _l2, _n2, _dt2 = _prepare(shifted, _INERTIA)
    assert qd2[0] == pytest.approx(qd_now[0]), (
        "the first state's velocity moved when only later samples changed — the "
        "feature is reading the future"
    )


def test_the_instrument_floor_suppresses_the_gate():
    # At the quantum the residual is fitting quantization, and warning about that
    # would be warning about the encoder.
    at_floor = ModelCompleteness(
        joint="j",
        parametric_rms_rad=_QUANTUM * 0.5,
        hybrid_rms_rad=_QUANTUM * 0.01,
        quantum_rad=_QUANTUM,
        n_samples=1000,
    )
    assert at_floor.absorbed_fraction > 0.9  # would otherwise warn loudly
    assert at_floor.at_instrument_floor
    assert residual_gates([at_floor]) == []


def test_the_hybrid_never_scores_worse_than_the_parametric_model():
    _model, score = _fit(_synthetic(extra=lambda q, qd: -_EXTRA_STIFFNESS * q))
    assert score.hybrid_rms_rad <= score.parametric_rms_rad


def test_without_a_residual_the_reading_is_zero_absorbed_not_a_missing_field():
    score = _score(_synthetic(extra=lambda q, qd: -_EXTRA_STIFFNESS * q))
    assert score.hybrid_rms_rad == score.parametric_rms_rad
    assert score.absorbed_fraction == pytest.approx(0.0)


# --- multi-step rollout ---------------------------------------------------


def test_horizon_one_reproduces_the_single_step_result():
    capture = _synthetic(extra=lambda q, qd: -_EXTRA_STIFFNESS * q)
    assert _score(capture, horizon=1).parametric_rms_rad == pytest.approx(
        _score(capture).parametric_rms_rad
    )


def test_a_longer_rollout_compounds_the_parametric_error():
    capture = _synthetic(extra=lambda q, qd: -_EXTRA_STIFFNESS * q)
    short = _score(capture, horizon=1).parametric_rms_rad
    long_ = _score(capture, horizon=20).parametric_rms_rad
    assert long_ > 10.0 * short


def test_a_residual_on_real_structure_stays_bounded_under_rollout():
    """The discriminating measurement, and it is a *pair* of growth rates.

    Against genuine unmodelled structure the parametric error compounds hard
    while the residual keeps correcting as it is integrated forward, so the two
    growth rates diverge. This is what a longer horizon buys — not a collapse in
    the absorbed fraction, which the noise case below shows stays put.
    """
    capture = _synthetic(extra=lambda q, qd: -_EXTRA_STIFFNESS * q)
    _m1, short = _fit(capture, horizon=1)
    _m2, long_ = _fit(capture, horizon=20)
    parametric_growth = long_.parametric_rms_rad / short.parametric_rms_rad
    hybrid_growth = long_.hybrid_rms_rad / short.hybrid_rms_rad
    assert parametric_growth > 20.0
    assert hybrid_growth < parametric_growth / 3.0


def test_against_noise_both_errors_compound_at_the_same_rate():
    # The contrast: the residual shaves a constant share off and does nothing
    # about the divergence, so the two growth rates track each other.
    capture = _synthetic(noise=3.0 * _QUANTUM)
    _m1, short = _fit(capture, horizon=1)
    _m2, long_ = _fit(capture, horizon=20)
    parametric_growth = long_.parametric_rms_rad / short.parametric_rms_rad
    hybrid_growth = long_.hybrid_rms_rad / short.hybrid_rms_rad
    assert hybrid_growth == pytest.approx(parametric_growth, rel=0.6)
    assert parametric_growth < 5.0


def test_the_absorbed_fraction_alone_does_not_discriminate():
    # Documents why the growth-rate pair is the reading. On noise the absorbed
    # fraction is substantial and roughly horizon-independent, because the
    # residual is cancelling how the damping term amplifies velocity noise.
    capture = _synthetic(noise=3.0 * _QUANTUM)
    fractions = [_fit(capture, horizon=h)[1].absorbed_fraction for h in (1, 20)]
    assert all(0.1 < f < 0.6 for f in fractions), fractions
    assert abs(fractions[0] - fractions[1]) < 0.25


def test_the_horizon_is_recorded_so_two_readings_are_not_confused():
    score = _score(_synthetic(), horizon=7)
    assert score.horizon == 7
    assert score.to_dict()["horizon"] == 7


def test_windows_do_not_overlap():
    # Overlapping windows would count the same error many times and make a long
    # horizon look more precise than it is.
    capture = _synthetic(n=1200)
    usable = _score(capture, horizon=1).n_samples
    for horizon in (2, 5, 20):
        score = _score(capture, horizon=horizon)
        assert score.n_samples == (usable // horizon) * horizon


def test_a_horizon_longer_than_the_capture_is_refused():
    with pytest.raises(CalibrationError, match="fewer than the horizon"):
        _score(_synthetic(n=30), horizon=100)


def test_a_zero_horizon_is_refused():
    with pytest.raises(CalibrationError, match="horizon must be >= 1"):
        _score(_synthetic(n=100), horizon=0)


# --- fairness to the physics ----------------------------------------------


def test_omitting_the_load_calibration_handicaps_the_parametric_model():
    """The bug this test exists to prevent, in numbers.

    Treating the raw load register as newton-metres scales the parametric model's
    torque by an arbitrary factor. It does not make the residual better; it makes
    the physics look worse, and the absorbed fraction is a *ratio* between the two.
    """
    capture = _synthetic()
    fair = _score(capture, k_t=1.0, load_offset=0.0)
    handicapped = _score(capture, k_t=0.18, load_offset=0.0)
    assert handicapped.parametric_rms_rad > 10.0 * fair.parametric_rms_rad


def test_the_load_offset_is_applied():
    capture = _synthetic()
    with_offset = _score(capture, load_offset=0.5)
    assert with_offset.parametric_rms_rad > _score(capture).parametric_rms_rad


# --- refusals -------------------------------------------------------------


def test_a_capture_too_short_for_a_step_is_refused():
    tiny = _Capture(t=np.array([0.0, 0.1]), q=np.zeros(2), load_proxy=np.zeros(2))
    with pytest.raises(CalibrationError, match="at least three"):
        _score(tiny)


def test_non_monotonic_timestamps_are_refused():
    capture = _synthetic(n=50)
    capture.t = np.zeros_like(capture.t)
    with pytest.raises(CalibrationError, match="non-positive sample"):
        _score(capture)


def test_a_non_positive_inertia_is_refused_by_name():
    with pytest.raises(CalibrationError, match="map a torque to an acceleration"):
        model_completeness(
            "j",
            _synthetic(n=50),
            inertia=0.0,
            damping=_DAMPING,
            friction=_FRICTION,
            quantum_rad=_QUANTUM,
        )


# --- reporting ------------------------------------------------------------


def test_to_dict_materializes_the_derived_readings_and_round_trips():
    score = _score(_synthetic(extra=lambda q, qd: -_EXTRA_STIFFNESS * q))
    d = score.to_dict()
    assert json.loads(json.dumps(d)) == d
    assert "absorbed_fraction" in d and "at_instrument_floor" in d


def test_the_fit_reports_its_final_loss_so_convergence_is_checkable():
    model, _score_ = _fit(
        _synthetic(extra=lambda q, qd: -_EXTRA_STIFFNESS * q), steps=50
    )
    assert np.isfinite(model.final_loss)
    assert model.hidden == 8


# --- the real unit --------------------------------------------------------


@pytest.mark.slow
def test_the_real_captures_physics_largely_suffices():
    """On LK-0001 the parametric model is within ~2 encoder quanta.

    Which is what a healthy unit should look like, and the reason this metric is
    worth reporting: it does not cry wolf on a robot that is fine.
    """
    robot = zoo.load("so100")
    fields = JointCalibration.__dataclass_fields__
    rows = json.loads(pathlib.Path(_CAL).read_text())["calibrations"]
    cals = [
        JointCalibration(**{k: v for k, v in c.items() if k in fields}) for c in rows
    ]
    doc = json.loads((_DATA / "commanded_shoulder_pan.json").read_text())
    capture = load_commanded_capture(doc, robot, calibration=_CAL)
    result = identify_closed_loop(robot, capture)
    (row,) = [c for c in cals if c.joint == "shoulder_pan"]
    score = model_completeness(
        "shoulder_pan",
        capture,
        inertia=axis_inertia_feature(robot, cals, "shoulder_pan")(()),
        damping=float(result.fitted.get("shoulder_pan.damping", 0.0)),
        friction=float(result.fitted.get("shoulder_pan.friction", 0.0)),
        k_t=result.residuals["K_t"],
        load_offset=result.residuals["load_offset"],
        quantum_rad=1.0 / row.ticks_per_rad,
    )
    quanta = score.parametric_rms_rad / score.quantum_rad
    assert quanta < 4.0, f"parametric model is {quanta:.1f} encoder quanta off"
