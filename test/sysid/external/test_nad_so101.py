# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""External validation of model completeness against NAD's SO-101 subset.

The claim :mod:`jaxterity.sysid.hybrid` makes is that a **growing parametric
residual means something the model does not represent is present**. Every test of
that claim so far has been on synthetic data (where the extra term is one we
injected ourselves) or on LK-0001 (our own unit, our own capture protocol, and no
ground truth about what is missing).

NAD supplies the missing ingredient: the same arm, logged by other people on
another unit, with the payload deliberately stepped **empty → 200 g → 300 g →
400 g → 500 g**. Our SO-101 model carries no payload term, so the payload is
*labelled unmodelled physics* — a known quantity of exactly the thing the metric
claims to detect, with the label supplied by someone with no stake in our result.

The test fits the parametric parameters **once, on the empty-payload training
trajectories only**, then holds them fixed and scores the held-out test
trajectory of every payload. If the metric works, its error must rise with mass
it was never told about.

**Opt-in / Mode B.** NAD is not vendored (the upstream repository declares no
license — see ``fetch_nad.py``), so this test *skips* unless the data is present.

    python test/sysid/external/fetch_nad.py
    pytest -m external_data

Recorded outcomes live in ``RESULTS.md``.

Citation: Z. Dou et al., "NeuralActuator: Neural Actuation Modeling for Robot
Dynamics and External Force Perception", RSS 2026 (arXiv 2607.11734).
"""

from __future__ import annotations

import csv
import dataclasses
from pathlib import Path

import numpy as np
import pytest

from jaxterity.sysid import model_completeness

pytestmark = pytest.mark.external_data

DATA_DIR = Path(__file__).resolve().parent / "_nad_data" / "so101"

#: The joint the payload actually loads. SO-101 joint 2 is the shoulder lift: it
#: carries the arm and whatever the gripper is holding against gravity, so a mass
#: at the end effector shows up there. Joint 1 is the base yaw, whose gravity
#: torque is identically zero, which is the same reason
#: :func:`jaxterity.calibrate.axis_inertia_feature` exists.
JOINT_INDEX = 2
JOINT_NAME = "shoulder_lift"

#: The task with a quasi-static hold, so the trajectories are comparable across
#: payloads. ``pick_and_place`` moves through different poses per run, which would
#: confound payload with pose (see T-235).
TASK = "go_up_and_stay_still"
PAYLOADS = ("empty", "200g", "300g", "400g", "500g")
PAYLOAD_GRAMS = {"empty": 0, "200g": 200, "300g": 300, "400g": 400, "500g": 500}

#: Shoulder-lift axis inertia, kg·m². Approximate and *deliberately held fixed*:
#: it scales both the parametric and hybrid errors identically, so the monotonic
#: trend this test asserts is unaffected by getting it somewhat wrong. It is not
#: taken from our own URDF because the point is to avoid tuning anything per
#: payload.
INERTIA = 0.02

#: STS3215 encoder quantum, rad (4096 counts / 2*pi).
QUANTUM_RAD = 2.0 * np.pi / 4096.0

#: Coulomb velocity scale, matching the closed-loop fit's default.
COULOMB_EPS = 0.01


@dataclasses.dataclass
class _Capture:
    """The capture surface :func:`model_completeness` reads."""

    t: np.ndarray
    q: np.ndarray
    load_proxy: np.ndarray


def _require_data() -> None:
    if not DATA_DIR.is_dir():
        pytest.skip(
            "NAD SO-101 subset not present. Fetch it with "
            "`python test/sysid/external/fetch_nad.py`."
        )


def _load(split: str, payload: str) -> list[_Capture]:
    directory = DATA_DIR / TASK / payload / split
    captures = []
    for path in sorted(directory.glob("*.csv")):
        rows = list(csv.DictReader(path.open()))
        captures.append(
            _Capture(
                t=np.array([float(r["timestamp"]) for r in rows]),
                q=np.array([float(r[f"pos{JOINT_INDEX}"]) for r in rows]),
                load_proxy=np.array([float(r[f"load{JOINT_INDEX}"]) for r in rows]),
            )
        )
    return captures


def _fit_parametric(captures: list[_Capture]) -> dict[str, float]:
    """Least squares for ``(k_t, load_offset, damping, friction)``.

    In the one-step position domain, so it is fitted against the same quantity
    the completeness metric scores: the residual position error is linear in the
    torque, and the torque is linear in all four parameters.
    """
    design, target = [], []
    for capture in captures:
        dt = float(np.median(np.diff(capture.t)))
        velocity = np.diff(capture.q) / np.diff(capture.t)
        q, qd = capture.q[1:-1], velocity[:-1]
        load, q_next = capture.load_proxy[1:-1], capture.q[2:]
        target.append((q_next - q - qd * dt) * 2.0 * INERTIA / dt**2)
        design.append(
            np.column_stack([load, np.ones_like(load), -qd, -np.tanh(qd / COULOMB_EPS)])
        )
    coefficients, *_ = np.linalg.lstsq(
        np.vstack(design), np.concatenate(target), rcond=None
    )
    return {
        "k_t": float(coefficients[0]),
        "load_offset": float(coefficients[1]),
        "damping": float(coefficients[2]),
        "friction": float(coefficients[3]),
    }


@pytest.fixture(scope="module")
def parameters():
    """Fitted on the **empty** payload's training set, and never refitted."""
    _require_data()
    return _fit_parametric(_load("train", "empty"))


def _score(payload: str, parameters: dict[str, float], horizon: int) -> float:
    """Mean parametric position RMS over a payload's held-out test set."""
    errors = [
        model_completeness(
            JOINT_NAME,
            capture,
            inertia=INERTIA,
            quantum_rad=QUANTUM_RAD,
            coulomb_eps=COULOMB_EPS,
            horizon=horizon,
            **parameters,
        ).parametric_rms_rad
        for capture in _load("test", payload)
    ]
    return float(np.mean(errors))


@pytest.fixture(scope="module")
def curve(parameters):
    _require_data()
    return {
        horizon: {p: _score(p, parameters, horizon) for p in PAYLOADS}
        for horizon in (1, 10)
    }


def test_the_dataset_is_the_one_that_was_pinned():
    _require_data()
    from fetch_nad import present_and_valid  # noqa: PLC0415

    assert present_and_valid(DATA_DIR.parent), (
        "the local NAD copy does not match the pinned manifest; re-fetch it"
    )


def test_the_parametric_fit_on_empty_is_at_the_instrument_floor(parameters):
    """Precondition: our model does fit their empty-payload data.

    Without this the rest proves nothing — a model that fits nothing would also
    show error rising with payload.
    """
    _require_data()
    empty = _score("empty", parameters, horizon=1)
    assert empty < 2.0 * QUANTUM_RAD, (
        f"one-step error on the empty payload is {empty * 1e3:.3f} mrad, "
        f"{empty / QUANTUM_RAD:.1f} encoder quanta — the model does not fit "
        "their data, so the payload trend below would not be attributable"
    )


def test_parametric_error_rises_monotonically_with_unmodelled_payload(curve):
    """The validation. Ground truth supplied by someone else's labels."""
    _require_data()
    for horizon, scores in curve.items():
        ordered = [scores[p] for p in PAYLOADS]
        assert ordered == sorted(ordered), (
            f"horizon {horizon}: error did not rise monotonically with payload: "
            + ", ".join(f"{p}={scores[p] * 1e3:.3f}mrad" for p in PAYLOADS)
        )


def test_the_heaviest_payload_is_clearly_separated_from_empty(curve):
    _require_data()
    assert curve[1]["500g"] > 1.5 * curve[1]["empty"]
    assert curve[10]["500g"] > 2.5 * curve[10]["empty"]


def test_the_rollout_amplifies_the_signal(curve):
    """Independent confirmation that the multi-step horizon earns its keep.

    Compounding the error over ten steps separates the payloads further than
    single-step prediction does — on data that has nothing to do with the
    synthetic captures the horizon was developed against.
    """
    _require_data()
    single = curve[1]["500g"] / curve[1]["empty"]
    rolled = curve[10]["500g"] / curve[10]["empty"]
    assert rolled > single


def test_the_trend_tracks_mass_rather_than_merely_being_ordered(curve):
    # Ordering alone could come from anything monotone in run index. A positive
    # correlation against the actual gram values is the stronger statement.
    _require_data()
    grams = np.array([PAYLOAD_GRAMS[p] for p in PAYLOADS], dtype=float)
    for horizon, scores in curve.items():
        errors = np.array([scores[p] for p in PAYLOADS])
        correlation = float(np.corrcoef(grams, errors)[0, 1])
        assert correlation > 0.9, f"horizon {horizon}: r = {correlation:.3f}"
