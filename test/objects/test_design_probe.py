# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Differentiable probe design — the optimized heft must out-inform the naive
one, deterministically, and never leave the joint-limit envelope.

Cost discipline: everything runs on the cartpole (the lightest zoo model);
each distinct probe/fit pairing is an XLA compile of an MJX simulate.
"""

from __future__ import annotations

import numpy as np
import pytest

from jaxterity import zoo
from jaxterity.calibrate.active import (
    design_coast_excitation,
    validate_excitation,
)
from jaxterity.errors import CalibrationError
from jaxterity.objects import (
    PayloadLoopbackLink,
    RigidPayload,
    attach_payload,
    design_probe,
    identify_payload,
    probe_information,
)

pytest.importorskip("mujoco")

_TRUE_TIP_MASS = 0.2
_NOISE = 1e-4
_FIT = ("mass",)


@pytest.fixture(scope="module")
def cartpole():
    return zoo.load("cartpole")


@pytest.fixture(scope="module")
def tip(cartpole):
    return RigidPayload(attach_link="pole", mass=0.05, offset=(0.0, 0.0, 0.6))


@pytest.fixture(scope="module")
def naive(cartpole):
    """A deliberately weak starting probe: a slow kick, early samples."""
    return design_coast_excitation(cartpole, speed=0.02, sample_times=(0.05, 0.1))


@pytest.fixture(scope="module")
def optimized(cartpole, tip, naive):
    return design_probe(cartpole, tip, fit=_FIT, excitation0=naive)


def test_optimized_probe_is_more_informative(cartpole, tip, naive, optimized):
    """The ascent's whole point: strictly more Fisher information."""
    info_naive = probe_information(cartpole, tip, naive, fit=_FIT)
    info_opt = probe_information(cartpole, tip, optimized, fit=_FIT)
    assert info_opt > info_naive


def test_optimized_probe_beats_naive_on_final_covariance(
    cartpole, tip, naive, optimized
):
    """The kill-gate: a measurably tighter posterior from the same fit."""

    def covariance(excitation):
        link = PayloadLoopbackLink(
            cartpole, tip, true_mass=_TRUE_TIP_MASS, seed=3, noise_std=_NOISE
        )
        result = identify_payload(cartpole, link, tip, excitation=excitation, fit=_FIT)
        assert result.covariance is not None
        return float(result.covariance["payload.mass"])

    assert covariance(optimized) < covariance(naive)


def test_design_probe_is_deterministic(cartpole, tip, naive, optimized):
    """No RNG anywhere: the same inputs return bit-identical probes."""
    again = design_probe(cartpole, tip, fit=_FIT, excitation0=naive)
    assert again.q0 == optimized.q0
    assert again.qd0 == optimized.qd0
    assert again.sample_times == optimized.sample_times


def test_optimized_probe_respects_limits(cartpole, tip, optimized):
    """The returned probe passes the same hard gate as any excitation."""
    attached = attach_payload(cartpole, tip)
    validate_excitation(attached, optimized)  # must not raise


def test_prior_covariance_shifts_the_objective(cartpole, tip, naive):
    """A previous fit's covariance folds in (Bayesian D-optimal) and changes
    the information score."""
    flat = probe_information(cartpole, tip, naive, fit=_FIT)
    with_prior = probe_information(
        cartpole, tip, naive, fit=_FIT, prior_covariance={"payload.mass": 1e-4}
    )
    assert with_prior != flat
    assert with_prior > flat  # extra (prior) information can only add


def test_unknown_objective_is_refused(cartpole, tip):
    with pytest.raises(CalibrationError, match="objective"):
        design_probe(cartpole, tip, objective="vibes-optimal")


def test_probe_moves_from_the_start(naive, optimized):
    """The ascent actually moved the probe (not a no-op wrapper)."""
    assert not np.allclose(
        np.asarray(optimized.qd0), np.asarray(naive.qd0)
    ) or not np.allclose(np.asarray(optimized.q0), np.asarray(naive.q0))
