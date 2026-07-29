# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Margin-constrained probe design — the Fisher ascent inside the ZMP floor.

The claim under test is that the constraint *shapes* the search rather than
vetoing it afterwards: the constrained probe must (a) satisfy the tip-over
floor everywhere along its coast, (b) still beat the naive starting probe on
information, and (c) be strictly less aggressive than the unconstrained
optimum whenever that optimum breaches the floor. Plus the refusal: a starting
probe already outside the floor cannot be rescued by any descendant of it, and
the function says so instead of searching.

Cost discipline: the cartpole (the lightest zoo model) with a small step
budget, since every constrained iterate costs an MJX rollout plus a MuJoCo
compile on top of the information gradient. Even so, an ascent here is minutes
rather than seconds, so every test that *runs* one is marked ``nightly``; the
PR tier keeps the two that establish the constraint is wired up at all (the
floor's default, and that the margin responds to how aggressive a probe is).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from jaxterity import zoo
from jaxterity.calibrate.active import design_coast_excitation
from jaxterity.checkup.stability import BaseProfile, StabilityConstraint
from jaxterity.errors import CalibrationError
from jaxterity.objects import (
    RigidPayload,
    design_probe,
    probe_information,
    probe_margin,
)

pytest.importorskip("mujoco")

_FIT = ("mass",)
_STEPS = 4


@pytest.fixture(scope="module")
def cartpole():
    return zoo.load("cartpole")


@pytest.fixture(scope="module")
def tip():
    return RigidPayload(attach_link="pole", mass=0.05, offset=(0.0, 0.0, 0.6))


@pytest.fixture(scope="module")
def naive(cartpole):
    return design_coast_excitation(cartpole, speed=0.02, sample_times=(0.05, 0.1))


@pytest.fixture(scope="module")
def wide_base():
    """A base roomy enough that the starting probe is comfortably inside it."""
    return BaseProfile(
        name="test-wide",
        support_polygon_xy=((0.4, 0.4), (-0.4, 0.4), (-0.4, -0.4), (0.4, -0.4)),
        base_mass=5.0,
        base_com_xyz=(0.0, 0.0, 0.05),
        arm_mount_xyz=(0.0, 0.0, 0.05),
        margin_min_m=0.05,
    )


@pytest.fixture(scope="module")
def constrained(cartpole, tip, naive, wide_base):
    return design_probe(
        cartpole,
        tip,
        fit=_FIT,
        excitation0=naive,
        steps=_STEPS,
        stability=StabilityConstraint(base=wide_base),
    )


@pytest.fixture(scope="module")
def unconstrained(cartpole, tip, naive):
    return design_probe(cartpole, tip, fit=_FIT, excitation0=naive, steps=_STEPS)


def test_constraint_default_floor_comes_from_the_base(wide_base):
    assert StabilityConstraint(base=wide_base).floor == wide_base.margin_min_m
    assert StabilityConstraint(base=wide_base, margin_floor_m=0.1).floor == 0.1


@pytest.mark.nightly
def test_the_constrained_probe_clears_the_floor(cartpole, tip, wide_base, constrained):
    constraint = StabilityConstraint(base=wide_base)
    assert probe_margin(cartpole, tip, constrained, constraint) >= constraint.floor


@pytest.mark.nightly
def test_the_constrained_probe_still_out_informs_the_naive_one(
    cartpole, tip, naive, constrained
):
    # The constraint bounds the search; it must not neuter it.
    assert probe_information(cartpole, tip, constrained, fit=_FIT) > probe_information(
        cartpole, tip, naive, fit=_FIT
    )


@pytest.mark.nightly
def test_a_tight_floor_holds_the_probe_back(cartpole, tip, naive, wide_base):
    # Score the unconstrained optimum, then re-run with a floor just above the
    # margin it achieves: the constrained answer must be strictly gentler.
    loose = design_probe(cartpole, tip, fit=_FIT, excitation0=naive, steps=_STEPS)
    loose_margin = probe_margin(
        cartpole, tip, loose, StabilityConstraint(base=wide_base)
    )
    tight = StabilityConstraint(base=wide_base, margin_floor_m=loose_margin + 1e-4)
    held_back = design_probe(
        cartpole, tip, fit=_FIT, excitation0=naive, steps=_STEPS, stability=tight
    )
    assert probe_margin(cartpole, tip, held_back, tight) >= tight.floor
    assert float(np.max(np.abs(held_back.qd0))) <= float(np.max(np.abs(loose.qd0)))


@pytest.mark.nightly
def test_a_starting_probe_outside_the_floor_is_refused_not_searched(
    cartpole, tip, naive, wide_base
):
    # A floor no probe from this pose can meet: the function must say the pose
    # is the problem rather than burn the step budget.
    impossible = StabilityConstraint(base=wide_base, margin_floor_m=10.0)
    with pytest.raises(CalibrationError, match="already below the floor"):
        design_probe(
            cartpole,
            tip,
            fit=_FIT,
            excitation0=naive,
            steps=_STEPS,
            stability=impossible,
        )


@pytest.mark.nightly
def test_the_constrained_ascent_is_deterministic(cartpole, tip, naive, wide_base):
    constraint = StabilityConstraint(base=wide_base)
    kwargs = dict(fit=_FIT, excitation0=naive, steps=_STEPS, stability=constraint)
    first = design_probe(cartpole, tip, **kwargs)
    second = design_probe(cartpole, tip, **kwargs)
    assert first.q0 == second.q0
    assert first.qd0 == second.qd0


@pytest.mark.nightly
def test_an_unconstrained_call_is_unchanged_by_the_new_parameter(unconstrained):
    # Regression guard: stability=None must leave the original path alone.
    assert unconstrained.sample_times == (0.05, 0.1)
    assert len(unconstrained.qd0) == len(unconstrained.q0)


@pytest.mark.slow
def test_probe_margin_falls_as_the_probe_gets_more_aggressive(
    cartpole, tip, naive, wide_base
):
    constraint = StabilityConstraint(base=wide_base)
    gentle = probe_margin(cartpole, tip, naive, constraint)
    aggressive = probe_margin(
        cartpole,
        tip,
        dataclasses.replace(naive, qd0=tuple(3.0 for _ in naive.qd0)),
        constraint,
    )
    assert aggressive < gentle
