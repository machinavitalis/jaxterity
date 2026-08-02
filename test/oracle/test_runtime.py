# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""The oracle: simulate to answer, with the answer's dispersion attached.

The contract under test, in order: a plan is well-formed or refused; the
posterior is only as wide as the fit that was actually attached; an answer
carries the worst case *and* the fraction of the posterior behind it; a
diverged integration is raised rather than reported as a margin; alternatives
are ranked; and the coupled tip-over screen joins the margin set when a base is
declared.

Cost discipline: the two-DOF cartpole twin carries the thermal/torque
questions; the LeKiwi (arm + shared battery) twin appears once, for the bus and
tip-over paths, as a module-scoped fixture.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaxterity import zoo
from jaxterity.calibrate.active import design_coast_excitation
from jaxterity.checkup.stability import LEKIWI_BASE
from jaxterity.errors import JaxterityError
from jaxterity.objects import PayloadLoopbackLink, RigidPayload, SemanticPrior
from jaxterity.oracle import (
    TIP_OVER,
    envelope_from_mapping,
    hold_plan,
    identify_object,
    posterior_twins,
    simulate_plan,
    what_if,
)
from jaxterity.selfcheck import SafetyEnvelope
from jaxterity.twin import Twin, lekiwi_twin

pytest.importorskip("mujoco")

#: Representative thermal leaves for the cartpole test twin.
_C_TH, _R_TH, _K_PWR = 15.0, 4.0, 6.0
#: A ceiling the hot hold crosses and the gentle one does not, K.
_CEILING_K = 320.0


@pytest.fixture(scope="module")
def cart_twin():
    robot = zoo.calibrated("cartpole")
    n = robot.build_system(actuation=True)._nv
    return Twin(
        robot,
        serial="CP-ORACLE",
        C_th=jnp.full(n, _C_TH),
        R_th=jnp.full(n, _R_TH),
        k_pwr=jnp.full(n, _K_PWR),
    )


@pytest.fixture(scope="module")
def uncertain_twin(cart_twin):
    """The same unit, with a fit's covariance attached (a real posterior)."""
    n = cart_twin.n_servos
    return cart_twin.with_thermal(
        param_uncertainty=(
            ("offbody.R_th", tuple([1.0] * n)),
            ("offbody.k_pwr", tuple([2.0] * n)),
        )
    )


@pytest.fixture(scope="module")
def envelope():
    return SafetyEnvelope(temp_ceiling_k=_CEILING_K, torque_margin_frac=0.9)


@pytest.fixture(scope="module")
def gentle(cart_twin):
    return hold_plan(
        "gentle", cart_twin, [0.05] * cart_twin.n_servos, n_steps=20, dt=0.01
    )


@pytest.fixture(scope="module")
def hot(cart_twin):
    return hold_plan("hot", cart_twin, [1.5] * cart_twin.n_servos, n_steps=20, dt=0.01)


# --- plans ------------------------------------------------------------------


def test_hold_plan_shapes_and_defaults(cart_twin):
    plan = hold_plan("hold", cart_twin, [0.1, 0.2], n_steps=5, dt=0.01)
    assert plan.n_steps == 5
    assert plan.duration_s == pytest.approx(0.05)
    assert len(plan.tau[0]) == cart_twin.n_servos
    # Temperature starts at the unit's ambient unless told otherwise.
    assert plan.T0 == tuple([cart_twin.T_amb] * cart_twin.n_servos)


def test_hold_plan_rejects_wrong_width(cart_twin):
    with pytest.raises(JaxterityError, match="one commanded torque per DOF"):
        hold_plan("bad", cart_twin, [0.1, 0.2, 0.3], n_steps=2, dt=0.01)


# --- the posterior ----------------------------------------------------------


def test_posterior_without_covariance_is_a_single_point(cart_twin):
    """No fit uncertainty → no band, and the answer says so instead of faking one."""
    draws, note = posterior_twins(cart_twin, 8, seed=0)
    assert len(draws) == 1
    assert "no fitted off-body covariance" in note


def test_posterior_draws_are_distinct_physical_and_deterministic(uncertain_twin):
    draws, note = posterior_twins(uncertain_twin, 6, seed=3)
    assert len(draws) == 6
    assert "R_th" in note and "k_pwr" in note
    r_th = np.array([float(d.R_th[0]) for d in draws])
    assert np.unique(r_th).size == r_th.size
    assert np.all(r_th > 0.0)  # clipped to stay physical
    again, _ = posterior_twins(uncertain_twin, 6, seed=3)
    assert np.allclose(r_th, [float(d.R_th[0]) for d in again])
    # The unsampled leaves stay at their point value.
    assert np.allclose(np.asarray(draws[0].C_th), np.asarray(uncertain_twin.C_th))


# --- answers ----------------------------------------------------------------


def test_gentle_hold_passes_and_binds_to_the_unit(cart_twin, gentle, envelope):
    answer = simulate_plan(cart_twin, gentle, envelope, n_draws=4)
    assert answer.ok and answer.outcome == "pass"
    assert answer.pass_fraction == 1.0
    assert answer.twin_handle == cart_twin.attestation_handle
    assert {b.name for b in answer.margins} == {"servo_temperature", "torque"}
    assert answer.margin("servo_temperature").units == "K"


def test_hot_hold_is_vetoed_with_a_time_to_breach(cart_twin, hot, envelope):
    answer = simulate_plan(cart_twin, hot, envelope, n_draws=1)
    assert answer.outcome == "veto"
    assert answer.binding_constraint == "servo_temperature"
    assert answer.time_to_breach_s is not None and answer.time_to_breach_s > 0.0
    assert answer.margin("servo_temperature").worst > _CEILING_K


def test_dispersion_is_the_honesty(uncertain_twin, envelope):
    """A wide fit gives a wide band — and separates 'no' from 'we cannot tell'.

    The hold is chosen so the ceiling sits *inside* the posterior's steady-state
    spread: the same command is safe for the lucky end of this unit's own
    parameter interval and unsafe for the unlucky end, which is precisely the
    case a point prediction cannot express.
    """
    n = uncertain_twin.n_servos
    warm = hold_plan("warm", uncertain_twin, [1.0] * n, n_steps=20, dt=0.01)
    answer = simulate_plan(uncertain_twin, warm, envelope, n_draws=12, seed=1)
    band = answer.margin("servo_temperature")
    assert band.hi > band.lo  # the posterior actually spread the outcome
    assert band.worst >= band.median
    # Some draws survive the ceiling the worst case crosses.
    assert 0.0 < answer.pass_fraction < 1.0
    assert answer.outcome == "veto"


def test_point_twin_answer_has_zero_width_band(cart_twin, hot, envelope):
    answer = simulate_plan(cart_twin, hot, envelope, n_draws=8)
    band = answer.margin("servo_temperature")
    assert band.lo == pytest.approx(band.hi)
    assert answer.n_draws == 1


def test_diverged_rollout_is_raised_not_reported(cart_twin, envelope):
    """A fixed-step blow-up is an integration failure, not a physical verdict."""
    coarse = hold_plan(
        "coarse", cart_twin, [5.0] * cart_twin.n_servos, n_steps=40, dt=0.5
    )
    with pytest.raises(JaxterityError, match="diverged"):
        simulate_plan(cart_twin, coarse, envelope, n_draws=1)


def test_empty_and_malformed_questions_are_refused(cart_twin, gentle, envelope):
    with pytest.raises(JaxterityError, match="coverage"):
        simulate_plan(cart_twin, gentle, envelope, coverage=1.5)
    wide = hold_plan("wide", cart_twin, [0.1] * cart_twin.n_servos, n_steps=2, dt=0.01)
    mismatched = type(wide)(
        name="mismatched",
        q0=wide.q0,
        qd0=(0.0,),
        T0=wide.T0,
        tau=wide.tau,
        dt=wide.dt,
    )
    with pytest.raises(JaxterityError, match="servos"):
        simulate_plan(cart_twin, mismatched, envelope, n_draws=1)


# --- comparison -------------------------------------------------------------


def test_what_if_ranks_the_gentler_plan_first(cart_twin, gentle, hot, envelope):
    comparison = what_if(cart_twin, [hot, gentle], envelope, n_draws=1)
    assert comparison.best == "gentle"
    assert comparison.ranking == ("gentle", "hot")
    assert comparison.answer("hot").outcome == "veto"
    assert "gentle" in comparison.detail


def test_what_if_requires_unique_named_candidates(cart_twin, gentle, envelope):
    with pytest.raises(JaxterityError, match="unique"):
        what_if(cart_twin, [gentle, gentle], envelope, n_draws=1)
    with pytest.raises(JaxterityError, match="at least one"):
        what_if(cart_twin, [], envelope)


# --- envelope adapter -------------------------------------------------------


def test_envelope_mapping_refuses_a_mistyped_limit():
    env = envelope_from_mapping({"temp_ceiling_k": 340.0, "bus_floor_v": 11.0})
    assert env.temp_ceiling_k == 340.0
    with pytest.raises(JaxterityError, match="unknown limit"):
        envelope_from_mapping({"temp_ceiling": 340.0})


# --- the coupled tip-over screen -------------------------------------------


@pytest.fixture(scope="module")
def mobile_twin():
    return lekiwi_twin("LK-ORACLE")


def test_declared_base_adds_a_tip_over_margin(mobile_twin):
    """With a base declared, the answer covers what a joint-limit check cannot."""
    plan = hold_plan(
        "arm free swing",
        mobile_twin,
        [0.02] * mobile_twin.n_servos,
        n_steps=40,
        dt=0.005,
    )
    envelope = SafetyEnvelope(temp_ceiling_k=343.15, bus_floor_v=11.0)
    answer = simulate_plan(mobile_twin, plan, envelope, base=LEKIWI_BASE, n_draws=1)
    band = answer.margin(TIP_OVER)
    assert band is not None and band.units == "m"
    # The unsupported arm falls forward off the base: this is the T-229 finding.
    assert answer.outcome == "veto"
    assert answer.binding_constraint == TIP_OVER
    assert band.worst < LEKIWI_BASE.margin_min_m


# --- the identification verb ------------------------------------------------


def test_identify_object_reports_params_intervals_and_provenance():
    cartpole = zoo.load("cartpole")
    tip = RigidPayload(attach_link="pole", mass=0.05, offset=(0.0, 0.0, 0.6))
    link = PayloadLoopbackLink(cartpole, tip, true_mass=0.2)
    answer = identify_object(
        cartpole,
        link,
        tip,
        prior=SemanticPrior(label="tip mass", mass_kg=(0.12, 0.35), source="vlm/test"),
        excitation=design_coast_excitation(cartpole, speed=0.3),
        fit=("mass",),
        uncertainty="laplace",
    )
    assert answer.params["payload.mass"] == pytest.approx(0.2, abs=1e-3)
    assert answer.intervals is not None
    lo, hi = answer.intervals["payload.mass"]
    assert lo <= answer.params["payload.mass"] <= hi
    assert answer.recipe == "objects.payload+semantic_prior"
    assert any("semantic prior" in w for w in answer.warnings)
    assert "payload.mass" in answer.summary()
