# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Scene enrollment: vet under the worst case, heft, load the twin, vet the job.

The claims under test are the ones the arc rests on: the vetting assumes the
*heaviest* object the prior allows (never the guess's midpoint), a pose that
cannot be made safe is refused rather than probed slowly, the heft tightens the
prior's bracket into an interval, the loaded unit is a different attested
configuration, and the technique verdict is taken on the loaded unit — the one
that will actually move.

Cost discipline: the full enrollment runs **once** (module-scoped) and several
tests assert against it; the refusal path only pays for probe simulations.
"""

from __future__ import annotations

import numpy as np
import pytest

from jaxterity.checkup.stability import LEKIWI_BASE, StabilityConstraint
from jaxterity.errors import CalibrationError
from jaxterity.objects import (
    PayloadLoopbackLink,
    RigidPayload,
    SemanticPrior,
    enroll_object,
    vet_probe,
    worst_case_payload,
)
from jaxterity.oracle import TIP_OVER, hold_plan
from jaxterity.selfcheck import SafetyEnvelope
from jaxterity.twin import lekiwi_twin

pytest.importorskip("mujoco")

#: A tucked pose whose static tip-over margin clears the LeKiwi floor.
_TUCKED = (0.0, -1.2, -0.8, 0.0, 0.0, 0.0)
#: The model's zero pose — extended forward, the 2026-07-28 tip-over pose.
_EXTENDED = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
#: What the simulated gripper is really holding, kg.
_PLANTED_MASS = 0.42
#: Probe speeds tried in the tests (short list — each is an MJX rollout).
_SPEEDS = (0.30, 0.10)


@pytest.fixture(scope="module")
def twin():
    return lekiwi_twin("LK-SCENE-TEST")


@pytest.fixture(scope="module")
def payload():
    return RigidPayload("gripper_frame_link", mass=0.1, name="tote")


@pytest.fixture(scope="module")
def prior():
    return SemanticPrior.from_material(
        "small plastic tote", "plastic", (0.10, 0.08, 0.06), source="vlm/test"
    )


@pytest.fixture(scope="module")
def stability():
    return StabilityConstraint(base=LEKIWI_BASE)


@pytest.fixture(scope="module")
def enrollment(twin, payload, prior, stability):
    link = PayloadLoopbackLink(twin.robot, prior.seed(payload), true_mass=_PLANTED_MASS)
    techniques = [
        hold_plan(
            "brisk", twin, [0.3] * twin.n_servos, n_steps=30, dt=0.005, q0=_TUCKED
        ),
        hold_plan(
            "gentle", twin, [0.05] * twin.n_servos, n_steps=30, dt=0.005, q0=_TUCKED
        ),
    ]
    return enroll_object(
        twin,
        link,
        payload,
        prior,
        techniques,
        SafetyEnvelope(temp_ceiling_k=343.15, bus_floor_v=11.0),
        stability=stability,
        speeds=_SPEEDS,
        pose_rad=_TUCKED,
        n_draws=2,
    )


# --- the worst case ---------------------------------------------------------


def test_worst_case_takes_the_prior_upper_bound(payload, prior):
    """Vetting never assumes the guess's midpoint — the heavy end is the risk."""
    heavy = worst_case_payload(payload, prior)
    assert float(heavy.mass) == pytest.approx(prior.mass_kg[1])
    assert float(heavy.mass) > prior.seed_mass
    assert heavy.attach_link == payload.attach_link and heavy.name == payload.name


def test_vetting_reports_the_worst_case_it_assumed(enrollment, prior):
    assert enrollment.vetting.approved
    assert enrollment.vetting.worst_case_mass_kg == pytest.approx(prior.mass_kg[1])
    assert enrollment.vetting.margin_m >= enrollment.vetting.floor_m


def test_an_unsafe_pose_is_refused_not_probed_slowly(twin, payload, prior, stability):
    """From the extended pose no probe speed clears the floor — say so."""
    vetting = vet_probe(
        twin.robot, payload, prior, stability, speeds=_SPEEDS, pose_rad=_EXTENDED
    )
    assert not vetting.approved
    assert vetting.margin_m < vetting.floor_m
    assert "no probe approved" in vetting.detail
    # And the enrollment stops there rather than hefting an object that may tip
    # the unit.
    link = PayloadLoopbackLink(twin.robot, prior.seed(payload), true_mass=_PLANTED_MASS)
    with pytest.raises(CalibrationError, match="Re-pose the arm"):
        enroll_object(
            twin,
            link,
            payload,
            prior,
            [hold_plan("x", twin, [0.05] * twin.n_servos, n_steps=4, dt=0.005)],
            SafetyEnvelope(temp_ceiling_k=343.15),
            stability=stability,
            speeds=_SPEEDS,
            pose_rad=_EXTENDED,
            n_draws=1,
        )


def test_enrollment_needs_a_technique_to_vet(twin, payload, prior, stability):
    link = PayloadLoopbackLink(twin.robot, prior.seed(payload), true_mass=_PLANTED_MASS)
    with pytest.raises(CalibrationError, match="at least one candidate technique"):
        enroll_object(
            twin,
            link,
            payload,
            prior,
            [],
            SafetyEnvelope(temp_ceiling_k=343.15),
            stability=stability,
            pose_rad=_TUCKED,
        )


# --- the heft ---------------------------------------------------------------


def test_heft_tightens_the_prior_into_an_interval(enrollment, prior):
    """The bracket goes in; a measurement with an interval comes out."""
    mass = enrollment.identification.params["tote.mass"]
    assert mass == pytest.approx(_PLANTED_MASS, abs=5e-3)
    assert prior.mass_kg[0] < mass < prior.mass_kg[1]
    lo, hi = enrollment.identification.intervals["tote.mass"]
    assert lo <= mass <= hi
    # The measurement is much sharper than the guess that seeded it.
    assert (hi - lo) < 0.1 * (prior.mass_kg[1] - prior.mass_kg[0])


def test_the_seeded_fit_declares_itself(enrollment):
    assert enrollment.identification.recipe == "objects.payload+semantic_prior"
    assert any("semantic prior" in w for w in enrollment.identification.warnings)


# --- the mini-twin and the coupled verdict ---------------------------------


def test_loaded_unit_is_a_new_attested_configuration(enrollment, twin):
    assert enrollment.twin_handle != twin.attestation_handle
    assert enrollment.loaded_twin.serial == twin.serial
    # The object is a real body in the loaded model.
    assert "tote" in enrollment.loaded_twin.robot.link_map
    assert float(enrollment.loaded_twin.robot.link_map["tote"].mass) == pytest.approx(
        enrollment.identification.params["tote.mass"], abs=5e-3
    )


def test_the_technique_is_vetted_on_the_loaded_unit(enrollment):
    """Tip-over with the load is what separates the two techniques."""
    assert enrollment.technique == "gentle"
    assert enrollment.comparison.ranking == ("gentle", "brisk")
    rejected = enrollment.comparison.answer("brisk")
    assert rejected.outcome == "veto"
    assert rejected.binding_constraint == TIP_OVER
    assert enrollment.verdict.margin(TIP_OVER) is not None
    assert enrollment.verdict.twin_handle == enrollment.twin_handle


def test_enrollment_summary_carries_the_arc(enrollment):
    text = enrollment.summary() + " " + enrollment.detail
    assert "tote.mass" in text
    assert "prior bracket" in text
    assert "probe approved" in text


def test_probe_speed_is_a_weak_lever_next_to_the_pose(twin, payload, prior, stability):
    """Measured here: from a tucked pose, speed barely moves the margin.

    Over 0.10-0.30 rad/s the worst-case margin moves under a millimetre, while
    the same probe from the extended pose is tens of millimetres worse (see
    ``test_an_unsafe_pose_is_refused_not_probed_slowly``). So the vetting's
    speed derate is a fine adjustment and the *pose* is the real lever — which
    is why the refusal points at re-posing rather than at slowing down.
    """
    from jaxterity.objects import probe_speed_margin_curve

    curve = probe_speed_margin_curve(
        twin.robot,
        payload,
        prior,
        stability,
        speeds=(0.30, 0.10),
        pose_rad=_TUCKED,
    )
    assert np.all(np.isfinite(curve))
    assert np.all(curve >= stability.floor)
    assert float(np.ptp(curve)) < 1e-3  # < 1 mm across the speed range
