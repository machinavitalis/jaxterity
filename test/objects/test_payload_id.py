# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Grasped-payload identification — recovery, honesty gates, attestation.

The payload loopback is a seeded simulated arm carrying a *planted* payload:
it runs the excitation through the payload-attached robot's free dynamics, so
:func:`identify_payload` must recover the planted mass/CoM from the same
telemetry channel the joint-level active lane uses.

Cost discipline: every distinct observer is a fresh XLA compile of an MJX
simulate, so the flagship so100 fit runs **once** (module-scoped fixture) and
several tests assert against it; the honesty-gate and determinism tests run on
the much lighter cartpole. A real capture replays through the same fit via
``ReplayLink``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaxterity import zoo
from jaxterity.calibrate.active import Excitation, design_coast_excitation
from jaxterity.errors import JaxterityError
from jaxterity.objects import (
    PayloadLoopbackLink,
    RigidPayload,
    attach_payload,
    identify_payload,
    payload_identifiability,
)

pytest.importorskip("mujoco")

#: Planted ground truth for the so100 recovery tests (kg, m).
_TRUE_MASS = 0.15
_TRUE_COM = (0.012, -0.008, 0.02)
#: Planted tip mass for the cartpole tests (kg).
_TRUE_TIP_MASS = 0.2


@pytest.fixture(scope="module")
def robot():
    return zoo.load("so100")


@pytest.fixture(scope="module")
def payload():
    return RigidPayload(attach_link="gripper_frame_link", mass=0.05)


@pytest.fixture(scope="module")
def excitation(robot):
    return design_coast_excitation(robot, speed=0.1)


@pytest.fixture(scope="module")
def so100_result(robot, payload, excitation):
    """The flagship noise-free so100 fit — computed once, asserted by many."""
    link = PayloadLoopbackLink(robot, payload, true_mass=_TRUE_MASS, true_com=_TRUE_COM)
    return identify_payload(robot, link, payload, excitation=excitation)


@pytest.fixture(scope="module")
def cartpole():
    return zoo.load("cartpole")


@pytest.fixture(scope="module")
def tip(cartpole):
    return RigidPayload(attach_link="pole", mass=0.05, offset=(0.0, 0.0, 0.6))


@pytest.fixture(scope="module")
def tip_excitation(cartpole):
    return design_coast_excitation(cartpole, speed=0.3)


# --- structural attach ------------------------------------------------------


def test_attach_payload_adds_welded_body(robot, payload):
    """Attaching adds one link + one FIXED joint and no DOFs."""
    attached = attach_payload(robot, payload)
    assert len(attached.links) == len(robot.links) + 1
    assert len(attached.joints) == len(robot.joints) + 1
    assert attached.link_map["payload"].name == "payload"
    weld = attached.joint_map["payload_weld"]
    assert weld.joint_type.value == "fixed"
    assert weld.parent_link == "gripper_frame_link"
    # A welded payload adds no articulation.
    assert (
        attached.build_system(actuation=True)._nv
        == robot.build_system(actuation=True)._nv
    )


def test_attach_payload_changes_attestation_handle(robot, payload):
    """The payload body is part of the model description → new handle."""
    attached = attach_payload(robot, payload)
    assert attached.attestation_handle != robot.attestation_handle


def test_attach_payload_rejects_unknown_link(robot):
    with pytest.raises(JaxterityError, match="gripper_of_theseus"):
        attach_payload(robot, RigidPayload(attach_link="gripper_of_theseus"))


def test_attach_payload_rejects_name_collision(robot):
    with pytest.raises(JaxterityError, match="already"):
        attach_payload(
            robot,
            RigidPayload(attach_link="gripper_frame_link", name="base_link"),
        )


def test_payload_prior_mass_must_be_positive():
    """A zero prior mass would zero the CoM sensitivity columns — refused."""
    with pytest.raises(JaxterityError, match="positive"):
        RigidPayload(attach_link="gripper_frame_link", mass=0.0)


def test_prior_defaults_round_trip(robot):
    """Priors land on the attached link exactly as given."""
    p = RigidPayload(
        attach_link="gripper_frame_link",
        mass=0.11,
        com=(0.01, 0.02, 0.03),
        name="ball",
    )
    attached = attach_payload(robot, p)
    ball = attached.link_map["ball"]
    assert float(ball.mass) == pytest.approx(0.11)
    assert np.allclose(np.asarray(ball.com), [0.01, 0.02, 0.03])
    assert np.asarray(ball.inertia).shape == (3, 3)
    assert float(jnp.linalg.eigvalsh(ball.inertia)[0]) > 0.0


# --- loopback recovery (the kill-gate) --------------------------------------


def test_recovers_planted_payload_so100(so100_result):
    """One end-to-end call recovers the planted payload mass and CoM."""
    result = so100_result
    assert result.recipe_name == "objects.payload"
    assert float(result.fitted["payload.mass"]) == pytest.approx(_TRUE_MASS, abs=1e-3)
    com = np.asarray(result.fitted["payload.com"])
    for i in range(3):
        assert float(com[i]) == pytest.approx(_TRUE_COM[i], abs=1e-3)
    # A clean, well-posed fit carries no warnings (the T-219/C-119 contract).
    assert result.warnings == ()


def test_recovers_planted_payload_cartpole(cartpole, tip, tip_excitation):
    """Second zoo robot: a mass clamped to the cartpole tip is recovered."""
    link = PayloadLoopbackLink(cartpole, tip, true_mass=_TRUE_TIP_MASS)
    result = identify_payload(
        cartpole,
        link,
        tip,
        excitation=tip_excitation,
        fit=("mass",),
        uncertainty=None,
    )
    assert float(result.fitted["payload.mass"]) == pytest.approx(
        _TRUE_TIP_MASS, abs=1e-3
    )


def test_seeded_loopback_gives_bit_identical_fits(cartpole, tip, tip_excitation):
    """Seeded noise → bit-identical fitted parameters across runs."""

    def run():
        link = PayloadLoopbackLink(
            cartpole, tip, true_mass=_TRUE_TIP_MASS, seed=7, noise_std=1e-4
        )
        return identify_payload(
            cartpole,
            link,
            tip,
            excitation=tip_excitation,
            fit=("mass",),
            uncertainty=None,
        )

    a, b = run(), run()
    assert float(a.fitted["payload.mass"]) == float(b.fitted["payload.mass"])
    assert a.telemetry_hash == b.telemetry_hash


# --- honesty gates -----------------------------------------------------------


def test_degenerate_probe_is_flagged_before_any_fit(cartpole, tip):
    """A quasi-static probe cannot excite the payload inertia — say so up front."""
    barely = Excitation(q0=(0.0, 0.0), qd0=(0.0, 0.0), sample_times=(0.001,))
    report = payload_identifiability(
        cartpole, tip, barely, fit=("mass", "com", "inertia_diag")
    )
    assert not report.identifiable
    with pytest.raises(Exception, match="identifiab"):
        report.raise_if_unidentifiable("quasi-static probe")


def test_degenerate_probe_warning_reaches_the_result(cartpole, tip):
    """identify_payload folds the identifiability verdict into the result."""
    barely = Excitation(q0=(0.0, 0.0), qd0=(0.0, 0.0), sample_times=(0.001,))
    link = PayloadLoopbackLink(cartpole, tip, true_mass=_TRUE_TIP_MASS)
    result = identify_payload(
        cartpole,
        link,
        tip,
        excitation=barely,
        fit=("mass", "com", "inertia_diag"),
        iterations=2,
        uncertainty=None,
    )
    assert any("identifiab" in w or "weakly identified" in w for w in result.warnings)


def test_unphysical_fit_is_bounded_and_flagged(cartpole, tip, tip_excitation):
    """A planted negative mass cannot be fitted: pinned at 0 and named."""
    link = PayloadLoopbackLink(cartpole, tip, true_mass=-0.08)
    result = identify_payload(
        cartpole,
        link,
        tip,
        excitation=tip_excitation,
        fit=("mass",),
        uncertainty=None,
    )
    assert float(result.fitted["payload.mass"]) >= 0.0
    assert any("payload.mass" in w and "bound" in w for w in result.warnings)


def test_well_posed_probe_reports_identifiable(cartpole, tip, tip_excitation):
    """The coast probe carries information about the tip mass."""
    report = payload_identifiability(cartpole, tip, tip_excitation, fit=("mass",))
    assert report.identifiable
    assert report.rank == 1
    assert report.names == ("payload.mass",)


# --- result plumbing ---------------------------------------------------------


def test_result_applies_to_attached_robot_with_provenance(robot, payload, so100_result):
    """The fit is a SysidResult that calibrates the payload-attached robot."""
    attached = attach_payload(robot, payload)
    calibrated = so100_result.apply_to(attached)
    assert calibrated.attestation_handle != attached.attestation_handle
    assert any(r[0] == "objects.payload" for r in calibrated.provenance)
    assert float(calibrated.link_map["payload"].mass) == pytest.approx(
        _TRUE_MASS, abs=1e-3
    )


def test_interval_expands_vector_leaves_componentwise(so100_result):
    """interval() names each CoM component instead of crashing on the vector."""
    intervals = so100_result.interval()
    assert intervals is not None
    assert "payload.mass" in intervals
    for i in range(3):
        lo, hi = intervals[f"payload.com[{i}]"]
        assert lo <= float(np.asarray(so100_result.fitted["payload.com"])[i]) <= hi


def test_twin_with_payload_reattests(payload, so100_result):
    """A payload (and its re-fit) is a new unit version: the handle moves."""
    from jaxterity.twin import so101_twin

    twin = so101_twin("SN-payload-test-001")
    bare_handle = twin.attestation_handle

    held = twin.with_payload(payload)
    assert held.attestation_handle != bare_handle

    calibrated = twin.with_payload(payload, so100_result)
    assert calibrated.attestation_handle != held.attestation_handle
    assert calibrated.attestation_handle != bare_handle
    # The payload adds no servos — the off-body model is untouched.
    assert calibrated.n_servos == twin.n_servos
