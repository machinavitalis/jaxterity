# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Semantic priors (T-243): seed, box, and the honesty of a seeded fit.

A prior is a *declared guess*: it moves where the fit starts and where it may
go, and the result must say that it did. These tests cover the three claims —
the conversion from "what a vision model can say" to numbers, the box actually
constraining the search, and the provenance/declaration of a seeded fit — plus
the value claim itself (a seeded fit beats the same fit started from a bad
prior, under an identical iteration budget).

Cost discipline: the cartpole carries every fit here (one DOF pair, cheap MJX
compile); the so100 fits live in ``test_payload_id.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from jaxterity import zoo
from jaxterity.calibrate.active import design_coast_excitation
from jaxterity.errors import JaxterityError
from jaxterity.objects import (
    PayloadLoopbackLink,
    RigidPayload,
    SemanticPrior,
    identify_payload,
)

pytest.importorskip("mujoco")

#: Planted tip mass for the cartpole fits, kg.
_TRUE_TIP_MASS = 0.2


@pytest.fixture(scope="module")
def cartpole():
    return zoo.load("cartpole")


@pytest.fixture(scope="module")
def tip(cartpole):
    """A deliberately *bad* declared prior: 4x below the planted mass."""
    return RigidPayload(attach_link="pole", mass=0.05, offset=(0.0, 0.0, 0.6))


@pytest.fixture(scope="module")
def tip_excitation(cartpole):
    return design_coast_excitation(cartpole, speed=0.3)


# --- the guess → numbers conversion -----------------------------------------


def test_from_material_brackets_mass_by_density_and_fill():
    """Density range x box volume x fill bracket — the documented arithmetic."""
    prior = SemanticPrior.from_material(
        "aluminum block", "aluminum", (0.05, 0.04, 0.03), source="vlm/test"
    )
    volume = 0.05 * 0.04 * 0.03
    assert prior.mass_kg[0] == pytest.approx(2600.0 * volume * 0.4)
    assert prior.mass_kg[1] == pytest.approx(2850.0 * volume * 1.0)
    # The CoM of a rigid body lies inside its own bounding box.
    assert prior.com_halfwidth_m == (0.025, 0.02, 0.015)
    # Principal moments bounded by a solid box at the heaviest plausible mass.
    m = prior.mass_kg[1]
    assert prior.inertia_max is not None
    assert prior.inertia_max[0] == pytest.approx(m * (0.04**2 + 0.03**2) / 12.0)


def test_seed_mass_is_the_geometric_mean():
    """An order-of-magnitude bracket gets an order-of-magnitude midpoint."""
    prior = SemanticPrior(label="?", mass_kg=(0.05, 5.0), source="vlm/test")
    assert prior.seed_mass == pytest.approx(0.5)


def test_unknown_material_and_bad_geometry_are_refused():
    with pytest.raises(JaxterityError, match="unknown material"):
        SemanticPrior.from_material("x", "unobtanium", (1.0, 1.0, 1.0), source="s")
    with pytest.raises(JaxterityError, match="positive side lengths"):
        SemanticPrior.from_material("x", "steel", (0.1, 0.0, 0.1), source="s")
    with pytest.raises(JaxterityError, match="positive, ordered bracket"):
        SemanticPrior(label="x", mass_kg=(0.5, 0.1), source="s")
    with pytest.raises(JaxterityError, match="confidence"):
        SemanticPrior(label="x", mass_kg=(0.1, 0.5), source="s", confidence=1.5)


def test_seed_preserves_structure_and_moves_the_mass(tip):
    prior = SemanticPrior(label="tip", mass_kg=(0.15, 0.30), source="vlm/test")
    seeded = prior.seed(tip)
    assert seeded.attach_link == tip.attach_link
    assert seeded.name == tip.name
    assert seeded.offset == tip.offset
    assert float(seeded.mass) == pytest.approx(prior.seed_mass)
    assert float(seeded.mass) != float(tip.mass)


def test_bounds_follow_the_fit_layout(tip):
    prior = SemanticPrior(
        label="tip",
        mass_kg=(0.15, 0.30),
        source="vlm/test",
        com_halfwidth_m=(0.01, 0.02, 0.03),
    )
    seeded = prior.seed(tip)
    lower, upper = prior.bounds(seeded, ("mass", "com"))
    assert lower[0] == pytest.approx(0.15)
    assert upper[0] == pytest.approx(0.30)
    # CoM box centred on the payload's own CoM prior (the origin here).
    assert np.allclose(lower[1:4], [-0.01, -0.02, -0.03])
    assert np.allclose(upper[1:4], [0.01, 0.02, 0.03])


def test_bounds_keep_physical_nonnegativity_where_the_prior_is_silent(tip):
    """No moment bound declared → still floored at 0, still unbounded above."""
    prior = SemanticPrior(label="tip", mass_kg=(0.15, 0.30), source="vlm/test")
    seeded = prior.seed(tip)
    lower, upper = prior.bounds(seeded, ("mass", "inertia_diag"))
    assert np.all(lower[1:] >= 0.0)
    assert np.all(np.isinf(upper[1:]))


# --- the fit: seeding, bounding, declaring ----------------------------------


def test_prior_seeded_fit_recovers_the_planted_mass(cartpole, tip, tip_excitation):
    """A correct guess plus telemetry lands on the truth, and says it was seeded."""
    prior = SemanticPrior(
        label="steel tip mass, ~200 g",
        mass_kg=(0.12, 0.35),
        source="vlm/test",
        confidence=0.7,
    )
    link = PayloadLoopbackLink(cartpole, tip, true_mass=_TRUE_TIP_MASS)
    result = identify_payload(
        cartpole,
        link,
        tip,
        excitation=tip_excitation,
        fit=("mass",),
        uncertainty=None,
        prior=prior,
    )
    assert float(result.fitted["payload.mass"]) == pytest.approx(
        _TRUE_TIP_MASS, abs=1e-3
    )
    # Declared: in the provenance the attestation handle sees...
    assert result.recipe_name == "objects.payload+semantic_prior"
    # ...and in a note naming the guess, its source and its bracket.
    assert any("semantic prior" in w for w in result.warnings)
    assert any("vlm/test" in w for w in result.warnings)


def test_prior_beats_a_bad_start_under_the_same_budget(cartpole, tip, tip_excitation):
    """The value claim: the seed escapes a bad linearization point.

    The declared payload prior (0.05 kg) is 4x below the planted 0.2 kg; the
    step clip means a fixed iteration budget cannot cross that gap from the bad
    start. The semantic guess moves the start, not the data.
    """

    def fit(prior):
        link = PayloadLoopbackLink(cartpole, tip, true_mass=_TRUE_TIP_MASS)
        return float(
            identify_payload(
                cartpole,
                link,
                tip,
                excitation=tip_excitation,
                fit=("mass",),
                iterations=2,
                uncertainty=None,
                prior=prior,
            ).fitted["payload.mass"]
        )

    unseeded = fit(None)
    seeded = fit(SemanticPrior(label="tip", mass_kg=(0.12, 0.35), source="vlm/test"))
    assert abs(seeded - _TRUE_TIP_MASS) < abs(unseeded - _TRUE_TIP_MASS)


def test_a_wrong_prior_pins_the_fit_and_the_result_says_so(
    cartpole, tip, tip_excitation
):
    """A confidently wrong guess must not masquerade as a measurement."""
    wrong = SemanticPrior(
        label="foam replica, ~30 g",
        mass_kg=(0.02, 0.05),
        source="vlm/test",
        confidence=0.9,
    )
    link = PayloadLoopbackLink(cartpole, tip, true_mass=_TRUE_TIP_MASS)
    result = identify_payload(
        cartpole,
        link,
        tip,
        excitation=tip_excitation,
        fit=("mass",),
        uncertainty=None,
        prior=wrong,
    )
    mass = float(result.fitted["payload.mass"])
    # The box held: the telemetry wanted more mass, the prior refused.
    assert mass <= 0.05 + 1e-9
    assert any("pinned at its semantic-prior bound" in w for w in result.warnings)


def test_unseeded_fit_is_unchanged(cartpole, tip, tip_excitation):
    """No prior → the pre-T-243 behaviour, provenance included."""
    link = PayloadLoopbackLink(cartpole, tip, true_mass=_TRUE_TIP_MASS)
    result = identify_payload(
        cartpole,
        link,
        tip,
        excitation=tip_excitation,
        fit=("mass",),
        uncertainty=None,
    )
    assert result.recipe_name == "objects.payload"
    assert not any("semantic prior" in w for w in result.warnings)
