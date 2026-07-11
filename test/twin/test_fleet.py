# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Manufacturing-variance fleets.

LHS stratification, fleet sampling semantics (bounds, identity,
reproducibility), and the integration that matters downstream: the sampled
fleet's declared parameters build a HealthyManifold that contains fresh
healthy units and flags a drifted one.
"""

from __future__ import annotations

import numpy as np
import pytest

from jaxterity.errors import JaxterityError
from jaxterity.selfcheck import HealthyManifold
from jaxterity.twin import (
    g1_drift_parameters,
    healthy_population,
    latin_hypercube,
    sample_g1_units,
    unitree_g1_twin,
)

pytest.importorskip("mujoco")


def test_latin_hypercube_is_stratified():
    """One sample per stratum per dimension — the LHS property."""
    n, d = 16, 5
    cube = latin_hypercube(n, d, np.random.default_rng(0))
    assert cube.shape == (n, d)
    assert np.all((cube >= 0.0) & (cube < 1.0))
    strata = np.floor(cube * n).astype(int)
    for dim in range(d):
        assert sorted(strata[:, dim]) == list(range(n)), dim


@pytest.fixture(scope="module")
def fleet():
    return sample_g1_units(12, seed=42)


def test_fleet_units_are_distinct_bounded_and_reproducible(fleet):
    nominal = unitree_g1_twin("G1-NOMINAL")
    assert len(fleet) == 12
    serials = {u.serial for u in fleet}
    assert len(serials) == 12 and "G1-0001" in serials
    handles = {u.attestation_handle for u in fleet}
    assert len(handles) == 12  # distinct parameters + serials ⇒ distinct units

    nominal_friction = np.asarray(
        [float(nominal.robot.joint_map[j].friction) for j in nominal.servo_names]
    )
    for unit in fleet:
        friction = np.asarray(
            [float(unit.robot.joint_map[j].friction) for j in unit.servo_names]
        )
        ratio = friction / nominal_friction
        assert np.all((ratio > 0.85 - 1e-9) & (ratio < 1.15 + 1e-9))
        kt = np.asarray(unit.K_t_scale)
        assert np.all((kt > 0.9 - 1e-9) & (kt < 1.1 + 1e-9))

    # Same seed ⇒ the same fleet, bit-for-bit (handles included).
    again = sample_g1_units(12, seed=42)
    assert [u.attestation_handle for u in again] == [
        u.attestation_handle for u in fleet
    ]


def test_fleet_spread_validation():
    with pytest.raises(JaxterityError, match="n_units"):
        sample_g1_units(0)
    with pytest.raises(JaxterityError, match="friction_spread"):
        sample_g1_units(2, friction_spread=1.5)


def test_population_builds_a_working_manifold(fleet):
    """The B2→B3 handoff: the sampled fleet defines 'healthy'."""
    population = healthy_population(fleet)
    assert len(population) == len(fleet)
    schema_paths = {p.path for p in g1_drift_parameters()}
    assert set(population[0]) == schema_paths

    manifold = HealthyManifold.from_samples(population)
    # Fresh healthy units (a different seed) mostly sit inside …
    fresh = healthy_population(sample_g1_units(10, seed=7))
    covered = sum(manifold.contains(s) for s in fresh)
    assert covered >= 8, covered
    # … while a unit with a failing gearbox does not.
    sick = dict(fresh[0])
    sick["robot.right_knee_joint.friction"] = np.asarray(0.9)  # ≫ healthy box
    assert not manifold.contains(sick)
