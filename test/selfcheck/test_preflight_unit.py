# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Pre-flight on the REAL calibrated unit's twin.

Pre-flight verifies a candidate trajectory against a unit's own twin and
binds the verdict to that twin's attestation handle. Here the twin is the real
calibrated LeKiwi arm (`so101_twin` reads the now-real `so100` calibration).
The gate point: a pass is bound to *one* calibration. When the unit is
recalibrated (a re-fit shifts the friction → a new attestation handle), a prior
pass no longer applies to it; preflight must be re-run against the new twin.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from jaxterity.selfcheck import SafetyEnvelope
from jaxterity.selfcheck.preflight import Scenario, constant_policy, preflight
from jaxterity.twin import Twin, so101_twin


@pytest.fixture(scope="module")
def unit():
    # The real calibrated unit: so101_twin loads zoo.calibrated("so100"), which is
    # now a real closed-loop friction fit (Step 3), not the synthetic default.
    return so101_twin("LK-REAL-001")


def _scenario(twin):
    nv = twin.n_servos
    return Scenario(
        name="reach-and-hold",
        q0=tuple([0.0] * nv),
        qd0=tuple([0.0] * nv),
        T0=tuple([float(twin.T_amb)] * nv),
        n_steps=12,
        dt=0.02,
    )


def _env(twin, rise=120.0):
    return SafetyEnvelope(temp_ceiling_k=twin.T_amb + rise)


def test_real_unit_calibration_is_applied(unit):
    # The unit's robot carries the real fitted joint friction (not frictionless).
    assert float(unit.robot.joint_map["shoulder_pan"].damping) > 0.0


def test_safe_trajectory_passes_and_binds_to_the_unit_handle(unit):
    nv = unit.n_servos
    report = preflight(
        constant_policy(jnp.full(nv, 0.3)), unit, _env(unit), [_scenario(unit)]
    )
    assert report.passed
    assert report.twin_handle == unit.attestation_handle


def test_unsafe_trajectory_is_rejected_naming_the_constraint(unit):
    nv = unit.n_servos
    report = preflight(
        constant_policy(jnp.full(nv, 2.5)), unit, _env(unit), [_scenario(unit)]
    )
    assert not report.passed
    assert report.failures[0].decision.binding_constraint == "servo_temperature"


def _recalibrated(unit):
    # A re-fit nudges the identified friction (drift/aging caught by the online
    # re-fit) → a fresh robot → a new twin with a new attestation handle.
    refit_robot = unit.robot.with_parameters(
        {"shoulder_pan.damping": jnp.asarray(0.55)}
    )
    return Twin(
        refit_robot,
        serial=unit.serial,
        C_th=unit.C_th,
        R_th=unit.R_th,
        k_pwr=unit.k_pwr,
        T_amb=unit.T_amb,
        servo_names=unit.servo_names,
    )


def test_recalibration_invalidates_a_prior_pass(unit):
    nv = unit.n_servos
    policy = constant_policy(jnp.full(nv, 0.3))
    prior = preflight(policy, unit, _env(unit), [_scenario(unit)])
    assert prior.passed

    recal = _recalibrated(unit)
    # The recalibrated unit has a different attestation handle, so the prior pass
    # (bound to the old handle) no longer applies — it must be re-verified.
    assert recal.attestation_handle != unit.attestation_handle
    assert prior.twin_handle != recal.attestation_handle

    # Re-running preflight binds the verdict to the new handle.
    after = preflight(policy, recal, _env(recal), [_scenario(recal)])
    assert after.twin_handle == recal.attestation_handle
    assert after.twin_handle != prior.twin_handle
