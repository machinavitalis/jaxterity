# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Windowed drift re-fit — recovery on the G1 unit twin.

The plant drifts (right-knee friction up, right-knee/left-hip torque
constants drooped); the self-model collects one hold-and-wiggle window
(:mod:`jaxterity.calibrate.excitation`) from the drifted plant, re-fits, and
must recover the planted values, shrink the prediction residual, and roll
the attestation chain. Plus the validation surface (thermal paths deflect
to the thermal recipe, joint-name mismatches raise).
"""

from __future__ import annotations

import numpy as np
import pytest

from jaxterity.errors import JaxterityError

pytest.importorskip("mujoco")

from jaxterity.calibrate import (  # noqa: E402
    design_hold_excitation,
    refit_drift_parameters,
    simulate_hold_window,
)
from jaxterity.data.schema import JointTelemetry  # noqa: E402
from jaxterity.twin import g1_drift_parameters, unitree_g1_twin  # noqa: E402

#: Planted plant-side drift the fit must recover.
_TRUE_FRICTION = {"right_knee_joint": 0.55, "left_hip_pitch_joint": 0.45}
_TRUE_KT = {"right_knee_joint": 0.88, "left_hip_pitch_joint": 0.93}

#: The dynamics channels of the declared schema (thermal goes to setup).
_DYNAMICS_PARAMS = tuple(p for p in g1_drift_parameters() if p.path != "offbody.R_th")


@pytest.fixture(scope="module")
def window_and_twin():
    """One hold-and-wiggle window from the drifted plant + the (stale) twin."""
    twin = unitree_g1_twin("G1-TEST")
    names = twin.servo_names

    # The drifted plant: friction up on two joints, torque constants drooped.
    plant_robot = twin.robot.with_parameters(
        {f"{j}.friction": v for j, v in _TRUE_FRICTION.items()}
    )
    kt_true = np.ones(twin.n_servos)
    for joint, value in _TRUE_KT.items():
        kt_true[names.index(joint)] = value

    # The routine is designed on the *nominal* model (the unit's self-model)
    # and executed on the drifted plant — the telemetry difference IS the
    # drift the re-fit must recover.
    excitation = design_hold_excitation(twin.robot)
    telemetry = simulate_hold_window(excitation, plant_robot, K_t_scale=kt_true)
    return telemetry, twin


def test_windowed_refit_recovers_planted_drift(window_and_twin):
    telemetry, twin = window_and_twin
    result = refit_drift_parameters(twin, telemetry, _DYNAMICS_PARAMS, iterations=8)

    # The fit explains the window better than the stale model. (The bulk of
    # the *before* RMS is the central-difference noise floor shared by both,
    # so the recovery asserts below are the sharp oracle; this one just says
    # the fit moved in the right direction.)
    assert result.residual_rms_after < result.residual_rms_before, (
        result.residual_rms_before,
        result.residual_rms_after,
    )

    # Planted friction drift recovered (schema-bounded, prior-anchored GN).
    fitted = result.sysid_result.fitted
    for joint, true_value in _TRUE_FRICTION.items():
        got = float(fitted[f"robot.{joint}.friction"])
        assert got == pytest.approx(true_value, abs=0.05), (joint, got, true_value)

    # Planted torque-constant droop recovered on the drifted joints …
    names = twin.servo_names
    kt = np.asarray(fitted["offbody.K_t_scale"])
    for joint, true_value in _TRUE_KT.items():
        got = float(kt[names.index(joint)])
        assert got == pytest.approx(true_value, abs=0.05), (joint, got, true_value)
    # … while channels the window can't identify (idle wrists, ankle roll)
    # stay anchored at nominal instead of wandering to the schema bounds —
    # a wandering unidentified parameter would later read as drift.
    drifted = {names.index(j) for j in _TRUE_KT}
    others = np.asarray([k for k in range(len(names)) if k not in drifted])
    assert np.all(np.abs(kt[others] - 1.0) < 0.1), kt[others]

    # The rolling attestation: provenance on both chains, a new handle.
    new = result.twin
    assert new.attestation_handle != twin.attestation_handle
    assert new.robot.provenance[-1][0] == "sysid.windowed_drift"
    assert new.provenance[-1][0] == "sysid.windowed_drift"
    assert new.provenance[-1][2] == telemetry.hash().hex()
    # Input twin untouched (ADR-004).
    assert twin.robot.provenance == ()


def test_excitation_window_is_limit_impact_free(window_and_twin):
    """The collected window respects the design contract: clear of limits."""
    telemetry, twin = window_and_twin
    lower = np.asarray(
        [j.limits.lower for j in twin.robot.joints if j.name in telemetry.joint_names]
    )
    upper = np.asarray(
        [j.limits.upper for j in twin.robot.joints if j.name in telemetry.joint_names]
    )
    q = np.asarray(telemetry.q)
    assert np.all(q > lower[None, :] + 0.02)
    assert np.all(q < upper[None, :] - 0.02)


def test_excitation_rejects_floating_base():
    from jaxterity import zoo

    with pytest.raises(JaxterityError, match="fixed-base"):
        design_hold_excitation(zoo.unitree_g1.load(floating_base=True))


def test_thermal_paths_deflect_to_the_thermal_recipe(window_and_twin):
    telemetry, twin = window_and_twin
    thermal_only = tuple(p for p in g1_drift_parameters() if p.path == "offbody.R_th")
    with pytest.raises(JaxterityError, match="fit_servo_thermal"):
        refit_drift_parameters(twin, telemetry, thermal_only)


def test_joint_name_mismatch_raises(window_and_twin):
    telemetry, twin = window_and_twin
    renamed = JointTelemetry(
        time=np.asarray(telemetry.time),
        sample_rate_hz=telemetry.sample_rate_hz,
        joint_names=("bogus_joint",) + telemetry.joint_names[1:],
        q=np.asarray(telemetry.q),
        qd=np.asarray(telemetry.qd),
        tau=np.asarray(telemetry.tau),
    )
    with pytest.raises(JaxterityError, match="do not match"):
        refit_drift_parameters(twin, renamed, _DYNAMICS_PARAMS)
