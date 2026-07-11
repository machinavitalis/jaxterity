# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""SysidResult + SysidRecipe protocol tests (ADR-010)."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from jaxterity import zoo
from jaxterity.robot import CalibrationState
from jaxterity.sysid import SysidProtocol, SysidRecipe, SysidResult

pytest.importorskip("mujoco")


class _StubRecipe:
    """A recipe satisfying the SysidRecipe protocol (returns a fixed fit)."""

    recipe_name = "stub"
    recipe_version = "1.0"
    protocol = SysidProtocol(
        name="stub",
        excitation="none",
        required_columns=(),
        duration_s=0.0,
        sample_rate_hz=0.0,
    )

    def __call__(self, telemetry, robot, hints=None) -> SysidResult:
        return SysidResult(
            fitted={"cart.mass": jnp.asarray(2.0)},
            telemetry_hash="deadbeef",
            recipe_name=self.recipe_name,
            recipe_version=self.recipe_version,
            residuals={"joint_position": 0.01},
        )


def test_stub_satisfies_recipe_protocol():
    assert isinstance(_StubRecipe(), SysidRecipe)


def test_result_is_not_a_tuple():
    """SysidResult is a typed object, not unpackable (ADR-010, invariant 4)."""
    r = SysidResult(fitted={}, recipe_name="x", recipe_version="0", telemetry_hash="")
    with pytest.raises(TypeError):
        _a, _b = r


def test_apply_to_calibrates_and_records_provenance():
    robot = zoo.load("cartpole")
    h0 = robot.attestation_handle
    cal = _StubRecipe()(None, robot).apply_to(robot)
    assert cal.calibration_state is CalibrationState.CALIBRATED
    assert float(cal.link_map["cart"].mass) == 2.0
    assert robot.attestation_handle == h0  # input unchanged (ADR-004)
    assert cal.attestation_handle != h0


def test_handle_reflects_recipe_version_and_telemetry_hash():
    """Two fits with identical params but different recipe version / telemetry
    hash yield different attestation handles (the provenance is in the handle)."""
    robot = zoo.load("cartpole")

    def _fit(version: str, telem: str):
        return SysidResult(
            fitted={"cart.mass": jnp.asarray(2.0)},
            telemetry_hash=telem,
            recipe_name="r",
            recipe_version=version,
        ).apply_to(robot)

    base = _fit("1.0", "aaa").attestation_handle
    assert _fit("2.0", "aaa").attestation_handle != base  # recipe version matters
    assert _fit("1.0", "bbb").attestation_handle != base  # telemetry hash matters
