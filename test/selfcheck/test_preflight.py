# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Minimal pre-flight verification — pass/fail, naming, handle binding.

A candidate policy is an open-loop torque trajectory rolled through the unit's
twin per scenario; the realized rollout is checked against the SafetyEnvelope. A
pass records the twin's attestation handle, so it binds to one twin version.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from jaxterity.selfcheck import (
    SafetyEnvelope,
    Scenario,
    constant_policy,
    preflight,
)
from jaxterity.twin import Twin, so101_twin

pytest.importorskip("mujoco")


@pytest.fixture(scope="module")
def twin() -> Twin:
    return so101_twin("SO101-0001")


@pytest.fixture(scope="module")
def scenario(twin):
    nv = twin.n_servos
    nq = twin.robot.build_system()._nq
    return Scenario(
        name="hold-at-rest",
        q0=tuple([0.0] * nq),
        qd0=tuple([0.0] * nv),
        T0=tuple([twin.T_amb] * nv),
        n_steps=10,
        dt=0.02,
    )


def _env(twin, rise=35.0):
    return SafetyEnvelope(temp_ceiling_k=twin.T_amb + rise)


def test_empty_scenarios_raises(twin):
    with pytest.raises(ValueError, match="scenario"):
        preflight(constant_policy(jnp.zeros(twin.n_servos)), twin, _env(twin), [])


def test_in_envelope_policy_passes(twin, scenario):
    nv = twin.n_servos
    report = preflight(constant_policy(jnp.full(nv, 0.3)), twin, _env(twin), [scenario])
    assert report.passed
    assert not report.failures
    assert all(r.passed for r in report.results)


def test_out_of_envelope_policy_fails_naming_scenario_and_constraint(twin, scenario):
    nv = twin.n_servos
    report = preflight(constant_policy(jnp.full(nv, 2.0)), twin, _env(twin), [scenario])
    assert not report.passed
    assert len(report.failures) == 1
    failed = report.failures[0]
    assert failed.name == "hold-at-rest"
    assert failed.decision.binding_constraint == "servo_temperature"
    assert "hold-at-rest" in report.summary()


def test_report_records_twin_handle(twin, scenario):
    nv = twin.n_servos
    report = preflight(constant_policy(jnp.full(nv, 0.3)), twin, _env(twin), [scenario])
    assert report.twin_handle == twin.attestation_handle


def test_report_reflects_a_different_twin(twin, scenario):
    nv = twin.n_servos
    a = preflight(constant_policy(jnp.full(nv, 0.3)), twin, _env(twin), [scenario])
    other = twin.with_serial("SO101-9999")
    b = preflight(constant_policy(jnp.full(nv, 0.3)), other, _env(other), [scenario])
    assert a.twin_handle != b.twin_handle


def test_deterministic_for_fixed_inputs(twin, scenario):
    nv = twin.n_servos
    policy = constant_policy(jnp.full(nv, 1.0))
    a = preflight(policy, twin, _env(twin), [scenario])
    b = preflight(policy, twin, _env(twin), [scenario])
    assert a.passed == b.passed
    assert a.results[0].decision.factor == b.results[0].decision.factor


def test_passes_only_when_all_scenarios_pass(twin):
    nv = twin.n_servos
    nq = twin.robot.build_system()._nq
    base = dict(q0=tuple([0.0] * nq), qd0=tuple([0.0] * nv), n_steps=10, dt=0.02)
    cool = Scenario(name="cool-start", T0=tuple([twin.T_amb] * nv), **base)
    hot = Scenario(name="hot-start", T0=tuple([twin.T_amb + 40.0] * nv), **base)
    # already-hot start breaches the ceiling immediately → fails that scenario.
    report = preflight(
        constant_policy(jnp.full(nv, 0.3)), twin, _env(twin), [cool, hot]
    )
    assert not report.passed
    assert [r.name for r in report.failures] == ["hot-start"]


# --- CLI ------------------------------------------------------------------


def test_cli_preflight_pass(capsys):
    from jaxterity.cli import main

    rc = main(["preflight", "--hold-torque", "0.3"])
    assert rc == 0
    assert "PASS" in capsys.readouterr().out


def test_cli_preflight_fail(capsys):
    from jaxterity.cli import main

    rc = main(["preflight", "--hold-torque", "2.0"])
    assert rc == 1
    assert "FAIL" in capsys.readouterr().out
