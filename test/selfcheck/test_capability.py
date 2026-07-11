# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Live capability estimation.

Under sustained heating the published envelope must shrink measurably and
monotonically, and it must be accepted by the existing SafetyMonitor — a
controller that respected the nominal envelope now respects the shrunk one.
"""

from __future__ import annotations

import numpy as np
import pytest

from jaxterity.selfcheck import (
    CapabilityEstimator,
    LimitModel,
    SafetyAction,
    SafetyMonitor,
)
from jaxterity.twin import TwinRollout, so101_twin

pytest.importorskip("mujoco")


@pytest.fixture(scope="module")
def twin():
    return so101_twin("SO101-CAP")


@pytest.fixture
def limits(twin):
    # A wide thermal span so the temperature check doesn't bind in the torque test.
    return LimitModel(
        temp_ceiling_k=twin.T_amb + 60.0,
        nominal_torque_margin_frac=0.9,
        min_torque_margin_frac=0.05,
    )


def _temps(twin, value):
    return np.full(twin.n_servos, value)


def test_capability_factor_bounds(twin, limits):
    est = CapabilityEstimator(twin, limits)
    assert est.capability_factor(_temps(twin, twin.T_amb)) == pytest.approx(1.0)
    assert est.capability_factor(_temps(twin, limits.temp_ceiling_k)) == pytest.approx(
        0.0
    )
    mid = est.capability_factor(_temps(twin, twin.T_amb + 30.0))
    assert 0.0 < mid < 1.0


def test_envelope_shrinks_monotonically_with_heating(twin, limits):
    est = CapabilityEstimator(twin, limits)
    margins = [
        est.estimate(_temps(twin, twin.T_amb + d)).torque_margin_frac
        for d in (0.0, 10.0, 20.0, 30.0, 40.0, 50.0)
    ]
    # Strictly decreasing while above the floor.
    for a, b in zip(margins, margins[1:]):
        assert b < a
    assert margins[0] == pytest.approx(0.9)


def test_margin_floored_not_zero_at_ceiling(twin, limits):
    est = CapabilityEstimator(twin, limits)
    env = est.estimate(_temps(twin, limits.temp_ceiling_k + 5.0))
    assert env.torque_margin_frac == pytest.approx(limits.min_torque_margin_frac)


def test_hottest_servo_binds(twin, limits):
    est = CapabilityEstimator(twin, limits)
    temps = _temps(twin, twin.T_amb + 5.0)
    temps[3] = twin.T_amb + 50.0  # one hot servo
    # Capability equals the hottest servo's headroom, not the mean.
    hot_only = est.capability_factor(_temps(twin, twin.T_amb + 50.0))
    assert est.capability_factor(temps) == pytest.approx(hot_only)


def _prediction(twin, temp_value):
    """A minimal forecast at a benign temperature (isolates the torque check)."""
    n = twin.n_servos
    nq = twin.robot.build_system()._nq
    steps = 4
    temperature = np.full((steps + 1, n), temp_value)
    q = np.zeros((steps + 1, nq))
    qd = np.zeros((steps + 1, n))
    return TwinRollout(q=q, qd=qd, temperature=temperature, bus_voltage=None)


def test_shrunk_envelope_is_accepted_and_enforced_by_safety_monitor(twin, limits):
    """A moderate action the cool envelope passes is vetoed under the hot one."""
    est = CapabilityEstimator(twin, limits)
    # A benign forecast temperature so the *torque* constraint binds, not thermal.
    pred = _prediction(twin, twin.T_amb + 15.0)
    # 1 N·m of a 10 N·m effort → frac 0.1. Small enough that the sustained-hold
    # steady state (k_pwr·R_th·τ² ≈ 24 K rise) stays well under the 60 K ceiling,
    # so the torque margin — not temperature — is what differs cool vs hot.
    action = np.full(twin.n_servos, 1.0)

    cool_env = est.estimate(_temps(twin, twin.T_amb))  # margin ~0.9
    hot_env = est.estimate(_temps(twin, twin.T_amb + 50.0))  # margin ~0.15
    assert hot_env.torque_margin_frac < cool_env.torque_margin_frac

    cool_decision = SafetyMonitor(twin, cool_env).evaluate(pred, action=action)
    hot_decision = SafetyMonitor(twin, hot_env).evaluate(pred, action=action)

    # The cool envelope passes the action; the shrunk one derates/vetoes it.
    assert cool_decision.action is SafetyAction.OK
    assert hot_decision.action in (SafetyAction.DERATE, SafetyAction.VETO)
    assert hot_decision.factor < cool_decision.factor
