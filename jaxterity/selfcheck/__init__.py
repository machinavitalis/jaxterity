# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Predictive self-check: forecast a unit's near future and veto unsafe commands.

The self-check runs the per-unit :class:`~jaxterity.twin.Twin` *beside* the
controller. :class:`~jaxterity.selfcheck.runtime.CoProcessor` rolls the
twin forward each tick and publishes the predicted servo temperature, bus
voltage, and joint trajectories; the safety monitor and pre-flight check
consume those forecasts to derate/veto commands a body-only model would
wave through.
"""

from .capability import CapabilityEstimator, LimitModel
from .monitor import SafetyAction, SafetyDecision, SafetyEnvelope, SafetyMonitor
from .preflight import (
    PreflightReport,
    Scenario,
    ScenarioResult,
    constant_policy,
    preflight,
)
from .residual import AnomalyReport, HealthyManifold, ResidualMonitor
from .runtime import CoProcessor, RunReport, TickReport

__all__ = [
    "CoProcessor",
    "TickReport",
    "RunReport",
    "ResidualMonitor",
    "AnomalyReport",
    "HealthyManifold",
    "SafetyMonitor",
    "SafetyEnvelope",
    "SafetyDecision",
    "SafetyAction",
    "preflight",
    "Scenario",
    "ScenarioResult",
    "PreflightReport",
    "constant_policy",
    "CapabilityEstimator",
    "LimitModel",
]
