# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""The oracle surface: what a unit can do, and what happens if it tries.

Three kinds of question, one surface. **Statically**, from the model alone:
:func:`capability_sheet` renders a unit's physical limits **from its model**
with per-field provenance (fitted / derived / nominal / representative), and
:func:`assess` performs the quantitative task-vs-limits comparison that
frontier agents measurably skip, returning proceed / refuse / uncertain with
margins and the binding constraint. Limits may come from a measured sheet
(the product thesis) or be declared externally (benchmark interop).

**Dynamically**, by simulating the unit's twin (:mod:`~jaxterity.oracle.runtime`):
:func:`simulate_plan` answers "what happens if I run this?" with per-constraint
margins and the dispersion of those margins across the unit's own parameter
posterior, :func:`what_if` ranks alternatives on the same draws, and
:func:`identify_object` asks the arm what it is holding. The static surface
bounds what to attempt; the simulated one prices a specific attempt.
"""

from .capability import CapabilityField, CapabilitySheet, capability_sheet
from .feasibility import Assessment, Limits, TaskSpec, assess
from .runtime import (
    TIP_OVER,
    MarginBand,
    ObjectAnswer,
    OracleAnswer,
    Plan,
    WhatIfAnswer,
    envelope_from_mapping,
    hold_plan,
    identify_object,
    object_answer,
    plan_from_trajectory,
    posterior_twins,
    simulate_plan,
    what_if,
)

__all__ = [
    "Assessment",
    "CapabilityField",
    "CapabilitySheet",
    "Limits",
    "MarginBand",
    "ObjectAnswer",
    "OracleAnswer",
    "Plan",
    "TIP_OVER",
    "TaskSpec",
    "WhatIfAnswer",
    "assess",
    "capability_sheet",
    "envelope_from_mapping",
    "hold_plan",
    "identify_object",
    "object_answer",
    "plan_from_trajectory",
    "posterior_twins",
    "simulate_plan",
    "what_if",
]
