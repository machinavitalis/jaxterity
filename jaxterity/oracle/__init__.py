# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""The oracle surface: capability sheets and quantitative feasibility verdicts.

Two callables an agent (or a human) can consult before a robot moves:
:func:`capability_sheet` renders a unit's physical limits **from its model**
with per-field provenance (fitted / derived / nominal / representative), and
:func:`assess` performs the quantitative task-vs-limits comparison that
frontier agents measurably skip, returning proceed / refuse / uncertain with
margins and the binding constraint. Limits may come from a measured sheet
(the product thesis) or be declared externally (benchmark interop).
"""

from .capability import CapabilityField, CapabilitySheet, capability_sheet
from .feasibility import Assessment, Limits, TaskSpec, assess

__all__ = [
    "Assessment",
    "CapabilityField",
    "CapabilitySheet",
    "Limits",
    "TaskSpec",
    "assess",
    "capability_sheet",
]
