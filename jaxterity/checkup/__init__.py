# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Per-unit health checkup: one command from captures to a report card.

``jaxterity.checkup`` composes the existing calibration lane — commanded-
excitation captures, the closed-loop friction fit, the battery Thévenin fit —
into a per-unit **report card** (:class:`CheckupReport`): fitted parameters
with confidence intervals, held-out fit quality versus the frictionless URDF
baseline, coarse actuation-health gates, an attestation handle, and a
versioned triage score. Runs persist in a serial-keyed :class:`CheckupStore`
so parameter drift across checkups is a first-class series, and
:func:`render_report_html` turns a run (plus its history) into a
self-contained HTML report. The CLI surface is ``jaxterity checkup``.
"""

from .html import render_report_html
from .plan import (
    PLAN_SCHEMA,
    PLAN_SCHEMA_V1,
    CapturePlan,
    PlanEntry,
    build_capture_plan,
    validate_capture_motion,
)
from .probe import (
    SegmentScore,
    StabilityOfFit,
    commanded_velocity,
    fit_stability,
    segment_identifiability,
)
from .protocol import (
    POSE_TOLERANCE_TICKS,
    PROTOCOL_SCHEMA,
    BatteryProtocol,
    CaptureProtocol,
    JointProtocol,
    capture_protocol,
)
from .report import (
    SCHEMA,
    BatteryCheckup,
    CheckupReport,
    Gate,
    JointCheckup,
    run_checkup,
)
from .stability import (
    LEKIWI_BASE,
    BaseProfile,
    StabilityConstraint,
    StabilityReport,
    zmp_margins,
)
from .store import CheckupStore

__all__ = [
    "LEKIWI_BASE",
    "PLAN_SCHEMA",
    "PLAN_SCHEMA_V1",
    "POSE_TOLERANCE_TICKS",
    "PROTOCOL_SCHEMA",
    "BatteryProtocol",
    "CaptureProtocol",
    "JointProtocol",
    "SegmentScore",
    "StabilityOfFit",
    "capture_protocol",
    "commanded_velocity",
    "fit_stability",
    "segment_identifiability",
    "SCHEMA",
    "BaseProfile",
    "BatteryCheckup",
    "CapturePlan",
    "CheckupReport",
    "CheckupStore",
    "Gate",
    "JointCheckup",
    "PlanEntry",
    "StabilityConstraint",
    "StabilityReport",
    "build_capture_plan",
    "render_report_html",
    "run_checkup",
    "validate_capture_motion",
    "zmp_margins",
]
