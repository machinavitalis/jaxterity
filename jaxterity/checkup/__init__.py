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

from .backlash import (
    BACKLASH_CAPTURE_SCHEMA,
    LostMotion,
    ReversalFit,
    backlash_summary,
    fit_reversal,
    load_backlash_capture,
    reversal_lost_motion,
)
from .html import render_report_html
from .plan import (
    PLAN_SCHEMA,
    PLAN_SCHEMA_V1,
    CapturePlan,
    PlanEntry,
    build_capture_plan,
    plan_information,
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
from .uncertainty import (
    BIAS_DOMINANT_FRACTION,
    CONFIDENCE_RELATIVE_LIMIT,
    TotalUncertainty,
    length_ladder,
    total_uncertainty,
    uncertainty_gates,
    uncertainty_summary,
)
from .wheels import (
    WHEEL_CAPTURE_SCHEMA,
    WheelHealth,
    load_wheel_capture,
    wheel_gates,
    wheel_health,
    wheel_summary,
)

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
    "WHEEL_CAPTURE_SCHEMA",
    "BACKLASH_CAPTURE_SCHEMA",
    "LostMotion",
    "ReversalFit",
    "backlash_summary",
    "fit_reversal",
    "load_backlash_capture",
    "reversal_lost_motion",
    "BIAS_DOMINANT_FRACTION",
    "CONFIDENCE_RELATIVE_LIMIT",
    "TotalUncertainty",
    "length_ladder",
    "total_uncertainty",
    "uncertainty_gates",
    "uncertainty_summary",
    "WheelHealth",
    "capture_protocol",
    "load_wheel_capture",
    "wheel_gates",
    "wheel_health",
    "wheel_summary",
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
    "plan_information",
    "render_report_html",
    "run_checkup",
    "validate_capture_motion",
    "zmp_margins",
]
