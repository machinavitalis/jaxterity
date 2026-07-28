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
from .report import (
    SCHEMA,
    BatteryCheckup,
    CheckupReport,
    Gate,
    JointCheckup,
    run_checkup,
)
from .store import CheckupStore

__all__ = [
    "SCHEMA",
    "BatteryCheckup",
    "CheckupReport",
    "CheckupStore",
    "Gate",
    "JointCheckup",
    "render_report_html",
    "run_checkup",
]
