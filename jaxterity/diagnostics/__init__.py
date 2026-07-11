# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Sim-to-real diagnostics — ``parity_report``, the Rerun adapters, and
attestation-handle helpers.

``RobotRerunLogger`` / ``log_robot_trajectory`` stream a robot's 3D
articulated state + per-joint signals to Rerun; both keep ``rerun-sdk`` (and
MuJoCo) imports lazy, so importing this package never requires the optional
visualization stack.
"""

from .attestation import ALGORITHM_VERSION, compute_handle
from .attestation_log import AttestationLog, AttestationRecord
from .drift_view import DriftView, log_drift_view, prepare_drift_view
from .evidence import SCHEMA_VERSION, EvidenceRecord, build_evidence
from .parity import ParityReport, parity_report
from .rerun_robot import RobotRerunLogger, log_robot_trajectory

__all__ = [
    "compute_handle",
    "ALGORITHM_VERSION",
    "AttestationLog",
    "AttestationRecord",
    "EvidenceRecord",
    "build_evidence",
    "SCHEMA_VERSION",
    "ParityReport",
    "parity_report",
    "RobotRerunLogger",
    "log_robot_trajectory",
    "DriftView",
    "prepare_drift_view",
    "log_drift_view",
]
