# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Layered code-as-policies API — S1 (high-level) through S4 (raw
torque/encoder) interfaces written for agent use.

- :class:`S4` — raw actuator + encoder interface.
- :class:`S3` — task-level: set task priorities, step WBC / MPC.
- :mod:`s2` — primitive skills: ``move_joint``, ``track_trajectory``.
- :mod:`s1` — high-level: ``reach``, ``pick``, ``walk_to``.

Each layer is a thin wrapper over the one below; the contract is that an
LLM can use just one layer's documented surface (the docstrings, not the
implementation) to assemble a working policy.
"""

from . import eval_harness, s1, s2
from .feedforward import (
    TrackingReport,
    TrackingTrajectory,
    design_tracking_trajectory,
    feedforward_tracking,
)
from .s3 import S3
from .s4 import S4

__all__ = [
    "S4",
    "S3",
    "s2",
    "s1",
    "eval_harness",
    "design_tracking_trajectory",
    "feedforward_tracking",
    "TrackingTrajectory",
    "TrackingReport",
]
