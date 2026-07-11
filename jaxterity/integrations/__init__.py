# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Adapters that connect Jaxterity to external robot-learning ecosystems.

Jaxterity ships no policy code and no training stack (CONTEXT.md): training
lives upstream in libraries the user already runs. This subpackage is the thin,
documented seam where a Jaxterity-calibrated robot and its simulated rollouts
meet those libraries' on-disk and in-memory contracts.

- :mod:`jaxterity.integrations.lerobot` — read/write the
  `LeRobot <https://github.com/huggingface/lerobot>`_ ``LeRobotDataset`` format,
  so demonstrations collected in a Jaxterity sim are consumable by LeRobot
  policies (ACT, diffusion, …) and, conversely, real LeRobot teleoperation
  datasets can be replayed against a calibrated Jaxterity robot.
"""

from __future__ import annotations

from . import lerobot

__all__ = ["lerobot"]
