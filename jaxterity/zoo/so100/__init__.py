# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""SO-100 / SO-101 reference robot — a 6-DoF desktop manipulator (LeRobot
ecosystem).

Source: TheRobotStudio/SO-ARM100 (``Simulation/SO101/so101_new_calib.urdf``),
Apache-2.0. The vendored URDF is unmodified; its mesh references are kept as
opaque handles (geometry is not loaded — inertials drive the dynamics).
"""

from __future__ import annotations

import pathlib

from ...robot import Robot, from_urdf

URDF_PATH = pathlib.Path(__file__).with_name("so101.urdf")
SOURCE = "https://github.com/TheRobotStudio/SO-ARM100 (Simulation/SO101)"
LICENSE = "Apache-2.0"


def load() -> Robot:
    """Load the (uncalibrated) SO-100/SO-101 6-DoF manipulator.

    Returns
    -------
    Robot
        Six revolute joints (shoulder_pan/lift, elbow_flex, wrist_flex/roll,
        gripper) plus a fixed gripper frame, built from the vendored URDF.
    """
    return from_urdf(URDF_PATH)


__all__ = ["load", "URDF_PATH", "SOURCE", "LICENSE"]
