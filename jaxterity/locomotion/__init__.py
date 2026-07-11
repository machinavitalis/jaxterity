# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Locomotion controllers for legged zoo robots.

Currently: the Unitree G1 joystick walking controller — a vendored,
pretrained MuJoCo Playground policy reimplemented in pure JAX over the
vendored mesh-free locomotion scene. See
:mod:`jaxterity.locomotion.g1_joystick`.
"""

from .g1_joystick import (
    G1JoystickController,
    G1JoystickPolicy,
    WalkRollout,
    load_walk_model,
)

__all__ = [
    "G1JoystickController",
    "G1JoystickPolicy",
    "WalkRollout",
    "load_walk_model",
]
