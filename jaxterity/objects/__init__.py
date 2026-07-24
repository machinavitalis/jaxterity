# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Object-side identification: what a held object's dynamics reveal.

A camera cannot see mass, mass distribution, stiffness, or restitution — two
visually identical objects can demand different motions. This subpackage
recovers those invisible parameters from interaction, reusing the active-sysid
lane's honesty guards throughout (identifiability before the fit, physical
bounds during it, named uncertainty after it):

- :mod:`~jaxterity.objects.payload` — a grasped object's mass/CoM/inertia
  from the robot's own joint telemetry (the arm is the force scale).
"""

from .payload import (
    PayloadLoopbackLink,
    RigidPayload,
    attach_payload,
    identify_payload,
    payload_identifiability,
)

__all__ = [
    "RigidPayload",
    "attach_payload",
    "PayloadLoopbackLink",
    "payload_identifiability",
    "identify_payload",
]
