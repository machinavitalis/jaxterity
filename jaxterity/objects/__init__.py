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
- :mod:`~jaxterity.objects.probe` — differentiable optimal experiment design:
  gradient-optimize the probe itself for Fisher information about exactly
  those parameters.
- :mod:`~jaxterity.objects.contact` — contact/impact parameters: restitution
  from bounce times (differentiated *through the impacts* via the saltation
  lane), stiffness/damping from a squeeze's force–deformation trace.
"""

from .contact import (
    identify_restitution,
    identify_squeeze,
    predict_bounce_times,
)
from .payload import (
    PayloadLoopbackLink,
    RigidPayload,
    attach_payload,
    identify_payload,
    payload_identifiability,
)
from .probe import design_probe, probe_information

__all__ = [
    "RigidPayload",
    "attach_payload",
    "PayloadLoopbackLink",
    "payload_identifiability",
    "identify_payload",
    "design_probe",
    "probe_information",
    "identify_restitution",
    "identify_squeeze",
    "predict_bounce_times",
]
