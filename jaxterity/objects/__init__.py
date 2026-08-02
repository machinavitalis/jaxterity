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
- :mod:`~jaxterity.objects.priors` — a declared semantic guess (a VLM naming
  the object) turned into a seed and a search box for the fit, recorded in the
  result's provenance so a seeded fit reads as seeded.
- :mod:`~jaxterity.objects.scene` — scene enrollment: the whole arc, from a
  vision prior through a worst-case-vetted heft to an object mini-twin and the
  coupled robot+object verdict on the technique that follows.
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
from .priors import MATERIAL_DENSITY_KG_M3, SemanticPrior
from .probe import design_probe, probe_information, probe_margin
from .scene import (
    DEFAULT_PROBE_SPEEDS,
    ObjectEnrollment,
    ProbeVetting,
    enroll_object,
    probe_speed_margin_curve,
    vet_probe,
    worst_case_payload,
)

__all__ = [
    "RigidPayload",
    "attach_payload",
    "PayloadLoopbackLink",
    "payload_identifiability",
    "identify_payload",
    "SemanticPrior",
    "MATERIAL_DENSITY_KG_M3",
    "design_probe",
    "probe_margin",
    "probe_information",
    "identify_restitution",
    "identify_squeeze",
    "predict_bounce_times",
    "ObjectEnrollment",
    "ProbeVetting",
    "DEFAULT_PROBE_SPEEDS",
    "enroll_object",
    "vet_probe",
    "worst_case_payload",
    "probe_speed_margin_curve",
]
