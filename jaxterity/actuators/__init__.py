# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Calibrated actuator-model library.

An :class:`Actuator` is a Jaxonomy ``LeafSystem`` with the calibration metadata
every motor model needs: a :class:`~jaxterity.sysid.protocol.SysidProtocol`
(how to collect identification data), a :meth:`parameter_spec` (which leaves a
sysid recipe may fit), and :meth:`validate` (a structured residual report). The
base lives in ``actuators._base`` so concrete actuators can import it without
inducing a circular import through this ``__init__``.

:class:`IdealTorqueActuator` is the trivial reference — applies a
commanded torque exactly. :class:`DCMotorWithFriction` is the first physically
realistic motor (WPILib SysId model).

The library is organized by what a block *is*, because that is also what its
calibration recipe is:

- **Motors** — :class:`DCMotorWithFriction`, :class:`BLDC`, :class:`Servo`,
  :class:`QuasiDirectDrive` (the humanoid/quadruped case: a low-ratio stage
  behind a big-gap motor), :class:`Stepper` (open-loop, with the pull-out
  cliff), :class:`PneumaticCylinder` (chamber pressure, not current).
- **Transmissions** — :class:`Gearbox` (two-quadrant efficiency + loss),
  :class:`HarmonicDrive` (the single-encoder first cut),
  :class:`PositionalBacklash` (true angular play, dual-encoder),
  :class:`BeltDrive` (compliance first), :class:`LeadScrew` (rotary→linear,
  often non-backdrivable), :class:`CableActuator`,
  :class:`SeriesElasticActuator`.

Each block's module docstring states what it does *not* model; read it before
assuming a fitted number means what its name suggests.
"""

from ._base import Actuator, IdealTorqueActuator
from .backlash import (
    PositionalBacklash,
    backlash_torque,
    deadzone,
    engagement,
    lost_motion,
)
from .belt import BeltDrive, belt_force, belt_resonance_hz
from .bldc import BLDC
from .cable import CableActuator
from .dc_motor import DCMotorWithFriction
from .gearbox import Gearbox, gearbox_torque
from .harmonic_drive import HarmonicDrive
from .lead_screw import LeadScrew, is_backdrivable, screw_efficiency, screw_force
from .pneumatic import PneumaticCylinder, chamber_pressure_rate, orifice_flow
from .qdd import QuasiDirectDrive, qdd_torque_envelope
from .series_elastic import SeriesElasticActuator
from .servo import Servo
from .stepper import Stepper, pull_out_margin, stepper_torque

__all__ = [
    "Actuator",
    "IdealTorqueActuator",
    "DCMotorWithFriction",
    "Servo",
    "BLDC",
    "HarmonicDrive",
    "SeriesElasticActuator",
    "CableActuator",
    # transmissions
    "BeltDrive",
    "Gearbox",
    "LeadScrew",
    "PositionalBacklash",
    "backlash_torque",
    "belt_force",
    "belt_resonance_hz",
    "deadzone",
    "engagement",
    "gearbox_torque",
    "is_backdrivable",
    "lost_motion",
    "screw_efficiency",
    "screw_force",
    # actuation families
    "PneumaticCylinder",
    "QuasiDirectDrive",
    "Stepper",
    "chamber_pressure_rate",
    "orifice_flow",
    "pull_out_margin",
    "qdd_torque_envelope",
    "stepper_torque",
]
