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
"""

from ._base import Actuator, IdealTorqueActuator
from .bldc import BLDC
from .cable import CableActuator
from .dc_motor import DCMotorWithFriction
from .harmonic_drive import HarmonicDrive
from .series_elastic import SeriesElasticActuator
from .servo import Servo

__all__ = [
    "Actuator",
    "IdealTorqueActuator",
    "DCMotorWithFriction",
    "Servo",
    "BLDC",
    "HarmonicDrive",
    "SeriesElasticActuator",
    "CableActuator",
]
