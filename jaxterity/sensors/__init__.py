# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Calibrated sensor-model library.

A :class:`Sensor` is a Jaxonomy ``LeafSystem`` symmetric to ``Actuator``: it
carries a :class:`~jaxterity.sysid.protocol.SysidProtocol`, a
:meth:`parameter_spec`, and :meth:`validate`. The base lives in
``sensors._base`` so concrete sensors can import it without inducing a
circular import through this ``__init__``.

:class:`IdealEncoder` is the trivial reference. :class:`IMU` is the
first realistic sensor (accel + gyro with bias / noise / misalignment).
"""

from ._base import IdealEncoder, Sensor
from .encoder import Encoder
from .ft_sensor import FTSensor
from .imu import IMU
from .magnetometer import Magnetometer

__all__ = ["Sensor", "IdealEncoder", "IMU", "Encoder", "FTSensor", "Magnetometer"]
