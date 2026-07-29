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

Two groups live here. **Physical sensors** (:class:`IMU`, :class:`Encoder`,
:class:`FTSensor`, :class:`Magnetometer`) model an instrument measuring the
world. **Measurement channels** (:class:`CurrentSense`,
:class:`TemperatureSensor`, :class:`BusChannel`) model the path *between* a
true quantity and the number a fit consumes — gain, offset, bandwidth,
quantization, transport latency. They exist because every one of those errors,
left unmodeled, is absorbed into a physical parameter and reported as a
property of the robot: a current gain becomes a torque constant, a sensor lag
becomes thermal mass, a bus latency becomes joint damping.
"""

from ._base import IdealEncoder, Sensor
from .bus_channel import BusChannel, phase_lag_deg
from .current_sense import (
    CurrentSense,
    adc_step,
    quantization_step_estimate,
    quantize,
)
from .encoder import Encoder
from .ft_sensor import FTSensor
from .imu import IMU
from .magnetometer import Magnetometer
from .temperature_sensor import TemperatureSensor, quantization_floor

__all__ = [
    "Sensor",
    "IdealEncoder",
    "IMU",
    "Encoder",
    "FTSensor",
    "Magnetometer",
    # measurement channels
    "BusChannel",
    "CurrentSense",
    "TemperatureSensor",
    "adc_step",
    "phase_lag_deg",
    "quantization_floor",
    "quantization_step_estimate",
    "quantize",
]
