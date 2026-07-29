# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Telemetry schema, dataset loaders, and mock-data generators (Pydantic
models in ``data.schema``).
"""

from .loaders import (
    load_telemetry,
    save_csv,
    save_hdf5,
    save_parquet,
)
from .schema import (
    ActuatorChirpTelemetry,
    ActuatorCommandTelemetry,
    ActuatorRampTelemetry,
    ActuatorStepTelemetry,
    BacklashTelemetry,
    BusLatencyTelemetry,
    BusVoltageTelemetry,
    ChannelSpec,
    CurrentSenseTelemetry,
    EncoderTelemetry,
    FTSensorTelemetry,
    IMUTelemetry,
    JointTelemetry,
    LinearTransmissionTelemetry,
    MagnetometerTelemetry,
    MocapTelemetry,
    PneumaticTelemetry,
    SeriesElasticTelemetry,
    Telemetry,
    TemperatureSensorTelemetry,
    ThermalTelemetry,
    TorqueStepTelemetry,
    TransmissionTelemetry,
    ValidationReport,
)

__all__ = [
    "Telemetry",
    "JointTelemetry",
    "IMUTelemetry",
    "FTSensorTelemetry",
    "MagnetometerTelemetry",
    "MocapTelemetry",
    "ActuatorCommandTelemetry",
    "ActuatorChirpTelemetry",
    "ActuatorRampTelemetry",
    "ActuatorStepTelemetry",
    "EncoderTelemetry",
    "SeriesElasticTelemetry",
    "TransmissionTelemetry",
    "BacklashTelemetry",
    "LinearTransmissionTelemetry",
    "TorqueStepTelemetry",
    "PneumaticTelemetry",
    "CurrentSenseTelemetry",
    "TemperatureSensorTelemetry",
    "BusLatencyTelemetry",
    "ThermalTelemetry",
    "BusVoltageTelemetry",
    "ChannelSpec",
    "ValidationReport",
    "load_telemetry",
    "save_csv",
    "save_parquet",
    "save_hdf5",
]
