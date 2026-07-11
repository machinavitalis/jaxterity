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
    BusVoltageTelemetry,
    ChannelSpec,
    EncoderTelemetry,
    FTSensorTelemetry,
    IMUTelemetry,
    JointTelemetry,
    MagnetometerTelemetry,
    MocapTelemetry,
    SeriesElasticTelemetry,
    Telemetry,
    ThermalTelemetry,
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
    "ThermalTelemetry",
    "BusVoltageTelemetry",
    "ChannelSpec",
    "ValidationReport",
    "load_telemetry",
    "save_csv",
    "save_parquet",
    "save_hdf5",
]
