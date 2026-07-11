# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""The data-collection protocol an actuator/sensor declares for identification.

Each calibrated block (`Actuator`/`Sensor`) carries a :class:`SysidProtocol`
describing *how* to collect the telemetry that identifies its parameters — the
excitation signal, required telemetry columns, duration, sample rate, and
hardware setup. A sysid recipe consumes telemetry collected per
this protocol and produces a `SysidResult`.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class SysidProtocol(BaseModel):
    """A documented procedure for collecting parameter-identification telemetry.

    Args
    ----
    name : str
        Short identifier for the procedure (e.g. ``"voltage-ramp"``).
    excitation : str
        The excitation signal to apply (e.g. ``"chirp"``, ``"step"``, ``"ramp"``,
        ``"none"``).
    required_columns : tuple[str, ...]
        Telemetry channels the recipe needs (e.g. ``("q", "tau")``).
    duration_s : float
        Recommended recording duration in seconds.
    sample_rate_hz : float
        Recommended sample rate in hertz.
    setup_notes : str
        Hardware setup notes (e.g. ``"actuator free, no load"``).
    """

    model_config = ConfigDict(frozen=True)

    name: str
    excitation: str
    required_columns: tuple[str, ...]
    duration_s: float
    sample_rate_hz: float
    setup_notes: str = ""


__all__ = ["SysidProtocol"]
