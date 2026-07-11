# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Sensor base + trivial IdealEncoder reference.

The base lives in its own module (not ``sensors/__init__.py``) so concrete
sensors like :class:`~jaxterity.sensors.imu.IMU` can import :class:`Sensor`
from the sub-package without inducing a circular import. Mirrors the
``actuators/_base.py`` layout.
"""

from __future__ import annotations

import abc

from jaxonomy.framework.leaf_system import LeafSystem

from ..data.schema import ValidationReport
from ..parameter import ParameterSpec
from ..sysid.protocol import SysidProtocol


class Sensor(LeafSystem, abc.ABC):
    """Abstract base for calibrated sensor models.

    Subclasses set :attr:`sysid_protocol` and implement :meth:`parameter_spec`
    and :meth:`validate`. Instantiating ``Sensor`` directly raises ``TypeError``.
    """

    #: How to collect telemetry that identifies this sensor's parameters.
    sysid_protocol: SysidProtocol

    @abc.abstractmethod
    def parameter_spec(self) -> ParameterSpec:
        """Return the identifiable parameter leaves a sysid recipe may fit."""

    @abc.abstractmethod
    def validate(self, telemetry) -> ValidationReport:
        """Return a structured residual report of this model against telemetry."""


class IdealEncoder(Sensor):
    """Measures its input exactly (no bias, noise, quantization, or delay).

    Input port ``measured`` (e.g. a joint position) is passed straight through to
    output port ``reading``.

    Args
    ----
    name : str
        Block name.
    """

    sysid_protocol = SysidProtocol(
        name="ideal_encoder",
        excitation="none",
        required_columns=(),
        duration_s=0.0,
        sample_rate_hz=0.0,
        setup_notes="Ideal encoder: reading == measured; nothing to identify.",
    )

    def __init__(self, name: str = "ideal_encoder"):
        super().__init__(name=name)
        self.declare_input_port(name="measured")
        self.declare_output_port(self._reading, name="reading", requires_inputs=True)

    def _reading(self, time, state, *inputs, **params):
        return inputs[0]

    def parameter_spec(self) -> ParameterSpec:
        """An ideal encoder has no identifiable parameters."""
        return ParameterSpec(())

    def validate(self, telemetry) -> ValidationReport:
        """Ideal encoder: no residual to report."""
        return ValidationReport()


__all__ = ["Sensor", "IdealEncoder"]
