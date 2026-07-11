# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Actuator base class and the trivial ``IdealTorqueActuator`` reference.

The base lives in its own module (not ``actuators/__init__.py``) so concrete
actuators like :class:`~jaxterity.actuators.dc_motor.DCMotorWithFriction` can
import :class:`Actuator` from the sub-package without inducing a circular
import through ``__init__.py``.
"""

from __future__ import annotations

import abc

import jax.numpy as jnp
from jaxonomy.framework.leaf_system import LeafSystem

from ..data.schema import ValidationReport
from ..parameter import ParameterSpec
from ..sysid.protocol import SysidProtocol


class Actuator(LeafSystem, abc.ABC):
    """Abstract base for calibrated actuator models.

    Subclasses set :attr:`sysid_protocol` and implement :meth:`parameter_spec`
    and :meth:`validate`. Instantiating ``Actuator`` directly raises ``TypeError``.
    """

    #: How to collect telemetry that identifies this actuator's parameters.
    sysid_protocol: SysidProtocol

    @abc.abstractmethod
    def parameter_spec(self) -> ParameterSpec:
        """Return the identifiable parameter leaves a sysid recipe may fit."""

    @abc.abstractmethod
    def validate(self, telemetry) -> ValidationReport:
        """Return a structured residual report of this model against telemetry."""


class IdealTorqueActuator(Actuator):
    """Applies a commanded torque exactly (no dynamics, nothing to identify).

    Output port ``tau`` returns the dynamic parameter ``tau`` (a length-``n_dof``
    torque vector), suitable for wiring into a ``Robot``'s torque input.

    Args
    ----
    n_dof : int
        Number of actuated degrees of freedom (output length).
    tau : array-like | None
        Initial commanded torque (default zeros).
    name : str
        Block name.
    """

    sysid_protocol = SysidProtocol(
        name="ideal",
        excitation="none",
        required_columns=(),
        duration_s=0.0,
        sample_rate_hz=0.0,
        setup_notes="Ideal actuator: applies the commanded torque exactly; "
        "there is nothing to identify.",
    )

    def __init__(self, n_dof: int, tau=None, name: str = "ideal_torque"):
        super().__init__(name=name)
        tau0 = jnp.zeros(n_dof) if tau is None else jnp.asarray(tau, dtype=float)
        self.declare_dynamic_parameter("tau", tau0)
        self.declare_output_port(self._torque, name="tau", requires_inputs=False)

    def _torque(self, time, state, *inputs, **params):
        return params["tau"]

    def parameter_spec(self) -> ParameterSpec:
        """An ideal actuator has no identifiable physical parameters."""
        return ParameterSpec(())

    def validate(self, telemetry) -> ValidationReport:
        """Ideal actuator: no residual to report."""
        return ValidationReport()


__all__ = ["Actuator", "IdealTorqueActuator"]
