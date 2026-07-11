# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Off-body physics blocks the body (MJX) model never sees.

A calibrated robot is more than its rigid-body dynamics: a *specific unit* runs
hot under load and (on a shared battery) sags its bus voltage. These effects
live off the MJX model, so they are provided as standalone Jaxonomy
``LeafSystem``s that compose into the per-unit twin:

- :class:`~jaxterity.physics.thermal.ServoThermal` — single-node lumped
  servo-thermal model (winding/case temperature under a torque²-proxy of
  dissipated power). Bench SO-101 + LeKiwi.
- :class:`~jaxterity.physics.battery.BatterySag` — Thévenin bus-voltage sag
  under aggregate servo current. LeKiwi-only.

Both are pure/JAX-traceable and differentiable; see each module's docstring for
the model, units, and (for the thermal block) the identifiability contract.
"""

from .battery import BatterySag, bus_voltage
from .thermal import (
    ServoThermal,
    dissipated_power,
    thermal_response,
    thermal_steady_state,
)

__all__ = [
    "ServoThermal",
    "dissipated_power",
    "thermal_steady_state",
    "thermal_response",
    "BatterySag",
    "bus_voltage",
]
