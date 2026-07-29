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

Those two are *lumped causal shortcuts*: one rail, one thermal node. When the
question is a network — which servo sags most, how a 5 V regulator loads the
pack, whether the winding or the housing is the hot one — the model belongs in
the acausal (Modelica-style) layer instead, built on Jaxonomy's electrical and
thermal domains:

- :mod:`jaxterity.physics.power_tree` — :class:`PowerTree`: pack (SoC-dependent
  OCV, aging), harness resistance per branch, DC-DC regulators, charger.
- :mod:`jaxterity.physics.thermal_network` — :class:`ThermalNetwork`: two-node
  winding/housing per servo, driver/amplifier nodes, inter-servo conduction,
  ambient coupling.

and the structural/environmental effects the body model leaves out:

- :mod:`jaxterity.physics.compliance` — suspension/wheel compliance (which
  tightens a tip-over margin), joint torsional compliance, rolling resistance.

The acausal builders need ``sympy`` (``pip install 'jaxterity[acausal]'``);
importing this package without it is fine, building a network is not.
"""

from .battery import BatterySag, bus_voltage
from .compliance import (
    LEKIWI_SUSPENSION,
    JointCompliance,
    RollingResistance,
    SuspensionProfile,
    compliant_margin,
    fit_coast_down,
    margin_floor_credit,
    rolling_resistance_force,
    stiffness_from_resonance,
    suspension_tilt,
    torsional_resonance_hz,
)
from .power_tree import (
    LEKIWI_POWER_TREE,
    CellAging,
    CellSpec,
    Charger,
    ChargerSpec,
    DCDCConverter,
    LoadSpec,
    PowerTree,
    PowerTreeSolution,
    PowerTreeSpec,
    RegulatorSpec,
    dc_solve,
    fit_power_tree,
    servo_currents,
)
from .thermal import (
    ServoThermal,
    dissipated_power,
    thermal_response,
    thermal_steady_state,
)
from .thermal_network import (
    AmbientSpec,
    ConductionSpec,
    DriverSpec,
    ServoThermalSpec,
    ThermalNetwork,
    ThermalNetworkSpec,
    fit_two_node,
    steady_state_temperatures,
    two_node_response,
    two_node_steady_state,
    validate_two_node,
)

__all__ = [
    "ServoThermal",
    "dissipated_power",
    "thermal_steady_state",
    "thermal_response",
    "BatterySag",
    "bus_voltage",
    # power tree (acausal electrical)
    "CellAging",
    "CellSpec",
    "Charger",
    "ChargerSpec",
    "DCDCConverter",
    "LEKIWI_POWER_TREE",
    "LoadSpec",
    "PowerTree",
    "PowerTreeSolution",
    "PowerTreeSpec",
    "RegulatorSpec",
    "dc_solve",
    "fit_power_tree",
    "servo_currents",
    # thermal network (acausal thermal)
    "AmbientSpec",
    "ConductionSpec",
    "DriverSpec",
    "ServoThermalSpec",
    "ThermalNetwork",
    "ThermalNetworkSpec",
    "fit_two_node",
    "steady_state_temperatures",
    "two_node_response",
    "two_node_steady_state",
    "validate_two_node",
    # structure / environment
    "JointCompliance",
    "LEKIWI_SUSPENSION",
    "RollingResistance",
    "SuspensionProfile",
    "compliant_margin",
    "fit_coast_down",
    "margin_floor_credit",
    "rolling_resistance_force",
    "stiffness_from_resonance",
    "suspension_tilt",
    "torsional_resonance_hz",
]
