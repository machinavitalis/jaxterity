# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Active (hardware-in-the-loop) calibration.

Where ``jaxterity.sysid`` provides the *fit* (recipes, forward-mode Jacobians,
`SysidResult`) and ``examples/sysid_so101.py`` drives it offline, this package
closes the loop on a *physical* unit: it excites the robot over its servo bus,
captures the motion, and feeds the same fit — all from one
``jaxterity calibrate --active --port …`` invocation.

The transport is abstracted behind :class:`~jaxterity.calibrate.active.ActiveLink`
so the path is gate-testable: :class:`~jaxterity.calibrate.active.LoopbackLink`
is a seeded simulated arm; :class:`~jaxterity.calibrate.active.SerialBusLink` is
the real serial-bus transport (validated on hardware separately).
"""

from .active import (
    ActiveLink,
    Excitation,
    LoopbackLink,
    ReplayLink,
    SerialBusLink,
    active_identify,
    design_coast_excitation,
    dof_joint_names,
    excitation_identifiability,
    validate_excitation,
)
from .battery import BusLoadCapture, fit_battery_sag, load_bus_load_trace
from .closed_loop import (
    CommandedCapture,
    cross_trajectory_validation,
    identify_closed_loop,
    load_commanded_capture,
)
from .excitation import (
    HoldExcitation,
    design_hold_excitation,
    simulate_hold_window,
)
from .thermal import calibrate_twin_thermal, fit_servo_thermal
from .windowed import WindowedFitResult, refit_drift_parameters

__all__ = [
    "ActiveLink",
    "LoopbackLink",
    "ReplayLink",
    "SerialBusLink",
    "Excitation",
    "design_coast_excitation",
    "validate_excitation",
    "excitation_identifiability",
    "active_identify",
    "dof_joint_names",
    "CommandedCapture",
    "load_commanded_capture",
    "identify_closed_loop",
    "cross_trajectory_validation",
    "BusLoadCapture",
    "load_bus_load_trace",
    "fit_battery_sag",
    "fit_servo_thermal",
    "calibrate_twin_thermal",
    "refit_drift_parameters",
    "WindowedFitResult",
    "HoldExcitation",
    "design_hold_excitation",
    "simulate_hold_window",
]
