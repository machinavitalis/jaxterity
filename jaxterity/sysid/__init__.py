# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""System-identification recipes — functions ``(telemetry, robot, hints) ->
SysidResult`` producing a calibrated ``Robot``.

This package defines the shared sysid types: :class:`SysidProtocol` (how to
collect identification data), :class:`SysidResult` (the structured fit
outcome), and the :class:`SysidRecipe` protocol. It also provides
:func:`parameter_jacobian`, the forward-mode differentiation primitive that lets
recipes get gradients w.r.t. robot parameters *through* MJX's constraint solver
(joint limits / contacts), where reverse-mode autodiff cannot. Concrete recipes
include ``actuator``, ``inertial``, ``full_body``, and ``contact``.
"""

from .diff import parameter_jacobian
from .hybrid import (
    ABSORBED_WARN_FRACTION,
    HUBER_DELTA_RAD,
    ModelCompleteness,
    ResidualModel,
    fit_residual,
    model_completeness,
    residual_gates,
)
from .protocol import SysidProtocol
from .result import SysidRecipe, SysidResult
from .uncertainty import (
    bootstrap_covariance,
    laplace_covariance,
    per_parameter_variance,
)

#: Designed-excitation + identifiability-checked recipes, lazily loaded.
_RECIPE_NAMES = frozenset(
    {
        "IdentifiabilityReport",
        "identifiability_from_jacobian",
        "design_dc_motor_excitation",
        "dc_motor_identifiability",
        "dc_motor_recipe",
        "design_bldc_excitation",
        "bldc_identifiability",
        "bldc_recipe",
    }
)

__all__ = [
    "ABSORBED_WARN_FRACTION",
    "HUBER_DELTA_RAD",
    "ModelCompleteness",
    "ResidualModel",
    "fit_residual",
    "model_completeness",
    "residual_gates",
    "SysidProtocol",
    "SysidResult",
    "SysidRecipe",
    "parameter_jacobian",
    "laplace_covariance",
    "bootstrap_covariance",
    "per_parameter_variance",
    "actuator",
    "inertial",
    "full_body",
    "contact",
    *sorted(_RECIPE_NAMES),
]


def __getattr__(name):
    # Lazy imports of the recipes — their implementations live in private
    # modules (_actuator.py / _robot_recipes.py) to break a circular import:
    # the actuators and sensors sub-packages' _base modules import from
    # .protocol, which re-triggers this __init__ before it has finished. The
    # private file names avoid the package/submodule attribute collision a
    # public name would cause (e.g. ``sysid.actuator`` would otherwise
    # resolve to the submodule rather than the function).
    if name == "actuator":
        from ._actuator import actuator as _actuator

        return _actuator
    if name in ("inertial", "full_body", "contact"):
        from . import _robot_recipes

        return getattr(_robot_recipes, name)
    if name in _RECIPE_NAMES:
        # recipes.py imports the actuator fitters (which import .protocol), so it
        # too is loaded lazily, after this package finishes initializing.
        from . import recipes

        return getattr(recipes, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
