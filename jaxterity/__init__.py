# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Jaxterity — the JAX-native robotics layer on top of Jaxonomy.

The core is in place: ``jaxterity.robot`` (``Robot``, ``Joint``,
``Link``, ``Frame``, URDF/MJCF import, MJX-backed ``to_diagram``), ``jaxterity.sysid``
(``SysidResult`` and forward-mode ``parameter_jacobian``), ``jaxterity.zoo``
(reference robots), the attestation handle, calibrated actuator/sensor libraries,
sysid recipes, and whole-body control. See ``AGENTS/CONTEXT.md``
for the architecture.

Submodules import heavy optional deps (JAX, MuJoCo) lazily, so importing the top
package stays cheap; pull in what you need (e.g. ``from jaxterity.robot import
Robot``).
"""

__version__ = "1.0.1"

__all__ = ["__version__"]
