# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Cartpole reference robot — the minimum-viable zoo entry for learning the
workflow (a cart on a slider with a freely-swinging pole).

Source: bulletphysics/bullet3 (``examples/pybullet/gym/pybullet_data/
cartpole.urdf``), zlib license. See ``README.md`` and ``AGENTS/LIBRARIES.md``.
"""

from __future__ import annotations

import pathlib

from ...errors import JaxterityError
from ...robot import Robot, from_urdf

URDF_PATH = pathlib.Path(__file__).with_name("cartpole.urdf")
SOURCE = "https://github.com/bulletphysics/bullet3 (pybullet_data/cartpole.urdf)"
LICENSE = "zlib"

#: Link names in the vendored Cartpole URDF, used by :func:`reduced_params`.
_CART_LINK = "cart"
_POLE_LINK = "pole"


def load() -> Robot:
    """Load the (uncalibrated) Cartpole reference robot.

    Returns
    -------
    Robot
        Links ``slideBar`` (fixed root) → ``cart`` (prismatic) → ``pole``
        (continuous), built from the vendored URDF.
    """
    return from_urdf(URDF_PATH)


def reduced_params(robot: Robot) -> dict[str, float]:
    """Closed-form Cartpole parameters, read from a (calibrated) ``robot``.

    Jaxility deploys Cartpole through a four-state **closed-form** point-mass
    model, not the MJX dynamics: MJX's constraint-solver ``while_loop`` cannot
    be lowered to acados' fixed-size SQP graph (jaxility ADR-016, *"closed-form
    per-robot dynamics is the contract"*). For the deployed binary to stay
    faithful to *this* robot — so calibrating a mass or a length actually
    changes what runs on the target, and the attestation handle and the lowered
    dynamics move together (Invariant 1, *"one model, one truth"*) — the
    closed-form's scalars are sourced here from the robot's fitted leaves rather
    than hand-hardcoded downstream.

    Returns the four scalars the canonical frictionless cartpole model uses:
    ``g`` (``robot.gravity``), ``mc`` (cart mass), ``mp`` (pole mass), and
    ``L`` (hinge→pole-CoM distance, the pole link's CoM z-offset).

    Joint damping/friction are intentionally **excluded**: the deployment
    closed-form — and jaxility's host-equivalence reference — is the
    frictionless point-mass cartpole; viscous terms are a stabilising margin the
    LQR/MPC absorbs, not part of the lowered plant. They remain in the MJX
    simulation model, so sim and deployment differ there by design.
    """
    lm = robot.link_map
    for name in (_CART_LINK, _POLE_LINK):
        if name not in lm:
            raise JaxterityError(
                f"reduced_params expects a Cartpole topology with a {name!r} "
                f"link; got links {sorted(lm)}."
            )
    return {
        "g": float(robot.gravity),
        "mc": float(lm[_CART_LINK].mass),
        "mp": float(lm[_POLE_LINK].mass),
        "L": float(lm[_POLE_LINK].com[2]),
    }


__all__ = ["load", "reduced_params", "URDF_PATH", "SOURCE", "LICENSE"]
