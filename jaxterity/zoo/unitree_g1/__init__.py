# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Unitree G1 reference robot — a 29-DoF humanoid (the zoo's first biped).

Source: google-deepmind/mujoco_menagerie (``unitree_g1/g1.xml``), itself derived
from unitreerobotics/unitree_ros (``g1_29dof_rev_1_0.xml``). Upstream model is
(c) Unitree Robotics under **BSD-3-Clause** — see the vendored ``LICENSE`` in
this directory and ``AGENTS/LIBRARIES.md``.

The vendored ``g1.xml`` is a **mesh-free reduction**: the kinematic tree, joints
(with explicit per-joint ``armature`` baked back from the mujoco_playground pin
— the menagerie source carries only a placeholder ``armature="0.01"``;
``frictionloss`` is the menagerie uniform ``0.3``), and explicit ``<inertial>``
elements are kept; all ``<geom>``/``<mesh>``/``<site>`` geometry, actuators,
sensors, and the keyframe are stripped, because Jaxterity drives dynamics from
inertials, not geometry (LIBRARIES.md, same policy as the SO-100 URDF).

Two body topologies are available:

- :func:`load` (default) returns the **fixed-base** humanoid — the pelvis is
  welded to the world, exactly like the fixed-base zoo arms. This is the form
  the ``nq == nv`` zoo free-fall regression exercises.
- ``load(floating_base=True)`` returns the **floating-base** humanoid — a
  ``FLOATING`` joint (ADR-018) connects the world to the pelvis, giving the
  6-DoF free base a real biped needs. Use this for whole-body / locomotion work.
"""

from __future__ import annotations

import dataclasses
import pathlib

import jax.numpy as jnp

from ...errors import JaxterityError
from ...robot import (
    CollisionGeom,
    Frame,
    FramedQuantity,
    Joint,
    JointLimits,
    JointType,
    Position,
    Robot,
    Transform,
    from_mjcf,
)

#: Vendored mesh-free MJCF. Named ``MJCF_PATH``; ``URDF_PATH`` aliases it so the
#: generic zoo machinery (golden-path lookup) — which addresses every entry's
#: asset via ``URDF_PATH`` — works unchanged for this MJCF-sourced robot.
MJCF_PATH = pathlib.Path(__file__).with_name("g1.xml")
URDF_PATH = MJCF_PATH
SOURCE = "https://github.com/google-deepmind/mujoco_menagerie (unitree_g1/g1.xml)"
LICENSE = "BSD-3-Clause"

#: The world-welded root body of the imported (fixed-base) model.
_ROOT_LINK = "pelvis"

#: Foot collision boxes, one per ankle-roll link (MuJoCo half-size convention,
#: in the link's body frame). Values match the box the MuJoCo Playground G1
#: locomotion stack uses (``g1_mjx_feetonly.xml``: pos ``0.04 0 -0.029``, size
#: ``0.09 0.03 0.008``), so contact behaves like the model the vendored walking
#: policy was trained on.
_FOOT_GEOMS = {
    "left_ankle_roll_link": CollisionGeom(
        kind="box",
        size=(0.09, 0.03, 0.008),
        pos=(0.04, 0.0, -0.029),
        name="left_foot",
    ),
    "right_ankle_roll_link": CollisionGeom(
        kind="box",
        size=(0.09, 0.03, 0.008),
        pos=(0.04, 0.0, -0.029),
        name="right_foot",
    ),
}


def _with_floating_base(robot: Robot) -> Robot:
    """Attach a 6-DoF :class:`JointType.FLOATING` base (world → pelvis).

    ``from_mjcf`` imports the upstream ``<freejoint>`` as a welded root (ADR-018:
    importers don't yet round-trip free joints), so the floating base is
    reconstructed here as a single ``FLOATING`` joint — the supported way to
    express a free base.
    """
    free = Joint(
        name="floating_base",
        joint_type=JointType.FLOATING,
        parent_link="world",
        child_link=_ROOT_LINK,
        axis=FramedQuantity(Position(jnp.array([0.0, 0.0, 1.0])), Frame("world")),
        limits=JointLimits(lower=-1.0, upper=1.0, effort=1.0, velocity=1.0),
        origin=Transform.identity(Frame("world")),
        damping=jnp.asarray(0.0),
        friction=jnp.asarray(0.0),
    )
    return dataclasses.replace(robot, joints=(free,) + robot.joints)


def _with_foot_contact(robot: Robot) -> Robot:
    """Attach foot collision boxes + the world floor plane."""
    links = tuple(
        dataclasses.replace(link, collision=(_FOOT_GEOMS[link.name],))
        if link.name in _FOOT_GEOMS
        else link
        for link in robot.links
    )
    return dataclasses.replace(robot, links=links, ground_plane=True)


def load(floating_base: bool = False, contact: bool = False) -> Robot:
    """Load the (uncalibrated) Unitree G1 29-DoF humanoid.

    Args
    ----
    floating_base : bool, default False
        If ``False``, the pelvis is welded to the world (fixed base,
        ``nq == nv == 29``) — the form the zoo regression harness uses. If
        ``True``, a ``FLOATING`` joint gives the pelvis a 6-DoF free base
        (``nq == 36``, ``nv == 35``; ADR-018), as a real biped requires.
    contact : bool, default False
        Attach foot collision boxes (one per ankle-roll link, matching the
        MuJoCo Playground locomotion model) and a world floor plane, enabling
        foot–ground contact. Requires ``floating_base=True`` — the
        fixed-base form welds the pelvis at the world origin, which would put
        the legs *below* the floor.

    Returns
    -------
    Robot
        29 revolute joints — per leg: hip pitch/roll/yaw, knee, ankle
        pitch/roll; waist yaw/roll/pitch; per arm: shoulder pitch/roll/yaw,
        elbow, wrist roll/pitch/yaw — built from the vendored mesh-free MJCF,
        optionally on a floating base, optionally with foot–ground contact.

    Raises
    ------
    JaxterityError
        If ``contact=True`` without ``floating_base=True``.

    Examples
    --------
    >>> from jaxterity import zoo
    >>> robot = zoo.load("unitree_g1")              # fixed-base, 29 DoF
    >>> humanoid = zoo.unitree_g1.load(floating_base=True)  # 6-DoF free base
    >>> walker = zoo.unitree_g1.load(floating_base=True, contact=True)
    """
    if contact and not floating_base:
        raise JaxterityError(
            "unitree_g1.load(contact=True) requires floating_base=True: the "
            "fixed-base form welds the pelvis at the world origin, so the legs "
            "would sit below the z=0 floor plane. Load with "
            "load(floating_base=True, contact=True) and set the free-base "
            "height in q0 (pelvis ≈ 0.79 m for standing)."
        )
    robot = from_mjcf(MJCF_PATH)
    if floating_base:
        robot = _with_floating_base(robot)
    if contact:
        robot = _with_foot_contact(robot)
    return robot


__all__ = ["load", "MJCF_PATH", "URDF_PATH", "SOURCE", "LICENSE"]
