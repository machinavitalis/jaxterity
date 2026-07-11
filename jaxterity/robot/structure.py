# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Kinematic structure: ``Joint`` and ``Link``.

A robot's kinematic tree is a graph of :class:`Link` (rigid bodies with inertial
parameters) connected by :class:`Joint` (the articulation between two links).
Both are ``equinox.Module`` PyTrees: their *identifiable* parameters (link
inertials; joint damping/friction/armature) are JAX array leaves in SI units, while
structural metadata (names, joint type, limits) is static (ADR-007). Each
exposes :meth:`parameter_spec` naming the leaves a sysid recipe may fit.

Getting the field set right here avoids URDF-importer rework downstream.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..parameter import ParameterSpec
from .frame import FramedQuantity, Transform


class JointType(str, enum.Enum):
    """The supported joint articulations."""

    REVOLUTE = "revolute"
    PRISMATIC = "prismatic"
    FIXED = "fixed"
    CONTINUOUS = "continuous"
    PLANAR = "planar"
    FLOATING = "floating"


#: Collision primitive kinds the MJCF backend can lower, with the number of
#: ``size`` entries MuJoCo expects for each (its half-size convention).
_GEOM_SIZE_ARITY = {"sphere": 1, "capsule": 2, "cylinder": 2, "box": 3}


@dataclasses.dataclass(frozen=True)
class CollisionGeom:
    """A primitive collision geometry rigidly attached to a link (static).

    Structural, not identifiable — collision shapes are part of the model
    description (they change the attestation handle through the lowered MJCF),
    not parameters a sysid recipe fits. Attach a tuple of these as a
    :class:`Link`'s ``collision`` field and the MJCF backend emits one MuJoCo
    ``<geom>`` per entry, enabling contact (e.g. foot–ground for a biped).
    Links whose ``collision`` is ``None`` or an opaque URDF handle
    (a mesh filename string) stay geometry-free, exactly as before.

    Args
    ----
    kind : str
        One of ``"sphere"``, ``"capsule"``, ``"cylinder"``, ``"box"``.
    size : tuple[float, ...]
        MuJoCo half-size convention, m: sphere ``(radius,)``; capsule and
        cylinder ``(radius, half_length)``; box ``(half_x, half_y, half_z)``.
    pos : tuple[float, float, float]
        Geom centre in the link's body frame, m. Default the origin.
    quat : tuple[float, float, float, float] | None
        Geom orientation in the body frame as a ``(w, x, y, z)`` quaternion;
        ``None`` (default) means the identity.
    name : str | None
        Optional MJCF geom name (useful for addressing contacts).
    friction : tuple[float, float, float] | None
        MuJoCo ``(sliding, torsional, rolling)`` friction coefficients;
        ``None`` (default) keeps MuJoCo's defaults.

    Raises
    ------
    ValueError
        If ``kind`` is unknown or ``size`` has the wrong arity for it.
    """

    kind: str
    size: tuple[float, ...]
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat: tuple[float, float, float, float] | None = None
    name: str | None = None
    friction: tuple[float, float, float] | None = None

    def __post_init__(self):
        if self.kind not in _GEOM_SIZE_ARITY:
            raise ValueError(
                f"CollisionGeom kind {self.kind!r} is not supported. The MJCF "
                f"backend lowers {sorted(_GEOM_SIZE_ARITY)}; use one of those "
                "(meshes stay opaque handles, per the inertials-drive-dynamics "
                "policy)."
            )
        expected = _GEOM_SIZE_ARITY[self.kind]
        if len(self.size) != expected:
            raise ValueError(
                f"CollisionGeom {self.kind!r} needs {expected} size entr"
                f"{'y' if expected == 1 else 'ies'} (MuJoCo half-size "
                f"convention), got {len(self.size)}: {self.size!r}."
            )


@dataclasses.dataclass(frozen=True)
class JointLimits:
    """Static joint limits (SI).

    Args
    ----
    lower, upper : float
        Position limits (rad for revolute/continuous, m for prismatic).
    effort : float
        Maximum |torque| (N·m) or |force| (N).
    velocity : float
        Maximum |velocity| (rad/s or m/s).
    """

    lower: float
    upper: float
    effort: float
    velocity: float


class Link(eqx.Module):
    """A rigid body with identifiable inertial parameters.

    Inertials are JAX array leaves in the link's body frame (SI); geometry is an
    opaque static handle. ``mass``/``com``/``inertia`` are the fittable subset.

    Args
    ----
    name : str
        Unique link name (static).
    body_frame : Frame
        The frame inertials are expressed in (static).
    mass : Float[Array, ""]
        Mass in kg.
    com : Float[Array, "3"]
        Centre of mass in the body frame (m).
    inertia : Float[Array, "3 3"]
        Inertia tensor about the COM in the body frame (kg·m²).
    visual, collision : Any
        Geometry (static); ``None`` if absent. URDF import stores opaque
        handles (mesh filename strings — not lowered). A ``collision`` that is
        a tuple of :class:`CollisionGeom` is lowered to MuJoCo ``<geom>``
        elements, enabling contact.
    """

    name: str = eqx.field(static=True)
    body_frame: Any = eqx.field(static=True)
    mass: Float[Array, ""]
    com: Float[Array, "3"]
    inertia: Float[Array, "3 3"]
    visual: Any = eqx.field(static=True, default=None)
    collision: Any = eqx.field(static=True, default=None)

    def parameter_spec(self) -> ParameterSpec:
        """The identifiable inertial parameters: ``mass``, ``com``, ``inertia``."""
        return ParameterSpec(("mass", "com", "inertia"))


class Joint(eqx.Module):
    """The articulation between a parent and child link.

    Structure (type, parent/child, limits) is static; the joint ``axis`` and
    ``origin`` carry the kinematic geometry; ``damping``, ``friction``, and
    ``armature`` are the identifiable dynamic parameters.

    Args
    ----
    name : str
        Unique joint name (static).
    joint_type : JointType
        The articulation type (static).
    parent_link, child_link : str
        Names of the connected links (static).
    axis : FramedQuantity
        The joint axis as a direction in a frame (a unit 3-vector). Structural —
        not identifiable.
    limits : JointLimits
        Position/effort/velocity limits (static).
    origin : Transform
        Parent→child transform at zero configuration.
    damping : Float[Array, ""]
        Viscous damping coefficient (N·m·s/rad or N·s/m). Identifiable.
    friction : Float[Array, ""]
        Coulomb friction magnitude (N·m or N). Identifiable.
    armature : Float[Array, ""]
        Reflected rotor inertia added to the joint's own DOF (kg·m² for a
        revolute joint, kg for a prismatic one) — MuJoCo's per-DOF
        ``armature``. Identifiable. Default 0 (MuJoCo's default; URDF has no
        equivalent element).
    """

    name: str = eqx.field(static=True)
    joint_type: JointType = eqx.field(static=True)
    parent_link: str = eqx.field(static=True)
    child_link: str = eqx.field(static=True)
    axis: FramedQuantity
    limits: JointLimits = eqx.field(static=True)
    origin: Transform
    damping: Float[Array, ""]
    friction: Float[Array, ""]
    armature: Float[Array, ""] = eqx.field(default_factory=lambda: jnp.asarray(0.0))

    def parameter_spec(self) -> ParameterSpec:
        """The identifiable dynamic parameters: ``damping``, ``friction``,
        ``armature``.

        A ``FLOATING`` (free) base has no identifiable joint dynamics, so it
        exposes none.
        """
        if self.joint_type is JointType.FLOATING:
            return ParameterSpec(())
        return ParameterSpec(("damping", "friction", "armature"))


__all__ = ["JointType", "JointLimits", "Link", "Joint", "CollisionGeom"]
