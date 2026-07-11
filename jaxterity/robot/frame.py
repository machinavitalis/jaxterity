# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Coordinate frames and frame-annotated quantities.

Frame errors — an IMU mounted 90° off, a wrench expressed in the body frame
instead of the world frame — are the most common bug class in real robotics
deployments, and an LLM author is at least as likely to make one as a human
(ADR-005). Jaxterity therefore makes the frame part of the value: every public
pose, twist, or wrench is a :class:`FramedQuantity` carrying an explicit
:class:`Frame`, and arithmetic across mismatched frames raises :class:`FrameError`
at the point of misuse rather than producing a silently wrong number.

Transform naming follows the field convention ``T_target_source``:
``T_world_body`` maps a point expressed in ``body`` to one expressed in
``world`` (``p_world = T_world_body.apply(p_body)``).

Static frame typing (catching a mismatch at ``mypy`` time) is a stretch goal —
frames are runtime values, so the guarantee here is **runtime-checked**. The
type annotations are best-effort documentation.
"""

from __future__ import annotations

import dataclasses
from typing import Generic, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..errors import FrameError, JaxterityError


@dataclasses.dataclass(frozen=True)
class Frame:
    """A named coordinate frame.

    Two frames are equal iff their names are equal; otherwise they are distinct.
    A ``Frame`` carries no numeric data — it is pure, hashable metadata, stored
    as a static field on the quantities and transforms that reference it (so it
    never becomes a traced array leaf).

    Args
    ----
    name : str
        The frame's name, e.g. ``"world"``, ``"base_link"``, ``"foot_left"``.

    Examples
    --------
    >>> Frame("world") == Frame("world")
    True
    >>> Frame("world") == Frame("body")
    False
    """

    name: str

    def __repr__(self) -> str:
        return f'Frame("{self.name}")'


# --- quantity value types --------------------------------------------------
# Each is an equinox.Module (a JAX PyTree) whose leaves are SI-unit arrays. They
# are the ``T`` in ``FramedQuantity[T]``.


class Position(eqx.Module):
    """A 3-D position in metres. Leaves: ``xyz``."""

    xyz: Float[Array, "3"]


class Twist(eqx.Module):
    """A spatial velocity: ``linear`` (m/s) and ``angular`` (rad/s)."""

    linear: Float[Array, "3"]
    angular: Float[Array, "3"]


class Wrench(eqx.Module):
    """A spatial force: ``force`` (N) and ``torque`` (N·m)."""

    force: Float[Array, "3"]
    torque: Float[Array, "3"]


T = TypeVar("T")


class FramedQuantity(eqx.Module, Generic[T]):
    """A quantity (:class:`Position` / :class:`Twist` / :class:`Wrench`) tagged
    with the :class:`Frame` it is expressed in.

    Addition and subtraction require the **same frame** and the same value type;
    a frame mismatch raises :class:`FrameError`. The frame is a static field, so
    a ``FramedQuantity`` is vmap- and grad-safe (only the underlying arrays are
    traced leaves).

    Args
    ----
    value : T
        The quantity (a PyTree of SI-unit arrays).
    frame : Frame
        The frame ``value`` is expressed in.

    Frame
    -----
    Carried explicitly in ``.frame``; arithmetic preserves it and rejects
    mismatches.

    Examples
    --------
    >>> a = FramedQuantity(Position(jnp.zeros(3)), Frame("world"))
    >>> b = FramedQuantity(Position(jnp.ones(3)), Frame("world"))
    >>> (a + b).frame
    Frame("world")
    """

    value: T
    frame: Frame = eqx.field(static=True)

    def _check_compatible(self, other: "FramedQuantity") -> None:
        if not isinstance(other, FramedQuantity):
            raise JaxterityError(
                f"Cannot combine a FramedQuantity with {type(other).__name__}. "
                "Both operands must be FramedQuantity."
            )
        if self.frame != other.frame:
            raise FrameError(
                f"Cannot combine quantities in different frames: "
                f"{self.frame.name!r} and {other.frame.name!r}. Express both in "
                f"the same frame first (e.g. transform one into {self.frame.name!r})."
            )
        if type(self.value) is not type(other.value):
            raise JaxterityError(
                f"Cannot combine FramedQuantity[{type(self.value).__name__}] with "
                f"FramedQuantity[{type(other.value).__name__}]; value types must match."
            )

    def __add__(self, other: "FramedQuantity[T]") -> "FramedQuantity[T]":
        self._check_compatible(other)
        value = jax.tree_util.tree_map(jnp.add, self.value, other.value)
        return FramedQuantity(value, self.frame)

    def __sub__(self, other: "FramedQuantity[T]") -> "FramedQuantity[T]":
        self._check_compatible(other)
        value = jax.tree_util.tree_map(jnp.subtract, self.value, other.value)
        return FramedQuantity(value, self.frame)


class Transform(eqx.Module):
    """A rigid-body transform ``T_target_source`` between two frames.

    Maps quantities expressed in ``source`` to ``target``: for a point,
    ``p_target = rotation @ p_source + translation``.

    Args
    ----
    rotation : Float[Array, "3 3"]
        The rotation matrix (``R_target_source``, orthonormal).
    translation : Float[Array, "3"]
        The translation in ``target`` (metres).
    source, target : Frame
        The frames this transform maps between.

    Frame
    -----
    ``source`` → ``target``. :meth:`apply` requires its argument to be in
    ``source`` and returns a quantity in ``target``.

    Examples
    --------
    >>> T = Transform.identity(Frame("world"))
    >>> (T @ T.inverse()).rotation.shape
    (3, 3)
    """

    rotation: Float[Array, "3 3"]
    translation: Float[Array, "3"]
    source: Frame = eqx.field(static=True)
    target: Frame = eqx.field(static=True)

    @classmethod
    def identity(cls, frame: Frame) -> "Transform":
        """The identity transform on ``frame`` (``T_frame_frame``)."""
        return cls(jnp.eye(3), jnp.zeros(3), source=frame, target=frame)

    def inverse(self) -> "Transform":
        """Return ``T_source_target`` (the inverse rigid-body transform)."""
        rt = self.rotation.T
        return Transform(
            rt, -rt @ self.translation, source=self.target, target=self.source
        )

    def __matmul__(self, other: "Transform") -> "Transform":
        """Compose two transforms: ``T_a_b @ T_b_c == T_a_c``.

        Requires ``self.source == other.target`` (the inner frames match).
        """
        if self.source != other.target:
            raise FrameError(
                f"Cannot compose T_{self.target.name}_{self.source.name} with "
                f"T_{other.target.name}_{other.source.name}: the inner frames "
                f"{self.source.name!r} and {other.target.name!r} must match. "
                "Composition is T_a_b @ T_b_c."
            )
        return Transform(
            self.rotation @ other.rotation,
            self.rotation @ other.translation + self.translation,
            source=other.source,
            target=self.target,
        )

    def apply(self, quantity: FramedQuantity) -> FramedQuantity:
        """Express a point ``quantity`` (in ``source``) in ``target``.

        Args
        ----
        quantity : FramedQuantity[Position]
            A position in this transform's ``source`` frame.

        Returns
        -------
        FramedQuantity[Position]
            The same point, expressed in ``target``.

        Raises
        ------
        FrameError
            If ``quantity.frame`` is not this transform's ``source``.
        JaxterityError
            If ``quantity`` is not a position (twist/wrench adjoints land with
            the WBC work).

        Frame
        -----
        Input: ``source``. Output: ``target``.
        """
        if quantity.frame != self.source:
            raise FrameError(
                f"Transform T_{self.target.name}_{self.source.name} expects a "
                f"quantity in frame {self.source.name!r}, got "
                f"{quantity.frame.name!r}."
            )
        if not isinstance(quantity.value, Position):
            raise JaxterityError(
                f"Transform.apply currently supports Position only, got "
                f"{type(quantity.value).__name__}. Twist/Wrench adjoint "
                "transforms land with the WBC primitives."
            )
        p = self.rotation @ quantity.value.xyz + self.translation
        return FramedQuantity(Position(p), self.target)


__all__ = [
    "Frame",
    "Position",
    "Twist",
    "Wrench",
    "FramedQuantity",
    "Transform",
]
