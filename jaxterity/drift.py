# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Scheduled plant-side parameter drift.

Hardware ages: gear friction grows, motor torque constants droop, bearings
wear. A :class:`DriftSchedule` models that as deterministic, piecewise-linear
trajectories over *simulated* time for selected plant parameters, applied to
an MJX model with :meth:`DriftSchedule.apply_to_mjx`. It is the "real world"
side of a drift study: the plant drifts on a fixed schedule while a
self-model either tracks it (online re-fit) or goes stale — which is the
gap the windowed re-fit and the residual monitor measure.

Design points:

- **Deterministic and shared.** A schedule is static data (breakpoints +
  values). Two rollouts given the same schedule see byte-identical plants —
  the precondition for honest baseline-vs-tracking comparisons.
- **jit-safe.** ``apply_to_mjx(model, t)`` works under ``jit``/``vmap`` with a
  traced ``t`` (piecewise-linear interpolation via ``jnp.interp``; ends are
  clamped, so before the first breakpoint the parameter holds its initial
  value and after the last it holds the final one).
- **Episodic by construction.** The intended use is the time-compressed
  pattern: pick the window's epoch ``t``, apply the schedule once, roll the
  window with the frozen drifted plant (e.g.
  ``G1JoystickController.rollout(model=...)``, which takes the plant as a
  traced argument and reuses its compiled kernel). Continuous within-rollout
  drift is out of scope — real drift is orders of magnitude slower than a
  telemetry window.
- **Faults are just schedules.** A slow ramp into an effectively-locked joint
  (friction loss ramped to hundreds of N·m) models a seizing gearbox; no
  special fault machinery.

Supported fields: ``dof_frictionloss`` / ``dof_damping`` (per-DOF, absolute
or scaled) and ``actuator_gain`` (per-actuator multiplicative scale of the
affine actuator gain/bias rows — the torque-constant-droop model for MuJoCo
``<position>``/``<motor>`` actuators).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax.numpy as jnp
import numpy as np

from .errors import JaxterityError

#: MJX model fields a channel may target, with the value semantics.
_FIELDS = ("dof_frictionloss", "dof_damping", "actuator_gain")
_MODES = ("set", "scale")


@dataclasses.dataclass(frozen=True)
class DriftChannel:
    """One drifting plant parameter: a piecewise-linear trajectory in time.

    Args
    ----
    field : str
        ``"dof_frictionloss"`` or ``"dof_damping"`` (per-DOF), or
        ``"actuator_gain"`` (per-actuator multiplicative scale — torque-
        constant droop).
    index : int
        DOF index (for ``dof_*``) or actuator index (for ``actuator_gain``).
        Resolve from names with :func:`joint_dof_index` /
        :func:`actuator_index`.
    times : tuple[float, ...]
        Strictly increasing breakpoints, s (simulated time).
    values : tuple[float, ...]
        Parameter value at each breakpoint, linearly interpolated between and
        clamped outside. Units: N·m (frictionloss), N·m·s/rad (damping),
        dimensionless scale (``actuator_gain``).
    mode : str
        ``"set"`` (values are absolute) or ``"scale"`` (values multiply the
        model's nominal value). ``actuator_gain`` is inherently a scale and
        must use ``"scale"``.

    Raises
    ------
    JaxterityError
        On an unknown field/mode, a length mismatch, or non-increasing times.
    """

    field: str
    index: int
    times: tuple[float, ...]
    values: tuple[float, ...]
    mode: str = "set"

    def __post_init__(self):
        if self.field not in _FIELDS:
            raise JaxterityError(
                f"DriftChannel field {self.field!r} is not supported; expected "
                f"one of {_FIELDS}."
            )
        if self.mode not in _MODES:
            raise JaxterityError(
                f"DriftChannel mode {self.mode!r} is not supported; expected "
                f"one of {_MODES}."
            )
        if self.field == "actuator_gain" and self.mode != "scale":
            raise JaxterityError(
                "DriftChannel(field='actuator_gain') models torque-constant "
                "droop as a multiplicative factor; use mode='scale' with "
                "dimensionless values (nominal 1)."
            )
        if len(self.times) != len(self.values) or len(self.times) < 1:
            raise JaxterityError(
                f"DriftChannel needs matching, non-empty times/values; got "
                f"{len(self.times)} times and {len(self.values)} values."
            )
        if any(b <= a for a, b in zip(self.times, self.times[1:])):
            raise JaxterityError(
                f"DriftChannel times must be strictly increasing, got {self.times}."
            )

    def value(self, t) -> jnp.ndarray:
        """The channel's value at simulated time ``t`` (clamped ends)."""
        return jnp.interp(
            jnp.asarray(t, dtype=float),
            jnp.asarray(self.times, dtype=float),
            jnp.asarray(self.values, dtype=float),
        )


@dataclasses.dataclass(frozen=True)
class DriftSchedule:
    """A fixed set of :class:`DriftChannel` trajectories for one plant.

    Args
    ----
    channels : tuple[DriftChannel, ...]
        The drifting parameters. Multiple channels may target distinct
        indices of the same field; two channels on the same (field, index)
        raise (the composition would be order-dependent).

    Raises
    ------
    JaxterityError
        On duplicate (field, index) targets.

    Examples
    --------
    A knee gearbox seizing over days 10–18 (B3-style fault)::

        DAY = 86400.0
        schedule = DriftSchedule((
            DriftChannel(
                field="dof_frictionloss",
                index=joint_dof_index(model, "right_knee_joint"),
                times=(10 * DAY, 18 * DAY),
                values=(0.1, 500.0),
            ),
        ))
        drifted = schedule.apply_to_mjx(mjx_model, t=14 * DAY)
    """

    channels: tuple[DriftChannel, ...]

    def __post_init__(self):
        targets = [(c.field, c.index) for c in self.channels]
        duplicates = {t for t in targets if targets.count(t) > 1}
        if duplicates:
            raise JaxterityError(
                f"DriftSchedule has multiple channels on the same target(s) "
                f"{sorted(duplicates)}; merge each into one channel (stacked "
                "channels would compose order-dependently)."
            )

    def apply_to_mjx(self, model: Any, t) -> Any:
        """The plant at simulated time ``t``: the model with drifted fields.

        Pure and jit/vmap-safe (``t`` may be traced). The input model is
        unchanged; channel order does not matter (one channel per target).

        Args
        ----
        model : mjx.Model
            The nominal plant (``mjx.put_model`` output). ``"scale"``-mode
            channels scale *this* model's values — pass the nominal model,
            not an already-drifted one.
        t : float | jax.Array
            Simulated time, s.

        Returns
        -------
        mjx.Model
            A new model with the scheduled values written in.
        """
        for channel in self.channels:
            value = channel.value(t)
            if channel.field == "actuator_gain":
                # Scale the affine actuator rows: gainprm[i, 0] is the gain
                # (kp for a <position> actuator) and biasprm[i, 1:3] the
                # state-feedback terms — scaling all three scales the
                # delivered torque, the torque-constant-droop model.
                gainprm = model.actuator_gainprm
                biasprm = model.actuator_biasprm
                i = channel.index
                model = model.replace(
                    actuator_gainprm=gainprm.at[i, 0].set(gainprm[i, 0] * value),
                    actuator_biasprm=biasprm.at[i, 1:3].set(biasprm[i, 1:3] * value),
                )
                continue
            current = getattr(model, channel.field)
            target = value if channel.mode == "set" else current[channel.index] * value
            model = model.replace(
                **{channel.field: current.at[channel.index].set(target)}
            )
        return model


def joint_dof_index(mj_model: Any, joint_name: str) -> int:
    """The DOF index of a (hinge/slide) joint in a ``mujoco.MjModel``.

    Args
    ----
    mj_model : mujoco.MjModel
        The compiled model (not the MJX copy — name tables live here).
    joint_name : str
        The joint's MJCF name.

    Returns
    -------
    int
        Index into ``dof_*`` arrays.

    Raises
    ------
    JaxterityError
        If the joint does not exist.
    """
    try:
        joint = mj_model.joint(joint_name)
    except KeyError as e:
        raise JaxterityError(
            f"joint_dof_index: no joint named {joint_name!r} in the model. "
            f"Known joints: {[mj_model.joint(i).name for i in range(mj_model.njnt)]}."
        ) from e
    return int(np.asarray(joint.dofadr)[0])


def actuator_index(mj_model: Any, actuator_name: str) -> int:
    """The index of an actuator in a ``mujoco.MjModel``.

    Args
    ----
    mj_model : mujoco.MjModel
        The compiled model.
    actuator_name : str
        The actuator's MJCF name (in the vendored G1 walk scene, actuators
        are named after their joints).

    Returns
    -------
    int
        Index into ``actuator_*`` arrays.

    Raises
    ------
    JaxterityError
        If the actuator does not exist.
    """
    try:
        return int(mj_model.actuator(actuator_name).id)
    except KeyError as e:
        raise JaxterityError(
            f"actuator_index: no actuator named {actuator_name!r} in the "
            f"model. Known actuators: "
            f"{[mj_model.actuator(i).name for i in range(mj_model.nu)]}."
        ) from e


__all__ = [
    "DriftChannel",
    "DriftSchedule",
    "joint_dof_index",
    "actuator_index",
]
