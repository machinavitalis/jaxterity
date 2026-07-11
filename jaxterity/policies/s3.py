# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Task-level interface — ``policies.S3``.

S3 wraps the WBC solver and exposes per-tick "set task priority /
step WBC" controls. The agent's mental model: maintain a dict of tasks,
adjust weights, ask for the next ``qdd``. The caller (S4 or the user) is
responsible for integrating qdd and applying torques.

Solver caching: a fresh :class:`WBCSolver` is *not* constructed per
tick. Instead the solver is rebuilt only when the task set changes
(``add_task`` / ``remove_task`` / ``update_task_field``). This lets a user
wrap :meth:`step_wbc` in ``jax.jit`` without thrashing the compilation cache
on every call.

Step-MPC vs. step-horizon-shooter: the original ``step_mpc`` name
oversold a forward-Euler horizon shooter. The honest name is
:meth:`step_horizon_shooter`; ``step_mpc`` is kept as a deprecated alias
that emits a one-shot ``DeprecationWarning`` and forwards. Real direct-
shooting / direct-transcription MPC is a follow-up and will land
under the ``step_mpc`` name.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any

import jax.numpy as jnp

from ..wbc import JointAccelerationLimits, Task, WBCSolver
from .s4 import S4


class S3:
    """Task-level controller built on top of :class:`S4`.

    Args
    ----
    s4 : S4
        The raw-actuator layer this S3 drives.
    tasks : dict[str, Task]
        Initial task dictionary keyed by name.
    limits : JointAccelerationLimits or None
        Optional symmetric qdd box bounds.
    """

    def __init__(
        self,
        s4: S4,
        tasks: dict[str, Task] | None = None,
        *,
        limits: JointAccelerationLimits | None = None,
    ):
        self.s4 = s4
        self._tasks: dict[str, Task] = dict(tasks or {})
        self._limits = limits
        self._solver: WBCSolver | None = None  # lazy cache; rebuilt on task changes

    # --- task management ---------------------------------------------

    def _invalidate(self) -> None:
        """Mark the cached WBCSolver as stale (call after any task mutation)."""
        self._solver = None

    def set_task_priority(self, name: str, w: float) -> None:
        """Update the soft-priority weight of an existing task in-place."""
        if name not in self._tasks:
            raise KeyError(f"S3.set_task_priority: unknown task {name!r}.")
        self.update_task_field(name, weight=float(w))

    def update_task_field(self, name: str, **fields: Any) -> None:
        """Replace one or more fields on an existing task in-place.

        Public surface for the higher layers (S2 / MCP tools) that need to
        nudge a task's setpoint or gains without reaching into ``_tasks``.
        Tasks are frozen dataclasses; this wraps ``dataclasses.replace`` and
        invalidates the WBCSolver cache so the next ``step_wbc`` rebuilds
        with the new field values.

        Raises
        ------
        KeyError
            If ``name`` is not a known task.
        """
        if name not in self._tasks:
            raise KeyError(f"S3.update_task_field: unknown task {name!r}.")
        old = self._tasks[name]
        self._tasks[name] = dataclasses.replace(old, **fields)  # type: ignore[call-arg]
        self._invalidate()

    def add_task(self, task: Task) -> None:
        """Register a new task in the WBC stack."""
        self._tasks[task.name] = task
        self._invalidate()

    def remove_task(self, name: str) -> None:
        """Drop a task from the WBC stack."""
        if self._tasks.pop(name, None) is not None:
            self._invalidate()

    # --- control ticks -----------------------------------------------

    def _ensure_solver(self) -> WBCSolver:
        """Return the cached solver, rebuilding from ``self._tasks`` if stale."""
        if self._solver is None:
            self._solver = WBCSolver(tasks=tuple(self._tasks.values()))
        return self._solver

    def step_wbc(self) -> jnp.ndarray:
        """Compute the next ``qdd`` via the soft-hierarchy WBC.

        Reads the current state from S4, asks the cached WBCSolver to solve,
        returns the n_dof-length ``qdd``. The caller (S4 or the user) is
        responsible for integrating + applying torques.
        """
        q, qd = self.s4.state()
        return self._ensure_solver().solve(q, qd, n_dof=q.shape[0], limits=self._limits)

    def step_horizon_shooter(
        self, *, horizon: int = 10, dt: float = 0.01
    ) -> jnp.ndarray:
        """Forward-Euler horizon shooter (honest name).

        Integrates ``horizon`` ticks of WBC qdd at fixed step ``dt`` and
        returns the predicted state ``concat([q, qd])`` at the horizon end.
        Useful as a single-shot lookahead for trajectory planning; it is
        *not* MPC (no horizon-wide optimization, no warm-starting). Real
        direct-shooting / direct-transcription MPC lands under :meth:`step_mpc`
        as a follow-up.
        """
        q, qd = self.s4.state()
        solver = self._ensure_solver()
        n = q.shape[0]
        for _ in range(horizon):
            qdd = solver.solve(q, qd, n_dof=n, limits=self._limits)
            qd = qd + dt * qdd
            q = q + dt * qd
        return jnp.concatenate([q, qd])

    def step_mpc(self, *, horizon: int = 10, dt: float = 0.01) -> jnp.ndarray:
        """Deprecated alias for :meth:`step_horizon_shooter`.

        The original ``step_mpc`` was a forward-Euler shooter, not MPC. The
        honest name is :meth:`step_horizon_shooter`; this alias survives so
        existing callers keep working but emits a one-shot
        :class:`DeprecationWarning`. Real MPC will reclaim the ``step_mpc``
        name as a follow-up.
        """
        warnings.warn(
            "S3.step_mpc was misnamed; it is a forward-Euler horizon shooter, "
            "not MPC. Use S3.step_horizon_shooter. The step_mpc name will be "
            "reclaimed by the real direct-shooting MPC follow-up.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.step_horizon_shooter(horizon=horizon, dt=dt)

    @property
    def tasks(self) -> dict[str, Task]:
        """Current task dict (mutable; returned view)."""
        return self._tasks


__all__ = ["S3"]
