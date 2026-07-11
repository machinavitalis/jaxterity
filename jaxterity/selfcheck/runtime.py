# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""The self-check co-processor runtime.

A :class:`CoProcessor` runs *beside* the controller, not in its loop. Each tick
it ingests the latest measured state (joint positions/velocities + per-servo
temperature) and the commanded action, rolls the unit's :class:`~jaxterity.twin.Twin`
forward over a short horizon, and publishes the predicted near future — servo
temperature, (LeKiwi) bus voltage, and joint trajectories — for the safety
monitor to veto on.

It is **soft real-time**: the rollout is compiled once (via
:meth:`~jaxterity.twin.Twin.compiled_rollout`, which builds the MJX dynamics a
single time) so a warmed co-processor ticks in microseconds. If a tick overruns
its deadline (``1 / rate_hz``) the result is **skipped and flagged** rather than
published late, and a deadline-miss counter advances — the co-processor never
blocks the control deadline. The rollout is pure, so a fixed input stream gives a
bit-identical prediction stream.

Honest scope: the twin runs in Python/JAX here (no Jaxility lowering). If the
host can't hold the target rate (the ``pi``-marked timing test, skipped by
``scripts/check.sh``), the escalation is a lowered co-processor — a later phase,
flagged here rather than stubbed.
"""

from __future__ import annotations

import time
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from ..twin import Twin, TwinRollout

#: A source yields ``(q, qd, T, action)`` for a tick: measured joint positions
#: (rad), velocities (rad/s), per-servo temperature (K), and the commanded
#: generalized force (N·m). It models the controller + arm feeding the co-processor.
Source = Callable[[int], "tuple"]
#: A publish sink receives each on-time prediction: ``publish(tick, prediction)``.
Publish = Callable[[int, TwinRollout], None]


class TickReport(NamedTuple):
    """The outcome of one co-processor tick.

    Fields
    ------
    tick : int
        Zero-based tick index.
    elapsed_s : float
        Wall-clock time the tick's compute took, s (by the co-processor's clock).
    late : bool
        True iff ``elapsed_s`` exceeded the deadline — the prediction was skipped
        (not published) and the miss counted.
    prediction : TwinRollout | None
        The published prediction, or ``None`` when the tick was late/skipped.
    """

    tick: int
    elapsed_s: float
    late: bool
    prediction: TwinRollout | None


class RunReport(NamedTuple):
    """Aggregate outcome of a co-processor run.

    Fields
    ------
    n_ticks : int
        Number of ticks attempted.
    deadline_misses : int
        How many ticks overran the deadline (were skipped-and-flagged).
    published : int
        How many ticks published an on-time prediction.
    reports : tuple[TickReport, ...]
        Per-tick detail.
    """

    n_ticks: int
    deadline_misses: int
    published: int
    reports: tuple[TickReport, ...]


class CoProcessor:
    """Predictive self-check co-processor: rolls a :class:`Twin` forward each tick.

    See the module docstring for the soft-real-time contract.

    Args
    ----
    twin : Twin
        The per-unit twin to roll forward.
    rate_hz : float
        Target tick rate, Hz (arm rates: tens–low-hundreds). Sets the deadline
        ``1 / rate_hz``.
    horizon_steps : int
        Number of forward steps per prediction.
    dt : float
        Forward-integration step, s (lookahead = ``horizon_steps * dt``).
    clock : Callable[[], float]
        Monotonic clock returning seconds (injectable for deterministic tests).
        Defaults to :func:`time.perf_counter`.
    warmup : bool
        Compile + run one throwaway prediction in ``__init__`` so the first real
        tick is hot (default True).
    """

    def __init__(
        self,
        twin: Twin,
        *,
        rate_hz: float,
        horizon_steps: int = 20,
        dt: float = 0.02,
        clock: Callable[[], float] = time.perf_counter,
        warmup: bool = True,
    ):
        if rate_hz <= 0:
            raise ValueError(f"rate_hz must be > 0, got {rate_hz}.")
        self.twin = twin
        self.rate_hz = float(rate_hz)
        self.deadline_s = 1.0 / float(rate_hz)
        self.horizon_steps = int(horizon_steps)
        self.dt = float(dt)
        self._clock = clock
        self._nv = twin.n_servos
        self._nq = twin.robot.build_system()._nq
        self._rollout = twin.compiled_rollout(self.dt)
        if warmup:
            self._warmup()

    def _warmup(self) -> None:
        """Compile the rollout by running it once on zeros (block until ready)."""
        z_q = jnp.zeros(self._nq)
        z_v = jnp.zeros(self._nv)
        T0 = jnp.full(self._nv, self.twin.T_amb)
        pred = self.predict(z_q, z_v, T0, z_v)
        jax.block_until_ready(pred)

    def predict(self, q, qd, T, action) -> TwinRollout:
        """Roll the twin forward over the horizon under a held commanded action.

        Args
        ----
        q : array-like, shape ``(nq,)``
            Measured joint positions, rad.
        qd : array-like, shape ``(nv,)``
            Measured joint velocities, rad/s.
        T : array-like, shape ``(n_servos,)``
            Measured per-servo temperature, K.
        action : array-like, shape ``(nv,)``
            Commanded generalized force held over the horizon, N·m.

        Returns
        -------
        TwinRollout
            The predicted trajectories over ``horizon_steps`` (+ the initial
            state). Arrays are not blocked — call :func:`jax.block_until_ready`
            if you need to time the compute.
        """
        action = jnp.asarray(action, dtype=float)
        tau_seq = jnp.broadcast_to(action, (self.horizon_steps, self._nv))
        return self._rollout(
            jnp.asarray(q, dtype=float),
            jnp.asarray(qd, dtype=float),
            jnp.asarray(T, dtype=float),
            tau_seq,
        )

    def run(
        self,
        source: Source,
        n_ticks: int,
        *,
        publish: Publish | None = None,
    ) -> RunReport:
        """Run ``n_ticks`` ticks beside a (simulated) controller + arm.

        Each tick pulls ``(q, qd, T, action)`` from ``source``, predicts, and —
        if the compute met the deadline — publishes. A tick that overruns is
        skipped (not published) and flagged; the deadline-miss counter advances
        and the loop continues without blocking. Deterministic for a fixed
        ``source`` and clock.

        Args
        ----
        source : Source
            ``source(tick) -> (q, qd, T, action)`` — the measured state + command.
        n_ticks : int
            Number of ticks to run.
        publish : Publish | None
            Optional sink called ``publish(tick, prediction)`` on each on-time
            tick.

        Returns
        -------
        RunReport
            Miss/publish counts and per-tick detail.
        """
        reports: list[TickReport] = []
        misses = 0
        published = 0
        for k in range(n_ticks):
            q, qd, T, action = source(k)
            t0 = self._clock()
            pred = self.predict(q, qd, T, action)
            jax.block_until_ready(pred)
            elapsed = self._clock() - t0
            late = elapsed > self.deadline_s
            if late:
                misses += 1
                reports.append(TickReport(k, elapsed, True, None))
                continue
            if publish is not None:
                publish(k, pred)
            published += 1
            reports.append(TickReport(k, elapsed, False, pred))
        return RunReport(n_ticks, misses, published, tuple(reports))


__all__ = ["CoProcessor", "TickReport", "RunReport"]
