# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Minimal pre-flight verification.

Before a candidate command program runs on a unit, :func:`preflight` simulates it
on that unit's :class:`~jaxterity.twin.Twin` across a small declared
:class:`Scenario` set, checks each realized rollout against a
:class:`~jaxterity.selfcheck.SafetyEnvelope` (via :class:`SafetyMonitor`), and
returns a :class:`PreflightReport`: pass/fail, the binding constraint and
worst-case margin per scenario, and the **twin attestation handle** — so a pass
binds to one specific twin version (recalibrate the unit and the pass no longer
applies).

The candidate "policy" is an open-loop torque trajectory ``policy(tick) ->
action`` (a trajectory/parameter set, *not* a re-lowered binary): each scenario
rolls the whole trajectory through the twin once, so — unlike the real-time
monitor — the realized joint path is the policy's actual path, and the
joint-limit check is meaningful. The thermal check additionally uses the
trajectory's peak per-servo torque for the sustained-hold steady state.

Honest scope: minimal = in-twin simulation + envelope check. No OTA
transport, no fleet orchestration, no signed deploy gate (Jaxility / commercial,
later work).
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Sequence

import jax.numpy as jnp

from ..twin import Twin
from .monitor import SafetyDecision, SafetyEnvelope, SafetyMonitor

#: A candidate policy: maps a tick index to a commanded generalized force (N·m).
Policy = Callable[[int], "jnp.ndarray"]


@dataclasses.dataclass(frozen=True)
class Scenario:
    """A declared pre-flight condition the policy is simulated from.

    Args
    ----
    name : str
        Human-readable scenario label (reported on failure).
    q0, qd0, T0 : tuple[float, ...]
        Initial joint positions (rad), velocities (rad/s), and per-servo
        temperature (K).
    n_steps : int
        Number of policy steps to simulate.
    dt : float
        Step size, s.
    """

    name: str
    q0: tuple[float, ...]
    qd0: tuple[float, ...]
    T0: tuple[float, ...]
    n_steps: int
    dt: float


@dataclasses.dataclass(frozen=True)
class ScenarioResult:
    """The pre-flight outcome for one scenario."""

    name: str
    decision: SafetyDecision

    @property
    def passed(self) -> bool:
        """True iff the policy was not vetoed in this scenario."""
        return not self.decision.vetoed


@dataclasses.dataclass(frozen=True)
class PreflightReport:
    """The aggregate pre-flight verdict, bound to a specific twin version.

    Args
    ----
    passed : bool
        True iff no scenario vetoed the policy.
    twin_handle : str
        The twin's attestation handle at verification time — a pass binds to it.
    results : tuple[ScenarioResult, ...]
        Per-scenario outcomes.
    """

    passed: bool
    twin_handle: str
    results: tuple[ScenarioResult, ...]

    @property
    def failures(self) -> tuple[ScenarioResult, ...]:
        """The scenarios that failed (vetoed)."""
        return tuple(r for r in self.results if not r.passed)

    def summary(self) -> str:
        """A one-line human/agent summary."""
        if self.passed:
            return (
                f"PASS ({len(self.results)} scenarios) — twin {self.twin_handle[:12]}"
            )
        named = ", ".join(
            f"{r.name}[{r.decision.binding_constraint}]" for r in self.failures
        )
        return f"FAIL — vetoed in: {named}"


def preflight(
    policy: Policy,
    twin: Twin,
    envelope: SafetyEnvelope,
    scenarios: Sequence[Scenario],
) -> PreflightReport:
    """Verify a candidate policy on a twin across a scenario set.

    Args
    ----
    policy : Policy
        ``policy(tick) -> action`` — the candidate open-loop torque trajectory.
    twin : Twin
        The unit to verify against. Its attestation handle is recorded.
    envelope : SafetyEnvelope
        The safety limits to check each realized rollout against. Pre-flight is
        the place to enable ``joint_margin_rad`` — the realized rollout is the
        policy's actual joint path.
    scenarios : Sequence[Scenario]
        The declared conditions to simulate from (at least one).

    Returns
    -------
    PreflightReport
        Pass/fail, per-scenario decisions, and the twin handle.

    Raises
    ------
    ValueError
        If ``scenarios`` is empty.
    """
    if not scenarios:
        raise ValueError("preflight needs at least one scenario.")
    monitor = SafetyMonitor(twin, envelope)
    results: list[ScenarioResult] = []
    for sc in scenarios:
        tau_seq = jnp.stack(
            [jnp.asarray(policy(t), dtype=float) for t in range(sc.n_steps)]
        )
        rollout = twin.rollout(
            jnp.asarray(sc.q0, dtype=float),
            jnp.asarray(sc.qd0, dtype=float),
            jnp.asarray(sc.T0, dtype=float),
            tau_seq,
            sc.dt,
        )
        # peak per-servo torque over the trajectory feeds the sustained-hold
        # (steady-state) thermal check; the realized path feeds the rest.
        action_repr = jnp.max(jnp.abs(tau_seq), axis=0)
        decision = monitor.evaluate(rollout, action=action_repr)
        results.append(ScenarioResult(sc.name, decision))
    passed = all(r.passed for r in results)
    return PreflightReport(passed, twin.attestation_handle, tuple(results))


def constant_policy(action) -> Policy:
    """A policy that commands the same ``action`` (N·m) every tick."""
    vec = jnp.asarray(action, dtype=float)
    return lambda _tick: vec


__all__ = [
    "preflight",
    "Scenario",
    "ScenarioResult",
    "PreflightReport",
    "Policy",
    "constant_policy",
]
