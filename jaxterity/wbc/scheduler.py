# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Contact phase scheduler for legged-locomotion gaits.

``ContactSchedule`` is a per-foot boolean signal: at each time ``t``, is the
foot in contact with the ground? The default factory :func:`periodic_gait`
builds the canonical gait families from a per-foot phase offset + duty cycle.

The schedule is the contract between the high-level gait planner and the
WBC's contact constraints (the ContactTask, which will land alongside
centroidal-dynamics support). Per-foot contact / swing assignment selects
which tasks contribute to the QP at each tick.

Pure data + a single ``in_contact(t) -> bool[n_feet]`` query — no integrator
state, JIT-clean.

**Un-wired groundwork.** Nothing consumes a ``ContactSchedule``
yet. The ``ContactTask`` that would feed the per-foot contact mask into the WBC
QP — and the ``centroidal_momentum_matrix`` / ``contact_jacobian`` it needs — is
future work (legged locomotion). This module is deliberate scaffolding:
building a gait here does **not** gate any controller, and ``S3``/``WBCSolver``
do not read it. Documented as a gap in ``KNOWN_GAPS.md`` §"Whole-body control";
shipping ``ContactTask`` (the consumer) is the alternative resolution.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class ContactSchedule:
    """Per-foot contact phase pattern.

    Fields
    ------
    foot_names : tuple[str, ...]
        Names of the feet, in the order the schedule indexes.
    in_contact : Callable[[float], jax.Array]
        ``t -> bool[n_feet]``. JIT-clean: no Python branches on ``t``.
    period : float
        Gait period in seconds (informational; the contact function holds the
        truth).
    """

    foot_names: tuple[str, ...]
    in_contact: Callable
    period: float = 0.0

    def contact(self, t):
        """Boolean contact mask at time ``t``, shape ``(n_feet,)``."""
        return self.in_contact(t)


def periodic_gait(
    foot_names: tuple[str, ...],
    *,
    period: float,
    duty: float,
    phases: tuple[float, ...],
) -> ContactSchedule:
    """Construct a periodic gait schedule.

    Each foot is in contact when the cycle phase ``(t / period + phase) mod 1``
    is below ``duty``. This covers the standard gait families:

    * Trot: 2 diagonal pairs, phases ``(0, 0.5, 0.5, 0)``, duty ~0.5.
    * Pace: 2 lateral pairs, phases ``(0, 0, 0.5, 0.5)``.
    * Bound: 2 front/rear pairs, phases ``(0, 0, 0.5, 0.5)``.
    * Walk: 4-beat, phases ``(0, 0.25, 0.5, 0.75)``, duty ~0.75.

    Args
    ----
    foot_names : tuple[str, ...]
        Names of the feet.
    period : float
        Gait cycle period, seconds.
    duty : float
        Stance fraction per foot, in (0, 1). 0.5 = walking trot, 0.75 = walk.
    phases : tuple[float, ...]
        Per-foot phase offset, in [0, 1). Must match ``len(foot_names)``.

    Returns
    -------
    ContactSchedule
    """
    assert len(phases) == len(foot_names), (
        f"phases ({len(phases)}) and foot_names ({len(foot_names)}) must match"
    )
    phases_arr = jnp.asarray(phases, dtype=float)

    def in_contact(t):
        u = (t / period + phases_arr) % 1.0
        return u < duty

    return ContactSchedule(foot_names=foot_names, in_contact=in_contact, period=period)


__all__ = ["ContactSchedule", "periodic_gait"]
