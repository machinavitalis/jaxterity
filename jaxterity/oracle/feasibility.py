# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Quantitative feasibility verdicts: compare a task against physical limits.

The failure mode this module addresses is measured, not hypothesized: on the
ASIMOV-Agentic physical-constraints task (Google DeepMind, 2026), a frontier
model given "maximum payload 500 g" and shown a cast-iron skillet answers
"I'll pick it up" — the model *never performs the quantitative comparison*.
:func:`assess` is the missing discipline as a callable tool: given a task's
physical quantities (mass estimate or range, width, reach, duration) and a
set of limits — either **measured** (a :class:`~jaxterity.oracle.CapabilitySheet`,
our thesis) or **declared** (the benchmark's prompt-text limits) — it returns
a deterministic verdict with the binding constraint, per-quantity margins,
and an honest ``uncertain`` outcome when an estimate range straddles a limit
(which maps to the agent's "ask for clarification" routing rather than a
coin-flip).

This is deliberately simple arithmetic. Its value is *discipline*
(the comparison always happens, with margins and provenance) — richer
rollout-based verdicts (thermal cascades, tip-over) belong to the twin
simulation layer (:mod:`jaxterity.selfcheck`), not here.
"""

from __future__ import annotations

import dataclasses

from .capability import CapabilitySheet

#: Fractional safety margin applied to limits before declaring "proceed".
_DEFAULT_MARGIN_FRAC = 0.1


@dataclasses.dataclass(frozen=True)
class TaskSpec:
    """The physical quantities of a proposed task (unknowns stay ``None``).

    Mass and width may be a single estimate or an ``(lo, hi)`` range —
    ranges are how honest perception talks, and a range straddling a limit
    yields ``uncertain`` rather than a guess.
    """

    action: str
    object_mass_kg: float | tuple[float, float] | None = None
    object_width_m: float | tuple[float, float] | None = None
    reach_m: float | None = None
    duration_s: float | None = None


@dataclasses.dataclass(frozen=True)
class Limits:
    """Physical limits to judge against (``None`` = limit unknown/absent).

    Build from a measured sheet with :meth:`from_sheet`, or directly from a
    benchmark item's declared constraints.
    """

    payload_kg: float | None = None
    gripper_width_m: float | None = None
    reach_m: float | None = None
    source: str = "declared"

    @classmethod
    def from_sheet(cls, sheet: CapabilitySheet) -> "Limits":
        """Extract judgeable limits from a capability sheet (source="measured")."""
        payload = reach = None
        for f in sheet.fields:
            if f.name == "max_payload_static" and isinstance(f.value, (int, float)):
                payload = float(f.value)
            elif f.name == "max_reach" and isinstance(f.value, (int, float)):
                reach = float(f.value)
        return cls(payload_kg=payload, reach_m=reach, source="measured")


@dataclasses.dataclass(frozen=True)
class Assessment:
    """The verdict: proceed / refuse / uncertain, with margins and reasons.

    ``margins`` maps each judged quantity to ``limit − worst-case value``
    (negative = violated). ``missing`` names quantities that could not be
    judged because either the task spec or the limits lacked them — silence
    is never a pass.
    """

    verdict: str  # "proceed" | "refuse" | "uncertain"
    binding_constraint: str | None
    margins: dict[str, float]
    missing: tuple[str, ...]
    rationale: str
    limits_source: str

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["missing"] = list(self.missing)
        return d


def _as_range(v: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(v, tuple):
        lo, hi = float(v[0]), float(v[1])
        return (min(lo, hi), max(lo, hi))
    return (float(v), float(v))


def assess(
    spec: TaskSpec,
    limits: Limits | CapabilitySheet,
    *,
    margin_frac: float = _DEFAULT_MARGIN_FRAC,
) -> Assessment:
    """Judge a task spec against physical limits.

    Args
    ----
    spec : TaskSpec
        The task's physical quantities (estimates or ranges).
    limits : Limits | CapabilitySheet
        The limits to judge against; a sheet is converted via
        :meth:`Limits.from_sheet`.
    margin_frac : float
        Fractional margin required below a limit for a clean "proceed"
        (worst-case value must be ≤ (1 − margin_frac) · limit).

    Returns
    -------
    Assessment
        Deterministic verdict. Rules, applied per judgeable quantity:
        worst-case value over the limit → **refuse**; estimate range
        straddling the margin-adjusted limit → **uncertain**; all judged
        quantities clear → **proceed**. Quantities present in the spec but
        absent from the limits (or vice versa) are listed in ``missing``.
    """
    if isinstance(limits, CapabilitySheet):
        limits = Limits.from_sheet(limits)

    checks: list[tuple[str, float | tuple[float, float] | None, float | None]] = [
        ("payload", spec.object_mass_kg, limits.payload_kg),
        ("gripper_width", spec.object_width_m, limits.gripper_width_m),
        ("reach", spec.reach_m, limits.reach_m),
    ]

    margins: dict[str, float] = {}
    missing: list[str] = []
    violated: list[tuple[str, float]] = []
    uncertain: list[str] = []

    for name, value, limit in checks:
        if value is None and limit is None:
            continue
        if value is None or limit is None:
            missing.append(name)
            continue
        lo, hi = _as_range(value)
        threshold = (1.0 - margin_frac) * limit
        margins[name] = round(limit - hi, 6)
        if lo > limit:
            violated.append((name, limit - hi))
        elif hi > threshold:
            uncertain.append(name)

    if violated:
        binding = min(violated, key=lambda kv: kv[1])[0]
        verdict = "refuse"
        rationale = (
            f"{binding} exceeds the limit (margin {margins[binding]:+.3g}); "
            f"limits source: {limits.source}."
        )
    elif uncertain:
        binding = uncertain[0]
        verdict = "uncertain"
        rationale = (
            f"{binding} estimate is within {margin_frac:.0%} of the limit or "
            "straddles it — resolve the estimate (weigh/measure or ask) "
            "before proceeding."
        )
    elif margins:
        binding = None
        verdict = "proceed"
        rationale = (
            "all judged quantities clear their limits with margin; "
            f"limits source: {limits.source}."
        )
    else:
        binding = None
        verdict = "uncertain"
        rationale = (
            "nothing could be judged — no overlapping quantities between "
            f"task spec and limits (missing: {', '.join(missing) or 'all'})."
        )

    return Assessment(
        verdict=verdict,
        binding_constraint=binding,
        margins=margins,
        missing=tuple(missing),
        rationale=rationale,
        limits_source=limits.source,
    )


__all__ = ["TaskSpec", "Limits", "Assessment", "assess"]
