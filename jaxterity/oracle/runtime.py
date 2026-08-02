# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""The oracle: simulate to *answer*, not to choose and not to permit.

Three simulation horizons sit in a robot stack, and they are different organs.
An MPC simulates to **choose** (its own controller, millisecond-scale). The
self-check gate simulates to **permit** —
:func:`~jaxterity.selfcheck.preflight` and
:class:`~jaxterity.selfcheck.SafetyMonitor` roll a candidate motion on the
unit's twin and pass, derate, or veto it. This module adds the third: a
*service* userland can call to ask a question and get a quantitative answer
back.

It is the simulated half of the oracle package. Its sibling
:mod:`~jaxterity.oracle.capability` answers statically, from the model alone —
what this unit *can* do, with per-field provenance — and
:mod:`~jaxterity.oracle.feasibility` compares a task against those limits. Use
the static surface to bound what to attempt; use this one to price a specific
attempt, cascades and parameter uncertainty included.

    "Do I overheat holding 2 kg through this segment?"
    "Which of these three grasps keeps the most tip-over margin?"
    "Does the bus stay above brownout if I run it at this pace?"

The answers are **per-unit and uncertainty-carrying**, which is the whole
point: the plan is evaluated not once but across a posterior of the unit's
fitted parameters (:func:`posterior_twins`), and the **dispersion of the
outcome is the honesty**. An oracle that reports a single number from a single
parameter set is telling you what a nominal robot would do; this one reports
what *this* unit would do and how much of that answer is the fit's ignorance.
Where the twin carries no fit covariance, the answer says so rather than
reporting a confident band it has not earned.

What makes the verdict multi-physics rather than rule-based: a rollout can
express cascades no analytic limit set can — hold a pose long enough and the
servos heat, the heat raises current draw, the shared bus sags, and the
brownout floor binds before any joint limit does. The margins reported here
come from the *measured* twin (thermal ceiling, bus floor, torque headroom,
joint limits) and, when a base profile is declared, from the coupled tip-over
(ZMP) margin of the realized motion.

Scope, stated honestly:

- The posterior is over the twin's **off-body** fitted leaves (thermal,
  actuation scale, battery), the ones :meth:`~jaxterity.twin.Twin.uncertainty`
  carries. Body/payload parameter uncertainty is not propagated here (see
  ``KNOWN_GAPS.md``); a payload's own mass interval is reported by the
  identification verb and enters the enrollment lane
  (:mod:`jaxterity.objects.scene`) as a worst-case, not as a sampled axis.
- Draws are clipped to stay physical (positive), so a posterior wide enough to
  cross zero shows a piled-up lower tail rather than an unphysical unit.
- Bands are **empirical quantiles over the draws**, not theoretical confidence
  intervals; with a handful of draws they are coarse, and they inherit whatever
  the Laplace/bootstrap covariance was worth.

The verbs are exposed as MCP tools in :mod:`jaxterity.mcp.oracle_tools`, so an
LLM planner can call them natively.

Units
-----
rad, rad/s, N·m, K, V, s, m (tip-over margin).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Mapping, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..calibrate.active import dof_joint_names
from ..errors import JaxterityError
from ..selfcheck.monitor import (
    SafetyAction,
    SafetyDecision,
    SafetyEnvelope,
    SafetyMonitor,
)
from ..twin import Twin, TwinRollout

#: Off-body leaves the posterior is drawn over (those a twin actually carries).
_OFFBODY_LEAVES = ("C_th", "R_th", "k_pwr", "K_t_scale", "V_oc", "R_int")

#: Severity order used to take the worst action across a posterior.
_ACTION_RANK = {SafetyAction.OK: 0, SafetyAction.DERATE: 1, SafetyAction.VETO: 2}

#: Name of the coupled tip-over constraint in an answer's margin set.
TIP_OVER = "tip_over"

#: Cap on ZMP evaluation samples per draw — the tip-over screen decimates a long
#: rollout rather than running forward kinematics at every step.
_ZMP_MAX_SAMPLES = 64


@dataclasses.dataclass(frozen=True)
class Plan:
    """A question posed as a motion: an initial condition plus commanded torques.

    The oracle's unit of work. Deliberately the same shape the gate vets — a
    plan you can ask about is a plan you can then run.

    Args
    ----
    name : str
        Human/agent-readable label, reported back in the answer.
    q0, qd0, T0 : tuple[float, ...]
        Initial joint positions (rad), velocities (rad/s), and per-servo
        temperature (K).
    tau : tuple[tuple[float, ...], ...]
        Commanded generalized force per step, N·m, shape ``(n_steps, nv)``
        (zero-order held over each step).
    dt : float
        Step size, s.

    Units
    -----
    rad, rad/s, K, N·m, s.
    """

    name: str
    q0: tuple[float, ...]
    qd0: tuple[float, ...]
    T0: tuple[float, ...]
    tau: tuple[tuple[float, ...], ...]
    dt: float

    @property
    def n_steps(self) -> int:
        """Number of commanded steps."""
        return len(self.tau)

    @property
    def duration_s(self) -> float:
        """Wall-clock length of the plan, s."""
        return self.n_steps * self.dt


def hold_plan(
    name: str,
    twin: Twin,
    tau,
    *,
    n_steps: int = 50,
    dt: float = 0.005,
    q0=None,
    qd0=None,
    T0=None,
) -> Plan:
    """Build the "hold this and see" plan — a constant commanded torque.

    The question behind most thermal and brownout answers ("can I carry this
    for the next N seconds?") is a sustained hold, so it gets a constructor.

    Args
    ----
    name : str
        Plan label.
    twin : Twin
        The unit (supplies the DOF count and ambient temperature).
    tau : array-like, shape ``(nv,)``
        The held commanded generalized force, N·m.
    n_steps : int
        Number of steps to hold.
    dt : float
        Step size, s. The twin integrates with a fixed-step RK4 and a
        free-moving arm diverges above ~10 ms (gravity alone is enough), so the
        default is deliberately small — lengthen the horizon with ``n_steps``,
        not with ``dt``.
    q0, qd0 : array-like or None
        Initial joint positions / velocities; default zeros.
    T0 : array-like or None
        Initial per-servo temperature, K; default the twin's ambient.

    Returns
    -------
    Plan
        The constant-torque plan.

    Units
    -----
    N·m, rad, rad/s, K, s.
    """
    nv = twin.n_servos
    nq = twin.robot.build_system(actuation=True)._nq
    tau_vec = np.asarray(tau, dtype=float).reshape(-1)
    if tau_vec.shape[0] != nv:
        raise JaxterityError(
            f"hold_plan: tau has {tau_vec.shape[0]} entries but the twin has "
            f"{nv} servos. Pass one commanded torque per DOF (N·m)."
        )
    q_start = np.zeros(nq) if q0 is None else np.asarray(q0, dtype=float)
    qd_start = np.zeros(nv) if qd0 is None else np.asarray(qd0, dtype=float)
    T_start = np.full(nv, twin.T_amb) if T0 is None else np.asarray(T0, dtype=float)
    return Plan(
        name=name,
        q0=tuple(float(v) for v in q_start),
        qd0=tuple(float(v) for v in qd_start),
        T0=tuple(float(v) for v in T_start),
        tau=tuple(tuple(float(v) for v in tau_vec) for _ in range(int(n_steps))),
        dt=float(dt),
    )


def plan_from_trajectory(name: str, plan_like: Plan, tau_seq) -> Plan:
    """Return ``plan_like`` with a different torque trajectory (same start)."""
    seq = np.atleast_2d(np.asarray(tau_seq, dtype=float))
    return dataclasses.replace(
        plan_like,
        name=name,
        tau=tuple(tuple(float(v) for v in row) for row in seq),
    )


@dataclasses.dataclass(frozen=True)
class MarginBand:
    """One constraint's outcome across the posterior: worst case plus spread.

    Args
    ----
    name : str
        Constraint name (``"servo_temperature"``, ``"bus_voltage"``,
        ``"tip_over"``, …).
    worst : float
        The worst value any draw produced, in the constraint's own SI unit —
        the number a worst-case gate would act on.
    median : float
        Median across draws.
    lo, hi : float
        Empirical quantile band across draws at the requested coverage.
    factor_worst : float
        Worst per-draw severity factor for this constraint, in ``[0, 1]``
        (1 = clear, 0 = breached), on the monitor's scale.
    units : str
        The unit ``worst``/``median``/``lo``/``hi`` are in.
    """

    name: str
    worst: float
    median: float
    lo: float
    hi: float
    factor_worst: float
    units: str

    def summary(self) -> str:
        """One line: worst case, median, and band."""
        return (
            f"{self.name}: worst {self.worst:.4g} {self.units} "
            f"(median {self.median:.4g}, band [{self.lo:.4g}, {self.hi:.4g}])"
        )


@dataclasses.dataclass(frozen=True)
class OracleAnswer:
    """The answer to one plan-shaped question, with its dispersion.

    Args
    ----
    question : str
        The plan's name.
    outcome : str
        ``"pass"`` / ``"derate"`` / ``"veto"`` — the **worst case** over the
        posterior, i.e. what the gate would do if the unit is at the unlucky
        end of its own uncertainty.
    pass_fraction : float
        Fraction of draws that were not vetoed — the confidence in the outcome,
        and the honest separator between "physics says no" (fraction 0) and
        "our fit is too loose to tell" (fraction near 1 with a veto worst case).
    binding_constraint : str
        The constraint with the least median headroom.
    margins : tuple[MarginBand, ...]
        Per-constraint band, worst first by severity.
    time_to_breach_s : float | None
        Shortest predicted time to a thermal breach across draws, s, if any.
    twin_handle : str
        The twin's attestation handle — an answer binds to one unit version.
    n_draws : int
        Posterior draws evaluated.
    coverage : float
        Quantile coverage of the bands.
    detail : str
        One-line human/agent explanation.
    """

    question: str
    outcome: str
    pass_fraction: float
    binding_constraint: str
    margins: tuple[MarginBand, ...]
    time_to_breach_s: float | None
    twin_handle: str
    n_draws: int
    coverage: float
    detail: str

    @property
    def ok(self) -> bool:
        """True iff even the unlucky end of the posterior passes."""
        return self.outcome == "pass"

    def margin(self, name: str) -> MarginBand | None:
        """The band for one constraint by name, or ``None``."""
        for band in self.margins:
            if band.name == name:
                return band
        return None

    def summary(self) -> str:
        """One line: outcome, confidence, binding constraint."""
        return (
            f"{self.question}: {self.outcome.upper()} "
            f"({self.pass_fraction:.0%} of {self.n_draws} draws pass; "
            f"binding {self.binding_constraint}) — {self.detail}"
        )


@dataclasses.dataclass(frozen=True)
class WhatIfAnswer:
    """A ranked comparison of alternative plans.

    Args
    ----
    answers : tuple[OracleAnswer, ...]
        One answer per candidate, in the order asked.
    best : str
        Name of the candidate with the most headroom (see :func:`what_if` for
        the ordering).
    ranking : tuple[str, ...]
        Candidate names, best first.
    detail : str
        One-line explanation of why the winner won.
    """

    answers: tuple[OracleAnswer, ...]
    best: str
    ranking: tuple[str, ...]
    detail: str

    def answer(self, name: str) -> OracleAnswer | None:
        """The answer for one candidate by name, or ``None``."""
        for a in self.answers:
            if a.question == name:
                return a
        return None


@dataclasses.dataclass(frozen=True)
class ObjectAnswer:
    """What an interaction revealed about a held object.

    Args
    ----
    params : dict[str, float]
        Identified scalar parameters (``"payload.mass"``, ``"payload.com[0]"``,
        …), SI.
    intervals : dict[str, tuple[float, float]] | None
        Wald interval per parameter at :attr:`coverage`; ``None`` when no
        uncertainty was estimated.
    coverage : float
        Interval coverage.
    recipe : str
        The recipe that produced the fit — ``"objects.payload"``, or
        ``"objects.payload+semantic_prior"`` when a declared guess seeded it.
    telemetry_hash : str
        Provenance of the telemetry the fit consumed.
    warnings : tuple[str, ...]
        The fit's structured warnings, verbatim (identifiability, pinned
        bounds, unresolved parameters, the prior declaration).
    """

    params: dict[str, float]
    intervals: dict[str, tuple[float, float]] | None
    coverage: float
    recipe: str
    telemetry_hash: str
    warnings: tuple[str, ...]

    def summary(self) -> str:
        """One line per identified parameter with its interval."""
        parts = []
        for key, value in self.params.items():
            if self.intervals is not None and key in self.intervals:
                lo, hi = self.intervals[key]
                parts.append(f"{key}={value:.4g} [{lo:.4g}, {hi:.4g}]")
            else:
                parts.append(f"{key}={value:.4g}")
        return f"{self.recipe}: " + ", ".join(parts)


# --- the posterior ---------------------------------------------------------


def posterior_twins(
    twin: Twin, n_draws: int, *, seed: int = 0
) -> tuple[tuple[Twin, ...], str]:
    """Draw a posterior sample of the unit's off-body parameters.

    Samples each fitted off-body leaf from a Gaussian at its point estimate
    with the marginal variance :meth:`~jaxterity.twin.Twin.uncertainty`
    carries, clipped to stay physical (positive). Leaves with no attached
    uncertainty are held at their point value: the posterior is only as wide as
    what was actually measured.

    Args
    ----
    twin : Twin
        The unit.
    n_draws : int
        Number of draws. Collapses to 1 when the twin carries no uncertainty at
        all — an ensemble of identical copies would be a fake band.
    seed : int
        PRNG seed (draws are deterministic in it).

    Returns
    -------
    tuple[tuple[Twin, ...], str]
        The drawn twins and a one-line note on what the posterior covers
        (reported in the answer's ``detail``).

    Units
    -----
    Inherited per leaf (J/K, K/W, W/(N·m)², dimensionless, V, Ω).
    """
    if n_draws < 1:
        raise JaxterityError(f"posterior_twins: n_draws must be >= 1, got {n_draws}.")
    uncertainty = twin.uncertainty()
    sampled = {
        leaf: uncertainty[f"offbody.{leaf}"]
        for leaf in _OFFBODY_LEAVES
        if f"offbody.{leaf}" in uncertainty and getattr(twin, leaf) is not None
    }
    if not sampled:
        return (twin,), (
            "posterior: none — this twin carries no fitted off-body covariance, "
            "so the answer is a point prediction and its band is zero width. "
            "Attach a fit's uncertainty (Twin.with_thermal(param_uncertainty=…)) "
            "for a band that means something."
        )
    rng = np.random.default_rng(seed)
    draws: list[Twin] = []
    for _ in range(n_draws):
        replacements = []
        for leaf, variance in sampled.items():
            point = np.asarray(getattr(twin, leaf), dtype=float)
            std = np.sqrt(np.maximum(np.asarray(variance, dtype=float), 0.0))
            std = np.broadcast_to(std, point.shape)
            drawn = point + std * rng.standard_normal(point.shape)
            # Physical floor: these leaves are positive by construction.
            drawn = np.maximum(drawn, 1e-6 * np.abs(point) + 1e-12)
            replacements.append(jnp.asarray(drawn))
        keys = tuple(sampled)

        def where(tree: Twin, keys: tuple[str, ...] = keys) -> list[Any]:
            return [getattr(tree, k) for k in keys]

        draws.append(eqx.tree_at(where, twin, replace=replacements))
    named = ", ".join(sorted(sampled))
    return tuple(draws), f"posterior over {named} ({n_draws} draws)"


def _stack_twins(twin: Twin, twins: Sequence[Twin]) -> tuple[Twin, Any]:
    """Stack a draw list into one batched twin plus its ``vmap`` axis tree."""
    fields = [f for f in _OFFBODY_LEAVES if getattr(twin, f) is not None]
    stacked_values = [
        jnp.stack([jnp.asarray(getattr(t, f), dtype=float) for t in twins])
        for f in fields
    ]

    def where(tree):
        return [getattr(tree, f) for f in fields]

    stacked = eqx.tree_at(where, twin, replace=stacked_values)
    none_tree = jax.tree_util.tree_map(lambda _: None, twin)
    axes = eqx.tree_at(
        where,
        none_tree,
        replace=[0] * len(fields),
        is_leaf=lambda x: x is None,
    )
    return stacked, axes


def _ensemble_rollout(twin: Twin, twins: Sequence[Twin], plan: Plan) -> TwinRollout:
    """Roll one plan through every drawn twin in a single ``vmap``."""
    q0 = jnp.asarray(plan.q0, dtype=float)
    qd0 = jnp.asarray(plan.qd0, dtype=float)
    T0 = jnp.asarray(plan.T0, dtype=float)
    tau = jnp.asarray(plan.tau, dtype=float)
    if len(twins) == 1:
        return twins[0].rollout(q0, qd0, T0, tau, plan.dt)
    stacked, axes = _stack_twins(twin, twins)
    return jax.vmap(lambda tw: tw.rollout(q0, qd0, T0, tau, plan.dt), in_axes=(axes,))(
        stacked
    )


def _require_finite(rollout: TwinRollout, plan: Plan) -> None:
    """Refuse to answer from a diverged rollout.

    The twin integrates with a *fixed-step* RK4, so an over-long step blows the
    body state up to ``inf``/``nan`` — and a margin computed from that would be
    silently meaningless (a NaN compares false against every limit). An
    unsimulable plan is itself an answer, so it is raised, not returned.
    """
    for label, array in (
        ("joint positions", rollout.q),
        ("joint velocities", rollout.qd),
        ("servo temperature", rollout.temperature),
    ):
        values = np.asarray(array)
        bad = ~np.isfinite(values)
        if not bad.any():
            continue
        per_step = bad.any(axis=-1)
        if per_step.ndim > 1:  # batched over draws
            per_step = per_step.any(axis=0)
        step = int(np.argmax(per_step))
        raise JaxterityError(
            f"simulate_plan: the rollout of plan {plan.name!r} diverged — "
            f"{label} became non-finite at step {step} (t = {step * plan.dt:.4g} "
            f"s) with dt = {plan.dt:g} s. The twin integrates with a fixed-step "
            "RK4, so this is an integration failure, not a physical verdict; "
            "a free-swinging arm typically needs dt <= 0.01 s. Shorten dt (and "
            "raise n_steps to keep the horizon) or reduce the commanded torque."
        )


def _slice_rollout(rollout: TwinRollout, i: int, n_draws: int) -> TwinRollout:
    """One draw's trajectories out of a batched (or unbatched) rollout."""
    if n_draws == 1:
        return rollout
    return TwinRollout(
        q=rollout.q[i],
        qd=rollout.qd[i],
        temperature=rollout.temperature[i],
        bus_voltage=None if rollout.bus_voltage is None else rollout.bus_voltage[i],
    )


# --- constraint bookkeeping ------------------------------------------------

_UNITS = {
    "servo_temperature": "K",
    "joint_limit": "rad",
    "bus_voltage": "V",
    "torque": "fraction",
    TIP_OVER: "m",
}
#: Constraints whose worst case is the *smallest* value seen (a floor), rather
#: than the largest (a ceiling).
_FLOOR_CONSTRAINTS = frozenset({"joint_limit", "bus_voltage", TIP_OVER})


def _band(
    name: str, values: Sequence[float], factors: Sequence[float], coverage: float
) -> MarginBand:
    """Summarize one constraint's per-draw values as a band."""
    arr = np.asarray(values, dtype=float)
    tail = (1.0 - coverage) / 2.0
    lo, hi = np.quantile(arr, [tail, 1.0 - tail])
    worst = float(arr.min()) if name in _FLOOR_CONSTRAINTS else float(arr.max())
    return MarginBand(
        name=name,
        worst=worst,
        median=float(np.median(arr)),
        lo=float(lo),
        hi=float(hi),
        factor_worst=float(np.min(np.asarray(factors, dtype=float))),
        units=_UNITS.get(name, "SI"),
    )


def _tip_over_margins(twin: Twin, rollout_q: np.ndarray, dt: float, base) -> float:
    """Worst ZMP margin (m) of one realized joint trajectory."""
    from ..checkup.stability import zmp_margins

    names = dof_joint_names(twin.robot)
    q = np.asarray(rollout_q, dtype=float)
    if q.shape[1] != len(names):
        raise JaxterityError(
            f"oracle tip-over screen: the rollout has {q.shape[1]} position "
            f"coordinates but the robot has {len(names)} actuated DOFs. The ZMP "
            "screen reads a joint trajectory in DOF order, so it does not yet "
            "cover floating-base or ball-joint models. Ask without `base=` for "
            "the thermal/bus/joint margins alone."
        )
    stride = max(1, math.ceil(q.shape[0] / _ZMP_MAX_SAMPLES))
    return float(zmp_margins(twin.robot, q[::stride], dt * stride, names, base).min())


def simulate_plan(
    twin: Twin,
    plan: Plan,
    envelope: SafetyEnvelope,
    *,
    base=None,
    n_draws: int = 16,
    seed: int = 0,
    coverage: float = 0.9,
) -> OracleAnswer:
    """Answer "what happens if I run this?" — outcome, margins, and confidence.

    Rolls ``plan`` through a posterior sample of the unit
    (:func:`posterior_twins`), scores every draw against ``envelope`` with
    :class:`~jaxterity.selfcheck.SafetyMonitor`, optionally adds the coupled
    tip-over (ZMP) margin of each realized motion, and reports the worst case
    together with the spread that produced it.

    Args
    ----
    twin : Twin
        The unit to ask about. Its attestation handle is recorded in the answer.
    plan : Plan
        The candidate motion.
    envelope : SafetyEnvelope
        The limits to score against (thermal ceiling, bus floor, joint margin,
        torque headroom).
    base : jaxterity.checkup.BaseProfile | None
        Declared mobile base. When given, each draw's realized joint
        trajectory is additionally screened for tip-over and a
        :data:`TIP_OVER` margin band joins the answer.
    n_draws : int
        Posterior draws. Ignored (collapsed to 1) when the twin carries no
        fitted covariance.
    seed : int
        PRNG seed for the draws.
    coverage : float
        Quantile coverage of the reported bands, in ``(0, 1)``.

    Returns
    -------
    OracleAnswer
        Worst-case outcome, per-constraint bands, pass fraction, twin handle.

    Raises
    ------
    JaxterityError
        If the plan is empty, its widths disagree with the twin, ``coverage``
        is out of range, or a tip-over screen is asked for on a model whose
        position coordinates are not one-per-DOF.

    Units
    -----
    K, V, rad, N·m, m (tip-over), s.

    Examples
    --------
    >>> from jaxterity.twin import so101_twin  # doctest: +SKIP
    >>> twin = so101_twin("SO101-0001")  # doctest: +SKIP
    >>> answer = simulate_plan(  # doctest: +SKIP
    ...     twin,
    ...     hold_plan("carry 2 kg", twin, [0.4] * twin.n_servos, n_steps=100,
    ...               dt=0.005),
    ...     SafetyEnvelope(temp_ceiling_k=343.15),
    ... )
    >>> answer.outcome  # doctest: +SKIP
    'pass'
    """
    if not (0.0 < coverage < 1.0):
        raise JaxterityError(
            f"simulate_plan: coverage must be in (0, 1); got {coverage!r}."
        )
    if plan.n_steps < 1:
        raise JaxterityError(
            f"simulate_plan: plan {plan.name!r} commands no steps. A question "
            "needs at least one commanded step to simulate."
        )
    nv = twin.n_servos
    if len(plan.qd0) != nv or len(plan.T0) != nv or len(plan.tau[0]) != nv:
        raise JaxterityError(
            f"simulate_plan: plan {plan.name!r} is {len(plan.tau[0])} wide with "
            f"qd0 {len(plan.qd0)} / T0 {len(plan.T0)}, but the twin has {nv} "
            "servos. Build the plan against this unit (see hold_plan)."
        )

    draws, posterior_note = posterior_twins(twin, n_draws, seed=seed)
    n = len(draws)
    rollouts = _ensemble_rollout(twin, draws, plan)
    _require_finite(rollouts, plan)
    action_repr = np.max(np.abs(np.asarray(plan.tau, dtype=float)), axis=0)

    decisions: list[SafetyDecision] = []
    values: dict[str, list[float]] = {}
    factors: dict[str, list[float]] = {}
    tip_values: list[float] = []
    for i in range(n):
        prediction = _slice_rollout(rollouts, i, n)
        decision = SafetyMonitor(draws[i], envelope).evaluate(
            prediction, action=action_repr
        )
        decisions.append(decision)
        for name, value in decision.worst_values.items():
            values.setdefault(name, []).append(float(value))
            factors.setdefault(name, []).append(float(decision.factors[name]))
        if base is not None:
            tip_values.append(
                _tip_over_margins(twin, np.asarray(prediction.q), plan.dt, base)
            )

    ranks = [_ACTION_RANK[d.action] for d in decisions]
    tip_factors: list[float] = []
    if base is not None:
        floor = float(base.margin_min_m)
        tip_factors = [
            float(np.clip(m / floor, 0.0, 1.0)) if floor > 0 else float(m >= 0.0)
            for m in tip_values
        ]
        values[TIP_OVER] = tip_values
        factors[TIP_OVER] = tip_factors
        for i, f in enumerate(tip_factors):
            ranks[i] = max(ranks[i], 2 if f <= 0.0 else (1 if f < 1.0 else 0))

    bands = tuple(
        sorted(
            (_band(name, values[name], factors[name], coverage) for name in values),
            key=lambda b: b.factor_worst,
        )
    )
    binding = min(
        factors, key=lambda k: float(np.median(np.asarray(factors[k], dtype=float)))
    )
    worst_rank = max(ranks)
    outcome = ("pass", "derate", "veto")[worst_rank]
    pass_fraction = float(np.mean([r < 2 for r in ranks]))
    breaches = [d.time_to_breach_s for d in decisions if d.time_to_breach_s is not None]
    binding_band = next(b for b in bands if b.name == binding)
    detail = (
        f"{binding_band.summary()}; {posterior_note}. Worst case is what a gate "
        "would enforce; the pass fraction says how much of it is this unit's "
        "own parameter uncertainty."
    )
    return OracleAnswer(
        question=plan.name,
        outcome=outcome,
        pass_fraction=pass_fraction,
        binding_constraint=binding,
        margins=bands,
        time_to_breach_s=min(breaches) if breaches else None,
        twin_handle=twin.attestation_handle,
        n_draws=n,
        coverage=float(coverage),
        detail=detail,
    )


def what_if(
    twin: Twin,
    plans: Sequence[Plan],
    envelope: SafetyEnvelope,
    *,
    base=None,
    n_draws: int = 16,
    seed: int = 0,
    coverage: float = 0.9,
) -> WhatIfAnswer:
    """Compare candidate plans and rank them by the headroom each keeps.

    The comparison verb: "which grasp keeps the most tip-over margin?", "is the
    slow version worth it?". Each candidate is answered independently by
    :func:`simulate_plan` on the *same* posterior draws (same ``seed``), so the
    ranking reflects the plans and not the sampling.

    Ordering, best first: fewest vetoed draws, then the largest worst-case
    severity factor on the binding constraint (i.e. the most headroom at the
    unlucky end of the posterior), then the earliest candidate.

    Args
    ----
    twin : Twin
        The unit.
    plans : Sequence[Plan]
        The candidates (at least one), with distinct names.
    envelope : SafetyEnvelope
        The limits every candidate is scored against.
    base : jaxterity.checkup.BaseProfile | None
        Declared mobile base for the tip-over screen (see
        :func:`simulate_plan`).
    n_draws, seed, coverage
        As in :func:`simulate_plan`.

    Returns
    -------
    WhatIfAnswer
        Per-candidate answers plus the ranking and the winner.

    Raises
    ------
    JaxterityError
        If ``plans`` is empty or two candidates share a name (the answer is
        keyed by name).

    Units
    -----
    As :func:`simulate_plan`.
    """
    if not plans:
        raise JaxterityError("what_if: needs at least one candidate plan.")
    names = [p.name for p in plans]
    if len(set(names)) != len(names):
        raise JaxterityError(
            f"what_if: candidate names must be unique; got {names}. The answer "
            "is keyed by name."
        )
    answers = tuple(
        simulate_plan(
            twin,
            plan,
            envelope,
            base=base,
            n_draws=n_draws,
            seed=seed,
            coverage=coverage,
        )
        for plan in plans
    )

    def key(item: tuple[int, OracleAnswer]) -> tuple[float, float, int]:
        index, answer = item
        binding = answer.margin(answer.binding_constraint)
        headroom = 0.0 if binding is None else binding.factor_worst
        return (-answer.pass_fraction, -headroom, index)

    ordered = [a for _, a in sorted(enumerate(answers), key=key)]
    best = ordered[0]
    runner_up = ordered[1] if len(ordered) > 1 else None
    if runner_up is None:
        detail = f"only candidate: {best.summary()}"
    else:
        detail = (
            f"{best.question} wins on {best.binding_constraint} "
            f"({best.pass_fraction:.0%} of draws pass) over {runner_up.question} "
            f"({runner_up.pass_fraction:.0%}, binding "
            f"{runner_up.binding_constraint})"
        )
    return WhatIfAnswer(
        answers=answers,
        best=best.question,
        ranking=tuple(a.question for a in ordered),
        detail=detail,
    )


def identify_object(
    robot,
    link,
    payload,
    *,
    prior=None,
    coverage: float = 0.95,
    **kwargs: Any,
) -> ObjectAnswer:
    """Ask the arm what it is holding: parameters, intervals, provenance.

    The identification verb. Thin by design — it runs
    :func:`jaxterity.objects.identify_payload` (optionally seeded and bounded
    by a declared :class:`~jaxterity.objects.SemanticPrior`) and flattens the
    result into the oracle's answer shape: scalar parameters, their Wald
    intervals, the recipe that produced them, and the fit's own warnings
    verbatim. Nothing is smoothed away — a pinned bound or an unresolved
    parameter reaches the caller as written.

    Args
    ----
    robot : jaxterity.robot.Robot
        The (calibrated) robot doing the holding.
    link : jaxterity.calibrate.active.ActiveLink
        The transport (loopback fake or real replay).
    payload : jaxterity.objects.RigidPayload
        The object: attach point, name, priors.
    prior : jaxterity.objects.SemanticPrior | None
        A declared semantic guess to seed and bound the fit.
    coverage : float
        Interval coverage, in ``(0, 1)``.
    **kwargs
        Forwarded to :func:`~jaxterity.objects.identify_payload` (``fit``,
        ``excitation``, ``iterations``, ``uncertainty``, …).

    Returns
    -------
    ObjectAnswer
        Identified parameters with intervals and provenance.

    Units
    -----
    kg, m, kg·m².
    """
    from ..objects import identify_payload

    result = identify_payload(robot, link, payload, prior=prior, **kwargs)
    return object_answer(result, coverage=coverage)


def object_answer(result, *, coverage: float = 0.95) -> ObjectAnswer:
    """Flatten a payload :class:`~jaxterity.sysid.SysidResult` into an answer.

    The shared adapter behind :func:`identify_object` and the enrollment lane
    (:mod:`jaxterity.objects.scene`), which needs the ``SysidResult`` itself to
    build the object's mini-twin and the answer for the record.

    Args
    ----
    result : jaxterity.sysid.SysidResult
        A payload fit.
    coverage : float
        Wald-interval coverage in ``(0, 1)``.

    Returns
    -------
    ObjectAnswer
        Scalar parameters, intervals, provenance, and the fit's warnings
        verbatim.

    Units
    -----
    kg, m, kg·m².
    """
    params: dict[str, float] = {}
    for key, value in result.fitted.items():
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            params[key] = float(arr)
        elif arr.ndim == 1:
            for i, v in enumerate(arr):
                params[f"{key}[{i}]"] = float(v)
        else:  # inertia tensor — report the principal diagonal
            for i in range(arr.shape[0]):
                params[f"{key}_diag[{i}]"] = float(arr[i, i])
    intervals = result.interval(coverage)
    return ObjectAnswer(
        params=params,
        intervals=intervals,
        coverage=float(coverage),
        recipe=result.recipe_name,
        telemetry_hash=result.telemetry_hash,
        warnings=tuple(result.warnings),
    )


def envelope_from_mapping(mapping: Mapping[str, Any]) -> SafetyEnvelope:
    """Build a :class:`~jaxterity.selfcheck.SafetyEnvelope` from a plain dict.

    The adapter the MCP tool layer uses: agents pass JSON, the envelope is a
    dataclass. Unknown keys are rejected rather than ignored, so a typo in a
    limit name never silently disables a check.

    Args
    ----
    mapping : Mapping[str, Any]
        Keys from :class:`~jaxterity.selfcheck.SafetyEnvelope`'s fields.

    Returns
    -------
    SafetyEnvelope
        The envelope.

    Raises
    ------
    JaxterityError
        On an unknown key.
    """
    fields = {f.name for f in dataclasses.fields(SafetyEnvelope)}
    unknown = set(mapping) - fields
    if unknown:
        raise JaxterityError(
            f"envelope_from_mapping: unknown limit(s) {sorted(unknown)}. "
            f"Known: {sorted(fields)}. A mistyped limit would silently leave "
            "that constraint unchecked, so it is refused."
        )
    return SafetyEnvelope(**dict(mapping))


__all__ = [
    "Plan",
    "MarginBand",
    "OracleAnswer",
    "WhatIfAnswer",
    "ObjectAnswer",
    "TIP_OVER",
    "hold_plan",
    "plan_from_trajectory",
    "posterior_twins",
    "simulate_plan",
    "what_if",
    "identify_object",
    "object_answer",
    "envelope_from_mapping",
]
