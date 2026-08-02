# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Scene enrollment: objects get checkups too.

A unit's checkup measures the *body*. This module runs the same ceremony on
what the body picks up, so the protected resource extends from the robot to the
world it touches:

1. **Perceive** — a declared :class:`~jaxterity.objects.SemanticPrior` (a
   vision model's guess) brackets what the object could be.
2. **Vet the probe under the prior's worst case** — the chicken-and-egg of
   probing an unknown heavy object is resolved the way T-229 resolved it for
   capture plans: score the candidate probe on the *heaviest* object the prior
   allows and derate it until the tip-over margin clears the base's floor
   (:func:`vet_probe`). A probe that cannot be vetted is refused, not run
   slowly and hoped over.
3. **Heft** — the vetted probe runs, and the object's mass/CoM are identified
   from the arm's own telemetry, seeded and bounded by the same prior. The
   posterior tightens: a bracket becomes an estimate with an interval.
4. **Instantiate the mini-twin** — the fitted object is welded onto the unit's
   twin (:meth:`~jaxterity.twin.Twin.with_payload`), which is a *new attested
   configuration*: the same arm holding a different object is a different unit
   version, so a pre-flight pass on the empty arm does not vouch for the loaded
   one.
5. **Simulate the coupled system** — candidate techniques are compared on the
   loaded twin with the tip-over screen active
   (:func:`jaxterity.oracle.what_if`), and the winner is the vetted technique.

The result (:class:`ObjectEnrollment`) is one entry in what the strategy calls a
*world ledger*: not a semantic label, but measured parameters with intervals and
provenance, bound to the unit version that measured them.

Honest scope: rigid, lumped objects only — no materials or processes (cutting,
drilling), no grip-slip model, and the worst case swept is the prior's heaviest
**mass** at the declared CoM, not every corner of its CoM box. Everything here
is sim-tier: with a
:class:`~jaxterity.calibrate.active.ReplayLink` the heft step consumes a real
capture, but the demo path uses the seeded loopback and says so.

Units
-----
kg, m, kg·m², rad/s, s.
"""

from __future__ import annotations

import dataclasses
from typing import Sequence

import numpy as np

from ..calibrate.active import (
    ActiveLink,
    Excitation,
    design_coast_excitation,
)
from ..errors import CalibrationError
from ..oracle import ObjectAnswer, OracleAnswer, Plan, WhatIfAnswer, object_answer
from ..oracle import what_if as _what_if
from ..selfcheck.monitor import SafetyEnvelope
from ..twin import Twin
from .payload import RigidPayload, identify_payload
from .priors import SemanticPrior
from .probe import probe_margin

#: Probe speeds tried, fastest first — the fastest probe carries the most
#: Fisher information, so vetting walks *down* from it (rad/s).
DEFAULT_PROBE_SPEEDS: tuple[float, ...] = (0.30, 0.20, 0.12, 0.08, 0.05, 0.03)


@dataclasses.dataclass(frozen=True)
class ProbeVetting:
    """Whether a probe may be run on an object we have only *guessed* at.

    Args
    ----
    approved : bool
        True iff some candidate probe cleared the tip-over floor under the
        prior's worst case.
    speed_rad_s : float
        The approved probe's initial joint velocity, rad/s (the fastest
        candidate that cleared).
    worst_case_mass_kg : float
        The mass the vetting assumed — the heaviest the prior allows.
    margin_m : float
        Worst tip-over margin of the approved probe under that worst case, m.
    floor_m : float
        The floor it had to clear, m.
    attempts : tuple[tuple[float, float], ...]
        ``(speed, margin)`` per candidate tried, in order.
    detail : str
        One-line human/agent explanation.

    Units
    -----
    rad/s, kg, m.
    """

    approved: bool
    speed_rad_s: float
    worst_case_mass_kg: float
    margin_m: float
    floor_m: float
    attempts: tuple[tuple[float, float], ...]
    detail: str


@dataclasses.dataclass(frozen=True)
class ObjectEnrollment:
    """One object's entry in the unit's ledger of measured things.

    Args
    ----
    label : str
        The prior's label — what it was believed to be going in.
    prior : SemanticPrior
        The declared guess that seeded and bounded everything downstream.
    vetting : ProbeVetting
        How the probe was made safe before it was run.
    identification : ObjectAnswer
        What the heft measured: parameters, intervals, provenance, warnings.
    loaded_twin : Twin
        The unit holding the fitted object — a new attested configuration.
    comparison : WhatIfAnswer
        The coupled robot+object ranking of the candidate techniques.
    technique : str
        Name of the winning (vetted) technique.
    detail : str
        One-line summary of the whole arc.
    """

    label: str
    prior: SemanticPrior
    vetting: ProbeVetting
    identification: ObjectAnswer
    loaded_twin: Twin
    comparison: WhatIfAnswer
    technique: str
    detail: str

    @property
    def verdict(self) -> OracleAnswer:
        """The winning technique's coupled answer."""
        answer = self.comparison.answer(self.technique)
        assert answer is not None  # the winner is always among the answers
        return answer

    @property
    def twin_handle(self) -> str:
        """Attestation handle of the loaded (object-carrying) unit."""
        return self.loaded_twin.attestation_handle

    def summary(self) -> str:
        """One line: what it turned out to be, and what to do with it."""
        return (
            f"{self.label}: {self.identification.summary()} | probe vetted at "
            f"{self.vetting.speed_rad_s:.3g} rad/s | technique "
            f"{self.technique!r} ({self.verdict.outcome})"
        )


def worst_case_payload(payload: RigidPayload, prior: SemanticPrior) -> RigidPayload:
    """The heaviest object the prior allows, for pre-interaction vetting.

    Args
    ----
    payload : RigidPayload
        The object as declared (attach link, name, grasp offset, CoM prior).
    prior : SemanticPrior
        The semantic guess.

    Returns
    -------
    RigidPayload
        The same object at the prior's upper mass bound, with its inertia
        scaled to match. The CoM box is **not** swept (see the module's honest
        scope).

    Units
    -----
    kg, m, kg·m².
    """
    heavy = SemanticPrior(
        label=prior.label,
        mass_kg=(prior.mass_kg[1], prior.mass_kg[1]),
        source=prior.source,
        com_halfwidth_m=prior.com_halfwidth_m,
        inertia_max=prior.inertia_max,
        confidence=prior.confidence,
    )
    return heavy.seed(payload)


def _coast_from_pose(
    robot,
    speed: float,
    pose_rad: Sequence[float] | None,
    sample_times: tuple[float, ...] | None,
) -> Excitation:
    """A coast probe at ``speed``, starting from ``pose_rad`` (zeros if None)."""
    base_probe = (
        design_coast_excitation(robot, speed=float(speed))
        if sample_times is None
        else design_coast_excitation(
            robot, speed=float(speed), sample_times=sample_times
        )
    )
    if pose_rad is None:
        return base_probe
    pose = tuple(float(v) for v in pose_rad)
    if len(pose) != base_probe.n_dof:
        raise CalibrationError(
            f"scene: the declared start pose has {len(pose)} joints but the "
            f"robot has {base_probe.n_dof} DOFs (in dof_joint_names order)."
        )
    return dataclasses.replace(base_probe, q0=pose)


def vet_probe(
    robot,
    payload: RigidPayload,
    prior: SemanticPrior,
    stability,
    *,
    speeds: Sequence[float] = DEFAULT_PROBE_SPEEDS,
    pose_rad: Sequence[float] | None = None,
    sample_times: tuple[float, ...] | None = None,
    options=None,
) -> ProbeVetting:
    """Find the fastest probe that stays upright even if the object is heavy.

    Walks ``speeds`` from fastest to slowest, scoring each candidate coast on
    the arm *holding the prior's heaviest object* with
    :func:`~jaxterity.objects.probe_margin`, and returns the first that clears
    the stability floor. This is the T-229 pattern generalized from capture
    plans to objects: the gate does not merely reject a probe, it shapes it.

    How much authority the speed derate actually has, measured rather than
    assumed (SO-100 arm on the representative ``LEKIWI_BASE``, tucked pose,
    0.70 kg worst case): across 0.10-0.30 rad/s the worst-case margin moved
    **under 1 mm**, while moving the *pose* from tucked to the model's extended
    zero pose cost tens of millimetres. So the speed sweep is a fine
    adjustment; the pose is the lever, and a refusal from this function should
    be read as "re-pose", not as "probe more slowly".

    Args
    ----
    robot : jaxterity.robot.Robot
        The bare (calibrated) arm.
    payload : RigidPayload
        The object as declared.
    prior : SemanticPrior
        The guess whose upper bound defines the worst case.
    stability : jaxterity.checkup.StabilityConstraint
        The base, optional suspension, and the margin floor.
    speeds : Sequence[float]
        Candidate probe speeds, rad/s, fastest first.
    pose_rad : Sequence[float] | None
        Arm pose the probe coasts from, rad, in DOF order. ``None`` uses the
        model's zero pose — which on an SO-100-style arm is *extended forward*,
        the pose the 2026-07-28 hardware finding tips a LeKiwi from. On a
        mobile base, declare a tucked pose: no probe speed rescues a pose whose
        static margin is already under the floor, and the refusal says so.
    sample_times : tuple[float, ...] | None
        Capture times for the coast, s. Default: the active lane's.
    options : jaxonomy.SimulatorOptions | None
        Integration options.

    Returns
    -------
    ProbeVetting
        The approved probe (or a refusal), with every attempt recorded.

    Raises
    ------
    CalibrationError
        If ``speeds`` is empty.

    Units
    -----
    rad/s, m, kg.
    """
    if not tuple(speeds):
        raise CalibrationError("vet_probe: needs at least one candidate speed.")
    heavy = worst_case_payload(payload, prior)
    floor = float(stability.floor)
    attempts: list[tuple[float, float]] = []
    for speed in speeds:
        excitation = _coast_from_pose(robot, speed, pose_rad, sample_times)
        margin = probe_margin(robot, heavy, excitation, stability, options=options)
        attempts.append((float(speed), float(margin)))
        if margin >= floor:
            return ProbeVetting(
                approved=True,
                speed_rad_s=float(speed),
                worst_case_mass_kg=float(heavy.mass),
                margin_m=float(margin),
                floor_m=floor,
                attempts=tuple(attempts),
                detail=(
                    f"probe approved at {speed:.3g} rad/s: worst-case margin "
                    f"{margin * 1000:.1f} mm >= floor {floor * 1000:.1f} mm with "
                    f"the prior's heaviest object ({float(heavy.mass):.3g} kg)"
                ),
            )
    best_speed, best_margin = max(attempts, key=lambda a: a[1])
    return ProbeVetting(
        approved=False,
        speed_rad_s=float(best_speed),
        worst_case_mass_kg=float(heavy.mass),
        margin_m=float(best_margin),
        floor_m=floor,
        attempts=tuple(attempts),
        detail=(
            f"no probe approved: the best candidate ({best_speed:.3g} rad/s) "
            f"still leaves {best_margin * 1000:.1f} mm, under the "
            f"{floor * 1000:.1f} mm floor, with the prior's heaviest object "
            f"({float(heavy.mass):.3g} kg)"
        ),
    )


def enroll_object(
    twin: Twin,
    link: ActiveLink,
    payload: RigidPayload,
    prior: SemanticPrior,
    techniques: Sequence[Plan],
    envelope: SafetyEnvelope,
    *,
    stability=None,
    base=None,
    speeds: Sequence[float] = DEFAULT_PROBE_SPEEDS,
    pose_rad: Sequence[float] | None = None,
    fit: tuple[str, ...] = ("mass",),
    iterations: int = 8,
    uncertainty: str | None = "laplace",
    coverage: float = 0.95,
    n_draws: int = 8,
    seed: int = 0,
    options=None,
) -> ObjectEnrollment:
    """Enroll a held object: vet, heft, instantiate, and vet the technique.

    The whole arc in one call (see the module docstring). Each stage's output
    is the next stage's input, and every stage's honesty carries through: a
    probe that cannot be vetted stops the process, a fit that pins itself to a
    prior bound says so in the record, and the technique verdict is evaluated on
    the *loaded* unit — the one that is actually going to move.

    Args
    ----
    twin : Twin
        The unit doing the holding.
    link : jaxterity.calibrate.active.ActiveLink
        The heft transport: a seeded loopback (simulated) or a
        :class:`~jaxterity.calibrate.active.ReplayLink` over a real capture.
    payload : RigidPayload
        The object as declared: attach link, name, grasp offset.
    prior : SemanticPrior
        The vision model's guess — seeds and bounds both the vetting and the
        fit.
    techniques : Sequence[Plan]
        Candidate ways to do the job, compared on the loaded unit.
    envelope : SafetyEnvelope
        Limits the techniques are scored against.
    stability : jaxterity.checkup.StabilityConstraint | None
        Tip-over constraint for the probe vetting. ``None`` skips vetting —
        legitimate only for a bolted-down arm, and recorded as such.
    base : jaxterity.checkup.BaseProfile | None
        Base profile for the coupled technique screen. Defaults to
        ``stability.base`` when a stability constraint is given.
    speeds : Sequence[float]
        Candidate probe speeds for :func:`vet_probe`, rad/s.
    pose_rad : Sequence[float] | None
        Arm pose the heft is performed from, rad, in DOF order (see
        :func:`vet_probe` — on a mobile base, declare a tucked one).
    fit : tuple[str, ...]
        Payload parameter groups to identify.
    iterations : int
        Levenberg–Marquardt iterations for the heft.
    uncertainty : str | None
        Uncertainty estimator for the fit (``"laplace"`` by default) — the
        interval is the point of the exercise, so ``None`` is discouraged.
    coverage : float
        Interval coverage for the reported parameters.
    n_draws, seed : int
        Posterior draws (and their seed) for the technique comparison.
    options : jaxonomy.SimulatorOptions | None
        Integration options for the probe simulations.

    Returns
    -------
    ObjectEnrollment
        The ledger entry: vetting, identification, loaded twin, technique
        ranking.

    Raises
    ------
    CalibrationError
        If no candidate probe clears the stability floor under the prior's
        worst case (the enrollment refuses rather than probing an object that
        may tip the unit), or if ``techniques`` is empty.

    Units
    -----
    kg, m, rad/s, N·m, K, V, s.
    """
    if not tuple(techniques):
        raise CalibrationError(
            "enroll_object: needs at least one candidate technique — the "
            "enrollment ends by vetting what you intend to do with the object."
        )
    if stability is not None:
        vetting = vet_probe(
            twin.robot,
            payload,
            prior,
            stability,
            speeds=speeds,
            pose_rad=pose_rad,
            options=options,
        )
        if not vetting.approved:
            raise CalibrationError(
                f"enroll_object: {vetting.detail}. Re-pose the arm (a tucked "
                "pose has more margin than an extended one), measure the base "
                "profile to thin the floor, or narrow the prior before "
                "interacting with this object."
            )
        excitation: Excitation = _coast_from_pose(
            twin.robot, vetting.speed_rad_s, pose_rad, None
        )
        base = base if base is not None else stability.base
    else:
        speed = float(tuple(speeds)[-1])
        vetting = ProbeVetting(
            approved=True,
            speed_rad_s=speed,
            worst_case_mass_kg=float(prior.mass_kg[1]),
            margin_m=float("inf"),
            floor_m=0.0,
            attempts=(),
            detail=(
                "no stability constraint declared — the probe was not vetted "
                "for tip-over. Valid for a bolted-down arm only; pass a "
                "StabilityConstraint for anything on a mobile base."
            ),
        )
        excitation = _coast_from_pose(twin.robot, speed, pose_rad, None)

    result = identify_payload(
        twin.robot,
        link,
        payload,
        excitation=excitation,
        fit=fit,
        iterations=iterations,
        uncertainty=uncertainty,
        prior=prior,
        options=options,
    )
    identification = object_answer(result, coverage=coverage)
    seeded = prior.seed(payload)
    loaded = twin.with_payload(seeded, result)

    comparison = _what_if(
        loaded,
        list(techniques),
        envelope,
        base=base,
        n_draws=n_draws,
        seed=seed,
    )
    mass = identification.params.get(f"{payload.name}.mass")
    if mass is not None:
        tightening = (
            f"prior bracket [{prior.mass_kg[0]:.3g}, {prior.mass_kg[1]:.3g}] kg "
            f"-> heft {mass:.3g} kg"
        )
        if identification.intervals is not None:
            lo, hi = identification.intervals[f"{payload.name}.mass"]
            tightening += f" [{lo:.3g}, {hi:.3g}]"
    else:
        tightening = f"prior {prior.label!r} (mass not among the fitted groups)"
    detail = f"{tightening}; {vetting.detail}; coupled verdict: {comparison.detail}"
    return ObjectEnrollment(
        label=prior.label,
        prior=prior,
        vetting=vetting,
        identification=identification,
        loaded_twin=loaded,
        comparison=comparison,
        technique=comparison.best,
        detail=detail,
    )


def probe_speed_margin_curve(
    robot,
    payload: RigidPayload,
    prior: SemanticPrior,
    stability,
    *,
    speeds: Sequence[float] = DEFAULT_PROBE_SPEEDS,
    pose_rad: Sequence[float] | None = None,
    options=None,
) -> np.ndarray:
    """Worst-case tip-over margin (m) per candidate probe speed.

    The curve behind :func:`vet_probe`'s decision, exposed for plotting and for
    tests that assert the monotone shape (a faster probe on a heavier object
    leaves less margin).

    Args
    ----
    robot : jaxterity.robot.Robot
        The bare arm.
    payload : RigidPayload
        The declared object.
    prior : SemanticPrior
        The guess whose upper mass bound is used.
    stability : jaxterity.checkup.StabilityConstraint
        Base and floor.
    speeds : Sequence[float]
        Candidate speeds, rad/s.
    pose_rad : Sequence[float] | None
        Arm pose the probes coast from, rad, in DOF order.
    options : jaxonomy.SimulatorOptions | None
        Integration options.

    Returns
    -------
    numpy.ndarray
        Margin per speed, m, in the order given.

    Units
    -----
    rad/s in; m out.
    """
    heavy = worst_case_payload(payload, prior)
    return np.array(
        [
            probe_margin(
                robot,
                heavy,
                _coast_from_pose(robot, float(s), pose_rad, None),
                stability,
                options=options,
            )
            for s in speeds
        ]
    )


__all__ = [
    "DEFAULT_PROBE_SPEEDS",
    "ProbeVetting",
    "ObjectEnrollment",
    "worst_case_payload",
    "vet_probe",
    "enroll_object",
    "probe_speed_margin_curve",
]
