# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Scene enrollment on a LeKiwi: an unknown object gets a checkup.

A mobile manipulator is asked to move a tote it has never seen. A camera can
say what it *looks* like; it cannot say what it weighs, and on a small base
with a high payload-to-robot mass ratio that is exactly the number that decides
whether the unit tips over. This demo runs the whole arc, in simulation:

1. **Vision prior** — "a small plastic tote, roughly 10x8x6 cm" becomes a mass
   bracket (:class:`~jaxterity.objects.SemanticPrior`). Wide, and honest about
   it: 0.16-0.70 kg.
2. **Vet the probe under the worst case** — before touching anything, the
   candidate heft is simulated with the *heaviest* object the prior allows and
   scored for tip-over on the declared base. The fastest probe that clears the
   floor is the one that runs (:func:`~jaxterity.objects.vet_probe`). This is
   the chicken-and-egg escape: you never need to know the mass to probe safely,
   only its upper bound.
3. **Heft** — the vetted probe runs and the arm's own telemetry identifies the
   mass, seeded and bounded by the same prior. The bracket collapses to an
   estimate with an interval.
4. **Object mini-twin** — the fitted object is welded onto the unit's twin,
   which changes its attestation handle: the loaded unit is a different attested
   configuration from the empty one.
5. **Coupled technique verdict** — candidate ways to do the job are compared on
   the *loaded* unit with the tip-over screen active
   (:func:`jaxterity.oracle.what_if`); the winner is what gets executed.

Honest scope: **everything here is simulated.** The heft runs through the
seeded loopback link with a planted mass, so this demo shows that the pipeline
recovers a known answer, not that a real tote was weighed; swap
:class:`~jaxterity.objects.PayloadLoopbackLink` for a
:class:`~jaxterity.calibrate.active.ReplayLink` over a real capture and the same
code consumes hardware telemetry. The LeKiwi base profile is representative
geometry, not a measured unit, and the twin's thermal defaults are
representative too — so the *ordering* of the verdicts is trustworthy well
before the absolute margins are.

Run: ``python -m examples.scene_enrollment_lekiwi``.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib

from jaxterity.checkup.stability import LEKIWI_BASE, StabilityConstraint
from jaxterity.objects import (
    PayloadLoopbackLink,
    RigidPayload,
    SemanticPrior,
    enroll_object,
    vet_probe,
)
from jaxterity.oracle import hold_plan
from jaxterity.selfcheck import SafetyEnvelope
from jaxterity.twin import lekiwi_twin

#: A tucked start pose (rad, DOF order) whose static ZMP margin clears the
#: LeKiwi floor with room to spare. The model's zero pose is extended forward —
#: the pose that rocked a real LeKiwi off its rear wheel on 2026-07-28 — so the
#: enrollment declares this one instead.
TUCKED_POSE_RAD = (0.0, -1.2, -0.8, 0.0, 0.0, 0.0)

#: What is actually in the gripper, for the simulated heft (kg). The pipeline
#: does not know it; recovering it is the point.
PLANTED_MASS_KG = 0.42


@dataclasses.dataclass(frozen=True)
class DemoResult:
    """Everything the demo asserts on (and prints)."""

    prior_bracket_kg: tuple[float, float]
    worst_case_mass_kg: float
    probe_speed_rad_s: float
    probe_margin_m: float
    probe_attempts: tuple[tuple[float, float], ...]
    identified_mass_kg: float
    mass_interval_kg: tuple[float, float]
    recipe: str
    empty_twin_handle: str
    loaded_twin_handle: str
    technique: str
    technique_outcome: str
    rejected_technique: str
    rejected_outcome: str
    binding_constraint: str
    heavy_label: str
    heavy_worst_case_mass_kg: float
    heavy_approved: bool
    heavy_best_margin_m: float
    run_hash: str


def run_demo(
    *,
    serial: str = "LK-0001",
    planted_mass_kg: float = PLANTED_MASS_KG,
    n_steps: int = 30,
    dt: float = 0.005,
    n_draws: int = 2,
    speeds: tuple[float, ...] = (0.30, 0.10, 0.03),
) -> DemoResult:
    """Run the enrollment arc and return its structured result.

    Args
    ----
    serial : str
        The unit's serial number (part of its attested identity).
    planted_mass_kg : float
        Ground truth for the simulated heft, kg.
    n_steps, dt : int, float
        Length and step of each candidate technique (s). The twin integrates
        with a fixed-step RK4; keep ``dt <= 0.01`` for a moving arm.
    n_draws : int
        Posterior draws for the technique comparison.
    speeds : tuple[float, ...]
        Candidate probe speeds, rad/s, fastest first.

    Returns
    -------
    DemoResult
        The vetting, the identification, the two twin handles, and the verdict.

    Units
    -----
    kg, m, rad/s, s.
    """
    twin = lekiwi_twin(serial)
    payload = RigidPayload("gripper_frame_link", mass=0.1, name="tote")

    # 1. What the camera can say.
    prior = SemanticPrior.from_material(
        "small plastic tote, ~10x8x6 cm",
        "plastic",
        (0.10, 0.08, 0.06),
        source="vlm/demo",
        confidence=0.6,
    )

    # The simulated arm actually holding something in that bracket.
    link = PayloadLoopbackLink(
        twin.robot, prior.seed(payload), true_mass=planted_mass_kg
    )

    # 5. Two ways to do the job, compared on the loaded unit.
    techniques = [
        hold_plan(
            "brisk lift",
            twin,
            [0.3] * twin.n_servos,
            n_steps=n_steps,
            dt=dt,
            q0=TUCKED_POSE_RAD,
        ),
        hold_plan(
            "gentle lift",
            twin,
            [0.05] * twin.n_servos,
            n_steps=n_steps,
            dt=dt,
            q0=TUCKED_POSE_RAD,
        ),
    ]
    envelope = SafetyEnvelope(
        temp_ceiling_k=343.15, bus_floor_v=11.0, torque_margin_frac=0.9
    )

    enrollment = enroll_object(
        twin,
        link,
        payload,
        prior,
        techniques,
        envelope,
        stability=StabilityConstraint(base=LEKIWI_BASE),
        speeds=speeds,
        pose_rad=TUCKED_POSE_RAD,
        n_draws=n_draws,
    )

    # 6. The gate has teeth: the same arm, same pose, a heavier guess. Speed is
    #    a weak lever when the margin is dominated by where the arm is holding
    #    the load, so the refusal points at the pose rather than the probe.
    heavy_prior = SemanticPrior.from_material(
        "large plastic tote, ~14x10x8 cm",
        "plastic",
        (0.14, 0.10, 0.08),
        source="vlm/demo",
        confidence=0.6,
    )
    heavy_vetting = vet_probe(
        twin.robot,
        payload,
        heavy_prior,
        StabilityConstraint(base=LEKIWI_BASE),
        speeds=speeds,
        pose_rad=TUCKED_POSE_RAD,
    )

    mass = enrollment.identification.params["tote.mass"]
    interval = (
        enrollment.identification.intervals["tote.mass"]
        if enrollment.identification.intervals is not None
        else (float("nan"), float("nan"))
    )
    loser = enrollment.comparison.ranking[-1]
    loser_answer = enrollment.comparison.answer(loser)
    assert loser_answer is not None
    digest = hashlib.blake2b(
        "|".join(
            [
                twin.attestation_handle,
                enrollment.twin_handle,
                enrollment.technique,
                enrollment.verdict.outcome,
                f"{mass:.6f}",
            ]
        ).encode(),
        digest_size=16,
    ).hexdigest()
    return DemoResult(
        prior_bracket_kg=(float(prior.mass_kg[0]), float(prior.mass_kg[1])),
        worst_case_mass_kg=enrollment.vetting.worst_case_mass_kg,
        probe_speed_rad_s=enrollment.vetting.speed_rad_s,
        probe_margin_m=enrollment.vetting.margin_m,
        probe_attempts=enrollment.vetting.attempts,
        identified_mass_kg=mass,
        mass_interval_kg=(float(interval[0]), float(interval[1])),
        recipe=enrollment.identification.recipe,
        empty_twin_handle=twin.attestation_handle,
        loaded_twin_handle=enrollment.twin_handle,
        technique=enrollment.technique,
        technique_outcome=enrollment.verdict.outcome,
        rejected_technique=loser,
        rejected_outcome=loser_answer.outcome,
        binding_constraint=loser_answer.binding_constraint,
        heavy_label=heavy_prior.label,
        heavy_worst_case_mass_kg=heavy_vetting.worst_case_mass_kg,
        heavy_approved=heavy_vetting.approved,
        heavy_best_margin_m=heavy_vetting.margin_m,
        run_hash=digest,
    )


def _print(result: DemoResult) -> None:
    lo, hi = result.prior_bracket_kg
    print("Scene enrollment — LeKiwi picks up an unknown tote (simulated heft)")
    print("=" * 72)
    print(f"1. vision prior            mass in [{lo:.3f}, {hi:.3f}] kg")
    print(f"2. probe vetted at         {result.probe_speed_rad_s:.2f} rad/s")
    print(
        f"   worst case assumed      {result.worst_case_mass_kg:.3f} kg "
        f"-> margin {result.probe_margin_m * 1000:.1f} mm "
        f"(floor {LEKIWI_BASE.margin_min_m * 1000:.0f} mm)"
    )
    for speed, margin in result.probe_attempts:
        print(f"     tried {speed:.2f} rad/s -> {margin * 1000:6.1f} mm")
    ilo, ihi = result.mass_interval_kg
    print(
        f"3. heft says               {result.identified_mass_kg:.3f} kg "
        f"[{ilo:.3f}, {ihi:.3f}]  ({result.recipe})"
    )
    print(f"4. empty unit              {result.empty_twin_handle[:16]}…")
    print(f"   loaded unit             {result.loaded_twin_handle[:16]}… (new version)")
    print(
        f"5. vetted technique        {result.technique!r} ({result.technique_outcome})"
    )
    print(
        f"   rejected                {result.rejected_technique!r} "
        f"({result.rejected_outcome}, binding {result.binding_constraint})"
    )
    verdict = "approved" if result.heavy_approved else "REFUSED"
    print(
        f"6. same pose, heavier guess ({result.heavy_label}, worst case "
        f"{result.heavy_worst_case_mass_kg:.2f} kg): {verdict} — best margin "
        f"{result.heavy_best_margin_m * 1000:.1f} mm"
    )
    if not result.heavy_approved:
        print(
            "   slowing the probe does not rescue it: the margin is dominated "
            "by where the load is held, so re-pose (or measure the base)."
        )
    print(f"run hash                   {result.run_hash}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--serial", default="LK-0001", help="unit serial number")
    parser.add_argument(
        "--mass",
        type=float,
        default=PLANTED_MASS_KG,
        help="planted object mass for the simulated heft, kg",
    )
    args = parser.parse_args()
    _print(run_demo(serial=args.serial, planted_mass_kg=args.mass))


if __name__ == "__main__":
    main()
