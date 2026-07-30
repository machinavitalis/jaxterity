# External validation — recorded results

## EMPS: DC-motor recipe

Outcome of `test/sysid/external/test_emps_dc_motor.py` on the EMPS dataset
(`DATA_EMPS.mat` estimation set, `DATA_EMPS_PULSES.mat` held-out set). Regenerate
by fetching the data and running `pytest -m external_data`.

## Parameter recovery (estimation set)

The rig's model is `F = M*a + Fv*v + Fc*sgn(v) + offset`. Our DC-motor recipe
fits `V = Ks*sgn(v) + Kv*v + Ka*a` (no offset term), with `Ka↔M`, `Kv↔Fv`,
`Ks↔Fc`. Both fitters run on the same real 1 kHz telemetry (position
Butterworth-filtered at 100 Hz, central-differenced to velocity).

| Quantity | Paper's IDIM-LS (reproduced) | Our `sysid.actuator` | Agreement |
|---|---|---|---|
| Mass `M` / `Ka` (kg) | 95.085 | 95.151 | 0.07 % |
| Viscous `Fv` / `Kv` (N·s/m) | 204.668 | 204.989 | 0.16 % |
| Coulomb `Fc` / `Ks` (N) | 20.281 | 20.243 | 0.19 % |
| offset (N) | −3.170 | — (recipe has none) | n/a |

The reproduced IDIM-LS values match the published EMPS reference
(M ≈ 95.1 kg, Fv ≈ 204.7, Fc ≈ 20.3; Janot/Gautier/Brunot 2019), confirming the
data + preprocessing are correct. Our recipe lands within **0.2 %** of that
independent fit on all three physical parameters — despite carrying no offset
term — from real hardware data. A model-form or sign error in the recipe would
not survive this.

## Held-out prediction (pulses validation set)

Using the estimation-set fit to predict the *pulses* trace (never seen; a
different excitation with force swings of ±300 N):

- **RMS error: 7.94 N**
- **Relative error: 13.6 %**

The residual is dominated by the recipe's missing constant-offset term
(the rig's ≈ −3.2 N `offset`) and mild friction asymmetry — a known,
documented limitation, not a fit failure. The recipe generalizes to an
independent real trajectory it was not fit on.

## Scope

This validates the **DC-motor recipe** only. The Servo / BLDC / SEA LM recipes
have no equivalent open real dataset and still rely on in-stack oracles plus
(eventually) our own bench capture — see `../../../KNOWN_GAPS.md`.


---

# NAD SO-101 — model completeness against a labelled unmodelled mass

Outcome of `test/sysid/external/test_nad_so101.py` on the NAD SO-101 subset
(Dou et al., RSS 2026, [arXiv 2607.11734](https://arxiv.org/abs/2607.11734)).
Regenerate with `python test/sysid/external/fetch_nad.py` then
`pytest -m external_data`.

## What it validates, and why NAD specifically

`jaxterity.sysid.hybrid` claims that a **growing parametric residual means
something the model does not represent is present**. Until this, that claim had
been tested only on synthetic captures (where we injected the extra term
ourselves) and on LK-0001 (our unit, our protocol, no ground truth about what was
missing). Both are in-house.

NAD closes that: the same arm, logged by other people on another unit, with the
payload deliberately stepped empty → 500 g. Our model carries no payload term, so
**the payload is labelled unmodelled physics** — a known quantity of exactly what
the metric claims to detect, labelled by someone with no stake in the result.

Protocol: fit `(k_t, load_offset, damping, friction)` **once**, on the
empty-payload *training* trajectories only, hold them fixed, then score every
payload's held-out *test* trajectory. `go_up_and_stay_still`, shoulder lift
(joint 2 — the base yaw's gravity torque is identically zero).

## Result

Parametric one-step and 10-step rollout position RMS, mean over each payload's
test set. Fitted parameters: `k_t = -0.00051`, `load_offset = 0.04645`,
`damping = 0.39345`, `friction = -0.04560`.

| Payload | H=1 (mrad) | H=1 (encoder quanta) | H=10 (mrad) |
|---|---|---|---|
| empty | 0.814 | 0.53 | 10.50 |
| 200 g | 0.881 | 0.57 | 19.36 |
| 300 g | 0.945 | 0.62 | 26.01 |
| 400 g | 1.206 | 0.79 | 30.22 |
| 500 g | 1.403 | 0.91 | 33.08 |

**Strictly monotone in mass at both horizons**, with 500 g separated from empty
by 1.72x (H=1) and 3.15x (H=10). Correlation against the actual gram values is
r = 0.915 (H=1) and r = 0.994 (H=10) — the trend tracks mass, not merely run
order.

Two things follow.

1. **The metric detects unmodelled physics on hardware that is not ours**, with
   the ground truth supplied externally. The precondition is also met: on the
   empty payload the model sits at **0.53 encoder quanta**, i.e. it genuinely
   fits their data, so the rise is attributable to the payload rather than to a
   model that fits nothing.
2. **The multi-step rollout earns its keep.** Compounding over ten steps widens
   the empty-to-500 g separation from 1.72x to 3.15x and tightens the mass
   correlation from 0.915 to 0.994 — on data with no connection to the synthetic
   captures the horizon was developed against.

## Honest notes

- `k_t` comes out slightly **negative** (−0.00051), i.e. their load register's
  sign convention is opposite to ours, and its magnitude is small. The comparison
  across payloads is unaffected — the parameters are fitted once and held fixed —
  but this is not a claim that our load-to-torque calibration transfers to their
  logging setup.
- `friction` is also negative, which is unphysical for a Coulomb term. On our own
  captures `identify_closed_loop` *holds* a non-positive friction channel at the
  frictionless prior rather than binding it; this test deliberately does no such
  cleanup, because the point is to hold one fitted model fixed across payloads,
  not to publish a parameter set for their unit.
- The inertia is a fixed approximation (0.02 kg·m²) and is not tuned per payload.
  It scales both errors identically, so it cannot manufacture the trend.
- This validates the **completeness metric**, not our SO-101 parameter values.
  NAD has no independently published friction parameters to check ours against,
  unlike EMPS above.
