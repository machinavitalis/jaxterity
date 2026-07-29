# Claims

**Evidence ledger.** Maps every claim made in a shippable surface
(`README.md`, `examples/**`, root-level `*.ipynb`) to the specific
evidence that substantiates it. Its inverse — what we do **not** yet
claim — is [`KNOWN_GAPS.md`](KNOWN_GAPS.md).

> **Doc-drift rule.** `CLAIMS.md`, `KNOWN_GAPS.md`, and the `README.md`
> maturity pointer form a three-document contract. A capability that lands
> moves text from `KNOWN_GAPS.md` to here and updates the README; the
> mechanical half (files exist, cross-references intact) is enforced by
> `test/test_top_level_docs.py`.

## The rule

Every claim made on a shippable surface should map to a row here. The
discipline:

- **For each claim** in a shippable surface: add a row, point it at an
  evidence file (a test or a runnable example), run the evidence, confirm
  green. If a row has no green evidence, the claim comes down.
- A shippable-surface edit that touches a claim is the natural moment to
  register it.

---

## Format

| ID | Claim (as written) | Where it appears | Evidence file | Status |
|---|---|---|---|---|
| C-001 | [exact wording of the claim] | [path or URL] | [path to test/example/log] | green / yellow / red |

**Status meaning:**

- **green** — evidence exists, runs in CI, passes. Claim is publishable.
- **yellow** — evidence exists but is not yet in CI, or runs locally but
  is not yet committed. Acceptable for short windows only.
- **red** — no evidence yet, evidence is failing, or the claim drifted
  from what the evidence shows. **The claim must come down.**

---

## Active claims

### Per-unit twin + predictive self-check

Evidence runs under `scripts/check.sh` (the `not nightly/hardware/pi`
tier); `hardware`/`pi` evidence is noted where the dev-host proxy is what CI runs.

| ID | Claim (as written) | Where it appears | Evidence file | Status |
|---|---|---|---|---|
| C-101 | A per-unit attested `Twin` composes a calibrated robot + off-body physics + serial; two units with identical params but different serials get distinct handles | `examples/selfcheck_demo_so101.py`, `examples/twin_so101.ipynb` | `test/twin/test_twin.py` | green |
| C-102 | Off-body physics blocks (`ServoThermal`, `BatterySag`) forward-sim against their analytic responses and fit-recover their parameters | docstrings / examples | `test/physics/test_thermal.py`, `test/physics/test_battery.py` | green |
| C-103 | Active sysid recovers planted params over a bus abstraction in one CLI call (`--port loopback`), bit-identically when seeded | `jaxterity calibrate --active` | `test/calibrate/test_active_sysid.py` | green |
| C-104 | The self-check co-processor rolls the twin forward beside a controller, skipping-and-flagging overruns; holds 20 Hz with zero misses on the dev host | `examples/selfcheck_demo_so101.py` | `test/selfcheck/test_runtime.py` (`pi` test = host proxy) | green |
| C-105 | The `SafetyMonitor` vetoes a sustained hold the body-only model passes, naming the servo-thermal constraint + time-to-ceiling; brownout vetoes on bus voltage | `examples/selfcheck_so101.py` | `test/selfcheck/test_monitor.py`, `test/selfcheck/test_lekiwi_brownout.py` | green |
| C-106 | `preflight` passes an in-envelope policy and fails an out-of-envelope one (naming scenario + constraint), binding the verdict to the twin handle | `jaxterity preflight` | `test/selfcheck/test_preflight.py` | green |
| C-107 | A unit's per-servo thermal is fit from telemetry and applied to the twin with provenance | — (API) | `test/calibrate/test_thermal_fit.py` | green |
| C-108 | The end-to-end self-check composes (active-ID → twin → co-processor → veto → preflight) and is reproducible from hashes | `examples/selfcheck_demo_so101.py` | `test/examples/test_selfcheck_demo.py` | green |
| C-109 | A per-unit closed-loop active-sysid fit from **real LeKiwi** telemetry (commanded multisine + inverse-dynamics equation error) predicts joint torque measurably better than the frictionless URDF baseline on a held-out split (45% shoulder_pan / 16% elbow_flex) | — (API: `jaxterity.calibrate.identify_closed_loop`) | `test/calibrate/test_closed_loop_sysid.py` (real telemetry under `test/calibrate/data/lekiwi/`) | green |
| C-110 | The identified plant friction generalizes to a **separately captured, different** real trajectory (28.9% shoulder_pan / 19.0% elbow_flex torque-error reduction), transferring only the plant params and refitting the per-capture measurement calibration | — (API: `jaxterity.calibrate.cross_trajectory_validation`) | `test/calibrate/test_closed_loop_crossval.py` | green |
| C-111 | The committed `so100` zoo calibration is a **real** fit (`synthetic:false`) from LeKiwi telemetry — `zoo.recalibrate("so100", captures, recipe="closed_loop")` writes per-joint friction/damping; `zoo.calibrated("so100")` applies it unchanged (data swap, not API change) | `jaxterity/zoo/so100/calibrated.json` | `test/zoo/test_recalibrate_closed_loop.py` | green |
| C-112 | The LeKiwi battery Thévenin sag (`V_oc`/`R_int`) is fit from a **real** aggregate multi-servo load trace (0.7 V measured sag); the battery-bearing twin's `SafetyMonitor` vetoes a sustained-load brownout (binding `bus_voltage`) that a body-only twin is blind to | — (API: `jaxterity.calibrate.fit_battery_sag`) | `test/selfcheck/test_lekiwi_brownout_real.py` | green |
| C-113 | A reduced NumPy self-check co-processor (thermal + battery forecast) runs on the **real Raspberry Pi** at rate with zero deadline misses (compute 500 Hz; integrated read+forecast loop 50 Hz), faithful to the JAX physics models | `scripts/lekiwi_reduced_coprocessor.py` | `test/selfcheck/test_reduced_coprocessor.py` (committed on-Pi timing reports) | green |
| C-114 | Pre-flight on the real calibrated unit's twin passes a safe trajectory (binding the verdict to the unit's attestation handle), rejects an unsafe one, and a re-fit (new handle) invalidates a prior pass | — (API: `jaxterity.selfcheck.preflight`) | `test/selfcheck/test_preflight_unit.py` | green |
| C-115 | The full self-check chain composes on the **real LeKiwi** (real self-ID → attested twin → co-processor 0 misses → thermal AND brownout vetoes a body-only model proceeds on → preflight rejects unsafe) and is reproducible from a `run_hash` | `examples/selfcheck_demo_lekiwi.py`, `examples/selfcheck_demo_lekiwi.ipynb` | `test/examples/test_selfcheck_demo_lekiwi.py` | green |
| C-116 | `jax.grad` works **through MJX's joint-limit constraint solver** with `implicit_vjp=True` (IFT wrap of the KKT residual): forward pass byte-equivalent to the raw solve; gradients match finite differences and forward mode on limit-active poses, rollouts, and `jaxonomy.simulate`; one backward pass yields all-parameter gradients (full SO-101 damping vector) | `jaxterity/robot/implicit.py` docstring, `Robot.functional_dynamics` / `build_system` / `to_diagram` | `test/gradients/test_implicit_vjp.py` | green |
| C-117 | Dynamics-consistent WBC (`solve_whole_body`) on the **floating-base G1** in double support: solved contact forces carry the robot's weight (Σfz ≈ m·g, 5%), the unactuated base rows of the equations of motion are satisfied, normals are unilateral, a swing foot transmits no force, and fixed-base `inverse_dynamics` torques round-trip through `mjx.forward` | `SKILL.md` §Whole-body control, `KNOWN_GAPS.md` §Whole-body control | `test/wbc/test_dynamics.py` | green |
| C-118 | The `lekiwi_twin` battery builder defaults (`V_oc` = 12.166 V, `R_int` = 0.0653 Ω duty-frame) are **fitted from the real committed aggregate-load capture** (r² = 0.66, 0.70 V measured sag), reproducible via `scripts/fit_twin_defaults.py`, and CI re-runs the fit so the constants cannot drift from the data | `jaxterity/twin/__init__.py` defaults, `KNOWN_GAPS.md` §Per-unit twin | `test/twin/test_default_provenance.py` | green |
| C-119 | Active sysid **fails loudly instead of returning plausible nonsense**: `excitation_identifiability` flags a degenerate probe before any fit; under noise that used to yield a silent negative damping, `active_identify` now returns non-negative fits with named warnings (bound-hit / weak-identifiability / unresolved-uncertainty), while a clean well-posed fit still recovers exactly with zero warnings | §7 of `examples/tutorial_unitree_g1.ipynb`, `examples/tutorial_unitree_g1_sysid.json` | `test/calibrate/test_active_sysid.py` | green |
| C-120 | **Grasped-payload identification in simulation**: `identify_payload` recovers a *planted* payload's mass + 3-axis CoM on so100 (and mass on cartpole) from joint telemetry via a welded MJX body and the active lane's projected LM, bit-identically when seeded; the T-219 guard set carries over — `payload_identifiability` flags a quasi-static probe as unable to excite the inertia *before* any fit, a planted negative mass pins at 0 with a named bound-hit warning, and a clean fit recovers with zero warnings; `Twin.with_payload` re-attests per held object (sim-tier only — no real-hardware payload capture yet) | `SKILL.md` §Object grounding, `KNOWN_GAPS.md` §Object-side identification | `test/objects/test_payload_id.py` | green |
| C-121 | **Differentiable probe design**: `design_probe` ascends the probe's predicted Fisher information about the payload parameters w.r.t. the probe's own initial state (forward-over-forward autodiff through the MJX simulation), deterministically and inside the joint-limit envelope (`validate_excitation` is the hard gate); on cartpole the optimized probe strictly beats the naive coast probe on both the information objective and the **final fitted-parameter covariance** from the same fit, and a previous fit's covariance folds in as a Bayesian prior (probe → fit → next-probe) | `SKILL.md` §Object grounding, `KNOWN_GAPS.md` §Object-side identification | `test/objects/test_design_probe.py` | green |
| C-122 | **Contact/impact identification through the saltation lane**: `identify_restitution` recovers a planted restitution (+ drop height) exactly from clean impact times, with d(impact time)/d(e, h₀) from `jaxonomy.multi_event_time_gradient` matching central FD within 5% on every bounce (the exactly-zero first-bounce e-sensitivity included); `identify_squeeze` recovers planted contact stiffness+damping from a force–deformation trace; the guard set holds — one impact cannot pin two parameters (refused with a fix), a quasi-static squeeze flags damping as unidentifiable, unphysical fits pin at 0 ≤ e ≤ 1 / k, c ≥ 0 by name (smooth/event path only — no MJX contact-solver gradients) | `SKILL.md` §Object grounding, `KNOWN_GAPS.md` §Object-side identification | `test/objects/test_contact_id.py` | green |
| C-123 | **The "same pixels, different physics" demo executes end to end in simulation**: one blind open-loop sling throw, tuned on the vision prior, misses two visually identical planted balls (0.30 kg solid / 0.04 kg hollow) by centimetres in opposite directions; a ~0.2 s heft recovers each ball's mass to 4 decimal places and CoM to sub-mm with honest per-component flags (the hollow ball's µm-scale CoM x/y are named unresolved rather than trusted); the throw re-tuned on the *identified* model lands both to ~0 cm; the toss/squeeze cameo recovers restitution and stiffness; and each held ball yields a distinct twin attestation handle — all committed with outputs (executed notebook, sim-tier, planted-truth loopbacks) | `examples/same_pixels_different_physics.ipynb` (executed, outputs committed), `examples/TUTORIALS.md` §Object grounding | `examples/same_pixels_different_physics.ipynb` | green |
| C-124 | **The demo's rigid-weld grasp is validated against stock-MuJoCo contact physics**: six friction pads (a four-pad seat + two-pad cap — the ball caught from opposite ends), a bounded gripper position servo closed quasi-statically, and a free-body ball; through the identification probe the ball stays gripped with millimetre-scale in-hand settle and the contact-held coast deviates from the welded model by ≤ ~0.7° on the main arm joints (~3° on wrist_roll); the probe clip renders from the contact simulation (jaw angle and ball pose from physics). Requires a representative gripper armature (0.002 kg·m²) — the zoo's zero-armature jaw makes MuJoCo's inertia-scaled constraint impedances unphysically soft, itself a documented fidelity finding | `examples/same_pixels_different_physics.ipynb` §The grasp for real, `KNOWN_GAPS.md` §Object-side identification | `examples/same_pixels_different_physics.ipynb` | green |
| C-125 | **One-command per-unit health checkup** (`jaxterity checkup` / `jaxterity.checkup`): composes a unit's commanded-excitation captures into a gated report card — per-joint closed-loop friction fits with 95% CIs, the battery Thévenin fit with a brownout-margin gate, response-ratio actuation checks, a versioned triage score, and the calibrated unit's attestation handle; runs persist in a serial-keyed store with per-parameter drift series and render to a self-contained HTML report; on the committed real LeKiwi captures the composed numbers agree with the standalone fits (shoulder_pan b ≈ 0.415 N·m·s/rad, held-out ≥ 45%; V_oc ≈ 12.166 V, R_int ≈ 65.3 mΩ), the report JSON round-trips, and the pipeline is deterministic given a fixed timestamp | `KNOWN_GAPS.md` §Per-unit health checkup | `test/checkup/test_checkup.py` | green |
| C-126 | **Checkup motions are tip-over-gated before they run** (`jaxterity checkup --plan-out`, `jaxterity.checkup.stability`): every planned capture motion is simulated against a declared base profile and must clear a quasi-dynamic ZMP margin floor — the *recorded poses of the real 2026-07-28 LeKiwi tip-over session* are the regression fixture, and the gate refuses exactly the captures during which the base physically rocked while accepting a full plan built from the bring-up choreography's tucked home pose (amplitude-derated where the band or margin requires); a statically unstable start pose is rejected outright; the on-device scripts consume the vetted plan and bound all transit speeds. Base profile is representative (declared, operator-overridable), and margins assume perfect tracking + rigid wheels — the floor carries that budget | `KNOWN_GAPS.md` §Per-unit health checkup | `test/checkup/test_stability.py` | green |

| C-127 | **Cross-embodiment block coverage is sim-tier and says so**: the transmission family (true positional backlash with a dual-encoder fit + model-free `lost_motion`, two-quadrant `Gearbox` where back-driving costs `1/η_b`, `BeltDrive`, `LeadScrew` with geometric backdrivability), the actuation families (`QuasiDirectDrive`'s N²-reflected inertia and speed-collapsing envelope, `Stepper`'s pull-out cliff, `PneumaticCylinder`'s ISO 6358 chambers), the measurement channels (`CurrentSense`, `TemperatureSensor` + `quantization_floor`, `BusChannel` + `phase_lag_deg`), the two acausal networks (`power_tree` — per-branch sag, regulator feedback, charger, checked against a closed-form DC solve; `thermal_network` — two-node winding/housing, driver, inter-servo conduction, checked against a closed-form conductance solve), the structural/environmental models (suspension compliance feeding `zmp_margins`, torsional compliance, coast-down rolling resistance), the aging priors, and the margin-constrained `design_probe` each recover their planted parameters from synthetic traces and *refuse* the captures that cannot identify them (a single-direction backlash reversal, a one-quadrant gearbox sweep, a constant-load power trace, a thermal soak, a downward-trending aging history, a starting probe already outside the tip-over floor). No hardware capture exists for any of these blocks | `SKILL.md` §Picking a block, `KNOWN_GAPS.md` §Calibrated actuator / sensor libraries | `test/actuators/test_transmission.py`, `test/actuators/test_families.py`, `test/sensors/test_channels.py`, `test/physics/test_power_tree.py`, `test/physics/test_thermal_network.py`, `test/physics/test_compliance.py`, `test/test_aging_priors.py`, `test/objects/test_probe_stability.py` (nightly tier — a constrained ascent is minutes, not seconds) | green |

The self-check is claimed **in simulation only** (C-108). An on-silicon run with
a recorded video is intentionally absent from this table and tracked as an open
gap in [`KNOWN_GAPS.md`](KNOWN_GAPS.md), not claimed here.

### Backfill

These pre-existing capabilities are claimed on shippable surfaces and have
evidence; they are being registered as rows here as those surfaces are edited:

- `Robot` import (URDF/MJCF) → `test/robot/`
- MJX-backed differentiable simulation → `test/gradients/`,
  `test/conservation/`, `test/determinism/`
- Forward-mode system identification end-to-end → `examples/sysid_so101.py`
  and `test/sysid/`
- Reference robot zoo → `test/zoo/`
- Attestation handle on a calibrated `Robot` → `test/robot/`

Each row, when registered, gets: exact claim wording, the shippable surface it
appears in, the evidence file, and a green/yellow/red status.

---

## Why this file exists

1. **Self-discipline.** The failure mode for an agent-developed library is
   that claims drift from reality without anyone noticing. This file is the
   forcing function: if a row cannot be filled, the claim cannot be made.
2. **External readiness.** When a downstream consumer or evaluator asks
   "how do you know that is true?", this file is the answer.

This file is public. Its inverse — what we do not yet claim — is
[`KNOWN_GAPS.md`](KNOWN_GAPS.md).
