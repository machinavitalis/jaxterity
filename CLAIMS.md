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
