# Independent-integrator sysid oracle traces

Frozen reference telemetry for the `sysid.actuator` LM recovery test
(`test/sysid/test_actuator_recipe_independent.py`).

## Why these exist

The sibling round-trip test (`test/sysid/test_actuator_recipe.py`) generates its
truth telemetry with the **same fixed-step RK4 the fitter uses**. That proves
self-consistency but not correctness: a shared discretization or sign error in
that RK4 would be baked into both the data and the fit, and the test would pass
while recovering the bug instead of the physics.

These traces break that circularity at two levels.

**Integrator independence** (`generate.py`) — produced by a **different
integrator**, SciPy's implicit adaptive `Radau` (L-stable, 5th order) at tight
tolerance, applied to a standalone transcription of each device's documented
ODE. The data and the fitter no longer share an integrator, so a
discretization/sign bug in the fitter's rollout cannot hide.

**Model-form independence** (`generate_jaxonomy.py`) — ground truth built from
**jaxonomy's physics**, not any transcription of the device ODE. For SEA and
BLDC this is genuine independent physics from acausal primitives (a two-inertia
drivetrain; an `IdealMotor` with its own R/L/back-EMF/torque law, current-loop P
folded into an effective resistance so no feedback crosses the acausal
boundary). The Servo is one level weaker — **implementation** independence: its
nonlinear saturation feedback can't be a passive acausal network, so it is a
jaxonomy causal block diagram on the BDF solver, an independent engine but a
transcription of the same equations.

## Files

| File | Device | Oracle | Excitation |
|---|---|---|---|
| `servo_step.npz` | `Servo` | Radau (integrator) | position steps, 200 Hz |
| `bldc_chirp.npz` | `BLDC` | Radau (integrator) | current chirp 0.5→50 Hz, 2 kHz |
| `sea_chirp.npz` | `SeriesElasticActuator` | Radau (integrator) | torque chirp 0.5→30 Hz, 2 kHz |
| `sea_jaxonomy.npz` | `SeriesElasticActuator` | jaxonomy acausal (physics) | torque chirp 0.5→30 Hz, 2 kHz |
| `bldc_jaxonomy.npz` | `BLDC` | jaxonomy acausal (physics) | current chirp 0.5→50 Hz, 2 kHz |
| `servo_jaxonomy.npz` | `Servo` | jaxonomy causal (implementation) | position steps, 200 Hz |

Each `.npz` holds the trace arrays plus a `meta_json` entry recording the oracle
settings, the truth parameters (what the fit must recover), and the warm-start
parameters (what the recipe is handed).

## Regenerating

```
python test/sysid/data/independent/generate.py           # Radau traces
python test/sysid/data/independent/generate_jaxonomy.py  # jaxonomy oracles
```

SciPy / jaxonomy are required to regenerate but **not** to run the test — the
test only loads the frozen `.npz`. Regenerate and review the diff whenever a
device model form changes.

## What this does NOT cover

All oracles are still within the jax\*y stack (jaxonomy is jaxterity's upstream
engine), so a physics error shared across the stack — or an effect no in-repo
model captures — would survive. The Servo oracle is only
implementation-independent, not physics-independent. Fully closing this needs
ground truth external to the stack: a published third-party reference log, or
real actuator-bench data. That gap is tracked in `KNOWN_GAPS.md`.
