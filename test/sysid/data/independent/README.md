# Independent-integrator sysid oracle traces

Frozen reference telemetry for the `sysid.actuator` LM recovery test
(`test/sysid/test_actuator_recipe_independent.py`).

## Why these exist

The sibling round-trip test (`test/sysid/test_actuator_recipe.py`) generates its
truth telemetry with the **same fixed-step RK4 the fitter uses**. That proves
self-consistency but not correctness: a shared discretization or sign error in
that RK4 would be baked into both the data and the fit, and the test would pass
while recovering the bug instead of the physics.

These traces break that circularity. They are produced by a **different
integrator** — SciPy's implicit, adaptive `Radau` (L-stable, 5th order) at tight
tolerance — applied to a standalone transcription of each device's documented
ODE. Because the data and the fitter no longer share an integrator, a
discretization/sign bug in the fitter's rollout can no longer hide.

## Files

| File | Device | Telemetry schema | Excitation |
|---|---|---|---|
| `servo_step.npz` | `Servo` | `ActuatorStepTelemetry` | position steps, 200 Hz |
| `bldc_chirp.npz` | `BLDC` | `ActuatorChirpTelemetry` | current chirp 0.5→50 Hz, 2 kHz |
| `sea_chirp.npz` | `SeriesElasticActuator` | `SeriesElasticTelemetry` | torque chirp 0.5→30 Hz, 2 kHz |

Each `.npz` holds the trace arrays plus a `meta_json` entry recording the
integrator settings, the truth parameters (what the fit must recover), and the
warm-start parameters (what the recipe is handed).

## Regenerating

```
python test/sysid/data/independent/generate.py
```

SciPy is required to regenerate but **not** to run the test — the test only loads
the frozen `.npz`. Regenerate and review the diff whenever a device model form
changes.

## What this does NOT cover

The transcription in `generate.py` is still *jaxterity's own* model equations, so
a physics model-form error copied into both the fitter and the transcription
would survive. Catching that needs telemetry whose ground truth is external to
jaxterity (a published third-party reference log, or real actuator-bench data).
That gap is tracked in `KNOWN_GAPS.md`.
