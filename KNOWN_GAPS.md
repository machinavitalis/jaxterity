# Known Gaps

This file documents what Jaxterity does **not** yet do, or does only
partially. It is intentionally public. We would rather tell you what is
missing than have you discover it mid-deployment.

This document is the inverse of [`CLAIMS.md`](CLAIMS.md): that file maps
what we claim works to its evidence; this file lists what we do not yet
claim. The two are kept **symmetric** — when a capability lands, text
moves from here into `CLAIMS.md`; it should never silently appear or
vanish from only one side.

> **Doc-drift rule.** `CLAIMS.md`, `KNOWN_GAPS.md`, and the `README.md`
> maturity pointer form a three-document contract. Closing a gap here that
> is user-visible should add a row to `CLAIMS.md` and update the README;
> the mechanical half of this rule (files exist, cross-references intact,
> gap-anchor phrases preserved) is enforced by
> `test/test_top_level_docs.py`.

If you hit something not listed here, please open an issue —
undocumented gaps are bugs in this file.

---

## Format

Each entry has the same shape:

- **Area**: the part of the system affected
- **Status**: `not yet implemented` / `partial` / `experimental` /
  `known limitation`
- **What works**: the part that does work today, if any
- **What doesn't**: the specific limitation
- **Workaround**: what to do in the meantime, if anything

---

## Currently known gaps

### Calibrated actuator / sensor libraries

- **Area**: the device libraries under `jaxterity/actuators/` and
  `jaxterity/sensors/`
- **Status**: partial
- **What works**: the device abstraction, the current device set, and
  their `validate()` surface exist and import.
- **What doesn't**: the library is **not yet complete** — coverage of
  real actuator/sensor families is still being filled in, and some
  `validate()` checks are shallower than their final form.
- **Workaround**: model the device explicitly as a Jaxonomy `LeafSystem`
  if the family you need is not yet covered.

### System-identification recipes

- **Area**: `jaxterity/sysid/` recipe layer
- **Status**: partial
- **What works**: forward-mode system identification is real and
  exercised end-to-end — see `examples/sysid_so101.py`, which differentiates
  *through* MJX's joint-limit constraint solver to recover joint damping.
  The per-device LM recipes (`sysid.actuator` for Servo / BLDC /
  SeriesElastic) recover their identifiable parameters both on self-generated
  round-trips and on **independent-integrator** traces — telemetry produced by
  a different integrator (SciPy `Radau`) than the fitter's own RK4, frozen
  under `test/sysid/data/independent/` and checked in
  `test/sysid/test_actuator_recipe_independent.py`. That rules out a
  discretization/sign bug shared between the fitter and its own data generator.
  There are additionally **cross-model** oracles built from jaxonomy (the
  upstream engine), not from a transcription of the device ODE at all: the SEA
  from acausal Inertia/Spring/Damper primitives, and the BLDC from an acausal
  `IdealMotor` (its own R/L/back-EMF/torque equations, with the current-loop P
  folded into an effective resistance so no feedback crosses the acausal
  boundary). A sign or coupling error in jaxterity's own SEA/BLDC equations is
  caught even if copied identically into the integrator transcription. The
  Servo additionally has an independent-*implementation* oracle (a jaxonomy
  causal block diagram on the BDF solver) — a distinct engine, though not
  independent physics, since its nonlinear feedback can't be a passive network.
  Beyond the stack entirely, the **DC-motor recipe** is validated against real
  hardware: on the EMPS positioning-rig benchmark it recovers the rig's
  mass+friction to within 0.2% of the paper's own identification and predicts a
  held-out trace (opt-in `external_data` test under `test/sysid/external/`;
  data fetched on demand, not vendored — see its README).
- **What doesn't**:
  - The higher-level **sysid recipe catalogue** (turnkey per-regime recipes)
    is **not yet complete**; some recipe paths (e.g. the contact recipe) are
    still being hardened.
  - **External-to-stack validation covers only the DC-motor recipe.** The
    in-stack cross-model oracles (jaxonomy) could miss a physics error shared
    across the stack; the EMPS check closes that for the **DC-motor** recipe
    against real hardware, but the **BLDC / SEA / Servo** LM recipes have no
    equivalent open real dataset (no clean public trace with commanded current +
    high-rate current, input torque + both-side positions, or a real position
    setpoint loop). External validation for those still needs our own bench
    capture. (The EMPS test is also opt-in, not a standing CI gate.)
  - **Transmission backlash is not fit.** `sysid.actuator` for
    `HarmonicDrive` holds `backlash_v` at its prior; the module docstring's
    "1-parameter LM fit on the velocity-rolloff residual" is not implemented
    (`jaxterity/sysid/_actuator.py`). Identifying it needs a dual-encoder
    (motor-side + joint-side) capture with torque-reversal excitation, which
    the current `TransmissionTelemetry` captures do not carry.
  - **BLDC electrical split (R / L / Ke) is weakly identified.** Under the
    free-shaft current-chirp protocol the stiff current loop pins only the
    `Kt/J` ratio (plus `b`, `Fc`); the recipe returns R/L/Ke but flags them as
    weakly constrained, and they have only ever been exercised on synthetic
    telemetry. Separating them needs a locked-rotor current-step protocol with
    phase-current sensing (≥1 kHz) — not present on the LeKiwi rig, whose
    captures expose a PWM-duty proxy, not current.
- **Workaround**: drive the forward-mode sysid primitives directly, as the
  SO-101 example does; trust `Kt/J`, `b`, `Fc` from the BLDC recipe and treat
  the electrical split as a prior until locked-rotor data exists.

### Whole-body control

- **Area**: `jaxterity/wbc/`
- **Status**: partial
- **What works**: the task DSL (`PostureTask` / `EndEffectorTask` / `CoMTask` /
  `MomentumTask` / `ContactTask`, soft-weighted hierarchy) with the JAX-traceable
  QP (`qp_solve`) and kinematic `WBCSolver`; the centroidal primitives
  (`contact_jacobian`, `centroidal_momentum_matrix`, …); the contact scheduler
  (`ContactSchedule` / `periodic_gait`) wired into the QP via
  `contact_tasks_from_schedule`; and the **dynamics-consistent** solve
  (`solve_whole_body`) — one QP over `[qdd, contact forces]` with the
  unactuated-row (floating-base) equations of motion as a hard equality,
  unilateral gated force bounds, and realizable-torque recovery
  (`inverse_dynamics` is the fixed-base special case). Exercised on the real G1
  in double support (`test/wbc/test_dynamics.py`: forces carry the robot's
  weight, base rows satisfied, swing foot transmits no force).
- **What doesn't**:
  - **No friction cone.** Tangential contact forces are bounded only by a
    magnitude cap; the pyramid/cone inequality needs general inequality rows
    that `qp_solve` does not yet expose.
  - **Soft hierarchy only.** Task priorities are weights, not a strict
    lexicographic cascade.
  - **No closed-loop walking.** The gait scheduler + WBC solve one tick;
    swing-foot trajectory generation, footstep planning, and a stabilized
    walking loop on the G1 are not yet built (the reduced-order LIP /
    capture-point models in `jaxterity.locomotion.reduced` are the planning
    ingredients).
- **Workaround**: for locomotion research use the vendored learned G1
  joystick policy (`jaxterity.locomotion`); for manipulator torque control the
  fixed-base path (`WBCSolver` + `inverse_dynamics`) is complete.

### Learned policies and the MJX path

- **Area**: `jaxterity/policies/`
- **Status**: known limitation
- **What works**: the layered code-as-policies API + the policy evaluation
  harness, and the S2 skills integrate **real MJX** via
  `S4.step_with_acceleration` (inverse dynamics → force → MJX RK4), not the
  `set_state` math facade — so the layered stack steps through the MJX
  dynamics. The eval baselines are non-vacuous.
- **What doesn't**: the deployment under-actuation case is not yet covered —
  the layered stack force-controls every DOF via `qfrc_applied` (the zoo robots
  carry no `<actuator>` limits); a robot with explicitly limited actuators would
  need a WBC dynamics-consistency constraint on the unactuated rows. Floating
  base in the layered stack is not yet supported.
- **Workaround**: for explicitly under-actuated hardware, add the
  consistency-constrained WBC before trusting the layered torques.

### Per-unit twin + predictive self-check

- **Area**: `jaxterity/twin/`, `jaxterity/selfcheck/`, and the active /
  thermal calibration in `jaxterity/calibrate/`
- **Status**: partial
- **What works**: the full software stack, exercised end-to-end in simulation
  (`examples/selfcheck_demo_so101.py`). A per-unit attested `Twin` composes a
  calibrated `Robot` with off-body physics (`ServoThermal`, and `BatterySag`
  for `lekiwi_twin`); active system identification runs over a bus abstraction
  with a seeded loopback fake (`jaxterity calibrate --active --port loopback`);
  the self-check `CoProcessor` rolls the twin forward beside a controller; the
  `SafetyMonitor` returns Ok/Derate/Veto (servo-thermal ceiling, bus-voltage
  floor, torque/joint margins) naming the binding constraint; `preflight`
  verifies a candidate policy against a twin; and a unit's per-servo thermal can
  be **fit from telemetry** (`fit_servo_thermal` / `calibrate_twin_thermal`).
  The full chain also runs on **real LeKiwi** artifacts
  (`examples/selfcheck_demo_lekiwi.py`): closed-loop active self-ID beats the
  frictionless baseline on held-out real data (16–45 % across all 6 arm joints),
  the `so100` zoo calibration is a real fit (`synthetic:false`), the battery sag
  is fit from a real aggregate-load trace, a reduced NumPy co-processor holds
  rate on the real Pi with zero deadline misses, and the twin vetoes a thermal
  overload **and** a brownout that a body-only model proceeds on; pre-flight
  rejects an unsafe trajectory; the run is reproducible from a `run_hash`.
- **What doesn't**:
  - **The MJX-body chain runs on the workstation, not on-device.** Twin
    rollout, vetoes, and pre-flight need JAX/MJX; the Pi runs NumPy only, so the
    on-device pieces are the capture + the reduced co-processor. The on-silicon
    end-to-end run with a recorded video is not yet claimed.
  - The brownout fires on an **aged-battery scenario** (real `R_int` × wear; a
    healthy bench pack won't brown out a realistic floor), and the overlay video
    is a rendered dashboard (+ Rerun `.rrd`), not a 3D Rerun-viewer capture.
  - **Gravity-loaded joints are weakly identified** by this excitation: the
    duty-cycle `load` proxy represents a large quasi-static gravity holding torque
    poorly, so the held-out improvement is modest (elbow ~16%) versus a
    gravity-neutral, dynamically-excited joint (shoulder_pan ~45%). Coulomb
    friction has a near-zero gradient and is held when its fit is non-positive.
  - The real serial bus is driven **read-only + commanded position-mode capture**
    on-device (validated); `SerialBusLink.run_excitation` (in-process drive) stays
    a stub by design — the Pi lacks the JAX/MJX stack, so the supported route is
    split capture → workstation fit.
  - There is **no LeKiwi mobile-base URDF** in the zoo: `lekiwi_twin` attaches
    the shared-battery model to the SO-100 *arm* (no wheels/base in the
    dynamics). The `lekiwi_twin` battery builder defaults (`V_oc`/`R_int`) are
    now **fitted** from the real committed aggregate-load trace
    (`scripts/fit_twin_defaults.py`, pinned by
    `test/twin/test_default_provenance.py`) — subject to the duty-vs-torque
    current-frame caveat in `calibrate/battery.py`; the per-servo
    torque→current gain `k_i` is *not* identified by that trace and stays
    representative.
  - **Thermal identifiability**: a temperature+torque trace fixes only
    `k_pwr/C_th` and `R_th·C_th`, so `C_th` is held known (from
    geometry/datasheet). `so101_twin`'s thermal defaults remain
    **representative by measurement**: running `fit_servo_thermal` on the real
    committed `temp_C` traces (`scripts/fit_twin_defaults.py`) shows they carry
    no usable thermal signal — the register is 1 °C-quantized and the 14–16 s
    excitations are far shorter than the servo thermal time constant. A
    per-unit thermal fit needs a dedicated several-minute heating capture.
  - The co-processor runs the twin in **Python/JAX** (no Jaxility lowering): it
    holds tens of Hz on a dev host; low-hundreds-Hz needs a shorter horizon or a
    lowered co-processor (future work). The monitor's joint-limit check is
    opt-in (meaningful only on a control-fine-horizon prediction).
  - Predict-and-veto only — there is **no recovery controller** (runtime
    assurance / simplex) yet; the safety envelope is operator-declared.
- **Workaround**: use the simulated self-check + loopback paths today; the hardware
  run and the lowered co-processor are tracked, not stubbed.

### Deployment to silicon (Jaxility)

- **Area**: the hand-off from a calibrated `Robot` to an embedded artifact
- **Status**: not yet implemented (in this repo)
- **What works**: the attestation handle that a calibrated `Robot` carries.
- **What doesn't**: Jaxterity does not itself produce embedded artifacts.
  Deployment is owned by **Jaxility** (the compiler / deployment factory,
  forthcoming), which consumes a calibrated Jaxterity model.
- **Workaround**: none — track the Jaxility repository for the deployment
  lane.
- **Tracking**: cross-repo; see the Jaxility project.

---

## Out of scope (by design, not a gap)

- **General-purpose simulation primitives.** `Diagram`, `Context`,
  `LeafSystem`, `simulate`, `fit_parameters`, `linearize`, the acausal
  layer, and the framework module are **imported from Jaxonomy**, not
  re-implemented here (see `AGENTS/DECISIONS.md` ADR-002). If you need a
  non-robotics simulation feature, it belongs upstream in Jaxonomy.
