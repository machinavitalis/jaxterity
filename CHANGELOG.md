# Changelog

A snapshot of what Jaxterity implements today, grouped by area. For what is
*not* yet done see [`KNOWN_GAPS.md`](KNOWN_GAPS.md); for the evidence behind
each capability see [`CLAIMS.md`](CLAIMS.md).

## [Unreleased]

- **Parametric-first hybrid** (`jaxterity.sysid.hybrid`): the interpretable fit
  stays in front, an optional small learned residual absorbs what it cannot, and
  the parametric residual's size is the reported signal. `residual_gates` warns
  when the learned residual fits much better than the physics — evidence the
  parameters have stopped describing the joint. Trained through the model on
  measured position with a Huber loss.

- **Lost motion from a slow reversal sweep**
  (`scripts/lekiwi_backlash_capture.py`, `jaxterity.checkup.backlash`): a
  single-encoder servo cannot observe motor-minus-joint angle, so what is
  measured is the command-to-measurement hysteresis width — total lost motion,
  an upper bound on gear play with the position-loop deadband and compliance
  inside it. Nothing in `LostMotion` is named backlash; single-direction and
  non-quasi-static sweeps are refused.

- **Capture amplitude designed rather than derated**:
  `build_capture_plan(..., optimize=True)` bisects each entry to the largest
  amplitude whose simulated ZMP margin clears the floor, off by default so
  committed plans reproduce. `plan_information` scores per channel, since the
  Coulomb channel saturates where the viscous one goes as amplitude².
  **Breaking:** `SegmentScore.information` was a column-normalized Gram diagonal
  and identically 1.0; it is now the unnormalized `XᵀX` diagonal.

- **Capture-length-honest uncertainty** (`jaxterity.checkup.uncertainty`): a
  least-squares interval says how well a parameter is pinned given the data, not
  how much the answer depends on how much data was taken — and sub-window
  re-fits land outside it. `TotalUncertainty` reports `sqrt(wald² + bias²)` with
  both components visible, `length_ladder` estimates the systematic term, and
  `uncertainty_gates` splits confidence from composition (both warn).

- **Pose-conditioned friction** (`jaxterity.calibrate.pose_conditioned`): a
  scalar viscous/Coulomb pair is a category error on any joint whose load
  depends on configuration. `CanonicalPose` / `require_same_pose` report at a
  declared pose and refuse a cross-pose comparison; `fit_pose_surface` regresses
  the parameter on the dominating pose variable — axis inertia for yaw axes,
  gravity torque for pitch — and reports `∂param/∂pose` with an interval.

- **Drive-wheel health** (`scripts/lekiwi_wheel_capture.py`,
  `jaxterity.checkup.wheels`): a bidirectional speed staircase and coast-down
  read into `WheelHealth` with `wheel_gates` / `wheel_summary` — speed tracking
  fails, duty symmetry warns, coast-down is reported but ungated. Temperature is
  a median plus a garbled fraction rather than a maximum, and stopping is
  thresholded above one speed quantum with window censoring flagged.

- **Thermal capture mode** (`scripts/lekiwi_thermal_capture.py`,
  `calibrate.load_thermal_capture`): ramp to a vetted pose, hold under torque,
  then cut torque for a cooling tail — the decay is what separates the cooling
  constant from the heating gain, which a 14–16 s commanded capture cannot see.
  Plan-gated, guard-clamped, torque off on every exit path.

- **Capture scripts release what they energize.** Load-trace centres are nudged
  inward so the vetted motion is one the guard band can execute, the load script
  takes its centre from the plan rather than the live position, SIGTERM routes
  into the torque-off path, and captures no longer finish holding under torque.

- **Pre-motion probe scoring** (`jaxterity.checkup.probe`):
  `segment_identifiability` builds the velocity columns the friction fit
  regresses from the commanded waveform alone, so a planned segment's
  conditioning and per-channel variance are known before the arm moves.
  `fit_stability` reports how far a re-fit on part of a capture moves it.

- **Protocol pinning for drift** (`jaxterity.checkup.protocol`): each run records
  a `CaptureProtocol` — per joint the waveform, tones, amplitude, duration,
  command rate, excitation centre and held pose — and `drift_warnings` names
  every boundary where it changed. The recipe is hashed exactly; poses compare
  against `POSE_TOLERANCE_TICKS`. Report schema `jaxterity-checkup-v2`; v1 reads
  back unpinned.

- **True-peak multisine normalization, versioned** (`schroeder_multisine(...,
  kind=)`): the `√(2n)` crest estimate is overshot 1.52× by the five-tone set,
  so a commanded amplitude understated the real swing; the scripts now divide by
  the sum's measured peak. Captures record `excitation.kind`, the plan schema is
  `jaxterity-checkup-plan-v2`, and a plan vetted for the other normalization is
  refused. The response-ratio gate uses recorded `goal_ticks`.

- **Cross-embodiment block coverage.** The block library is a
  calibration-recipe library, so covering block types is how a new embodiment
  becomes calibratable. Devices stay causal, physical networks acausal:
  - `physics.power_tree` — ECM pack with SoC-dependent OCV and `CellAging`,
    per-branch harness resistance, `DCDCConverter`, CC/CV `Charger`;
    `fit_power_tree` identifies the resistance sum a rail trace can resolve.
  - `physics.thermal_network` — two-node winding/housing per servo with driver
    nodes and inter-servo conduction; all four parameters are identifiable only
    when both nodes are instrumented.
  - Transmissions: `PositionalBacklash` (dual-encoder fit plus a model-free
    `lost_motion` check), `Gearbox`, `BeltDrive`, `LeadScrew`.
  - Actuation: `QuasiDirectDrive`, `Stepper` (with `pull_out_margin`),
    `PneumaticCylinder`.
  - Measurement channels: `CurrentSense`, `TemperatureSensor`, `BusChannel` —
    the paths whose errors otherwise land in a physical parameter.
  - `physics.compliance` wired into `zmp_margins`, `jaxterity.aging` for
    friction and backlash growth, and `design_probe(..., stability=...)`.
  - New `[acausal]` extra (sympy) and seven telemetry schemas. Sim-tier: no
    hardware capture for these blocks yet.

- **Stability-gated capture plans** (`jaxterity.checkup.stability` / `.plan`,
  `checkup --plan-out`): every planned motion — pose transit, per-joint
  multisine, battery load trace — is simulated against a declared `BaseProfile`
  and must clear a quasi-dynamic ZMP margin floor. Excitations derate or are
  refused, unstable poses are rejected, and the on-device scripts consume the
  vetted plan (`--plan`).

- **Per-unit health checkup** (`jaxterity checkup`, `jaxterity.checkup`): one
  command from a directory of on-device captures to a stored, gated report card
  — per-joint friction fits with 95 % CIs, a battery Thévenin fit with a
  brownout-margin gate, actuation-health gates, a versioned triage score and the
  unit's attestation handle. Runs persist in a serial-keyed `CheckupStore` with
  drift series; `--html` renders a self-contained report.

- **Demo notebook: "Same pixels, different physics"**
  (`examples/same_pixels_different_physics.ipynb`, executed with outputs): two
  visually identical balls, where a throw tuned on the vision prior misses both
  and the same throw re-tuned on a 0.2 s heft lands both. Covers probe scoring,
  `design_probe`, restitution and squeeze, and per-object attestation, with the
  grasp validated against a stock-MuJoCo contact simulation. Sim-tier.

- **Contact and impact identification** (`jaxterity.objects.contact`):
  `identify_restitution` fits restitution, and drop height if unknown, to
  observed impact times — differentiated through the impacts via the saltation
  lane. `identify_squeeze` fits Hunt–Crossley stiffness and damping to a
  force–deformation trace. Under-determined records are refused by name; MJX
  contact-solver gradients stay out of scope.

- **Differentiable probe design** (`jaxterity.objects.probe`): `design_probe`
  ascends the probe's predicted Fisher information with respect to its own
  initial joint state, through the MJX simulation, projected into joint limits
  and gated by `validate_excitation`. `prior_covariance` makes it Bayesian
  D-optimal; `probe_information` ranks candidate probes.

- **Grasped-payload identification** (`jaxterity.objects`): recover a held
  object's mass, CoM and optional principal inertia from the robot's own joint
  telemetry — the calibrated arm is the force scale that makes absolute
  parameters identifiable. `attach_payload` welds a `RigidPayload` into the
  lowered model as an MJX body with no new DOFs, `payload_identifiability`
  scores a probe before any fit, and `Twin.with_payload` re-attests per object.

## [1.1.0] - 2026-07-22

- **Wheels now ship the package data.** The 1.0.0/1.0.1 wheels contained only
  `.py` files — no zoo URDF/MJCF models, no `calibrated.json`, no policy or
  golden `.npz`, no vendored licenses, no web viewer templates — so every zoo
  robot failed to load from a PyPI install (`jaxterity.zoo.load(...)` raised
  `FileNotFoundError`; downstream, every `jaxility build <robot>` broke).
  Added the `[tool.setuptools.package-data]` manifest covering
  `zoo/*/{*.urdf,*.xml,*.json,*.npz,LICENSE*,README.md}` and `web/template/*`.
- **Excitation identifiability check + physical-bounds guards on active sysid
  (`active_identify`).** A planted-parameter G1 fit at `noise_std=1e-3` used to
  return a **negative damping** — physically impossible, a damper cannot inject
  energy — with an *empty* warnings list; the Laplace covariance knew the
  estimate was garbage but nothing surfaced it. Three changes close that hole:
  (a) a **design-time identifiability report** — new
  `jaxterity.calibrate.excitation_identifiability(robot, targets, excitation)`
  simulates the designed probe once at the prior (no hardware, no fit),
  differentiates the observation forward-mode, and scores the sensitivity
  Jacobian (SVD rank / condition number / weak parameters; `J^T J` is the
  excitation's Fisher information under iid Gaussian noise) so a probe that
  cannot identify a parameter is caught *before* minutes of LM or moving a real
  arm; (b) **physical-bounds guards** — the LM iterates are projected onto
  `>= 0` for physically non-negative fields (damping / armature / frictionloss /
  mass / inertia; `bounds=False` opts out), and a fit pinned at the bound is
  flagged by name; (c) **uncertainty warnings** — parameters whose standard
  error exceeds `warn_rel_std` (default 100%) of the fitted value are flagged
  as unresolved. `active_identify` scores identifiability from its own
  first-iteration Jacobian (no extra cost), reports the excitation condition
  number under `residuals`, and returns all three guard families in
  `SysidResult.warnings` — under-informed fits now fail loudly instead of
  returning plausible nonsense. Regression-tested on the noisy planted-fit
  scenario (`test/calibrate/test_active_sysid.py`); the G1 tutorial artifact
  (`examples/tutorial_unitree_g1_sysid.json`, §7 of the tutorial) is
  regenerated with the guards active.
- **Measured LeKiwi battery defaults on the twin builders.** The `lekiwi_twin`
  builder's `V_oc`/`R_int` defaults were representative placeholders; they are
  now **fitted from the real committed aggregate-load capture**
  (`V_oc = 12.166 V`, `R_int = 0.0653 Ω` in the duty-current frame; r² = 0.66
  on a real 0.70 V sag). `scripts/fit_twin_defaults.py` is the reproducible
  provenance, and `test/twin/test_default_provenance.py` re-runs the fit in CI
  so the defaults can never silently drift from the data. Honest scope: `k_i`
  is not identified by the duty-frame trace and stays representative, and the
  SO-101 **thermal** defaults stay representative *by measurement* — the fit
  was run on the real `temp_C` traces and they carry no usable thermal signal
  (1 °C quantization, 14–16 s excitations vs a minutes-scale time constant);
  a dedicated long-duration heating capture is the tracked path. The brownout
  unit test now exercises an aged-pack scenario (5× `R_int`), matching the
  real-trace test and demo — a healthy fitted pack barely sags, which is the
  physically honest outcome.
- **Dynamics-consistent whole-body control** (`jaxterity.wbc.solve_whole_body`).
  The kinematic `WBCSolver` answers "which qdd best satisfies the tasks"; the
  new solve answers the dynamic question — one QP over
  `[qdd, contact forces]` with the unactuated-row (floating-base) equations of
  motion `Su(M·qdd + h − ΣJcᵀλ) = 0` as a hard equality, unilateral
  contact-force bounds gated by each contact's schedule `active` gate (a swing
  foot transmits no force), and realizable actuator torques recovered by
  substitution. `inverse_dynamics` covers the fixed-base special case
  (qdd → tau for a manipulator), and `actuated_mask` derives the default
  actuation pattern (free-joint DOFs unactuated). Exercised on the real G1 in
  double support: the solved forces carry the robot's weight (Σfz ≈ m·g), the
  base rows of the equations of motion are satisfied, and a swing contact's
  force block is zero (`test/wbc/test_dynamics.py`). Named limitations: no
  friction cone (magnitude cap only) and soft task hierarchy only — both in
  `KNOWN_GAPS.md`. The stale "un-wired groundwork" framing on the contact
  scheduler and the fictitious `QPController` example in `SKILL.md` are
  replaced with the real `WBCSolver` / `solve_whole_body` API.
- **Reverse-mode autodiff through the MJX constraint solve** (`implicit_vjp=True`
  on `Robot.functional_dynamics` / `build_system` / `to_diagram`). MuJoCo's
  constraint solver is a data-dependent `lax.while_loop`, so `jax.grad` through
  `mjx.forward` raised and only forward-mode survived — at one pass per
  parameter. The new `jaxterity.robot.implicit` module wraps the solve with
  `jaxonomy.optimization.implicit_solver` (jaxonomy T-131): the forward pass
  runs MJX's solver untaped and byte-equivalent, and the backward pass solves
  the adjoint of the solver's own KKT condition, reconstructed from the
  constraint rows (equality, dof/tendon friction, and joint/tendon limits —
  contact rows are rejected at construction; the contact-complementarity
  residual is the tracked follow-up). One backward pass now yields gradients
  for *all* parameters and the state at once — verified against finite
  differences and forward mode on limit-active poses, RK4 rollouts, and
  `jaxonomy.simulate` (`test/gradients/test_implicit_vjp.py`). Pair with an
  explicit integrator (`rk4`): the `custom_vjp` forecloses forward-mode, which
  BDF's Newton loop needs even on the primal pass.
- **External-hardware validation of the DC-motor recipe (EMPS).** Added an
  opt-in `external_data` test (`test/sysid/external/`) that fits the
  `DCMotorWithFriction` recipe to the EMPS positioning-rig benchmark — real
  1 kHz lab data, ground truth outside the jax*y stack. It recovers the rig's
  mass and viscous/Coulomb friction to within 0.2% of the paper's own IDIM-LS
  identification and predicts a held-out validation trace (~13.6% RMS). The data
  is fetched on demand (`fetch_emps.py`, sha256-pinned), **not vendored**
  (unclear license), and the test skips unless present — excluded from
  `scripts/check.sh`/CI. Numbers recorded in `test/sysid/external/RESULTS.md`.
- **Independent-integrator validation for the `sysid.actuator` LM recipes.**
  The Servo / BLDC / SeriesElastic fitters were previously only checked against
  telemetry generated by the *same* fixed-step RK4 they use, so a shared
  discretization or sign error could pass silently. Added frozen reference
  traces produced by SciPy's implicit `Radau` integrator over a standalone
  transcription of each device ODE (`test/sysid/data/independent/`, regenerate
  with `generate.py`) and a recovery test
  (`test/sysid/test_actuator_recipe_independent.py`) that fits them through the
  recipe. There are additionally **cross-model** oracles (`generate_jaxonomy.py`)
  whose ground truth is jaxonomy physics, not a transcription of the device ODE:
  SEA from acausal Inertia/Spring/Damper, and BLDC from an acausal `IdealMotor`
  (current-loop P folded into an effective resistance so no feedback crosses the
  acausal boundary) — so a sign/coupling error in jaxterity's own SEA/BLDC
  equations is caught even if copied into the integrator transcription. The Servo
  gets an independent-*implementation* oracle (a jaxonomy causal block diagram on
  the BDF solver), a weaker level since its nonlinear feedback can't be a passive
  network. SciPy / jaxonomy are only needed to regenerate the traces, not to run
  the test.
  `KNOWN_GAPS.md` now also tracks the remaining sysid gaps explicitly (validation
  external to the stack, transmission backlash, the BLDC R/L/Ke electrical
  split).

## [1.0.1] - 2026-07-12

Packaging, dependency, and CI fixes; no public API changes.

- **`requires-python` is now `>=3.11`** (was `>=3.10`). `optimistix>=0.1`, a core
  dependency, requires Python 3.11+, so 1.0.0 was not actually installable on
  3.10 despite advertising it.
- **Support MuJoCo 3.10** and pin `mujoco>=3.10,<3.11`. 3.10 removed the legacy
  `mjData.qM` (a `mj_fullM` signature change) and switched the CG solver to
  Hager-Zhang, which shifts numerics; the inertia calls are updated and the
  goldens are validated on the 3.10 line.
- Free-fall zoo goldens are now compared through the numerical tolerance
  framework (`jaxterity.testing.tolerances`) rather than bit-exactly, so they are
  reproducible across CPU architectures (arm64 dev host / x86-64 CI).
- Correct the mypy configuration so type-checking passes across the CI matrix
  (it no longer pins an older target that rejected newer third-party stubs).
- CI is enabled and green (lint, types, tests on 3.11/3.12); the PR gate excludes
  the hardware/pi tests, matching `scripts/check.sh`.
- Fix the `self_maintenance_demo_so101` example to the current `ResidualMonitor`
  API; lint/format the tutorial notebooks.
- Bump the default `llm_eval` model id to `claude-sonnet-5`.

## [1.0.0] - 2026-07-11

First public release. The stable core — robot import, MJX-backed differentiable
dynamics, forward-mode system identification, the reference zoo, the attestation
handle, and the per-unit twin + self-check stack — is implemented, exercised in
CI, and backed by the evidence ledger ([`CLAIMS.md`](CLAIMS.md)). Some
subsystems are preview-stage (full whole-body control, the turnkey sysid-recipe
catalogue, complete actuator/sensor coverage); what is partial or not yet done
is tracked in [`KNOWN_GAPS.md`](KNOWN_GAPS.md).

## Robot model

- URDF and MJCF import → a `Robot` exposing an MJX-backed, differentiable
  Jaxonomy `Diagram` via `robot.to_diagram()`.
- Round-trip export to MJCF with calibrated parameters substituted.
- Identifiable inertial and joint parameters (`mass`, `com`, `inertia`,
  `damping`, `friction`, `armature`) threaded into the MJX model and fittable in
  place.
- Stable `attestation_handle`: a content hash over model structure + fitted
  parameters that the deployment compiler consumes as a provenance key.

## Simulation & autodiff

- Differentiable forward/inverse dynamics, forward kinematics, and contact via
  MJX.
- Forward-mode autodiff (`jax.jacfwd`) through MJX's joint-limit / contact
  constraint solver, where reverse-mode cannot trace the iterative solve.
- `jit`/`vmap`-safe robots and rollouts (batched trajectories, parallel reaches).

## System identification & calibration

- Forward-mode sysid: recover joint damping/friction and link inertials from
  telemetry via Levenberg–Marquardt on the differentiable rollout.
- Active sysid over a servo-bus abstraction (seeded loopback for tests; split
  capture → workstation fit for real hardware).
- Closed-loop sysid for geared servos: an inverse-dynamics equation-error fit
  (damping + Coulomb friction bound to the robot; measurement scale/offset
  reported), with a held-out split and cross-trajectory validation.
- Off-body calibration: per-servo thermal fit and battery Thévenin (`V_oc`,
  `R_int`) sag fit from telemetry.
- Every fit returns a `SysidResult` (fitted parameters, covariance, residuals,
  telemetry hash, recipe), applied with `.apply_to(robot)`.

## Actuator & sensor models

- Calibrated actuator blocks (DC motor with friction, servo, BLDC,
  harmonic-drive, series-elastic, cable/tendon) as Jaxonomy `LeafSystem`s, each
  with a `sysid_protocol` and a `validate(telemetry)` residual report.
- Sensor models: IMU, joint encoder (quantization + delay), six-axis
  force/torque (cross-axis coupling), magnetometer, and depth/lidar (simulation).

## Whole-body control & policies

- WBC primitives: a hierarchical QP with a task DSL (posture, end-effector,
  CoM, momentum, contact) and centroidal-dynamics helpers.
- A layered, agent-friendly "code-as-policies" API: raw torque/encoder access →
  mid-level task priorities and horizon shooting → reusable skills → high-level
  goals.
- Gym/Gymnax-compatible environment wrappers.
- A FastMCP server exposing the layered API and recipes as MCP tools.

## Per-unit digital twins & self-check

- Per-unit attested `Twin`: a calibrated robot + off-body physics (servo
  thermal, battery sag) + serial number, with its own attestation handle that is
  distinct per unit.
- A predictive self-check `CoProcessor` that rolls the twin forward beside a
  controller, holding rate with explicit deadline accounting.
- A `SafetyMonitor` returning Ok / Derate / Veto against an operator-declared
  envelope (servo-thermal ceiling, bus-voltage floor, torque/joint margins),
  naming the binding constraint and the analytic time-to-breach.
- `preflight`: accept or reject a candidate policy against a unit's twin,
  binding the verdict to the twin's attestation handle.
- A reduced NumPy co-processor (thermal + battery forecast) that runs on a
  Raspberry Pi at rate, faithful to the JAX physics models.
- Self-maintenance helpers: windowed re-fit tracking parameter drift in a linked
  attestation chain, residual fault surfacing, and a capability envelope that
  shrinks as a unit heats.

## Reference robot zoo

- Ready-to-run robots with import and dynamics goldens: Cartpole, the SO-100 /
  SO-101 arm (with a real closed-loop calibration from hardware telemetry),
  Crazyflie, and the Unitree G1.
- `zoo.load`, `zoo.calibrated`, and `zoo.recalibrate`.

## Diagnostics & visualization

- `parity_report` quantifying sim-vs-real divergence and pointing at the
  most-likely-mismatched parameters.
- A Rerun adapter streaming 3D link transforms and per-joint signals.
- Post-hoc MuJoCo rendering of mesh-free zoo trajectories through full-mesh
  models, with GIF/MP4 helpers.

## Integrations & tooling

- LeRobot interoperability: write and read datasets in the `LeRobotDataset`
  format.
- CLI entry points (`jaxterity calibrate`, `simulate`, `export`).
- Worked, executed example notebooks and scripts, one per zoo robot.
