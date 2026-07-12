# Jaxterity — Project Context

Read AGENTS/README.md first for navigation; this file is the orientation
document for any agent (human or AI) working on Jaxterity — what Jaxterity is,
its architecture and invariants. Then PATTERNS.md and DECISIONS.md.

If you have not yet read the Jaxonomy `AGENTS/CONTEXT.md`, read it before this
one. Jaxterity is built directly on top of Jaxonomy and inherits most of its
conventions, abstractions, and invariants. This document describes only what is
specific to Jaxterity.

## What Jaxterity is

Jaxterity is a Python library for modeling, calibrating, and controlling robots
in JAX. It is the robotics-specific layer that sits on top of Jaxonomy's
general-purpose hybrid dynamical simulation engine. It is aimed at engineers
and researchers who deploy robot controllers to real hardware — on humanoids,
quadrupeds, drones, and manipulators — where sim-to-real fidelity, reproducible
calibration, and attestation of what was deployed all matter.

Jaxterity's API is written to be **read fluently by coding agents and LLMs** as
well as humans: every public API, error message, docstring, and example is
designed to be legible to an agent working on behalf of a robotics engineer.

Jaxterity is distributed under the MIT license.

## Relationship to Jaxonomy and Jaxility

Jaxterity is the middle package of a three-package stack:

- **Jaxonomy** — the engine. General-purpose hybrid dynamical simulation in
  JAX. Not robotics-specific. Jaxterity imports it.
- **Jaxterity** — the robotics layer. URDF/MJCF import, articulated dynamics,
  calibrated actuator and sensor models, sysid recipes, whole-body control
  primitives, a layered code-as-policies API, and a reference robot zoo.
- **Jaxility** — the deployment artifact factory. Takes a Jaxterity model
  fitted to real telemetry and produces a self-contained C project that runs
  on a target Arm SoC, with a signed attestation manifest.

The architectural rule that draws the boundary: **if a feature is useful for
controls engineers working outside robotics (HVAC, battery management,
aerospace, energy systems), it belongs in Jaxonomy. If it only makes sense for
things with joints, actuators, contacts, sensors, and kinematic chains, it
belongs in Jaxterity.** When in doubt, default to Jaxonomy and add a
robotics-specific wrapper in Jaxterity.

Jaxterity does NOT vendor or re-implement Jaxonomy primitives. `Diagram`,
`Context`, `LeafSystem`, `simulate`, `fit_parameters`, `linearize`,
`simulate_batch`, the acausal layer, and the entire framework module are
imported from Jaxonomy. If you find yourself copying simulation code into
Jaxterity, the package boundary is being drawn wrong; stop and either
upstream the change into Jaxonomy or restructure the Jaxterity-side abstraction.

## Design philosophy: agent-legible, end-to-end, opinionated

Jaxterity is designed around three overlapping theses:

**1. The agent is the primary user.** Every function has an LLM-consumable
docstring with types, units, frame conventions, and at least one runnable
example. Error messages are structured for agent consumption: they name the
offending object, the violated invariant, the expected shape or unit, and the
likely fix. The repository itself should fit in an LLM's context window when
the SKILL.md and the layered API are loaded; the full source is a stretch goal
but not a hard requirement after the framework section grows.

**2. End-to-end through one problem, not building blocks for many.** Jaxterity
ships with worked, tested, deployable end-to-end demos for specific robots —
the Cartpole, the Crazyflie, the SO-100 manipulator, a bipedal humanoid. The
demos are not toys; they are the product. The library is what falls out of
making those demos clean, readable, and reusable. If a capability is not
demanded by at least one demo, it does not go into Jaxterity.

**3. Sim-to-real parity by construction, not by tuning.** The simulation model
a user calibrates against telemetry is the same model that gets handed to
Jaxility for deployment. Drift between the simulated and deployed
dynamics is impossible by construction — there is one model, parameterized,
and the deployed controller is generated from it. This single-model property
is what a separate sim model and deploy model cannot guarantee.

When in doubt about a design choice, prefer the option that (a) is more
agent-legible, (b) collapses the sim-to-deploy round-trip, or (c) makes the
end-to-end demo shorter and more readable.

## What Jaxterity is NOT

- **It is not a physics engine.** Jaxterity does not write a new contact
  solver, a new articulated-body algorithm, or a new collision system. It
  composes MuJoCo (via MJX) for forward kinematics, forward dynamics, contact,
  and collision, and adds Jaxonomy block wrappers that make those engines
  composable with the rest of a control diagram. (JaxSim was an originally
  intended composition partner but is not used — see ADR-016. PyRoki was
  originally listed as a kinematics partner but is not yet integrated — MJX
  covers FK natively; PyRoki is not on PyPI and has zero callers today —
  install it from source (`pip install "pyroki @
  git+https://github.com/chungmin99/pyroki.git"`) if a future IK use case
  lands — see LIBRARIES.md.)
- **It is not a policy library.** Jaxterity does not ship a PPO implementation,
  a VLA model, or an imitation-learning trainer. It ships the calibrated
  environments and the layered API that those external libraries (K-Sim,
  RSL-RL, LeRobot, ksim) target. The training stack is upstream; Jaxterity
  is downstream of training and upstream of deployment.
- **It is not a deployment toolchain.** Cross-compilation, acados export,
  CasADi lowering, LiteRT/ExecuTorch packaging, Arm SoC profiles, and
  attestation manifests live in Jaxility, not Jaxterity. Jaxterity
  produces the fitted model; Jaxility consumes it.
- **It is not a GUI tool.** No visual editor. Rerun and Mermaid are first-class
  output surfaces for visualization, but they are read-only for humans. The
  agent is the author of the model.
- **It is not a general-purpose mechanical/multi-domain modeling tool.** That
  is Jaxonomy's acausal layer. Jaxterity is robotics-specific.
- **It is not a successor to MuJoCo Playground or K-Sim.** It is complementary.
  An MJX environment is a valid input to a Jaxterity workflow; a K-Sim training
  loop is a valid consumer of a Jaxterity-calibrated environment. Jaxterity adds
  the calibration-to-deployment round-trip on top of the environments those
  tools produce.

## Scope — what Jaxterity owns

Jaxterity provides an opinionated robotics layer on top of Jaxonomy. Its
scope includes:

- **Robot model ingestion**
  - URDF import (joints, links, inertias, limits, collision geometry) →
    Jaxterity `Robot` object, which exposes a Jaxonomy `Diagram` view
  - MJCF import for MuJoCo-native models, with the same `Robot` interface
  - Round-trip export back to URDF/MJCF with calibrated parameters substituted
  - Reference robot zoo: Cartpole, Crazyflie, SO-100, Berkeley Humanoid Lite,
    Stompy mini, each with both unverified-from-URDF and calibrated-from-
    telemetry parameter sets
- **Articulated-body dynamics**
  - Forward kinematics, forward dynamics, inverse dynamics, contact Jacobians
  - Implementations delegated to **MJX** (rigid-body dynamics + contact +
    collision + FK). MJX covers the Featherstone role originally planned for
    JaxSim (ADR-016) and the FK role originally planned for PyRoki — see
    LIBRARIES.md.
  - Jaxterity does not re-implement these; it wraps them as Jaxonomy
    `LeafSystem`s so they compose with the rest of a diagram
- **Calibrated actuator models**
  - First-class blocks for: BLDC, brushed DC, servo, harmonic-drive, cycloidal,
    tendon-driven, series-elastic actuators
  - Friction models including Coulomb, viscous, Stribeck, and LuGre
  - All parameters are identifiable via `fit_parameters` from real telemetry
  - Each actuator block declares a `sysid_protocol` describing the data
    collection procedure that identifies its parameters (chirp, step, ramp,
    dynamometer fit)
- **Sensor models**
  - IMU (accelerometer + gyroscope + magnetometer), with bias, noise, and
    misalignment parameters
  - Joint encoders (absolute, incremental, magnetic) with quantization and
    delay
  - Force/torque sensors with cross-axis coupling
  - Depth cameras and lidar (simulation only; calibration belongs in the
    sensor's own driver)
  - Each sensor block declares a `sysid_protocol` analogous to actuators
- **System identification recipes**
  - `sysid.actuator` — fit a motor model from voltage-ramp / step / dyno data
  - `sysid.inertial` — fit link inertias from chirp excitation + mocap
  - `sysid.full_body` — joint trajectory fit using full-body dynamics
  - `sysid.contact` — friction and stiffness fit from impact data
  - Each recipe is a tested function that takes telemetry in a documented
    schema and returns a calibrated `Robot`
- **Whole-body control primitives**
  - Differentiable hierarchical QP (task-space WBC), built on OSQP-JAX or an
    equivalent JAX-traceable QP
  - Task DSL: `PostureTask`, `EndEffectorTask`, `CoMTask`, `MomentumTask`,
    `ContactTask`, with priorities and weights
  - Contact scheduling primitives for legged locomotion
  - Centroidal dynamics primitives for momentum-based control
- **Layered code-as-policies API**
  - S1 (high-level): `pick(object, pose)`, `walk_to(target)`, `reach(point)`
  - S2 (skill): `move_joint(j, q_des)`, `track_trajectory(traj)`
  - S3 (mid-level): `set_task_priority(task, w)`, `step_mpc()`
  - S4 (raw): `apply_torque(j, tau)`, `read_encoder(j)`
  - Each layer is fully documented for LLM use; lower layers are accessible
    from higher layers as escape hatches
- **Sim-to-real diagnostics**
  - `parity_report(sim_traj, real_traj)` — quantifies divergence and points
    at the most-likely-mismatched parameters
  - Rerun.io adapter for side-by-side sim/real trajectory visualization
  - `attestation_handle()` — produces a stable hash for a fitted `Robot` that
    Jaxility consumes as the provenance key
- **Agent affordances**
  - FastMCP server (`jaxterity.mcp`) exposing the above as tools callable
    from Claude Code, Cursor, and other MCP-aware clients
  - `@robot[<hash>]`, `@trajectory[<hash>]`, `@fit[<hash>]` stable-handle
    pattern for stateful artifacts referenced across multiple tool calls
  - Claude Code Skill (`SKILL.md` + `.claude/skills/jaxterity/`) shipped on
    day one
- **Reference data**
  - `jaxterity-data` (optional install) — telemetry datasets for the reference
    robot zoo, in a documented schema
  - `jaxterity.data.schema` — Pydantic models for telemetry, calibration
    artifacts, and attestation manifests

## Scope — what belongs elsewhere

- **General-purpose simulation primitives** — Jaxonomy. If a block is useful
  outside robotics, push it upstream.
- **Embedded codegen, acados/CasADi lowering, cross-compilation, attestation
  signing, Arm SoC profiles** — Jaxility. Jaxterity produces fitted
  models; it does not produce binaries.
- **Policy training (PPO, SAC, BC, diffusion policies, VLA fine-tuning)** —
  K-Sim, RSL-RL, LeRobot, or whatever the user already uses. Jaxterity
  exposes Gym/Gymnax-compatible environments and stops there.
- **Cloud hosting, multi-user collaboration, fleet management** — out of scope
  for this library. Jaxterity is a library, not a hosted service.
- **GUI / visual editor** — not in scope.
- **Robot hardware abstraction at the driver level (CAN, EtherCAT, motor
  controllers)** — out of scope. Jaxterity assumes telemetry has already been
  recorded into a CSV/Parquet/HDF5 file conforming to `jaxterity.data.schema`.

## Architectural heritage

Jaxterity inherits Jaxonomy's Drake-derived `System` / `Context` / `Diagram`
abstractions wholesale. Where Jaxonomy implements them in JAX, Jaxterity
specializes them for robotics.

In addition, Jaxterity draws specific patterns from a small set of upstream
projects (see `DECISIONS.md` for the full rationale):

- **MuJoCo Playground** — domain randomization patterns, sensor noise
  modeling, PD-controller defaults
- **PyRoki** — URDF → JAX forward kinematics, collision primitives, IK solver
  code paths. *Listed as a heritage influence only; not currently a runtime
  dependency. FK + collision are covered by MJX. PyRoki is not on PyPI and has
  zero callers today — install from source for a future IK use case — see
  LIBRARIES.md.*
- **REX / supergraph** — async-graph simulation patterns for sim-to-real
  latency and delay modeling (concepts only; implementation is fresh)
- **Pinocchio** — algorithm reference and ground-truth validation, not a
  runtime dependency
- **Drake** — trajectory-optimization problem structures and contact
  formulations as a validation reference
- **WPILib SysId** — motor model `V = Ks·sgn(ẋ) + Kv·ẋ + Ka·ẍ` and the
  data-collection procedure for actuator identification
- **Pink / Mink** — weighted task-hierarchy formulation for whole-body
  control

The full attribution and license-compatibility analysis lives in
`LIBRARIES.md`.

## Architecture overview

Top-level package layout (under `jaxterity/`):

- `robot/` — Core robot model abstractions. Defines `Robot`, `Joint`, `Link`,
  `Frame`, and the URDF/MJCF importers. Wraps these as Jaxonomy `Diagram`s
  via `robot.to_diagram()`. This is the foundation on which actuators,
  sensors, and controllers are layered.
- `actuators/` — Calibrated actuator-model library. Each actuator is a
  Jaxonomy `LeafSystem` with identifiable parameters, a documented
  `sysid_protocol`, and at least one validation test against canonical
  data (WPILib SysId reference traces are used where applicable).
- `sensors/` — Calibrated sensor-model library, same shape as `actuators/`.
- `sysid/` — System identification recipes. Each recipe is a function
  `(telemetry, robot, hints) -> calibrated_robot`. Recipes are tested
  end-to-end against the reference robot zoo telemetry.
- `wbc/` — Whole-body control. `QPController`, task DSL, contact scheduler,
  centroidal dynamics. Built on OSQP-JAX or an equivalent JAX-native QP
  solver.
- `policies/` — Layered code-as-policies API (S1–S4). This is the surface
  most agents will write against.
- `diagnostics/` — Sim-to-real parity tools, Rerun adapter, attestation
  handles.
- `data/` — Telemetry schema, dataset loaders, mock-data generators for
  tests.
- `mcp/` — FastMCP server. Wraps the layered API and the sysid recipes as
  MCP tools.
- `cli/` — Command-line entry points (`jaxterity calibrate`,
  `jaxterity simulate`, `jaxterity export`).
- `zoo/` — Reference robot zoo. Each robot is a subpackage with its URDF,
  its calibrated parameter set(s), its telemetry samples, and its
  end-to-end demo script.
- `testing/` — Test utilities, fixtures, golden-trajectory comparison
  helpers.

## Key abstractions and invariants

Understanding these is required before modifying any code outside the
demo scripts.

- **Robot** — The central Jaxterity object. Holds a URDF/MJCF source,
  a tree of `Joint`s and `Link`s, attached actuators and sensors, and a
  calibration state (uncalibrated, partially calibrated, fully calibrated).
  Exposes `.to_diagram()` to produce a Jaxonomy `Diagram` suitable for
  simulation. Robots are PyTrees and vmap-safe.
- **Joint / Link** — Pure data classes describing kinematic structure.
  Inertial parameters are `jaxonomy.Parameter` instances so they can be
  identified via `fit_parameters`.
- **Frame** — Named coordinate frame attached to a link. All Jaxterity APIs
  that take or return poses are explicit about which frame the pose lives
  in. There is no implicit world frame; the user (or the robot's URDF)
  declares it.
- **Actuator / Sensor** — Jaxonomy `LeafSystem` subclasses with extra
  metadata: a `sysid_protocol` attribute, a `parameter_spec` describing
  identifiable vs. fixed parameters, and a `validate(telemetry)` method
  that returns a structured residual report.
- **CalibratedRobot** — A `Robot` with all `Parameter`s bound to fitted
  values via `with_parameters()`. Has a stable `attestation_handle` —
  a content hash over the URDF, parameter values, sysid procedure, and
  data hashes used in the fit. This handle is the provenance key
  Jaxility consumes.
- **Task** — In WBC, a residual function plus a Jacobian plus a priority
  and a weight. Tasks compose into a hierarchical QP.
- **Telemetry** — A typed time-series object with documented columns,
  units, frames, and sample rates. The schema lives in
  `jaxterity.data.schema` and is enforced at load time.
- **SkillHandle** — `@robot[<hash>]`, `@trajectory[<hash>]`, `@fit[<hash>]`.
  Stable references to stateful artifacts. An MCP tool returning a handle
  caches the artifact server-side; subsequent tool calls accept the handle
  in lieu of re-passing the artifact.

### Invariants that must hold

These extend, not replace, the Jaxonomy invariants. All eight Jaxonomy
invariants apply transitively (no global mutable state, pure functions
on the simulation hot path, vmap safety, PyTree compliance, determinism,
end-to-end differentiability, backend neutrality where applicable, no
global environment leakage). The Jaxterity-specific additions:

1. **One model, one truth.** The calibrated `Robot` used for simulation
   is the same object handed to Jaxility. Any transformation between
   simulation and deployment must be a function of the `Robot`, not a
   re-creation from raw URDF. If two model objects exist for the same
   robot — one for sim, one for deploy — that is a bug.
2. **Frames are explicit.** No pose, velocity, force, or wrench may
   appear in a public API without an explicit frame annotation. The
   `Frame` type is required at every boundary.
3. **Units are explicit.** SI units throughout (radians, meters, seconds,
   kilograms, newtons). Telemetry loaders convert at the boundary; the
   internal representation never carries degrees, RPM, or pounds.
4. **Sysid produces a `Result`, not a tuple.** Every system identification
   call returns a `SysidResult` object containing fitted parameters,
   parameter covariance, residual statistics, the telemetry hash, the
   recipe version, and a structured warning list. Callers must not
   unpack into bare floats.
5. **Attestation handles are first-class.** A `CalibratedRobot` without a
   valid `attestation_handle` cannot be passed to Jaxility. The
   handle is computed once at calibration time and validated on every
   subsequent use.
6. **No silent contact smoothing.** Wherever Jaxterity approximates a
   non-smooth contact (e.g., for differentiability), the approximation
   is named (`SmoothedContact`, `SoftContact`) and its parameters are
   exposed. Hidden smoothing inside a generic-looking block is forbidden.
7. **Reference robot zoo is the ground truth.** Any change to a core
   abstraction must keep all zoo robots simulating, calibrating, and
   exporting correctly. Zoo regressions block release.
8. **The agent API is the user API.** A function intended only for
   internal use is named with a leading underscore and excluded from
   the MCP surface. A function exposed to agents must have an
   LLM-consumable docstring and at least one structured example.

## Supported versions and dependencies

- Python: 3.10+
- Jaxonomy: pinned to the current minor version (`>=2.3,<2.4` at v0.1;
  pin tightens as upstream stabilizes)
- JAX, NumPy, Diffrax: inherited from Jaxonomy's pins
- MuJoCo / MJX: required (forward kinematics + rigid-body dynamics + contact + collision)
- Optional: `[mocap]` (mocap loaders), `[rerun]` (Rerun.io visualization),
  `[zoo]` (reference robot data), `[mcp]` (MCP server), and `[all]`
  (everything installable from PyPI). PyRoki is a kinematics/IK library not
  used by core today; it is not on PyPI and is not declared as an extra —
  install it from source if needed (see LIBRARIES.md).

Optional extras follow Jaxonomy's pattern. The default install is intentionally
small; `[all]` is for contributors and CI.

## Repository layout at root

- `jaxterity/` — the package itself
- `test/` — pytest test suite (unit, integration, zoo end-to-end)
- `docs/` — MkDocs source (published via `mkdocs.yml`)
- `examples/` — standalone example scripts, one per zoo robot
- `validation/` — golden-trajectory regression tests against reference results
- `AGENTS/` — this directory; agent-oriented documentation and task specs
- `.claude/` — Claude Code project configuration and skills
- `SKILL.md` — root Claude Code
