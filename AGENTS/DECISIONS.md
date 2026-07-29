# Jaxterity — Architectural Decisions

This file records architectural decisions that have already been made.
Each entry is an Architectural Decision Record (ADR) in the lightweight
form: context, decision, alternatives considered, consequences. The
purpose is to prevent settled questions from being re-litigated every
session, and to leave a paper trail for the human reviewer.

When a decision changes, do not delete the ADR. Add a new ADR that
references the old one and marks it superseded. The ADR history is part
of the project's reasoning record.

ADRs follow Jaxonomy's `DEC-NNN` numbering convention. Jaxterity ADRs
start at `ADR-001` to avoid namespace collision with Jaxonomy's `DEC-`
series.

---

## ADR-001 — Three separate packages, three repositories

**Status:** Accepted

### Context

Jaxonomy, Jaxterity, and Jaxility could be structured as one
monorepo with three Python packages, as three separate repositories,
or as a single package with subpackages. Each option has costs.

### Decision

Three separate repositories, three separate PyPI packages, with
versioned `>=`/`<` dependencies between them.

- `jaxonomy` — the engine, MIT, depends only on JAX ecosystem.
- `jaxterity` — the robotics layer, MIT, depends on `jaxonomy`.
- `jaxility` — the deployment factory, MIT, depends on both.

### Alternatives considered

1. **Single repo, single package.** Forces users adopting only the
   simulation engine to install acados, Arm GCC, vendor SDKs. Rejected:
   the dependency surface is far too heavy for that use case.
2. **Monorepo with three packages.** Tempting for development velocity.
   Rejected for two reasons. First, the three packages have genuinely
   different user populations (controls engineers, deployment engineers,
   regulated-industry teams) and different rates of change; a monorepo
   forces them to ship together. Second, monorepo tooling complexity
   exceeds the velocity gain at this project's scale.
3. **One package, three subpackages with optional extras.** A milder
   version of (1). Same dependency-surface objection.

### Consequences

- Each repo owns its own CI, its own release cadence, its own
  changelog.
- Cross-repo refactors require coordinated PRs. The discipline of
  versioning interfaces between the packages is non-optional.
- Jaxonomy must publish a clear minor-version policy. Jaxterity pins
  `jaxonomy>=2.2,<3` (see `pyproject.toml`); tightening or loosening the
  pin is an ADR-grade decision.
- Local development uses `pip install -e .` in each repo, with the
  Jaxterity venv pointing at a local Jaxonomy checkout.

---

## ADR-002 — Jaxterity is the boundary; physics is upstream

**Status:** Accepted (the JaxSim half is superseded by ADR-016; the PyRoki
half is realised-narrower-than-planned — see "Realised in practice" below)

### Context

Jaxterity could re-implement articulated-body dynamics, kinematics, and
contact resolution natively in JAX, or it could compose existing
implementations (PyRoki, JaxSim, MJX). The native path gives more
control but multiplies the maintenance surface.

### Decision

Jaxterity does not re-implement physics. It composes PyRoki (URDF →
JAX FK and IK), JaxSim (Featherstone-style rigid-body dynamics), and
MJX (contact and collision). Jaxterity's contribution is the
calibration loop, the actuator and sensor libraries, the sysid
recipes, the layered code-as-policies API, and the attestation chain.

When the upstream implementations fall short, Jaxterity contributes
upstream rather than forking.

### Realised in practice

In practice **MJX absorbed both the JaxSim role (ADR-016) and the PyRoki
role**: MJX provides forward kinematics, forward dynamics, contact,
and collision in one JAX-traceable model. JaxSim was removed entirely
(ADR-016). PyRoki is not a declared extra — it is not on PyPI (PyPI rejects
direct-URL git deps) and has zero callers in `jaxterity/` today; it is
install-from-source, held in reserve only for a future IK use case where its
differentiable-IK +
collision-aware solver would outweigh a ~30-line MJX-Jacobian
reimplementation (see `AGENTS/LIBRARIES.md` for the per-capability
analysis). The principle of the ADR — Jaxterity composes, does not
reimplement — stands; the realised composition turned out to be
single-engine, not three-engine.

### Alternatives considered

1. **Native Jaxterity physics.** Would have replicated PyRoki, JaxSim,
   and MJX inside Jaxterity. Rejected: 6-12 months of work for
   capabilities already maintained competently by larger teams.
2. **Wrap only MJX.** Would have skipped JaxSim and PyRoki. Rejected
   at decision time on the assumption that MJX gradients would be
   weaker than JaxSim's; **subsequently adopted** in practice as
   forward-mode autodiff through MJX (ADR-017) proved sufficient and
   neither of the alternatives provided value the MJX-only path
   couldn't reach. This ADR option is effectively the realised design.
3. **MuJoCo native (not MJX).** Non-JAX path. Rejected: incompatible
   with autodiff through simulation, jaxterity's core technical property.

### Consequences

- Jaxterity's value-add is the integration layer and the deployment
  story, not the physics.
- A change to MJX can ripple through Jaxterity. The MJX version pin
  is tight and reviewed at every release.
- A future "JAX-native physics" project, if undertaken, would be a
  separate package consumed by Jaxterity behind the same wrapping
  layer.
- The maintenance burden in Jaxterity is proportional to the
  integration surface, not the physics complexity.

---

## ADR-003 — JAX-only; no NumPy or PyTorch backends in Jaxterity

**Status:** Accepted

### Context

Jaxonomy ships a backend-dispatch system (`MathDispatcher`,
`numpy_api as npa`) that allows JAX, NumPy, and partial PyTorch
backends. Jaxterity could inherit this, or it could be JAX-only.

### Decision

Jaxterity is JAX-only. The backend-dispatch alias `npa` is not used in
Jaxterity code. Where a NumPy implementation is needed (debugging,
boundary code, the `jaxterity.testing` utilities), `import numpy as np`
is explicit and the function is clearly marked as backend-restricted.

### Alternatives considered

1. **Inherit Jaxonomy's MathDispatcher.** Would keep the door open for
   a NumPy or PyTorch backend later. Rejected: Jaxterity's value
   proposition rests on autodiff-through-simulation, which is JAX-only
   in practice. The dispatcher would be dead weight 95% of the time.
2. **JAX + PyTorch.** Tempting because some users have PyTorch RL
   stacks. Rejected: those stacks consume Jaxterity environments via
   the Gym/Gymnax interface; they don't need PyTorch dispatch
   internally.

### Consequences

- The Jaxonomy `DEC-010`-style "every block uses `npa`" pattern is
  inverted: Jaxterity blocks use `jnp` directly.
- Less abstraction overhead, more direct code, better LLM legibility.
- A future NumPy or PyTorch backend would be a major version change.
- The lesson from "Lessons in retrospect" applies: Jaxonomy hedged on
  JAX early and the hedging cost real engineering. Jaxterity does
  not hedge.

---

## ADR-004 — `Robot` as the single source of truth across sim and deploy

**Status:** Accepted

### Context

A robot's model could exist as separate objects for simulation
(MuJoCo XML, MJX model), training (Gym env), and deployment (acados
problem definition). Each layer would maintain its own copy. The
alternative is one `Robot` object that flows through every layer.

### Decision

One `CalibratedRobot` per fitted robot. It is the input to
simulation, training, sysid, and (via Jaxility) deployment. Any
transformation between these uses is a function of the `Robot`, not a
re-construction from URDF.

This is invariant 1 from CONTEXT.md, locked in here as an ADR because
it constrains every other decision.

### Alternatives considered

1. **Separate sim and deploy models.** The default approach in the
   industry. Rejected: it is the root cause of the sim-to-real gap
   Jaxterity exists to close. The design principle is that the
   simulation model is the source for the deployed controller.
2. **One model object, but with separate sim and deploy parameter
   sets.** Tempting because the same robot might have different
   parameters under sim and under hardware. Rejected: this
   inadvertently reopens the drift problem. If hardware parameters
   differ from sim parameters after calibration, the calibration is
   incomplete, not a separate concern.

### Consequences

- `Robot.to_diagram()` is the simulation handoff.
- `Robot.attestation_handle` is the deployment handoff (Jaxility
  consumes the handle).
- Sysid produces a new `Robot`, never mutates the input.
- The handle algorithm must be stable enough to survive serialization,
  cross-machine reproducibility, and Jaxility's downstream use
  (ADR-008).

---

## ADR-005 — Frames are first-class in the type system

**Status:** Accepted

### Context

Frame errors are the single most common bug class in real-world
robotics deployments (the IMU mounted at 90° off, the foot wrench
expressed in the body frame instead of world). The type system can
either prevent this at compile time, or rely on documentation and
convention.

### Decision

Every pose, twist, wrench, and velocity on a public API surface is a
`FramedQuantity` with an explicit `Frame`. Arithmetic between
mismatched frames raises a `FrameError`. Bare arrays are permitted
only inside numerical internals.

### Alternatives considered

1. **Documentation-and-convention.** The Pinocchio approach. Cheaper
   to implement; relies on the user reading docs. Rejected:
   the user is often an LLM, and an LLM is even more likely than a
   human to make a frame error without type guardrails.
2. **Frames as runtime-checked strings.** Lower implementation effort
   than typed frames. Rejected: errors surface only when the offending
   line runs, often deep in a trained policy; type-checked frames
   surface errors at write time.

### Consequences

- Significant API design discipline. Every actuator, sensor, and
  controller declares its frame contract.
- Some Pythonic patterns (broadcasting frame-typed arrays, slicing)
  become more verbose. Acceptable.
- Telemetry loaders must convert frames at the boundary; raw mocap
  data in unknown frames is rejected at load time.

---

## ADR-006 — SI units, internal representation only

**Status:** Accepted

### Context

Robotics telemetry comes in mixed units (degrees, RPM, lbf, bar).
Internal representation can carry mixed units with conversion at API
boundaries, or it can be uniformly SI with conversion only at
ingestion and emission.

### Decision

Internal representation is uniformly SI (radians, meters, seconds,
kilograms, newtons, pascals, kelvins, amperes). Telemetry loaders
convert at the boundary. URDF importers convert at import time
(URDF degrees → radians). Sensor outputs in non-SI units carry their
conversion as identifiable `Parameter`s.

### Alternatives considered

1. **Pint or unyt unit-typed arrays.** Would add unit safety at the
   array level. Rejected: incompatible with JAX tracing
   (the JIT path does not survive unit-aware wrappers in the
   versions we care about). Revisit if a JAX-compatible unit library
   matures.
2. **Mixed internal units with conversion factors stored as
   parameters.** Would match some hardware vendor SDKs. Rejected:
   doubles the cognitive load on every function and every LLM
   reading the code.

### Consequences

- The unit-checker lint (see PATTERNS.md §7.3) enforces the
  contract.
- Telemetry schemas (Pydantic) declare per-column units explicitly.
- Sensor blocks that emit non-SI outputs (encoder counts, ADC raw
  values) carry the conversion factor and the factor is part of
  what sysid identifies.

---

## ADR-007 — `equinox.Module` for stateful objects

**Status:** Accepted

### Context

Stateful Jaxterity objects (`Robot`, `Joint`, `Link`, `Actuator`,
`Sensor`, `Task`, `SysidResult`) need to be JAX PyTrees, immutable,
and constructible from typed `__init__` signatures. Equinox provides
all three; the alternatives are hand-rolled PyTree registrations or
`flax.struct.PyTreeNode`.

### Decision

Use `equinox.Module` for every stateful Jaxterity object that needs
PyTree registration.

### Alternatives considered

1. **Hand-rolled `jax.tree_util.register_pytree_node` per class.**
   What Jaxonomy does in places. Rejected for Jaxterity: more
   boilerplate, more places for a `flatten`/`unflatten` mismatch
   bug. Equinox's `Module` is the industry default.
2. **`flax.struct.PyTreeNode`.** Equally viable. Rejected because
   Equinox is more widely adopted in the JAX scientific computing
   ecosystem (Diffrax uses it; PyRoki uses it).
3. **`dataclasses.dataclass(frozen=True)` with custom PyTree
   registration.** More effort, no advantage over Equinox.

### Consequences

- Equinox becomes a non-optional dependency.
- Class definitions are shorter and more LLM-legible.
- The `eqx.Module`-as-PyTree pattern is the lingua franca for the
  whole package.

---

## ADR-008 — Attestation handle is a versioned content hash

**Status:** Accepted

### Context

The attestation handle is the contract between Jaxterity and
Jaxility. It must be (a) stable across machines and Python
versions, (b) sensitive to anything that affects deployment behavior,
(c) insensitive to irrelevant changes (whitespace, comments), and
(d) versioned so that algorithm changes do not silently collide with
old handles.

### Decision

The handle is `BLAKE3(algorithm_version || URDF_canonical_bytes ||
parameter_values_in_canonical_order || actuator_block_versions ||
sensor_block_versions || sysid_recipe_versions || telemetry_hashes)`.

- `algorithm_version` is an integer that increments on any change to
  the handle computation.
- `URDF_canonical_bytes` is the URDF normalized through a documented
  canonicalization (whitespace collapsed, comments stripped, element
  attribute order sorted).
- Parameter values use a canonical ordering by parameter path.
- Block and recipe versions are taken from `__version__` on the
  respective modules.
- Telemetry hashes are the SHA-256 of the telemetry source bytes.

### Alternatives considered

1. **Plain SHA-256 over a serialized robot.** Rejected: silent
   collisions when the serialization changes. Versioning is required.
2. **A cryptographic signature instead of a hash.** Out of scope here.
   The handle is a content hash; signing it (if a deployment needs that)
   is a downstream concern layered on top, not a property of the handle
   itself.
3. **An opaque UUID assigned at calibration time.** Rejected: not
   reproducible. Two calibrations of the same robot with the same
   data would produce different handles. The whole point is content
   addressing.

### Consequences

- BLAKE3 is fast enough that handle computation is not a bottleneck.
- Bumping `algorithm_version` is a major-version release of
  Jaxterity (or a documented exception with migration tooling).
- Jaxility's deployment artifacts embed the handle they were
  built from. The runtime can refuse to run an artifact whose handle
  does not match the expected one.

---

## ADR-009 — Telemetry schema is Pydantic, not bare dicts

**Status:** Accepted

### Context

Sysid recipes consume telemetry. Telemetry could be passed as bare
`dict`s, as typed `NamedTuple`s, or as Pydantic models. The choice
affects how errors surface, how documentation is generated, and how
agents consume the API.

### Decision

Telemetry is typed Pydantic v2 models in `jaxterity.data.schema`. Every
column carries unit and frame metadata. Loaders convert at the
boundary and validate at construction time.

### Alternatives considered

1. **Bare `dict[str, jnp.ndarray]`.** The most common pattern in
   research code. Rejected: no validation, no documentation, no
   schema evolution.
2. **`typing.TypedDict`.** Lighter weight than Pydantic. Rejected:
   no runtime validation, weaker IDE support, no Pydantic-style JSON
   schema generation.
3. **`attrs` or `dataclasses`.** Roughly equivalent to Pydantic but
   without the JSON schema and validation story.

### Consequences

- Pydantic v2 is a required dependency.
- Schema versioning is explicit; old telemetry files are upgradable
  through documented migrations.
- The MCP server can publish JSON schemas for telemetry directly
  from Pydantic, which makes the agent-tool surface self-describing.

---

## ADR-010 — `SysidResult`, not tuples

**Status:** Accepted

### Context

System identification routines return fitted parameters and many
auxiliary quantities (covariance, residuals, warnings, the recipe
version, the telemetry hash). They could return tuples, dicts, or
typed result objects.

### Decision

Every sysid recipe returns a `SysidResult` object (an `equinox.Module`
with structured fields). Tuple unpacking is not supported. The
`SysidResult` carries an `apply_to(robot) -> Robot` method that
performs the calibration update and contributes the recipe version
and telemetry hash to the attestation chain.

### Alternatives considered

1. **`(params, covariance, residuals)` tuple.** Common but loses
   metadata (recipe version, telemetry hash) that the attestation
   chain needs.
2. **`dict[str, Any]`.** Loses type safety. Rejected.

### Consequences

- Sysid recipes are pleasant to use from agents because the result
  is fully introspectable.
- Adding a new field to `SysidResult` does not break callers (they
  ignore unknown fields).
- This is invariant 4 from CONTEXT.md, locked in here as an ADR.

---

## ADR-011 — Reference robot zoo lives in the package, not a separate dataset repo

**Status:** Accepted

### Context

The reference robot zoo (Cartpole, Crazyflie, SO-100, Berkeley
Humanoid Lite) could ship inside the `jaxterity` package or live in
a separate `jaxterity-data` package installed as an optional extra.

### Decision

The URDFs and calibrated parameter sets live inside `jaxterity` under
`jaxterity/zoo/`. The telemetry datasets live in a separate
`jaxterity-data` repository because they are large enough to dominate
the package wheel size.

`pip install jaxterity` gets the URDFs and calibrated params.
`pip install jaxterity[zoo]` adds the telemetry datasets.

### Alternatives considered

1. **Everything in the main package.** Simpler. Rejected because
   telemetry datasets push the wheel past 100 MB.
2. **Everything in `jaxterity-data`.** Cleaner separation. Rejected
   because tests in `jaxterity` need URDFs and calibrated params for
   the zoo regression suite; pushing those to a separate package
   forces a circular dependency.

### Consequences

- `jaxterity` wheel stays small (~5 MB target).
- `jaxterity[zoo]` is the path for users who want to reproduce
  calibrations.
- The zoo regression suite is always available.

---

## ADR-012 — Ship the Claude Code skill on day one

**Status:** Accepted

### Context

The choice is between treating the Claude Code skill as a follow-on
artifact (after v0.1 ships) or as a v0.1 deliverable.

### Decision

The Claude Code skill (`SKILL.md` and `.claude/skills/jaxterity/`)
ships as a v0.1 deliverable, not after v0.1. The skill informs every
public-API design decision from the outset.

This is consistent with "the agent is the primary user" (CONTEXT.md
design philosophy 1) and the cheat sheet's prioritization.

### Alternatives considered

1. **Skill as a follow-on.** Would have let v0.1 ship faster.
   Rejected: a skill written after the API is frozen will have to
   work around bad API shapes that an early skill would have caught.
2. **Skill only, no MCP server.** Skill-without-server means the
   agent can read docs but cannot take actions. Rejected: the
   action-taking case (sysid, simulate, export) is the higher-value
   surface.

### Consequences

- The skill ships with v0.1, not as a later add-on.
- Every public function ships with a docstring example that the
  skill references.
- Bumping the skill or MCP tools surface is a documented release-note
  event.

---

## ADR-013 — Pluggable extension points instead of hard-coded backends

**Status:** Accepted

### Context

A few surfaces — attestation signing, telemetry storage, fleet update
orchestration, compliance-evidence generation — are ones different
deployments will want to implement differently (a research lab, a hobbyist,
and a production fleet all have different needs). Baking one implementation
into the core would force that choice on everyone.

### Decision

`jaxterity` is fully MIT and defines these surfaces as documented
extension points — public interfaces with a working default (stub)
implementation. Downstream code registers an alternative implementation at
load time rather than patching the core.

### Alternatives considered

1. **Hard-code a single backend for each surface.** Rejected — it forces
   one storage/signing/orchestration choice on every user and makes the
   core harder to reuse across very different deployments.
2. **Leave the surfaces out entirely until needed.** Rejected — defining
   the interface now keeps the seam stable and lets the default stub ship
   immediately.

### Consequences

- Jaxterity defines extension points for: attestation signing, telemetry
  storage backends, fleet update orchestration, and compliance-evidence
  generation. These are public interfaces with stub implementations.
- The default stub is a fully working implementation for the common case;
  a downstream plugin can swap in a richer one against the same interface.

---

## ADR-014 — Stable-handle pattern (`@robot[<hash>]`) for MCP

**Status:** Accepted

### Context

Multi-step robotics workflows in MCP (calibrate → simulate → train →
export) pass large stateful objects between tool calls. Re-passing the
full `Robot` on every call is expensive and obscures the conversation.

### Decision

MCP tools that produce or consume stateful artifacts use stable handles
following the text-to-cad `@cad[<id>]` pattern. Jaxterity handles:

- `@robot[<hash>]` — a `CalibratedRobot`. Hash is the attestation
  handle (ADR-008).
- `@trajectory[<hash>]` — a simulated or recorded trajectory. Hash
  is content-addressed.
- `@fit[<hash>]` — a `SysidResult`. Hash includes the inputs.

The MCP server caches the artifacts server-side, keyed by hash.
Subsequent tool calls accept the handle in lieu of re-passing the
artifact.

### Alternatives considered

1. **Pass full JSON on every call.** Workable for small robots,
   breaks down on humanoids. Rejected.
2. **In-memory session state on the server.** Rejected: handles are
   stateless and portable across sessions, which matches the
   attestation chain semantics.

### Consequences

- The same hash that gates Jaxility also addresses MCP artifacts.
  One concept, two uses.
- The MCP server is a small typed cache; no session state to
  manage.
- Users can hand a handle to a colleague (or to their future self)
  and reproduce results.

---

## ADR-015 — Temporary subfolder placement during bootstrap (scoped deviation from ADR-001)

**Status:** Realised — Jaxterity extracted to the standalone
`machinavitalis/jaxterity` repo. Historical record
only; ADR-001 is back in force.

### Context

ADR-001 mandates three separate repositories. During the initial bootstrap,
Jaxterity was developed as a temporary subfolder inside the Jaxonomy
repository (`jaxonomy/jaxterity/`) — with its own `pyproject.toml`, `test/`
tree, `AGENTS/` directory, and `CHANGELOG.md`, but physically nested in the
engine's repo. This was a deliberate, time-boxed convenience authorized for the
bootstrap session: it let early development proceed against the
in-repo editable Jaxonomy without first publishing Jaxonomy to PyPI or standing
up a second repository and CI runner.

### Decision

Permit Jaxterity to live under `jaxonomy/jaxterity/` **only** through the
bootstrap, after which it is extracted to a standalone repository per ADR-001.
Direct consequences of the temporary arrangement, recorded so they are
auditable:

- `jaxonomy` is declared as an **unpinned** dependency (installed from the
  in-repo editable checkout, not PyPI). It tightens to `>=2.3,<2.4` (per
  ADR-001) on extraction.
- The GitHub Actions workflow is **inactive** during this period (GitHub runs
  only repository-root workflows); `scripts/check.sh` is the authoritative
  check.
- Bootstrap PRs merge into the **Jaxonomy** repository's `main`, and the
  bootstrap tag lives there.

### Alternatives considered

1. **Stand up the standalone repo immediately.** Rejected for the bootstrap:
   more setup overhead before any code exists, and Jaxonomy is not on PyPI yet,
   so cross-repo dependency resolution would require git installs from day one.
2. **Publish Jaxonomy to PyPI first, then start Jaxterity in its own repo.**
   Rejected as premature sequencing; it couples Jaxterity's start to an engine
   release decision.

### Consequences

- This ADR is the explicit, auditable record of the ADR-001 deviation. It does
  **not** supersede ADR-001; it is a scoped, temporary exception with a hard
  end condition.
- Extraction is itself work to schedule at the end of the bootstrap: new repo,
  CI activation, the `jaxonomy` pin bump, and git-history migration of
  `jaxonomy/jaxterity/`.
- **Accepted** by the maintainer. The deviation is authorized through the end
  of the bootstrap; the extraction work it implies is tracked as a pre-extraction
  item.
- **Realised**: the extraction completed
  (history-preserving `git filter-repo`, `jaxonomy>=2.2,<3` pin,
  `test/conftest.py` shim removed, GitHub Actions activated). The
  `AGENTS/EXTRACTION_CHECKLIST.md` playbook was deleted after the work
  landed — `git show ADR-015` reaches it via history. ADR-015 is now
  historical; ADR-001's separate-repos rule is the active norm.

---

## ADR-016 — MJX is the dynamics backend (revises ADR-002's JaxSim choice)

**Status:** Accepted (maintainer-directed);
**Realised final** — JaxSim removed from `pyproject.toml` entirely

### Context

ADR-002 chose JaxSim for Featherstone rigid-body dynamics and MJX for contact.
During the `Robot.to_diagram()` implementation, JaxSim turned out to be unusable in the
development/CI sandbox: it parses URDF/SDF through `rod`, which shells out to the
Gazebo `gz`/`ign` executable — a heavy system binary that is not present and not
pip-installable. (An in-memory `rod.Model` bypasses Gazebo but then requires
fixed-base configuration and more `rod` internals.) MJX, already a required
dependency, parses MJCF strings directly with no Gazebo dependency, is JAX-native
and differentiable, and exposes model arrays (`body_mass`, `dof_damping`, …) that
give a clean differentiable-parameter path.

### Decision

`Robot.to_diagram()` uses **MJX** as the dynamics backend: the kinematic tree is
lowered to MJCF (`jaxterity.robot.mjcf`), `mujoco.MjModel.from_xml_string` +
`mjx.put_model` build a differentiable model, and `mjx.forward` is wrapped as a
Jaxonomy `LeafSystem`. JaxSim is **removed from `pyproject.toml` entirely**
(the realised-final step): nothing in `jaxterity/`, `test/`, or
`examples/` ever imported it (`grep -rn jaxsim` returns zero hits in code);
the original ADR-002 rationale for keeping it (stronger parameter gradients
through Featherstone than through MJX) was empirically wrong — ADR-017's
forward-mode `parameter_jacobian` traverses MJX's constraint solver cleanly,
the SO-101 sysid demo recovers joint damping to ~2e-4 end-to-end, and the
robot recipes work without JaxSim. Re-add only if a
Gazebo-free `rod` build lands AND a measurable gradient-quality gap to MJX
is shown.

### Consequences

- MuJoCo/MJX and BLAKE3 (attestation, ADR-008) become **core** dependencies.
- **Reverse-mode caveat:** MJX runs an *iterative* constraint solver (a
  `lax.while_loop`) when a joint limit or contact is active, which `jax.grad`
  cannot traverse. Reverse-mode sysid therefore requires the constraint-free
  regime (e.g. unlimited/`continuous` joints, no active contact) or a
  forward-mode / BDF-adjoint path. Forward simulation is unaffected. This is a
  load-bearing constraint for the sysid recipes and is recorded in
  `LIBRARIES.md`.
- This ADR revises (does not delete) ADR-002's backend split; ADR-002 stays as
  the reasoning record.

---

## ADR-017 — Forward-mode autodiff for differentiable sysid through MJX constraints

**Status:** Accepted (maintainer-directed)

### Context

ADR-016 noted a load-bearing caveat: MJX runs an iterative constraint solver
under a `lax.while_loop` whenever a joint limit or contact is active, and JAX
**cannot reverse-differentiate a `while_loop`** (the backward pass needs a static
trip count). So `jax.grad` through `jaxonomy.simulate` raises on essentially
every real arm (cartpole's prismatic limit, the SO-101's six limited joints).
Empirically the constraint-free workaround is not enough on its own: a 6-DOF
reverse-mode adjoint also exhausts memory in the sandbox. Differentiable system
identification — the core value proposition — therefore had no working path on
real robots. Two facts changed the picture: (1) **forward-mode** autodiff
carries tangents *forward* through a `while_loop`, so it traverses the constraint
solver cleanly; (2) it builds no backward tape, so memory stays modest on
many-DOF arms. Verified: forward-mode gradients match finite differences on the
full SO-101 (mass and damping) and cross-check against reverse-mode on
constraint-free systems.

### Decision

Provide forward-mode parameter differentiation as the primary sysid gradient
path for constrained/real robots: `jaxterity.sysid.parameter_jacobian` builds the
robot diagram once, overrides identifiable parameters in the context, simulates
with the reverse-mode adjoint **off** (`enable_autodiff=False`, so JAX traces the
real ops), and differentiates with `jax.jacfwd`. Reverse-mode (`jax.grad` + the
BDF adjoint) remains the recommended path for constraint-free problems with many
parameters, where its cost (independent of parameter count) wins.

### Consequences

- Forward-mode cost scales with the number of differentiated **parameters** (one
  tangent sweep each) — well-matched to inertial/friction identification's modest
  counts; poorly matched to high-dimensional parameter vectors (use the
  constraint-free reverse-mode path there).
- **Active impacts stay non-differentiable.** A joint *slamming into* a limit or
  a hard contact making/breaking is a genuine kink; the gradient through it is
  ill-defined for forward-mode, reverse-mode, and finite differences alike.
  Identification excitation must stay within limits / free of hard contact
  (standard practice). The full implicit-differentiation of the contact solution
  (à la qpax/optimistix) remains future work, tracked for the contact recipe.
- Adds a gradient-vs-finite-difference tolerance to
  `jaxterity.testing.tolerances` (`gradient_rtol`/`gradient_atol`), looser than
  the state tolerances because the comparison is bounded by FD truncation, not
  the (exact) autodiff.

---

## ADR-018 — Floating base via a free joint + quaternion configuration map

**Status:** Accepted (maintainer-directed)

### Context

The fixed-root MJCF backend (ADR-016) could not represent a free-floating body —
blocking drones/quadrotors, legged robots, and even the free-rigid-body
conservation case (the deferred third conservation case). A free joint has 7 ``qpos``
(3 translation + a unit quaternion) but 6 ``qvel`` (``nq ≠ nv``), and its
configuration derivative is **not** ``q̇ = qd``: the orientation rate is
``½·quat⊗[0, ω_body]``. Jaxonomy integrates a generic ``dstate/dt`` with an ODE
solver, so the wrapper — not MuJoCo's special quaternion integrator — must supply
that map.

### Decision

A single ``JointType.FLOATING`` joint (parent ``"world"``, child = the root link)
makes a model floating-base. ``to_mjcf`` emits a MuJoCo ``<freejoint>`` on the
root body; ``_RobotSystem`` builds a static per-joint ``qpos→q̇`` map from the
MJX model (free joints get the quaternion rate, hinge/slide stay identity) and
applies it in ``ode``. Default ``q0`` is MuJoCo's ``qpos0`` (identity quaternion,
not zeros). The free joint exposes no identifiable parameters.

### Consequences

- A generic RK4/adaptive integrator keeps the quaternion on the unit sphere to
  high accuracy (the derivative is tangent to it); verified by the free-body
  conservation test (energy + angular momentum to the conservation bound over 10
  tumbling periods, no explicit renormalization). Very long horizons may
  eventually need re-normalization — revisit if a test shows drift.
- Scope: **one** floating base; **ball** and **planar** joints still raise.
  ``from_mjcf``/``from_urdf`` do not yet *import* free joints into ``FLOATING``
  (export works; round-tripping a floating-base model is future work).
- Floating-base robots are now expressible, but the Crazyflie/Humanoid zoo
  entries additionally need rotor/contact modelling (tracked separately).

---

## ADR-019 — LeRobot integration via the dataset format, not the package

**Status:** Accepted

### Context

The nanoLeRobot demo is the SO-100 end-to-end demo, and the SO-100/SO-101 is the
canonical arm of HuggingFace's open-source `LeRobot` stack. "LeRobot integration"
could mean (a) depend on the `lerobot` package and run its policies/trainers, or
(b) speak its on-disk `LeRobotDataset` contract and leave training to it. The
`lerobot` package pulls a heavy tree (`torch`, `datasets`, video codecs) and is
GPU-oriented; Jaxterity ships no trainer and no policy by design (CONTEXT.md "it
is not a policy library"), and its demos must run on a fresh clone in CI without
a GPU. The principle from ADR on `pyroki`/`jaxsim`: do not tax every install with
a large dependency nothing in core imports.

### Decision

Integrate at the **dataset-format** boundary. `jaxterity.integrations.lerobot`
reads and writes the `LeRobotDataset` v2.1 layout (`meta/info.json`,
`tasks/episodes/episodes_stats` sidecars, per-episode parquet with
`observation.state` / `action` as `list<float32>` columns). The SO-100 feature
columns use LeRobot's own motor order, so a dataset Jaxterity writes has the same
shape a real teleoperation recording does. Writing/reading needs only `pyarrow`;
the demo degrades gracefully — without `lerobot` it uses a dependency-light
parquet reader, so the whole nanoLeRobot pipeline (demos → dataset → behaviour
cloning → closed-loop reach) runs in the normal CI checks. The real package is an
opt-in `[lerobot]` extra and, when present, `load_lerobot_dataset` returns a true
`LeRobotDataset`; a self-skipping test exercises that path.

The nanoLeRobot demo's policy (a small goal-conditioned MLP behaviour-cloned on
the dataset) is **self-contained in the example**, exactly as nanojaxterity's PPO
is — it is the demo, not a library surface.

### Consequences

- Demonstrations collected against a *calibrated* Jaxterity SO-100 are directly
  consumable by LeRobot policies (ACT, diffusion, …) by pointing them at the
  written directory — the sim-to-real value (one calibrated model, CONTEXT.md
  invariant 1) flows into the LeRobot ecosystem without Jaxterity owning a
  trainer.
- `lerobot` stays out of `all` (same rationale as `pyroki`). The format is the
  stable contract; if LeRobot bumps its codebase version, the writer's
  `CODEBASE_VERSION` and `info.json` shape are the single place to update.
- The demo's reaching env is a stiff PD position servo on the calibrated MJX
  dynamics with distal-link inertia floored to the geared servos' reflected
  rotor inertia (a calibration stand-in, cf. ADR/cartpole) — chosen because a
  gravity-compensation feedforward is numerically unstable under explicit RK4 at
  the tiny vendored distal inertias. See `examples/nanolerobot.py`.

---

## ADR-020 — Devices are causal blocks; physical networks are acausal

**Status:** Accepted

### Context

Two of the gaps the LeKiwi and G1 work exposed are *networks*, not devices: the
power tree (one pack, one harness, many servos, a regulated compute rail) and
the thermal path (winding, housing, driver, a shared bracket). Both were being
modeled as lumped causal blocks — `BatterySag` is one rail with one resistance,
`ServoThermal` is one node — and both lumped models answer exactly one question
and mis-answer every neighbouring one: which servo sags most, whether the
winding or the case is hot, what a 5 V rail does to pack current.

The obvious extension is to keep writing causal blocks with more ports. That
fails for the reason acausal modeling exists: the *topology* is the thing that
changes between robots, and a causal block bakes in a solve order. A four-load
tree and a six-load tree with a regulator are the same physics and different
code. Jaxonomy already ships a Modelica-style acausal layer with electrical and
thermal domains and an index-reducing compiler.

### Decision

Split on whether the thing has a **causality** of its own:

- **Devices are causal Jaxonomy `LeafSystem`s.** A servo, a gearbox, a sensor, a
  bus channel: something goes in, something comes out, and the direction is a
  property of the hardware. This is every block under `actuators/` and
  `sensors/`.
- **Physical networks are acausal**, built on `jaxonomy.acausal`. A power tree
  and a thermal network are *declared* (`PowerTreeSpec`, `ThermalNetworkSpec`)
  and their equations derived. `jaxterity.physics.power_tree` and
  `.thermal_network` compile to an `AcausalSystem`, which is an ordinary
  `LeafSystem` again — so the network drops back into a normal diagram and the
  boundary is invisible downstream.
- **The body stays an implicit MJX monolith** (ADR-016): it is neither, and
  re-deriving articulated dynamics acausally would be re-implementing MJX.

The bridge between the two is explicit and thin: `servo_currents` maps torque to
current for the power tree, and the thermal network takes dissipated power in
watts. Neither network knows what a joint is.

Two Jaxterity-side acausal components fill gaps in Jaxonomy's electrical library
(`DCDCConverter`, `Charger`). They are robot-power-tree-shaped rather than
general electrical primitives; if a second consumer appears outside robotics,
the boundary rule (CONTEXT.md) says upstream them into Jaxonomy.

### Consequences

- `sympy` becomes an optional dependency (`[acausal]`), since Jaxonomy declares
  it only in its own extras. Importing `jaxterity.physics` without it works;
  *building* a network raises a `JaxterityError` naming the extra. The lumped
  `BatterySag` / `ServoThermal` path stays dependency-free, so the existing twin
  is untouched.
- Every network ships a **closed-form reference** for its steady state
  (`dc_solve`, `steady_state_temperatures`), and the compiled network is tested
  against it. This is not redundancy: the closed form is the fast path for duty
  studies and the fitter's forward model, and the agreement test is what makes a
  topology change detectable.
- Acausal models have real constraints the specs enforce up front rather than
  surfacing as compiler errors: a network needs at least one energy-storage
  element, and a capacitor may not sit directly across an ideal source or a
  regulated output (its state would be algebraically determined). `PowerTreeSpec`
  validates both.
- Conditioning: a power tree legitimately mixes milliohms with millifarads, so
  the compiler's ill-conditioning warning fires on healthy models. Both builders
  raise `condition_number_threshold`; a *genuinely* singular model still fails,
  in index reduction.

## Open questions (not yet decided)

These are explicitly tracked as undecided. Do not default them.

### OQ-1 — Which JAX-traceable QP for WBC?

OSQP-JAX, qpax, or a hand-rolled active-set QP. Decision deferred,
pending a benchmark across the four zoo robots.

### OQ-2 — Mocap loader formats

AMASS, LAFAN1, and a generic CSV format are obvious. Whether to
support BVH and FBX natively or via an external converter is
deferred.

### OQ-3 — Whether Jaxility consumes a serialized `Robot` or imports it directly

If Jaxility imports `jaxterity` directly, version coupling is
tighter but the dependency graph is simpler. If it consumes a
serialized format, the coupling is looser but a schema must be
maintained. Coordinated with Jaxility's ADR-001 (when written).

### OQ-4 — Cross-device determinism policy

Jaxonomy documents cross-device determinism as best-effort. Jaxterity
needs a stricter policy because the attestation chain depends on
hash stability. The exact contract (which operations are bit-exact
across CPU/GPU/TPU; which are ULP-bounded) is open.
