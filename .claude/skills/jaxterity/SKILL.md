# Jaxterity Skill

You are using Jaxterity, a JAX-native robotics stack for sim-to-silicon
deployment. This file is your operating manual. Read it before
suggesting any Jaxterity code.

## What Jaxterity does

Jaxterity takes a robot from URDF/MJCF to a calibrated simulation model
that matches real hardware telemetry, then hands that model to
Jaxility for deployment to embedded silicon. It is the robotics layer of a
larger JAX-native stack — for where it sits, what it imports, and what consumes
it, see `AGENTS/CONTEXT.md`.

## When to use Jaxterity

Use it when the user is:

- Importing a robot URDF or MJCF and wanting to simulate it in JAX.
- Calibrating a simulation model against real robot telemetry.
- Writing a controller (WBC, MPC, code-as-policies) for a robot.
- Preparing a calibrated model for embedded deployment (the handoff to
  Jaxility).
- Running parity diagnostics between a simulation and a real robot.

## When NOT to use Jaxterity

- **Pure simulation with no robotics structure.** Use Jaxonomy directly.
  Jaxterity is for things with joints, contacts, and actuators.
- **Training large policies (PPO, SAC, BC, diffusion, VLA).** Use K-Sim,
  RSL-RL, LeRobot, or another training library; expose a Jaxterity
  `Robot` as a Gymnax-compatible environment to the trainer.
- **Cross-compilation or embedded deployment.** That is Jaxility's
  job. Jaxterity produces the calibrated model; Jaxility produces
  the binary.
- **Pure rigid-body kinematics with no actuation or sysid.** Use PyRoki
  directly. Jaxterity layers on top.
- **General-purpose acausal modeling (HVAC, batteries, fluids).** Use
  Jaxonomy's `jaxonomy.acausal` directly.

## Decision tree for the typical workflow

When the user asks for help with Jaxterity, route through this tree
before writing code:

1. **Does the user have a URDF or MJCF in hand?**
   - Yes, URDF → `jaxterity.robot.from_urdf(path)`.
   - Yes, MJCF → `jaxterity.robot.from_mjcf(path)`.
   - No → Suggest starting with a reference robot from
     `jaxterity.zoo` (Cartpole, Crazyflie, SO-100, Berkeley Humanoid
     Lite). The zoo robots come with URDF, calibrated parameters, and
     a demo script.

2. **Does the user have telemetry from the real robot?**
   - Yes → Run the matching sysid recipe (see "Sysid recipes" below)
     to produce a `CalibratedRobot`.
   - No → Use the uncalibrated robot for simulation, but warn the user
     that sim-to-real parity is not guaranteed.

3. **What is the user trying to do with the (possibly calibrated) robot?**
   - Simulate a trajectory → `robot.to_diagram()`, then use
     Jaxonomy's `simulate()`.
   - Design a controller → Use the WBC primitives in
     `jaxterity.wbc` or the layered API in `jaxterity.policies`.
   - Train a policy → Expose the robot as a Gymnax env; hand to the
     user's training stack.
   - Deploy to hardware → Hand `robot.attestation_handle` to
     Jaxility.

4. **Is the user comparing simulation to real data?**
   - Use `jaxterity.diagnostics.parity_report(sim_traj, real_traj)`.

## Core API surface, in order of how often agents will use it

### Robot ingestion

```python
from jaxterity.robot import from_urdf, from_mjcf

robot = from_urdf("path/to/robot.urdf")
# robot is uncalibrated. Default actuators are IdealTorqueActuator
# per joint; default sensors are IdealEncoder per joint.
```

### Attaching realistic actuators and sensors

```python
from jaxterity.actuators import DCMotorWithFriction
from jaxterity.sensors import IMU, JointEncoder

robot = robot.with_actuator(
    joint="shoulder_left",
    actuator=DCMotorWithFriction.uncalibrated(),
)
robot = robot.with_sensor(IMU.uncalibrated(mount="base_link"))
robot = robot.with_sensor(JointEncoder.uncalibrated(joint="shoulder_left"))
```

`with_actuator` and `with_sensor` return new robots; they do not
mutate.

### Loading telemetry

```python
from jaxterity.data import load_telemetry

telemetry = load_telemetry("path/to/recording.parquet")
# Validates schema, units, and frames at load time. Raises SchemaError
# on mismatch.
```

### System identification

```python
from jaxterity.sysid import actuator, inertial, full_body, contact

# Per-actuator fit from voltage-ramp telemetry:
result = actuator(telemetry, robot, actuator="DCMotorWithFriction:shoulder_left")
calibrated_robot = result.apply_to(robot)

# For full-body sysid:
result = full_body(telemetry, robot)
calibrated_robot = result.apply_to(robot)
```

All recipes return a `SysidResult` with fitted parameters, covariance,
residuals, warnings, and `apply_to(robot) -> Robot`. **Do not unpack
the result as a tuple.** It is not a tuple.

### Simulation

```python
import jaxonomy

diagram = calibrated_robot.to_diagram()
context = diagram.create_context()
results = jaxonomy.simulate(diagram, context, stop_time=5.0)
```

The diagram is a Jaxonomy `Diagram`; everything Jaxonomy can do with a
`Diagram` (autodiff, `fit_parameters`, `linearize`, `simulate_batch`)
works on a Jaxterity-produced diagram.

### Whole-body control

```python
from jaxterity.wbc import (
    PostureTask, EndEffectorTask, WBCSolver, solve_whole_body,
    contact_jacobian, inverse_dynamics,
)

model = calibrated_robot.build_system()._mjx_model

# Tasks are frozen dataclasses; Cartesian tasks take closures over q so the
# kinematics recompute per tick. Weights set a soft hierarchy (10x weight
# ~ 10x tighter residual); there is no strict-priority cascade.
tasks = (
    PostureTask(name="home", q_desired=q_home, weight=0.1),
    EndEffectorTask(
        name="reach",
        pose_fn=lambda q: hand_position(model, q),          # (3,)
        jacobian_fn=lambda q: contact_jacobian(model, q, hand_body_id),
        x_desired=target_position_world,                     # frame: world
        weight=10.0,
    ),
)

# Kinematic solve: best qdd for the task stack.
qdd = WBCSolver(tasks=tasks).solve(q, qd, n_dof=model.nv)
tau = inverse_dynamics(model, q, qd, qdd)   # fixed-base torque command

# Dynamics-consistent solve (floating base / contacts): qdd, contact
# forces, and realizable torques in one QP — see jaxterity.wbc.dynamics.
sol = solve_whole_body(model, q, qd, tasks, contacts=stance_contacts)
tau = sol.tau
```

### Layered code-as-policies API

```python
from jaxterity.policies import S1, S2, S3, S4

# S1 — high-level
policy = S1(robot=calibrated_robot)
policy.pick(object_handle, pose=target_pose)

# S4 — raw torque
policy = S4(robot=calibrated_robot)
policy.apply_torque(joint="shoulder_left", tau=0.5)
```

Lower layers are always accessible from higher layers as escape
hatches. Reach for the highest layer that does the job.

### Deployment handoff

```python
# Hand the attestation handle to Jaxility:
handle = calibrated_robot.attestation_handle
# handle is a stable content hash; see Jaxility's SKILL.md for
# how to consume it.
```

## Frame and unit discipline

This is the single most common failure mode in robotics code. Watch
for it:

- **Every pose, velocity, force, and wrench carries an explicit `Frame`.**
  If you write `pos = jnp.array([1.0, 2.0, 3.0])` in Jaxterity code and
  hand it to a public API, you have a bug. Wrap it as
  `FramedQuantity(Position(pos), frame=Frame("world"))`.
- **SI units throughout.** Radians, meters, seconds, kilograms,
  newtons, pascals, kelvins, amperes. Telemetry loaders convert at
  the boundary. If you find yourself multiplying by `np.pi/180` inside
  Jaxterity code, you have a bug — the conversion should have happened
  at load time.

The error messages from `FrameError` and `UnitError` are structured and
should tell you how to fix the issue. Read them carefully before
guessing.

## Naming conventions

Match the field's standard usage:

- `q`, `qd`, `qdd` — joint positions, velocities, accelerations.
- `tau` — joint torques.
- `T_target_source` — transform from `source` frame to `target` frame.
  `T_world_body @ p_body = p_world`.
- `J` — Jacobian. `J_lin`, `J_ang` for linear and angular parts.
- `wrench`, `twist` — force-torque and linear-angular velocity pairs.
- `com`, `I` (yes, capital I) — center of mass and inertia tensor.

These names will look unpythonic. They are correct. Do not rename them
to `joint_positions`, `joint_velocities`, etc. — the names match the
literature and the rest of the robotics ecosystem.

## What you must NOT do

- **Re-implement physics.** Jaxterity wraps MJX (forward kinematics,
  forward dynamics, contact, collision). Do not write your own forward
  dynamics, contact solver, or Featherstone implementation. (PyRoki is
  reserved for a future IK use case — not currently wrapped; it is not on
  PyPI, so install it from source if you need it.)
- **Treat `SysidResult` as a tuple.** Always use `result.apply_to(robot)`
  or access named fields.
- **Mutate a `Robot`.** Always return a new one (`robot.with_...`).
- **Compare simulated and real trajectories without a parity report.**
  Use `parity_report`. It will tell you which parameter is the
  likely culprit.
- **Bypass the attestation handle for deployment.** Jaxility
  will refuse a `CalibratedRobot` without a valid handle.
- **Use bare arrays for poses on public APIs.** Use `FramedQuantity`.
- **Use degrees, RPM, or imperial units anywhere except in telemetry
  loaders.**

## Reference robots in the zoo

Each comes with URDF, calibrated parameters, telemetry, and a demo
script. When a user wants a starting point, point them at the closest
match:

| Robot                       | Use when…                                  |
|-----------------------------|--------------------------------------------|
| `jaxterity.zoo.cartpole`    | Learning the workflow; minimum viable demo |
| `jaxterity.zoo.crazyflie`   | Quadrotor / aerial; floating-base body + thrust-wrench model (`thrust_dynamics`), NMPC, STM32-class chip |
| `jaxterity.zoo.so100`       | 6-DoF manipulator, LeRobot ecosystem       |
| `jaxterity.zoo.unitree_g1` | Bipedal humanoid (29-DoF); `load(floating_base=True)` for a free base, WBC, Cortex-A class chip |

## Sysid recipes — quick reference

| Recipe                  | Telemetry needed                           | Identifies                          |
|-------------------------|--------------------------------------------|-------------------------------------|
| `sysid.actuator`        | Voltage-command + position response        | Motor model (Ks, Kv, Ka, friction)  |
| `sysid.inertial`        | Chirp excitation + mocap                   | Link masses, COMs, inertia tensors  |
| `sysid.full_body`       | Joint trajectory (no mocap required)       | Full kinematic parameter set        |
| `sysid.contact`         | Impact data (force/torque + position)      | Friction coefficients, stiffness    |

Each recipe documents its required telemetry schema in its docstring.
Read the docstring before calling.

## Picking a block: the library is a recipe library

Each block carries its identifiable parameters, the capture protocol that
identifies them (`block.sysid_protocol`), and a `validate()` that scores a
trace. Choosing the right block is therefore choosing the right *experiment*,
which is why the table below is organized by what you have, not by what the
part is called. **Everything in this table is sim-tier** — the models and
fitters are exercised against synthetic traces only (see `KNOWN_GAPS.md`
§Calibrated actuator / sensor libraries).

| You have | Use | It identifies |
|---|---|---|
| A geared joint, one encoder | `HarmonicDrive` | Ratio, direction efficiency, velocity-domain backlash proxy |
| A geared joint, motor **and** joint encoders | `PositionalBacklash` | True angular play + mesh stiffness (`lost_motion` cross-checks it model-free) |
| Torque + speed on both sides, both power directions | `Gearbox` | `η_f`, `η_b` (the reciprocal one — back-driving costs *more*), load-independent loss |
| A belt axis that rings | `BeltDrive` | Pulley ratio, belt stiffness/damping (`belt_resonance_hz` tells you where to chirp) |
| A screw axis | `LeadScrew` | Lead kinematics, two-quadrant efficiency; `is_backdrivable` answers from geometry |
| A low-ratio BLDC leg joint | `QuasiDirectDrive` | N²-reflected inertia, the speed-collapsing torque envelope |
| An open-loop stepper | `Stepper` | Holding/detent torque, speed droop; `pull_out_margin` is the number to plan against |
| A pneumatic cylinder | `PneumaticCylinder` | Valve conductance, critical ratio, dead volumes |
| A current reading you are turning into torque | `CurrentSense` | Gain, offset, bandwidth, quantization — before they become a wrong `K_t` |
| A servo reporting whole-degree temperatures | `TemperatureSensor` | Offset, sensor lag, step — and `quantization_floor` says what a fit *cannot* resolve |
| A bus between controller and joint | `BusChannel` | Latency, jitter, drop rate (`phase_lag_deg`: 5 ms at 10 Hz is 18° a fit will call damping) |

Two **networks** rather than devices, built on Jaxonomy's acausal layer
(`pip install 'jaxterity[acausal]'`):

```python
from jaxterity.physics import PowerTree, LEKIWI_POWER_TREE, dc_solve

# Closed-form DC operating point — fast enough for a duty study.
sol = dc_solve(LEKIWI_POWER_TREE, {"shoulder_pan": 1.5, "wrist_roll": 1.5})
# The electrically farthest servo sees the lowest voltage at equal current:
sol.load_voltage["wrist_roll"] < sol.load_voltage["shoulder_pan"]  # True

tree = PowerTree.build(LEKIWI_POWER_TREE)  # the compiled acausal system
```

- `jaxterity.physics.power_tree` — pack (SoC-dependent OCV, `CellAging`),
  per-branch harness resistance, DC-DC rails, charger. Use `dc_solve` for a
  duty study; build the network when you need the transient. It identifies the
  *sum* `R0 + R1 + R_main` from a rail trace and refuses to split it.
- `jaxterity.physics.thermal_network` — two-node winding/housing per servo,
  driver nodes, inter-servo conduction. `steady_state_temperatures` is the
  closed-form path; `fit_two_node` needs *both* nodes measured.

And three things that shape a plan rather than a model:

- `jaxterity.physics.compliance` — `zmp_margins(..., suspension=...)` subtracts
  the base tilt from a tip-over margin; `fit_coast_down` gets rolling
  resistance from one coast.
- `jaxterity.aging` — fit friction/backlash growth from a unit's re-fit
  history, project to a threshold, replay as a `DriftChannel`.
- `design_probe(..., stability=StabilityConstraint(base=...))` — the Fisher
  ascent, constrained to stay inside the tip-over floor.

## Object grounding — identify what the robot is holding

A camera cannot see a held object's mass or centre of mass; the arm's own
joint telemetry can (the calibrated arm is the force scale). Sim-tier today
(C-120) — the loopback proves recovery; a real capture replays through
`ReplayLink`.

```python
from jaxterity import zoo
from jaxterity.objects import (
    RigidPayload, PayloadLoopbackLink, identify_payload,
    payload_identifiability,
)

robot = zoo.calibrated("so100")
ball = RigidPayload(attach_link="gripper_frame_link", mass=0.1)  # prior

# 1. ALWAYS check the probe first — a quasi-static probe cannot excite
#    what you are asking for, and the report says so before anything moves.
report = payload_identifiability(robot, ball)          # default coast probe
report.raise_if_unidentifiable("payload probe")

# 2. Fit. fitted keys are leaf-shaped ("payload.mass", "payload.com");
#    covariance/warnings are per component ("payload.com[0]", ...).
link = PayloadLoopbackLink(robot, ball, true_mass=0.15)  # or ReplayLink(capture)
result = identify_payload(robot, link, ball)
result.interval()                                       # per-component CIs

# 3. Per-object attested twin: same arm + different object = different handle.
from jaxterity.twin import so101_twin
twin = so101_twin("SN-001").with_payload(ball, result)
```

Fit subsets honestly: `fit=("mass",)` when the articulation cannot see the
CoM (low-DOF robots), `fit=("mass", "com", "inertia_diag")` only with a rich
probe. `RigidPayload` priors are the LM start *and* the linearization point —
a zero prior mass is refused. The grasp is a rigid weld (no compliance
model); see `KNOWN_GAPS.md` §Object-side identification.

Let the robot *choose* its probe (differentiable experiment design, C-121):

```python
from jaxterity.objects import design_probe, probe_information

probe = design_probe(robot, ball)               # D-optimal ascent, in-limits
# probe -> fit -> next probe: target what the last fit left widest
probe2 = design_probe(robot, ball, prior_covariance=result.covariance)
probe_information(robot, ball, probe)           # rank candidate probes
```

Local ascent over coast probes only (pose + kick — not torque trajectories);
deterministic; second-order autodiff makes it minutes-scale on 6-DOF arms.

Contact parameters from toss/squeeze probes (C-122 — smooth/event path,
no MJX contact-solver gradients):

```python
from jaxterity.objects import identify_restitution, identify_squeeze

r = identify_restitution(impact_times)       # e (+ drop height if unknown)
s = identify_squeeze(deform, rate, force)    # k (N/m), c (N·s/m²)
```

Both refuse under-determined records by name (one impact can't pin two
parameters; a quasi-static squeeze can't resolve damping) and pin
unphysical fits at their bounds (0 ≤ e ≤ 1, k/c ≥ 0).

### Seed the fit with what the object *looks* like (C-132)

Inertial identification has bad local minima; a vision model's guess is the
cheapest way out. Declare it, and the fit records that you did:

```python
from jaxterity.objects import SemanticPrior

prior = SemanticPrior.from_material(          # or explicit mass_kg=(lo, hi)
    "small plastic tote, ~10x8x6 cm", "plastic", (0.10, 0.08, 0.06),
    source="vlm/claude-opus-5", confidence=0.6,
)
result = identify_payload(robot, link, ball, prior=prior)
result.recipe_name          # "objects.payload+semantic_prior" — declared
result.warnings             # the prior's label/source/bracket, and any pin
```

A parameter that ends pinned to a prior bound is named as such: that value is
the guess, not the measurement. Widen the bracket rather than lowering
`confidence` — the bracket is what actually bounds the search.

### Enroll an object before committing to it (C-132)

The whole arc — perceive, vet the probe under the prior's *worst case*, heft,
weld the object onto the twin, then vet the technique on the loaded unit:

```python
from jaxterity.checkup.stability import LEKIWI_BASE, StabilityConstraint
from jaxterity.objects import enroll_object
from jaxterity.oracle import hold_plan
from jaxterity.selfcheck import SafetyEnvelope

TUCKED = (0.0, -1.2, -0.8, 0.0, 0.0, 0.0)     # declare a pose with margin
enrollment = enroll_object(
    twin, link, ball, prior,
    techniques=[hold_plan("gentle", twin, [0.05] * 6, n_steps=30, dt=0.005,
                          q0=TUCKED),
                hold_plan("brisk", twin, [0.3] * 6, n_steps=30, dt=0.005,
                          q0=TUCKED)],
    envelope=SafetyEnvelope(temp_ceiling_k=343.15, bus_floor_v=11.0),
    stability=StabilityConstraint(base=LEKIWI_BASE), pose_rad=TUCKED,
)
enrollment.technique          # the vetted way to do the job
enrollment.twin_handle        # loaded unit != empty unit
```

`enroll_object` **refuses** (raises) when no probe clears the tip-over floor
under the prior's heaviest object — re-pose, don't probe slower (speed moves
the margin under 1 mm from a tucked pose; the pose moves it tens of mm).
Worked end to end in `examples/scene_enrollment_lekiwi.py`.

## The oracle — ask this unit's physics a question (C-130)

MPC simulates to choose; the gate simulates to permit; the oracle simulates to
**answer**. Use it when a planner needs a number ("do I overheat holding this
for the next minute?", "which of these keeps the most tip-over margin?"):

```python
from jaxterity.oracle import hold_plan, simulate_plan, what_if
from jaxterity.selfcheck import SafetyEnvelope
from jaxterity.checkup.stability import LEKIWI_BASE

plan = hold_plan("carry it", twin, [0.4] * twin.n_servos, n_steps=100, dt=0.005)
answer = simulate_plan(twin, plan, SafetyEnvelope(temp_ceiling_k=343.15),
                       base=LEKIWI_BASE, n_draws=16)
answer.outcome             # worst case over the posterior: pass/derate/veto
answer.pass_fraction       # how much of that is *uncertainty*, not physics
answer.margin("tip_over")  # per-constraint band: worst, median, [lo, hi]

best = what_if(twin, [plan_a, plan_b], envelope, base=LEKIWI_BASE).best
```

Read both numbers. A veto at `pass_fraction ≈ 0` is physics; a veto at
`pass_fraction ≈ 0.5` means this unit's fit is too loose to tell — measure
before you derate. A twin with no attached fit covariance reports a zero-width
band and says so; a `dt` too coarse for the fixed-step integrator raises rather
than returning a margin computed from `nan` (a moving arm wants `dt <= 0.01`).

## Runtime assurance — the reflex (C-129)

`preflight` and `SafetyMonitor` decide *whether* a command may run.
`ReflexController` decides *what actually reaches the actuators*, every tick,
from the same declared envelope:

```python
from jaxterity.selfcheck import ReflexController, SafetyMonitor, SimplexSwitch

reflex = ReflexController(twin, envelope)
decision = reflex.filter(q, qd, T, policy_tau)   # clamp any policy's command
decision.tau, decision.binding_constraint, decision.engaged

switch = SimplexSwitch(SafetyMonitor(twin, envelope), reflex)
step = switch.step(prediction, q, qd, T, policy_tau)
step.mode        # "pass" | "derate" | "reflex" (userland dropped, brake wins)
```

Build both from the *same* `Twin` — `SimplexSwitch` refuses a monitor and a
reflex bound to different unit versions. Check `decision.feasible`: when a stop
needs more braking torque than the thermal budget allows, the stop wins by
declared priority and the decision names the row that yielded. No WCET has been
measured (see `KNOWN_GAPS.md` §Per-unit twin).

## MCP tools available

Exposed by `jaxterity mcp serve` (the `[mcp]` extra). Handles are created once
and referenced by id (ADR-008), so artifacts are not re-passed per call:

- **Layered API** — `create_robot_handle(robot_name, q0) -> "@<hash>"`,
  `list_handles`, `release_handle`, `clear_registry`, then `s1_reach` /
  `s1_pick` / `s1_walk_to`, `s2_move_joint` / `s2_track_trajectory`,
  `s3_add_posture_task` / `s3_set_task_priority` / `s3_step_wbc`,
  `s4_read_encoder` / `s4_read_velocity` / `s4_apply_torque` / `s4_joint_names`.
- **Oracle verbs** — `create_twin_handle(unit, serial) -> "@twin-<hash>"`,
  `twin_info`, `list_twin_handles`, `release_twin_handle`,
  `clear_twin_registry`, then `oracle_simulate_plan`, `oracle_what_if`,
  `oracle_identify_payload` (C-131). Every answer carries the twin's
  attestation handle and per-constraint margins with their band; an
  identification also carries its transport (`loopback-simulated` vs a real
  replay), so a simulated answer cannot read as a measurement.
- `ping` — health check.

## Pitfalls & idioms (from consumer sessions)

- **`sysid.recipes.identifiability_from_jacobian` is generic.** It's a plain
  SVD + condition-number / rank check on any `(n_obs, n_params)` sensitivity
  matrix (returns an `IdentifiabilityReport`) — not bound to jaxterity classes.
  Build the Jacobian with `jax.jacfwd` of *any* forward model and pass it in to
  answer "is my excitation rich enough?" without re-implementing the algebra.
- **Mesh-free zoo models render blank.** The vendored dynamics models (G1,
  SO-101) are inertials-only (`ngeom≈0`), so `mujoco.Renderer` on
  `robot.to_mjcf()` yields empty frames. Replay the simulated `qpos(t)` through
  the full-mesh scene (`examples/assets/<robot>_mjcf/`) with
  `jaxterity.examples._robot_render.TrajectoryRenderer` (FK only). Check `ngeom`
  first.
- **Prefer `zoo.calibrated("so100")`** (real closed-loop calibration) over the
  uncalibrated `zoo.load`. There is no LeKiwi mobile-base model — for
  mobile-manipulation, author the planar kiwi-drive base as a jaxonomy
  `LeafSystem` and compose it with the MJX arm.
- **`Twin.rollout` integrates with a fixed-step RK4.** A free-swinging arm
  diverges at `dt = 0.02` *even under zero torque* (gravity alone); use
  `dt <= 0.01` and raise `n_steps` for the horizon you want. The oracle raises
  on a diverged rollout rather than returning margins computed from `nan`, but
  a bare `rollout` will hand you the `nan`s.
- **Pass `dynamics=` on any hot loop.** `Twin.state_derivative()` builds an MJX
  model per call when you don't; build it once
  (`f = twin.robot.functional_dynamics()`) and `jax.jit` the step, or use
  `Twin.compiled_rollout(dt)`. This is the difference between microseconds and
  a large fraction of a second per step.
- **A free arm's start pose decides its tip-over margin**, not its speed. The
  zoo SO-100's zero pose is extended forward — on a mobile base that is the
  pose that tips it. Declare a tucked pose for anything ZMP-gated
  (`(0, -1.2, -0.8, 0, 0, 0)` clears the LeKiwi floor with room).

## When to escalate to the human

- Frame ambiguity that cannot be resolved from the URDF (e.g., the
  IMU mount orientation is not documented).
- Telemetry that fails schema validation in a way that suggests
  hardware miscalibration, not file format.
- Sysid residuals above the documented threshold (`SysidResult` will
  flag this; the user needs to inspect).
- Attestation handle mismatch between calibration and deployment —
  this is always a human decision; never silently regenerate.
- Anything involving safety-critical deployment (robotic surgery,
  automotive, aerospace). Jaxterity's compliance affordances are
  scaffolding, not certification.

## Where to find more

- `AGENTS/CONTEXT.md` — the contributor-facing architectural orientation.
- `AGENTS/PATTERNS.md` — coding conventions.
- `AGENTS/DECISIONS.md` — ADRs explaining why things are the way they are.
- `docs/quickstart.md` — 10-minute guided tour.
- `examples/nanojaxterity/` — the end-to-end Cartpole demo.
- The zoo entry READMEs — per-robot details, source, license.
