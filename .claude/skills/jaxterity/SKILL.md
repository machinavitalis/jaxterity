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

## MCP tools available

If running under MCP, the following tools are exposed by the
`jaxterity.mcp` server:

- `load_robot(path) -> @robot[<hash>]`
- `load_telemetry(path) -> @telemetry[<hash>]`
- `calibrate(@robot, @telemetry, recipe) -> @fit[<hash>]`
- `apply_fit(@robot, @fit) -> @robot[<new_hash>]`
- `simulate(@robot, scenario) -> @trajectory[<hash>]`
- `parity_report(@trajectory_sim, @trajectory_real) -> ParityReport`
- `export_handle(@robot) -> attestation_handle`  (handoff to Jaxility)

Use the `@<type>[<hash>]` handle pattern instead of re-passing
artifacts. The MCP server caches them server-side.

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
