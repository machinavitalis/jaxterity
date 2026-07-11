# Unitree G1 (zoo reference robot)

A 29-DoF humanoid from Unitree Robotics — the zoo's first biped.

```python
from jaxterity import zoo

robot = zoo.load("unitree_g1")                 # fixed-base, 29 DoF (welded pelvis)
diagram = robot.to_diagram(q0=[0.0] * 29)

humanoid = zoo.unitree_g1.load(floating_base=True)  # 6-DoF free base (ADR-018)
diagram = humanoid.to_diagram()                # default qpos0 = identity quaternion
```

- **Joints (29 revolute):** per leg — `hip_pitch`, `hip_roll`, `hip_yaw`,
  `knee`, `ankle_pitch`, `ankle_roll`; waist — `waist_yaw`, `waist_roll`,
  `waist_pitch`; per arm — `shoulder_pitch`, `shoulder_roll`, `shoulder_yaw`,
  `elbow`, `wrist_roll`, `wrist_pitch`, `wrist_yaw`.
- **Source:** [`google-deepmind/mujoco_menagerie`](https://github.com/google-deepmind/mujoco_menagerie)
  — `unitree_g1/g1.xml`, itself derived from
  [`unitreerobotics/unitree_ros`](https://github.com/unitreerobotics/unitree_ros)
  (`g1_29dof_rev_1_0.xml`).
- **License:** BSD-3-Clause (© Unitree Robotics). The upstream `LICENSE` is
  vendored alongside this README. The vendored `g1.xml` is a **mesh-free
  reduction**: the kinematic tree, joints, and explicit `<inertial>` elements
  are kept; all `<geom>`/`<mesh>`/`<site>` geometry, actuators, sensors, and
  the keyframe are stripped, because Jaxterity drives dynamics from inertials,
  not geometry (same policy as the SO-100 URDF — see `AGENTS/LIBRARIES.md`).
- **Joint armature:** the menagerie `g1.xml` carries only a single
  placeholder `armature="0.01"`; the physically-faithful per-actuator-class
  reflected rotor inertias (0.0036–0.025 kg·m²) live in
  [`mujoco_playground`](https://github.com/google-deepmind/mujoco_playground)
  (`g1_mjx_feetonly.xml`, Apache-2.0 — the same upstream pin as `g1_walk.xml`).
  They are baked back as explicit per-joint `armature` attributes (the
  mesh-free reduction had collapsed them to one default). `frictionloss` stays
  the menagerie uniform `0.3`; `damping` stays `0` (this is scoped to
  armature). Honored by the importer.

## Body topologies

| `load(...)` | Base | `nq` / `nv` | Use |
|---|---|---|---|
| `load()` | pelvis welded to world | 29 / 29 | zoo `nq == nv` free-fall regression, fixed-base experiments |
| `load(floating_base=True)` | `FLOATING` joint (world → pelvis) | 36 / 35 | whole-body control, locomotion, anything needing a free base |
| `load(floating_base=True, contact=True)` | free base + foot boxes + floor | 36 / 35 | foot–ground contact (drop tests, contact-aware control) |

The floating base is reconstructed in code (a single `JointType.FLOATING`
joint) because the MJCF importer doesn't yet round-trip the upstream
`<freejoint>` (ADR-018). The contact form attaches one collision box per
ankle-roll link (pose/size matching the MuJoCo Playground locomotion model)
plus a world floor plane (`Robot.ground_plane`).

## Locomotion (vendored walking policy)

`jaxterity.locomotion.G1JoystickController` walks the G1 with a **pretrained
joystick policy** from
[`google-deepmind/mujoco_playground`](https://github.com/google-deepmind/mujoco_playground)
(Apache-2.0 — see `LICENSE-playground`), vendored at a pinned commit by
`scripts/vendor_g1_walk_scene.py`:

- `g1_walk.xml` — the mesh-free locomotion scene the policy was trained
  against (position actuators, per-class armature/damping, foot boxes + floor
  + contact pairs, pelvis IMU sensors, `home`/`knees_bent` keyframes).
- `g1_policy.npz` — the policy MLP weights, reimplemented in pure JAX.
- `g1_policy_golden.npz` — onnxruntime I/O pairs; the parity regression.

```python
from jaxterity.locomotion import G1JoystickController

controller = G1JoystickController()
trajectory = controller.rollout(command=[0.5, 0.0, 0.0], n_steps=100)  # 2 s walk
```

## Status

- ✅ Imports (`from_mjcf`), simulates via `to_diagram()`. Fixed-base free-fall
  golden committed (`golden_freefall.npz`); floating-base free-fall verified
  ballistic (½·g·t², unit-norm quaternion) in `test/zoo/test_unitree_g1.py`.
- ✅ **Foot–ground contact** (`contact=True`): Playground-matched foot boxes +
  floor; contact-arrested drop covered in `test/zoo/test_unitree_g1.py`.
- ✅ **Locomotion**: vendored pretrained joystick policy walks the vendored
  scene (forward walk, stand-in-place, and a drifted-plant walk covered in
  `test/locomotion/test_g1_joystick.py`).
- ⏳ **Uncalibrated.** No `calibrated.json` yet — `zoo.calibrated("unitree_g1")`
  raises until a sysid recipe is run against real telemetry and committed.
- ⏳ **Locomotion WBC** (model-based gait, vs the vendored learned policy) and
  the Cortex-A deployment lane stay follow-on work.
