# SO-100 / SO-101 (zoo reference robot)

A 6-DoF desktop manipulator from the SO-ARM100 project, widely used in the
LeRobot ecosystem.

```python
from jaxterity import zoo
robot = zoo.load("so100")             # uncalibrated 6-DoF arm
robot = zoo.calibrated("so100")       # with committed calibration (real fit)
diagram = robot.to_diagram(q0=[0, 0, 0, 0, 0, 0])
```

- **Joints:** `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`,
  `wrist_roll`, `gripper` (all revolute) + a fixed `gripper_frame`.
- **Source:** [`TheRobotStudio/SO-ARM100`](https://github.com/TheRobotStudio/SO-ARM100)
  — `Simulation/SO101/so101_new_calib.urdf`.
- **License:** Apache-2.0. The vendored URDF is unmodified; its `<mesh>`
  references are kept as opaque handles (the STL files are **not** vendored —
  geometry isn't needed for dynamics).

## Status

- ✅ Imports (`from_urdf`), simulates via `to_diagram()`, free-fall golden
  committed. The fixed `gripper_frame` link (zero inertia) is floored to a tiny
  valid inertia by the MJCF backend.
- ✅ **Calibrated** parameter set committed (`calibrated.json`): a real
  closed-loop friction/damping fit identified from LeKiwi telemetry
  (`recipe: active.closed_loop`).
- ✅ **nanoLeRobot demo** (`examples/nanolerobot.py`): the calibrated arm
  drives a stiff position-servo environment; a scripted expert's goal-reaching
  demonstrations are serialised to the **LeRobotDataset** format
  (`jaxterity.integrations.lerobot`) and a small behaviour-cloning policy is
  trained from them and reaches held-out goals in the calibrated sim.

## LeRobot interoperability

`observation.state` and `action` are the six joints **in LeRobot's own SO-100
motor order** (`shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll,
gripper`), so a dataset Jaxterity writes has the same column shape as a real
teleoperation recording. Install the `[lerobot]` extra to load Jaxterity-written
datasets through the genuine `LeRobotDataset`; without it, a dependency-light
reader keeps the demo and its tests runnable on just `pyarrow` (see ADR-019).
