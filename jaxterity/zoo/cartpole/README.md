# Cartpole (zoo reference robot)

The minimum-viable zoo entry: a cart on a horizontal slider carrying a
freely-swinging pole. Use it to learn the Jaxterity workflow.

```python
from jaxterity import zoo
robot = zoo.load("cartpole")          # uncalibrated Robot
diagram = robot.to_diagram(q0=[0.0, 0.1])   # cart at 0, pole 0.1 rad off vertical
```

- **Structure:** `slideBar` (fixed root) → `cart` (prismatic, x) → `pole`
  (continuous hinge, y). Continuous-state layout `[cart_x, pole_theta,
  cart_xdot, pole_thetadot]`.
- **Source:** [`bulletphysics/bullet3`](https://github.com/bulletphysics/bullet3)
  — `examples/pybullet/gym/pybullet_data/cartpole.urdf`.
- **License:** zlib (bullet3 main license; permissive, MIT-compatible). The
  vendored `cartpole.urdf` is unmodified.

## Status

- ✅ Imports (`from_urdf`), simulates via `to_diagram()`, free-fall golden
  committed (`golden_freefall.npz`).
- ⏳ **Uncalibrated.** A calibrated parameter set + telemetry lands later.
- Known: the URDF's inertias are placeholders (unit diagonal), typical of the
  bullet demo asset — fine for the workflow, not physically calibrated.
