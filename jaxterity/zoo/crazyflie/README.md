# Crazyflie 2.0 (zoo reference robot)

A 27 g nano-quadrotor from Bitcraze — the zoo's first flyer.

```python
from jaxterity import zoo
import jax.numpy as jnp

drone = zoo.load("crazyflie")          # floating-base free body (nq=7, nv=6)

f = zoo.crazyflie.thrust_dynamics()    # f(state, u) -> dstate, jit/vmap-able
state = jnp.array([0, 0, 0.1, 1, 0, 0, 0,   0, 0, 0, 0, 0, 0.0])  # level, at rest
u = zoo.crazyflie.hover_control()      # [m·g, 0, 0, 0] — thrust at weight
dstate = f(state, u)                   # vertical accel ≈ 0 (hover equilibrium)
```

- **Body:** a single rigid body (mass 0.027 kg) on a 6-DoF `FLOATING` base.
  A quadrotor has no meaningful fixed-base form (a welded body has zero DOF), so
  `load()` is floating-base only. `state = [x y z, quat(w,x,y,z), v(3), ω(3)]`.
- **Control:** a body **wrench** `u = [thrust (N), Mx, My, Mz (N·m)]` — total
  thrust along body +z plus three body moments, exactly the lumped model the
  upstream MJCF uses (a `body_thrust` + `x/y/z_moment` actuator at a site).
  `control_to_generalized_force` maps it to the free base's generalized force:
  the thrust is rotated into the world frame by the current attitude, the
  moments are already body-frame (the free joint's rotational coordinate).
- **Source:** [`google-deepmind/mujoco_menagerie`](https://github.com/google-deepmind/mujoco_menagerie)
  — `bitcraze_crazyflie_2/cf2.xml`.
- **License:** MIT (© 2024 whoenig). The upstream `LICENSE` is vendored
  alongside this README. The vendored `cf2.xml` is a **mesh-free reduction**:
  the single body + its explicit `<inertial>` are kept; all geometry, sites,
  actuators, and the upstream `<freejoint>` are stripped (Jaxterity drives
  dynamics from inertials — see `AGENTS/LIBRARIES.md`).

## Status

- ✅ Imports (`from_mjcf` + reconstructed floating base), simulates via
  `to_diagram()`. Free-fall golden committed (`golden_freefall.npz`); the
  thrust+moment wrench model is verified in `test/zoo/test_crazyflie.py`: hover
  thrust cancels gravity, thrust tilts with attitude (`m·g·sin θ` lateral),
  closed-loop hover holds altitude to <1 mm, and the dynamics jit/vmap.
- ⏳ **Uncalibrated.** No `calibrated.json` yet — `zoo.calibrated("crazyflie")`
  raises until a sysid recipe is run against real telemetry and committed.
- ⏳ **No four-rotor allocation / controller.** The body-wrench model is the
  flight-dynamics core; mapping individual rotor speeds → wrench (needs per-rotor
  thrust/drag coefficients) and a hover/trajectory NMPC are follow-on work, as
  is the STM32H7 deployment lane via Jaxility.
