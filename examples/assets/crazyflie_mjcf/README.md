# Crazyflie 2.0 — render assets (tutorial visualization only)

Full-mesh MuJoCo model used **only to render** jaxterity-produced trajectories in
the tutorials (the dynamics run on jaxterity's mesh-free
`jaxterity/zoo/crazyflie/cf2.xml`). Post-hoc rendering writes `qpos(t)` into this
model and calls `mj_forward` — MuJoCo does no physics here.

- **Source:** [`google-deepmind/mujoco_menagerie`](https://github.com/google-deepmind/mujoco_menagerie)
  `bitcraze_crazyflie_2/` (`cf2.xml` + `scene.xml` + `assets/*.obj`).
- **License:** MIT (© 2024 whoenig) — see `LICENSE`.
- **Modifications:** none to the meshes; `scene.xml` adds `offwidth/offheight`
  for offscreen rendering.
