# Unitree G1 — render assets (tutorial visualization only)

Full-mesh MuJoCo model used **only to render** jaxterity-produced trajectories in
the tutorials (the dynamics run on jaxterity's mesh-free
`jaxterity/zoo/unitree_g1/g1.xml`). Post-hoc rendering writes `qpos(t)` into this
model and calls `mj_forward` — MuJoCo does no physics here.

- **Source:** [`google-deepmind/mujoco_menagerie`](https://github.com/google-deepmind/mujoco_menagerie)
  `unitree_g1/` (`g1.xml` + `scene.xml`), derived upstream from
  `unitreerobotics/unitree_ros` `g1_29dof_rev_1_0.xml`.
- **License:** BSD-3-Clause (© Unitree Robotics) — see `LICENSE`.
- **Modifications:** the referenced visual STL meshes are **quadric-decimated to
  ≤ 8 000 faces each** (19 MB → 9 MB) — invisible at GIF/MP4 resolution; the
  `.xml` mesh references and kinematics are unchanged. `scene.xml` adds
  `offwidth/offheight` + a raking key light for offscreen rendering.
