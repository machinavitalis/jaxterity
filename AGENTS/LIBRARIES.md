# Jaxterity — Upstream Libraries, Attribution, and License Analysis

Jaxterity composes existing robotics and JAX libraries rather than
re-implementing physics (ADR-002). This file records, for every upstream
project Jaxterity depends on or borrows from: its role, how Jaxterity uses it
(runtime dependency, vendored code, or reference only), its reported license,
and the license-compatibility note for Jaxterity's MIT distribution.

## Methodology and caveat

- Licenses below were checked against each project's public repository on the
  date noted and are recorded as **reported**. Open-source licenses can change
  between releases. **Before any upstream code is vendored or copied into
  Jaxterity (as opposed to merely imported), re-confirm the license against the
  pinned version's `LICENSE`/`NOTICE` files** and record the exact commit/tag.
- "Incorporation mode" is the load-bearing column:
  - **dependency** — imported at runtime; we ship no upstream code. License
    compatibility is trivial (permissive licenses impose no obligation on a
    mere importer).
  - **vendored** — upstream source is copied into Jaxterity. The upstream
    license and notices travel with the copied files; this is where
    compatibility actually matters.
  - **reference** — used only as an algorithm/spec reference for clean-room
    implementation. No code is taken; no license obligation, but attribution
    is courteous and recorded here.
- pyroki and mujoco_playground licenses were verified directly; others are
  marked *reported — confirm at integration*.

## Permissive-license compatibility primer

MIT, BSD-2-Clause, and BSD-3-Clause are permissive and combine freely with
Jaxterity's MIT license; vendoring requires retaining the upstream copyright
and license text in the copied files. Apache-2.0 is also compatible to combine,
but vendoring Apache-2.0 code additionally requires preserving `NOTICE` content
and is one-directional (Apache-2.0 files keep their license; they are not
relicensed to MIT). None of the libraries below are copyleft (GPL/AGPL/LGPL);
if a future dependency is, it must be flagged in an ADR before adoption.

## Runtime dependencies and vendored sources

| Project | Role in Jaxterity | Incorporation | Upstream | License | Note |
|---------|-------------------|---------------|----------|---------|------|
| **MuJoCo / MJX** | **Dynamics backend** (forward dynamics) + contact/collision (ADR-016) | **core** dependency | google-deepmind/mujoco | Apache-2.0 *(reported)* | Patent grant + NOTICE; importer only. Parses MJCF strings directly (no Gazebo). **Reverse-mode caveat:** MJX's iterative constraint solver (active joint limits / contacts) runs under a `lax.while_loop`, which `jax.grad` cannot traverse — reverse-mode sysid needs the constraint-free regime (unlimited joints, no active contact) or forward-mode/BDF-adjoint. |
| **PyRoki** | URDF→JAX forward kinematics, collision primitives, IK | install-from-source only — **not on PyPI** (PyPI forbids `git+` deps) and **not a declared extra**; zero callers today, so `pip install "pyroki @ git+https://github.com/chungmin99/pyroki.git"` if the IK use case lands | **chungmin99/pyroki** | **MIT** *(verified)* | MIT → fully compatible to vendor into MIT; keep upstream copyright header on any copied file. NB the repo is `chungmin99/pyroki`, not `brentyi/pyroki`. **Not yet integrated.** FK and contact/collision are already covered by MJX (the dynamics backend, ADR-016). The remaining gap pyroki would close is **inverse kinematics** — both basic Cartesian IK (for the reach Cartesian-target follow-up) and differentiable-IK-through-implicit-function-theorem (no current consumer). Basic IK is reimplementable in ~30-50 lines on top of `mjx.jac_site_translation`; reuse only wins decisively for the differentiable / collision-aware variants. Re-evaluate when the first IK caller actually lands. |
| **Rerun** | Sim/real parity adapter + robot-agnostic 3D state streaming / signal inspection | dependency (`[rerun]`, `rerun-sdk`) | rerun-io/rerun | Apache-2.0 / MIT (dual) *(reported — confirm)* | Visualization only; not on the simulation/attestation path. `rerun-sdk` import stays lazy, so it's strictly optional. |
| **FastMCP** | MCP server framework | dependency (`[mcp]`) | jlowin/fastmcp | Apache-2.0 *(reported — confirm)* | Importer only. |
| **LeRobot** | `LeRobotDataset` read/write interop for the SO-100 nanoLeRobot demo | opt-in extra (`[lerobot]`); deliberately **not in `[all]`** (heavy `torch`+`datasets` tree) | huggingface/lerobot | Apache-2.0 *(reported — confirm)* | **Format interop, not code vendoring.** `jaxterity.integrations.lerobot` writes/reads the v2.1 on-disk schema with `pyarrow`; the `lerobot` package is only imported (when present) to load a written dataset through the genuine `LeRobotDataset`. No LeRobot source is copied. See ADR-019. |

## Reference-only influences (no code incorporated)

These shape Jaxterity's design (per CONTEXT.md "Architectural heritage") but no
source is taken; clean-room implementation.

| Project | What we take (concept) | Upstream | License *(reported)* |
|---------|------------------------|----------|----------------------|
| **MuJoCo Playground** | Domain-randomization patterns, sensor-noise modeling, PD defaults | google-deepmind/mujoco_playground | Apache-2.0 *(verified)* |
| **Pink / Mink** | Weighted task-hierarchy formulation for WBC | stephane-caron/pink, kevinzakka/mink | Apache-2.0 *(reported — confirm)* |
| **WPILib SysId** | Motor model `V = Ks·sgn(ẋ) + Kv·ẋ + Ka·ẍ` and the data-collection procedure | wpilibsuite/allwpilib + sysid | BSD-3-Clause *(reported — confirm)* |
| **Drake** | Trajectory-optimization problem structures, contact formulations (validation reference) | RobotLocomotion/drake | BSD-3-Clause *(reported — confirm)* |
| **Pinocchio** | Algorithm reference + ground-truth validation | stack-of-tasks/pinocchio | BSD-2-Clause *(reported — confirm)* |
| **REX / supergraph** | Async-graph patterns for sim-to-real latency/delay modeling (concepts only — implementation fresh) | bheijden/rex, bheijden/supergraph | *to confirm at integration* |

## Vendored data assets (reference robot zoo)

URDFs are vendored verbatim under `jaxterity/zoo/<robot>/` (mesh files are NOT
vendored for Jaxterity simulation — geometry is unused; inertials drive the
dynamics). The tutorial GIFs additionally vendor the **upstream SO-101 MuJoCo
MJCF + STL meshes** under `examples/assets/so101_mjcf/` so the renderer can
show the real arm rather than capsule proxies. Each upstream is permissive and
compatible with MIT redistribution with attribution.

| Robot | Asset | Upstream | License *(verified)* |
|-------|-------|----------|----------------------------------|
| **Cartpole** | `cartpole.urdf` | bulletphysics/bullet3 (`pybullet_data`) | **zlib** |
| **SO-100 / SO-101** | `so101.urdf` (zoo) + `so101_new_calib.xml` + `assets/*.stl` (tutorial render only) | TheRobotStudio/SO-ARM100 | **Apache-2.0** |
| **Unitree G1** | `g1.xml` (zoo, mesh-free reduction) + vendored `LICENSE` | google-deepmind/mujoco_menagerie `unitree_g1` ← unitreerobotics/unitree_ros `g1_29dof_rev_1_0.xml` | **BSD-3-Clause** (© Unitree Robotics) |
| **Crazyflie 2.0** | `cf2.xml` (zoo, mesh-free reduction) + vendored `LICENSE` | google-deepmind/mujoco_menagerie `bitcraze_crazyflie_2` | **MIT** (© 2024 whoenig) |
| **G1 / Crazyflie — render meshes** | `examples/assets/{unitree_g1_mjcf,crazyflie_mjcf}/` full-mesh models + `LICENSE` (**tutorial visualization only**) | google-deepmind/mujoco_menagerie (same upstreams) | **BSD-3-Clause** / **MIT** |

The G1 is the first biped and the Crazyflie the first flyer to
land. As with the SO-100, meshes are not vendored: each `.xml` is reduced to the
kinematic tree, joints, and `<inertial>` elements so it compiles without the
upstream meshes. BSD-3-Clause (G1) and MIT (Crazyflie) both vendor into MIT with
attribution — each upstream `LICENSE` travels in its `jaxterity/zoo/<robot>/`
directory, and the reduction provenance is recorded in a header comment at the
top of each vendored `.xml`.

For the **tutorial GIFs**, the *full-mesh* menagerie models are
additionally vendored under `examples/assets/{unitree_g1_mjcf,crazyflie_mjcf}/`
(same upstreams + licenses) — used only to post-hoc-render jaxterity-produced
trajectories with real geometry (the dynamics still run on the mesh-free zoo
models). The G1 visual STLs are quadric-decimated to ≤ 8 000 faces (19 MB → 9 MB;
invisible at GIF resolution); each directory carries its upstream `LICENSE` and a
`README.md` recording the provenance + modifications. This mirrors the SO-101
render assets already vendored under `examples/assets/so101_mjcf/`.

Both use the floating base (ADR-018, a `FLOATING` joint → MuJoCo `<freejoint>`),
reconstructed in code because the importers don't yet round-trip free joints
(`from_urdf`/`from_mjcf` weld them; export only today). The Crazyflie's rotors
enter as a **body thrust+moment wrench** mapped to the free base's generalized
force (`crazyflie.thrust_dynamics`), mirroring the upstream lumped actuator
model. What remains follow-on is the rest of each *full* demo — foot–ground
contact and locomotion control for the G1; four-rotor control allocation and a
hover/trajectory controller for the Crazyflie (geometry is stripped in both).

## Action items

- Confirm the *(reported — confirm)* rows against upstream `LICENSE` files when
  the corresponding capability is first implemented, and replace the
  caveat with a verified commit/tag.
- If any PyRoki code is vendored (ADR-002), copy the upstream MIT header into
  the vendored files and add a `THIRD_PARTY_NOTICES` entry.
- Resolve the REX/supergraph license before any pattern is implemented closely
  enough to risk derivation.
