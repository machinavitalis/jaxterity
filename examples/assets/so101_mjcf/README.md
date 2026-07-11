# SO-101 MuJoCo model (vendored, tutorial-only)

This directory contains a verbatim copy of the SO-101 MuJoCo description from
[TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101),
plus a small `scene_showcase.xml` wrapper that adds tutorial lighting / floor / camera.

- **Upstream:** `TheRobotStudio/SO-ARM100` @ `main`
- **License:** Apache-2.0 (see <https://github.com/TheRobotStudio/SO-ARM100/blob/main/LICENSE>)
- **Used by:** `examples/render_tutorial_gifs.py` — renders the SO-101 GIFs for
  `examples/calibrate_so101.ipynb`. Not used by the Jaxterity simulation path;
  Jaxterity loads its own URDF from `jaxterity/zoo/so100/so101.urdf` (also
  Apache-2.0, same upstream).

Files vendored verbatim:
- `so101_new_calib.xml` — joint chain, inertials, mesh references
- `assets/*.stl` — visual + collision meshes referenced by the MJCF

`scene_showcase.xml` is a Jaxterity-authored wrapper (MIT, in keeping with the
rest of this repo) that `<include>`s `so101_new_calib.xml` and adds a showcase
camera + lights.
