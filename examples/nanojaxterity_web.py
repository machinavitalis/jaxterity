# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""nanojaxterity_web — package the cartpole demo as a browser-runnable bundle.

Takes the *committed* nanojaxterity policy and the calibrated cartpole and writes
a static web bundle (HTML + JS + the policy as JSON + the renderable MJCF) that
runs the trained controller **in the browser**, no install required:

1. Load ``zoo.calibrated("cartpole")`` with the demo's physical override
   (``nanojaxterity.calibrated_cartpole``) and the committed actor policy.
2. Build a renderable MJCF (scene geoms + camera) for the MuJoCo-WASM backend.
3. Export the actor network, the action set, and the closed-form physics
   constants (``zoo.cartpole.reduced_params`` + the calibrated joint damping)
   into ``policy.json``.
4. Write the bundle with :func:`jaxterity.web.bundle.write_bundle`.

Run::

    python -m examples.nanojaxterity_web                 # -> *_web_dist/
    python -m examples.nanojaxterity_web --out /tmp/cp   # custom dir

Then serve it (ES modules + fetch need HTTP, not file://)::

    python -m http.server -d examples/nanojaxterity_web_dist 8000
    # open http://localhost:8000

The bundle is regenerable and gitignored; the static templates it is built from
live in ``jaxterity/web/template/``.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import equinox as eqx
import jax

import jaxterity  # noqa: F401  (import the package first; submodule-first import is fragile)
from jaxterity.web import bundle
from jaxterity.zoo import cartpole

_HERE = Path(__file__).resolve().parent
DEFAULT_OUT = _HERE / "nanojaxterity_web_dist"


def _load_nanojaxterity():
    """Load the sibling ``nanojaxterity.py`` as a module (it is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "nanojaxterity_demo", _HERE / "nanojaxterity.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def build(out_dir: Path) -> Path:
    """Build the cartpole web bundle at ``out_dir``; return the path."""
    nj = _load_nanojaxterity()
    render = _load_render()

    robot = nj.calibrated_cartpole()
    cfg = nj.Config()
    model = eqx.tree_deserialise_leaves(
        nj.POLICY_PATH, nj.ActorCritic(jax.random.PRNGKey(0))
    )
    mjcf = render.build_scene_mjcf(robot)
    payload = bundle.cartpole_payload(
        robot,
        model.actor,
        {
            "obs_scale": nj.OBS_SCALE,
            "actions": nj.ACTIONS,
            "force": cfg.force,
            "dt": cfg.dt,
            "n_substep": cfg.n_substep,
            "theta_fail": cfg.theta_fail,
            "x_fail": cfg.x_fail,
            "reset_theta": cfg.eval_start_theta,
        },
        reduced_params=cartpole.reduced_params(robot),
    )
    return bundle.write_bundle(out_dir, mjcf=mjcf, payload=payload)


def _load_render():
    spec = importlib.util.spec_from_file_location(
        "nanojaxterity_render", _HERE / "nanojaxterity_render.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    args = ap.parse_args()
    out = build(args.out)
    print(f"wrote browser bundle → {out}")
    print(
        f"serve it:  python -m http.server -d {out} 8000   # then open http://localhost:8000"
    )


if __name__ == "__main__":
    main()
