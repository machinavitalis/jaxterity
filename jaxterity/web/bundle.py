# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Write a static, browser-runnable web bundle for a Jaxterity demo.

The bundle is a directory of plain files — no build step, no server beyond a
static host:

    <out>/
      index.html        # the page (2D canvas render + controls)
      style.css
      policy.js         # the trained policy, ported to JS (parity-tested)
      physics.js        # closed-form integrator + MuJoCo-WASM backend loader
      app.js            # wiring: backend select, control loop, canvas draw
      policy.json       # exported policy weights + physics params (this module)
      cartpole.xml      # the renderable MJCF, for the MuJoCo-WASM backend
      MUJOCO_WASM.md    # how to enable the full-fidelity MuJoCo-WASM backend

``policy.json`` is the only data file this module computes; the rest are static
template assets copied verbatim from :data:`TEMPLATE_DIR`. The template reads
*everything* demo-specific (the network weights, the action set, the physics
constants, the failure bounds) from ``policy.json``, so the same page renders
any cartpole-shaped policy without editing JavaScript.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

#: Static template assets copied into every bundle.
TEMPLATE_DIR = Path(__file__).with_name("template")

#: Files copied verbatim from :data:`TEMPLATE_DIR` into the bundle.
_STATIC_FILES = (
    "index.html",
    "style.css",
    "policy.js",
    "physics.js",
    "app.js",
    "MUJOCO_WASM.md",
)


def mlp_to_dict(mlp, *, activation: str = "tanh") -> dict:
    """Serialise an ``equinox.nn.MLP`` to a JSON-friendly dict.

    Args
    ----
    mlp : equinox.nn.MLP
        The network. Each ``layers[i]`` must expose ``weight`` (``out, in``) and
        ``bias`` (``out,``). ``activation`` names the nonlinearity applied
        between layers (the final layer is linear).

    Returns
    -------
    dict
        ``{"activation", "layers": [{"W": [[...]], "b": [...]}, ...]}`` with the
        weights as nested Python floats (``W[i]`` is row ``i`` of the matrix, so
        ``y = W @ x + b`` in JS is ``y[i] = dot(W[i], x) + b[i]``).
    """
    layers = []
    for layer in mlp.layers:
        layers.append(
            {
                "W": np.asarray(layer.weight, dtype=np.float64).tolist(),
                "b": np.asarray(layer.bias, dtype=np.float64).tolist(),
            }
        )
    return {"activation": activation, "layers": layers}


def cartpole_payload(robot, actor_mlp, cfg, *, reduced_params) -> dict:
    """Build the ``policy.json`` payload for the nanojaxterity cartpole demo.

    Args
    ----
    robot : Robot
        The calibrated cartpole (read for the joint damping that the closed-form
        backend uses so the browser physics matches what the policy trained on).
    actor_mlp : equinox.nn.MLP
        The trained actor network (state → action logits).
    cfg : Config
        The demo config (``examples/nanojaxterity.py``'s ``Config``) — supplies
        ``OBS_SCALE``/``ACTIONS`` are read from the module, while ``force``,
        ``dt``, ``n_substep`` and the failure bounds come from here.
    reduced_params : dict
        ``{"g", "mc", "mp", "L"}`` from ``zoo.cartpole.reduced_params`` — the
        closed-form point-mass reduction.

    Returns
    -------
    dict
        The full payload written to ``policy.json``.
    """
    params = robot.parameters()
    cart_damping = float(params.get("slider_to_cart.damping", 0.0))
    pole_damping = float(params.get("cart_to_pole.damping", 0.0))
    return {
        "schema": "jaxterity.web/cartpole-actorcritic/1",
        "obs_scale": [float(s) for s in cfg["obs_scale"]],
        "actions": [float(a) for a in cfg["actions"]],
        "force": float(cfg["force"]),
        "mlp": mlp_to_dict(actor_mlp),
        "physics": {
            "g": float(reduced_params["g"]),
            "mc": float(reduced_params["mc"]),
            "mp": float(reduced_params["mp"]),
            "L": float(reduced_params["L"]),
            "cart_damping": cart_damping,
            "pole_damping": pole_damping,
            "dt": float(cfg["dt"]),
            "n_substep": int(cfg["n_substep"]),
            "theta_fail": float(cfg["theta_fail"]),
            "x_fail": float(cfg["x_fail"]),
        },
        "reset_theta": float(cfg["reset_theta"]),
    }


def write_bundle(
    out_dir: str | Path,
    *,
    mjcf: str,
    payload: dict,
    template_dir: str | Path | None = None,
) -> Path:
    """Write a complete web bundle to ``out_dir`` (created/overwritten).

    Args
    ----
    out_dir : str or Path
        Destination directory; replaced if it already exists.
    mjcf : str
        Renderable MJCF for the MuJoCo-WASM backend, written as ``cartpole.xml``.
    payload : dict
        The ``policy.json`` payload (see :func:`cartpole_payload`).
    template_dir : str or Path, optional
        Override the static-asset source (defaults to :data:`TEMPLATE_DIR`).

    Returns
    -------
    Path
        ``out_dir`` as a :class:`pathlib.Path`.
    """
    out_dir = Path(out_dir)
    src = Path(template_dir) if template_dir is not None else TEMPLATE_DIR
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    for name in _STATIC_FILES:
        shutil.copyfile(src / name, out_dir / name)
    (out_dir / "cartpole.xml").write_text(mjcf)
    (out_dir / "policy.json").write_text(json.dumps(payload, indent=2) + "\n")
    return out_dir


__all__ = ["TEMPLATE_DIR", "mlp_to_dict", "cartpole_payload", "write_bundle"]
