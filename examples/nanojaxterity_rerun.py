# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Rerun overlay for the nanojaxterity cartpole — the launch-video viz.

Rolls the trained policy out closed-loop on the calibrated cartpole and streams
it to Rerun (https://rerun.io): a live 2D cartpole (cart box, pole, commanded
force arrow) plus time-series of cart position, pole angle, and control force.
Open the recording in the Rerun viewer, arrange the panels, and screen-record
for the launch clip — this is the "3-minute Rerun overlay".

``rerun-sdk`` is an optional dependency (the ``rerun`` extra). The trajectory it
visualises is the same one the controller produces on the Pi 5; an on-target HIL
trace (``joint_position`` / ``actuator_torque`` JSON-Lines) has the same shape
and can be fed through :func:`log_trajectory` instead.

Run::

    python -m examples.nanojaxterity_rerun                 # spawn the live viewer
    python -m examples.nanojaxterity_rerun --save demo.rrd # write a recording
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import pathlib
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_DT = 0.02  # env timestep (s); matches nanojaxterity Config
_POLE_LEN = 1.0  # drawn pole length (m)
_CART_HALF = (0.18, 0.07)  # drawn cart half-extents (m)
_ARROW_SCALE = 0.05  # m per N


def _load_demo():
    spec = importlib.util.spec_from_file_location(
        "nanojaxterity_demo", _HERE / "nanojaxterity.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _require_rerun():
    try:
        import rerun as rr
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise SystemExit(
            'rerun-sdk is not installed. Install with `pip install "jaxterity[rerun]"`'
            " (or `pip install rerun-sdk`) to run the overlay."
        ) from exc
    return rr


def rollout(nj, robot, start, n_steps: int):
    """Greedy trained-policy rollout; return ``(states (n,4), forces (n,))``.

    ``forces`` is the cart force (N) the policy commanded each step — the same
    quantity the deployed controller would emit.
    """
    cfg = nj.Config()
    _, step = nj.make_env(robot, cfg)
    like = nj.ActorCritic(jax.random.PRNGKey(0))
    model = eqx.tree_deserialise_leaves(nj.POLICY_PATH, like)
    force_of = jnp.asarray(nj.ACTIONS) * cfg.force

    @eqx.filter_jit
    def run(model):
        def body(state, _):
            logits, _ = model(state)
            action = jnp.argmax(logits)
            nxt, _, _ = step(state, action)
            return nxt, (nxt, force_of[action])

        s0 = jnp.asarray(start, dtype=float)
        _, (traj, forces) = jax.lax.scan(body, s0, None, length=n_steps)
        return traj, forces

    traj, forces = run(model)
    return np.asarray(traj), np.asarray(forces)


def log_trajectory(states, forces, *, save_path=None, spawn=False) -> None:
    """Stream a cartpole ``(states (n,4), forces (n,))`` trajectory to Rerun.

    ``states`` columns are ``[cart_x, pole_theta, cart_xdot, pole_thetadot]``.
    With ``save_path`` writes a ``.rrd`` recording; with ``spawn`` opens the
    live viewer; with neither, buffers in-process (useful for tests).
    """
    rr = _require_rerun()
    rr.init("nanojaxterity-cartpole", spawn=spawn)
    if save_path is not None:
        rr.save(str(save_path))

    states = np.asarray(states, dtype=np.float64).reshape(-1, 4)
    forces = np.asarray(forces, dtype=np.float64).reshape(-1)
    for i in range(states.shape[0]):
        x, theta, x_dot, theta_dot = (float(v) for v in states[i])
        f = float(forces[i])
        rr.set_time("time", duration=i * _DT)

        # Time-series telemetry.
        rr.log("telemetry/cart_position_m", rr.Scalars(x))
        rr.log("telemetry/pole_angle_rad", rr.Scalars(theta))
        rr.log("telemetry/cart_velocity_mps", rr.Scalars(x_dot))
        rr.log("telemetry/pole_velocity_radps", rr.Scalars(theta_dot))
        rr.log("telemetry/control_force_N", rr.Scalars(f))

        # 2D cartpole: cart box, pole line (θ=0 is upright), force arrow.
        tip = [x + _POLE_LEN * math.sin(theta), _POLE_LEN * math.cos(theta)]
        rr.log("cartpole/cart", rr.Boxes2D(centers=[[x, 0.0]], half_sizes=[_CART_HALF]))
        rr.log("cartpole/pole", rr.LineStrips2D([[[x, 0.0], tip]]))
        if abs(f) > 1e-6:
            rr.log(
                "cartpole/force",
                rr.Arrows2D(origins=[[x, 0.12]], vectors=[[_ARROW_SCALE * f, 0.0]]),
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", metavar="PATH", help="write a .rrd recording")
    parser.add_argument("--steps", type=int, default=400, help="rollout steps")
    parser.add_argument(
        "--theta0", type=float, default=0.10, help="initial pole tilt (rad)"
    )
    args = parser.parse_args()

    nj = _load_demo()
    robot = nj.calibrated_cartpole()
    states, forces = rollout(nj, robot, [0.0, args.theta0, 0.0, 0.0], args.steps)
    peak = float(np.max(np.abs(states[:, 1])))
    log_trajectory(states, forces, save_path=args.save, spawn=args.save is None)
    where = args.save if args.save else "the live viewer"
    print(f"logged {len(states)} steps to {where}; max|θ|={peak:.4f} rad")


if __name__ == "__main__":
    main()
