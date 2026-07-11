# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Stream a zoo robot's 3D state + joint signals to Rerun.

Rolls a robot forward under its own MJX dynamics and streams every step to
[Rerun](https://rerun.io) via :class:`jaxterity.diagnostics.RobotRerunLogger`:
the articulated body in 3D (per-link transforms + the kinematic skeleton) plus
per-joint position/velocity/torque time series you can scrub and inspect.

Examples
--------
    # Live viewer (spawns Rerun):
    python examples/rerun_robot_demo.py --robot unitree_g1
    python examples/rerun_robot_demo.py --robot crazyflie   # hovers, then tilts
    python examples/rerun_robot_demo.py --robot so100

    # Headless: write an .rrd to open later (rerun your_file.rrd):
    python examples/rerun_robot_demo.py --robot unitree_g1 --save g1.rrd

Any robot but the Crazyflie is dropped under gravity (zero control); the
Crazyflie flies on its thrust+moment wrench (hover, then a brief pitch).
"""

from __future__ import annotations

import argparse

import numpy as np

from jaxterity import zoo
from jaxterity.diagnostics import log_robot_trajectory

_DT = 0.01
_STEPS = 200


def _require_rerun():
    try:
        import rerun  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            'rerun-sdk is not installed. Install with `pip install "jaxterity[rerun]"`.'
        ) from e


def _rollout(robot, *, floating: bool, crazyflie: bool):
    """Fixed-step RK4 rollout, returning per-step (qs, qds, taus)."""
    import jax
    import jax.numpy as jnp

    f = robot.functional_dynamics()
    diagram, sysm = robot.to_diagram(return_system=True)
    nq, nv = sysm._nq, sysm._nv

    # Initial state: model default (qpos0 keeps a free joint's quaternion valid).
    state0 = np.asarray(diagram.create_context()[sysm.system_id].continuous_state)

    if crazyflie:
        # Hover wrench → generalized force; add a brief pitch torque mid-flight.
        base = zoo.crazyflie.hover_control()

        def control(state, k):
            u = base.at[2].set(0.0008 * jnp.sin(0.05 * k))  # small pitch moment My
            return zoo.crazyflie.control_to_generalized_force(state, u)
    else:

        def control(state, k):
            return jnp.zeros(nv)

    @jax.jit
    def step(state, k):
        def deriv(s):
            return f(s, control(s, k))

        k1 = deriv(state)
        k2 = deriv(state + 0.5 * _DT * k1)
        k3 = deriv(state + 0.5 * _DT * k2)
        k4 = deriv(state + _DT * k3)
        return state + _DT / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    state = jnp.asarray(state0)
    qs, qds, taus = [], [], []
    for k in range(_STEPS):
        qs.append(np.asarray(state[:nq]))
        qds.append(np.asarray(state[nq : nq + nv]))
        taus.append(np.asarray(control(state, k)))
        state = step(state, k)
    return np.array(qs), np.array(qds), np.array(taus)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot",
        default="unitree_g1",
        choices=["cartpole", "so100", "unitree_g1", "crazyflie"],
    )
    parser.add_argument(
        "--save", default=None, help="write an .rrd here instead of spawning"
    )
    args = parser.parse_args()
    _require_rerun()

    crazyflie = args.robot == "crazyflie"
    floating = args.robot in ("crazyflie",)
    if args.robot == "unitree_g1":
        robot = zoo.unitree_g1.load(floating_base=True)
        floating = True
    else:
        robot = zoo.load(args.robot)

    qs, qds, taus = _rollout(robot, floating=floating, crazyflie=crazyflie)
    log_robot_trajectory(
        robot,
        qs,
        qds=qds,
        taus=taus,
        dt=_DT,
        namespace=args.robot,
        spawn=args.save is None,
        save_path=args.save,
    )
    if args.save:
        print(f"wrote {args.save} ({len(qs)} steps) — open with `rerun {args.save}`")


if __name__ == "__main__":
    main()
