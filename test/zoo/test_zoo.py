# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Reference-robot-zoo regression tests (CONTEXT.md invariant 7).

Every zoo robot must import, simulate a free-fall trajectory, and match its
committed golden bit-exactly (PATTERNS.md §6.6). Regenerate goldens with
``JAXTERITY_REGEN_GOLDEN=1 pytest test/zoo`` after a deliberate, justified change.
"""

from __future__ import annotations

import os

import jaxonomy
import numpy as np
import pytest
from jaxonomy import SimulatorOptions

from jaxterity import zoo

pytest.importorskip("mujoco")

_REGEN = bool(os.environ.get("JAXTERITY_REGEN_GOLDEN"))

# Per-robot free-fall initial conditions (a small perturbation so the robot
# moves under gravity). q/qd lengths match each robot's DOF count.
_SCENARIOS = {
    "cartpole": {"q0": [0.0, 0.1], "qd0": [0.0, 0.0], "t": 1.0},
    "so100": {"q0": [0.0] * 6, "qd0": [0.0] * 6, "t": 0.5},
    # Unitree G1: fixed-base (welded pelvis) 29-DoF humanoid. A small
    # left-knee flexion breaks symmetry so the legs/arms swing under gravity.
    "unitree_g1": {
        "q0": [0.0, 0.0, 0.0, 0.2] + [0.0] * 25,
        "qd0": [0.0] * 29,
        "t": 0.3,
    },
    # Crazyflie: floating-base nano-quadrotor, a single free body
    # (nq == 7: position + unit quaternion; nv == 6). Released from rest with
    # no thrust, it free-falls. q0 = [x y z, quat(w,x,y,z)].
    "crazyflie": {
        "q0": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "qd0": [0.0] * 6,
        "t": 0.3,
    },
}

_OPTS = SimulatorOptions(
    math_backend="jax", ode_solver_method="rk4", max_minor_step_size=0.01
)


def _simulate_freefall(name: str) -> np.ndarray:
    robot = zoo.load(name)
    sc = _SCENARIOS[name]
    diagram, sysm = robot.to_diagram(q0=sc["q0"], qd0=sc["qd0"], return_system=True)
    res = jaxonomy.simulate(
        diagram, diagram.create_context(), (0.0, sc["t"]), options=_OPTS
    )
    return np.asarray(res.context[sysm.system_id].continuous_state)


def _golden_path(name: str):
    return zoo.ROBOTS[name].URDF_PATH.with_name("golden_freefall.npz")


def test_all_zoo_robots_have_scenarios():
    """Every registered robot has a free-fall scenario (so it is tested)."""
    assert set(zoo.available()) <= set(_SCENARIOS)


@pytest.mark.parametrize("name", zoo.available())
def test_zoo_freefall_is_nontrivial(name):
    """The free-fall trajectory must actually move under gravity.

    Regression for the SO-100 entry, which was silently motionless (a dropped
    joint-origin rotation left every axis along gravity). A dead all-zero
    trajectory would otherwise satisfy the golden check vacuously, so guard the
    dynamics directly: the final configuration must differ from the initial one.
    """
    sc = _SCENARIOS[name]
    out = _simulate_freefall(name)
    # The configuration is the first nq entries; nq == len(q0) by construction
    # (== len(out)//2 for the fixed-base nq == nv robots, but a floating base
    # has nq == nv + 1, so split on q0's length, not the array's midpoint).
    nq = len(sc["q0"])
    moved = float(np.max(np.abs(out[:nq] - np.asarray(sc["q0"]))))
    assert moved > 1e-3, f"{name}: did not move under gravity (max |dq|={moved:.2e})"


@pytest.mark.parametrize("name", zoo.available())
def test_zoo_freefall_matches_golden(name):
    out = _simulate_freefall(name)
    assert np.all(np.isfinite(out)), f"{name}: non-finite free-fall trajectory"

    path = _golden_path(name)
    if _REGEN:
        np.savez(path, final_state=out)
        pytest.skip(f"regenerated golden for {name}")
    golden = np.load(path)["final_state"]
    # bit-exact same-device regression (PATTERNS §6.6 determinism)
    np.testing.assert_array_equal(out, golden)
