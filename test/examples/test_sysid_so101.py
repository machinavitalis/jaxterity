# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Verifies the SO-101 sysid example actually works (shippable-surface rule).

Two tiers: a fast check that the forward-mode Levenberg-Marquardt loop recovers a
known damping on a cheap pendulum, and a nightly run of the full SO-101 demo
asserting it identifies the planted coefficients and returns a calibrated result.
"""

from __future__ import annotations

import importlib.util
import pathlib

import jax.numpy as jnp
import jaxonomy
import numpy as np
import pytest
from jaxonomy import SimulatorOptions

from jaxterity.robot import (
    Frame,
    FramedQuantity,
    Joint,
    JointLimits,
    JointType,
    Link,
    Position,
    Robot,
    Transform,
)

pytest.importorskip("mujoco")

# Load the example module by path (examples/ is not an installed package).
_DEMO_PATH = pathlib.Path(__file__).resolve().parents[2] / "examples" / "sysid_so101.py"
_spec = importlib.util.spec_from_file_location("sysid_so101_demo", _DEMO_PATH)
demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo)


def _pendulum(damping: float) -> Robot:
    base = Link(
        name="base",
        body_frame=Frame("base"),
        mass=jnp.asarray(0.1),
        com=jnp.zeros(3),
        inertia=jnp.eye(3) * 1e-3,
    )
    bob = Link(
        name="bob",
        body_frame=Frame("bob"),
        mass=jnp.asarray(1.0),
        com=jnp.array([0.0, 0.0, -1.0]),
        inertia=jnp.eye(3) * 1e-6,
    )
    pivot = Joint(
        name="pivot",
        joint_type=JointType.CONTINUOUS,
        parent_link="base",
        child_link="bob",
        axis=FramedQuantity(Position(jnp.array([0.0, 1.0, 0.0])), Frame("base")),
        limits=JointLimits(lower=-3.14, upper=3.14, effort=100.0, velocity=100.0),
        origin=Transform.identity(Frame("base")),
        damping=jnp.asarray(damping),
        friction=jnp.asarray(0.0),
    )
    return Robot(links=(base, bob), joints=(pivot,), name="pendulum")


@pytest.mark.slow  # forward-mode LM loop; compile-dominated (~90s)
def test_identify_recovers_pendulum_damping():
    """The forward-mode LM loop recovers a known damping on a cheap pendulum."""
    true_damping = 0.4
    options = SimulatorOptions(
        math_backend="jax",
        ode_solver_method="rk4",
        enable_autodiff=False,
        max_minor_step_size=0.005,
        max_major_steps=200,
    )
    sample_times = (0.3, 0.6)

    def make_observe(robot):
        diagram, system = robot.to_diagram(q0=[0.6], qd0=[0.0], return_system=True)
        sid = system.system_id

        def observe(damping):
            ctx = diagram.create_context()
            sub = ctx[sid].with_parameter("pivot.damping", damping[0])
            ctx = ctx.with_subcontext(sid, sub)
            return jnp.concatenate(
                [
                    jaxonomy.simulate(diagram, ctx, (0.0, t), options=options)
                    .context[sid]
                    .continuous_state
                    for t in sample_times
                ]
            )

        return observe

    telemetry = np.asarray(
        make_observe(_pendulum(true_damping))(jnp.array([true_damping]))
    )
    estimate, history = demo.identify(telemetry, make_observe(_pendulum(0.0)), 1)

    assert history[-1] < history[0]  # cost decreased
    assert float(estimate[0]) == pytest.approx(true_damping, abs=5e-3)


@pytest.mark.nightly
def test_full_so101_demo_recovers_damping():
    """The full SO-101 demo identifies the planted damping and calibrates."""
    from jaxterity.robot import CalibrationState
    from jaxterity.sysid import SysidResult

    result = demo.main()
    assert isinstance(result, SysidResult)
    for value in result.fitted.values():
        assert float(value) == pytest.approx(demo.TRUE_DAMPING, abs=5e-3)
    # the result calibrates and records provenance
    calibrated = result.apply_to(demo.zoo.load("so100"))
    assert calibrated.calibration_state is CalibrationState.CALIBRATED
    assert calibrated.provenance[0][0] == "forward_mode_lm_damping"
