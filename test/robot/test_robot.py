# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Robot + to_diagram tests: analytical dynamics, autodiff, attestation."""

from __future__ import annotations

import math

import jax
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


def _pendulum(
    bob_mass: float = 1.0,
    damping: float = 0.0,
    friction: float = 0.0,
    joint_type: JointType = JointType.REVOLUTE,
) -> Robot:
    """A hanging pendulum: bob COM 1 m below a y-axis hinge (equilibrium at q=0).

    ``joint_type=CONTINUOUS`` gives an unlimited hinge (no MJX limit constraint),
    needed for reverse-mode autodiff — the iterative constraint solver MJX runs
    for a limited joint integrates under a ``lax.while_loop`` that ``jax.grad``
    cannot traverse.
    """
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
        mass=jnp.asarray(bob_mass),
        com=jnp.array([0.0, 0.0, -1.0]),  # 1 m below the pivot
        inertia=jnp.eye(3) * 1e-6,
    )
    pivot = Joint(
        name="pivot",
        joint_type=joint_type,
        parent_link="base",
        child_link="bob",
        axis=FramedQuantity(Position(jnp.array([0.0, 1.0, 0.0])), Frame("base")),
        limits=JointLimits(lower=-3.14, upper=3.14, effort=100.0, velocity=100.0),
        origin=Transform.identity(Frame("base")),
        damping=jnp.asarray(damping),
        friction=jnp.asarray(friction),
    )
    return Robot(links=(base, bob), joints=(pivot,), name="pendulum")


def _opts(autodiff: bool = False) -> SimulatorOptions:
    return SimulatorOptions(
        math_backend="jax",
        ode_solver_method="rk4",
        max_minor_step_size=0.002,
        enable_autodiff=autodiff,
    )


def test_to_mjcf_parses():
    import mujoco

    mjcf = _pendulum().to_mjcf()
    model = mujoco.MjModel.from_xml_string(mjcf)  # no Gazebo needed
    assert model.nq == 1 and model.nv == 1


def test_pendulum_matches_analytical_period():
    """Undamped small-angle pendulum returns to ~theta0 after one period
    T = 2*pi*sqrt(L/g) — the analytical reference (ADR-016 MJX dynamics)."""
    theta0 = 0.1
    L, g = 1.0, 9.81
    period = 2 * math.pi * math.sqrt(L / g)
    diagram, sysm = _pendulum().to_diagram(q0=[theta0], qd0=[0.0], return_system=True)
    res = jaxonomy.simulate(
        diagram, diagram.create_context(), (0.0, period), options=_opts()
    )
    theta_final = float(res.context[sysm.system_id].continuous_state[0])
    # small-angle approximation + RK4 integration error over one period.
    assert abs(theta_final - theta0) < 5e-3, theta_final


def test_joint_origin_rotation_orients_axis():
    """A joint's origin rotation must reach the MJCF (regression).

    Without it every joint frame collapses onto its parent's, so a robot whose
    URDF expresses each axis in a *rotated* joint frame (every real multi-axis
    arm) silently cannot move — this is exactly what left the SO-100 zoo entry
    motionless. Here the hinge axis is ``z`` in the joint frame, but the origin
    rotation tips it horizontal, so the offset bob must swing under gravity. Drop
    the rotation and the axis stays vertical → zero gravity torque → no motion.
    """
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
        com=jnp.array([0.5, 0.0, 0.0]),  # offset along x
        inertia=jnp.eye(3) * 1e-6,
    )
    # R_x(90°) maps the joint-frame z-axis to the world horizontal plane.
    rx90 = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    pivot = Joint(
        name="pivot",
        joint_type=JointType.REVOLUTE,
        parent_link="base",
        child_link="bob",
        axis=FramedQuantity(Position(jnp.array([0.0, 0.0, 1.0])), Frame("base")),
        limits=JointLimits(lower=-3.14, upper=3.14, effort=100.0, velocity=100.0),
        origin=Transform(rx90, jnp.zeros(3), source=Frame("bob"), target=Frame("base")),
        damping=jnp.asarray(0.0),
        friction=jnp.asarray(0.0),
    )
    robot = Robot(links=(base, bob), joints=(pivot,), name="tipped")
    assert "quat=" in robot.to_mjcf()  # the rotation reaches the MJCF
    diagram, sysm = robot.to_diagram(q0=[0.0], qd0=[0.0], return_system=True)
    res = jaxonomy.simulate(
        diagram, diagram.create_context(), (0.0, 0.5), options=_opts()
    )
    theta = float(res.context[sysm.system_id].continuous_state[0])
    assert abs(theta) > 0.1, f"bob did not swing under gravity (theta={theta})"


def test_damping_dissipates_energy():
    """A damped pendulum loses amplitude; an undamped one does not."""
    theta0 = 0.3
    out = {}
    for name, damping in (("undamped", 0.0), ("damped", 0.5)):
        diagram, sysm = _pendulum(damping=damping).to_diagram(
            q0=[theta0], qd0=[0.0], return_system=True
        )
        res = jaxonomy.simulate(
            diagram, diagram.create_context(), (0.0, 3.0), options=_opts()
        )
        x = np.asarray(res.context[sysm.system_id].continuous_state)
        out[name] = 0.5 * x[1] ** 2 + 9.81 * (1 - math.cos(x[0]))  # ~ specific energy
    assert out["damped"] < out["undamped"]


@pytest.mark.slow
def test_grad_through_simulate_wrt_damping():
    """jax.grad of a terminal cost flows into joint damping and matches FD.

    Uses the BDF adjoint (the reverse-mode-safe path; the adaptive explicit
    solvers integrate under a lax.while_loop that reverse-mode can't traverse).
    """
    diagram, sysm = _pendulum(damping=0.5, joint_type=JointType.CONTINUOUS).to_diagram(
        q0=[0.4], qd0=[0.0], return_system=True
    )
    sid = sysm.system_id
    adj = SimulatorOptions(
        math_backend="jax",
        ode_solver_method="bdf",
        enable_autodiff=True,
        rtol=1e-7,
        atol=1e-9,
        max_major_steps=300,
    )

    def loss(damping):
        ctx = diagram.create_context()
        sub = ctx[sid].with_parameter("pivot.damping", damping)
        ctx = ctx.with_subcontext(sid, sub)
        res = jaxonomy.simulate(diagram, ctx, (0.0, 1.0), options=adj)
        return res.context[sid].continuous_state[0] ** 2

    g = float(jax.grad(loss)(jnp.asarray(0.5)))
    fd = float((loss(jnp.asarray(0.51)) - loss(jnp.asarray(0.49))) / 0.02)
    assert np.isfinite(g)
    assert abs(g) > 1e-4  # damping genuinely affects the terminal state
    assert abs(g - fd) <= 1e-2 * (abs(fd) + 1e-3)


def test_armature_slows_the_dynamics():
    """Joint armature is threaded into MJX at runtime.

    Reflected rotor inertia adds to the joint-space inertia without adding
    gravitational torque, so a pendulum released from the same angle falls
    more slowly. Overridden through the context parameter (not a structural
    rebuild) to exercise the _RobotSystem runtime threading.
    """
    diagram, sysm = _pendulum().to_diagram(q0=[0.3], qd0=[0.0], return_system=True)
    sid = sysm.system_id

    def final_angle(armature: float) -> float:
        ctx = diagram.create_context()
        sub = ctx[sid].with_parameter("pivot.armature", jnp.asarray(armature))
        ctx = ctx.with_subcontext(sid, sub)
        res = jaxonomy.simulate(diagram, ctx, (0.0, 0.3), options=_opts())
        return float(res.context[sid].continuous_state[0])

    light, heavy = final_angle(0.0), final_angle(2.0)
    # Both fall from 0.3 rad toward 0; the armature-laden joint falls less far.
    assert heavy > light
    assert abs(heavy - light) > 1e-3


# --- attestation -----------------------------------------------------------


def test_attestation_handle_deterministic_and_sensitive():
    h1 = _pendulum().attestation_handle
    h2 = _pendulum().attestation_handle
    assert h1 == h2  # deterministic across builds
    assert len(h1) == 64  # BLAKE3 hex digest
    assert _pendulum(bob_mass=2.0).attestation_handle != h1  # sensitive to params


def test_with_parameters_is_immutable_and_changes_handle():
    robot = _pendulum()
    h0 = robot.attestation_handle
    fitted = robot.with_parameters({"bob.mass": 2.0, "pivot.damping": 0.3})
    # original unchanged (ADR-004: sysid never mutates)
    assert float(robot.link_map["bob"].mass) == 1.0
    assert robot.attestation_handle == h0
    # new robot reflects the change
    assert float(fitted.link_map["bob"].mass) == 2.0
    assert fitted.attestation_handle != h0


def test_with_parameters_rejects_unknown():
    from jaxterity.errors import JaxterityError

    with pytest.raises(JaxterityError, match="not a known link or joint"):
        _pendulum().with_parameters({"nope.mass": 1.0})
    with pytest.raises(JaxterityError, match="not an identifiable parameter"):
        _pendulum().with_parameters({"pivot.mass": 1.0})
