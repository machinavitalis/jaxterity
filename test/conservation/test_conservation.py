# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Energy-conservation property tests for the MJX dynamics wrapper.

Energy conservation in a closed, frictionless system is the cheapest,
hardest-to-fake check that ``Robot.to_diagram()`` integrates the right dynamics:
if a future change breaks the MJX wiring (wrong mass matrix, a stray force, a
sign error), a conservative system stops conserving and these tests fail before
any user sees it. We integrate two conservative robots over 10 oscillation
periods across every supported solver and assert the total mechanical energy
``E = ½·qd·M(q)·qd + Σ mᵢ·g·zᵢ`` drifts less than the committed tolerance.

The energy is recomputed from first principles at each recorded state (the dense
mass matrix and body COM heights), so it is independent of the *integrator* —
exactly the quantity under test. Joints are ``CONTINUOUS`` (unlimited): a closed
conservative system must have no joint-limit constraint adding or removing
energy.

Two tolerances play distinct roles. The **assertion** bound comes from
``jaxterity.testing.tolerances`` (the single source — PATTERNS §6.3). The
**integration** tolerance for the adaptive solvers is set tighter than that
bound on purpose, so the test measures the integrator's energy behaviour rather
than its truncation error; energy is quadratic in velocity, so it drifts a few×
faster than the state itself.

The third case — a free-flying rigid body — exercises the floating base
(a free joint, quaternion kinematics, ``nq ≠ nv``):
:func:`test_free_body_conserves_energy_and_momentum` checks that a torque-free
tumbling body conserves both kinetic energy and angular momentum.
"""

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
from jaxterity.testing import tolerances as tol

pytest.importorskip("mujoco")

#: Oscillation periods to integrate over (spec: "10+").
N_PERIODS = 10
#: Small-angle period of a 1 m pendulum under g = 9.81 (the time base).
_PERIOD = 2 * math.pi * math.sqrt(1.0 / 9.81)


# --- robot builders --------------------------------------------------------


def _link(name: str, mass: float, com: tuple[float, float, float]) -> Link:
    return Link(
        name=name,
        body_frame=Frame(name),
        mass=jnp.asarray(mass),
        com=jnp.asarray(com, dtype=float),
        inertia=jnp.eye(3) * 1e-6,  # point-mass rod; tiny isotropic inertia
    )


def _hinge(
    name: str, parent: str, child: str, origin: tuple[float, float, float] = (0, 0, 0)
) -> Joint:
    """An unlimited (``CONTINUOUS``) y-axis hinge — no joint-limit constraint."""
    return Joint(
        name=name,
        joint_type=JointType.CONTINUOUS,
        parent_link=parent,
        child_link=child,
        axis=FramedQuantity(Position(jnp.array([0.0, 1.0, 0.0])), Frame(parent)),
        limits=JointLimits(lower=-3.14, upper=3.14, effort=100.0, velocity=100.0),
        origin=Transform(
            jnp.eye(3),
            jnp.asarray(origin, dtype=float),
            source=Frame(parent),
            target=Frame(child),
        ),
        damping=jnp.asarray(0.0),
        friction=jnp.asarray(0.0),
    )


def _pendulum() -> Robot:
    """A single rigid pendulum: a 1 kg bob 1 m below an unlimited hinge."""
    return Robot(
        links=(_link("base", 0.1, (0, 0, 0)), _link("bob", 1.0, (0, 0, -1.0))),
        joints=(_hinge("pivot", "base", "bob"),),
        name="pendulum",
    )


def _double_pendulum() -> Robot:
    """A double pendulum: two 1 kg point masses on 1 m massless rods (chaotic)."""
    return Robot(
        links=(
            _link("base", 0.1, (0, 0, 0)),
            _link("link1", 1.0, (0, 0, -1.0)),
            _link("link2", 1.0, (0, 0, -1.0)),
        ),
        joints=(
            _hinge("p1", "base", "link1"),
            _hinge("p2", "link1", "link2", origin=(0, 0, -1.0)),
        ),
        name="double_pendulum",
    )


# --- energy ----------------------------------------------------------------


def _energy_fn(robot: Robot):
    """Return a vmapped ``state -> total mechanical energy`` for ``robot``.

    Recomputed from the MJX model directly (dense mass matrix + body COM
    heights), so it is independent of the integrator being tested.
    """
    import mujoco
    from mujoco import mjx

    model = mjx.put_model(mujoco.MjModel.from_xml_string(robot.to_mjcf()))
    nq = int(model.nq)
    nv = int(model.nv)
    g = float(robot.gravity)

    def energy(x):
        q, qd = x[:nq], x[nq : nq + nv]
        data = mjx.forward(model, mjx.make_data(model).replace(qpos=q, qvel=qd))
        kinetic = 0.5 * qd @ mjx.full_m(model, data) @ qd
        potential = jnp.sum(model.body_mass * g * data.xipos[:, 2])
        return kinetic + potential

    return jax.jit(jax.vmap(energy))


# --- solver configuration --------------------------------------------------

# Integration settings per solver: rk4 is fixed-step; the adaptive solvers run
# at a tolerance well below the assertion bound (see module docstring). These are
# integrator settings, not assertion tolerances — the latter are committed.
_SOLVER_CONFIG: dict[str, dict] = {
    "rk4": {"max_minor_step_size": 0.005},
    "dopri5": {"rtol": 1e-9, "atol": 1e-11},
    "bdf": {"rtol": 1e-9, "atol": 1e-11},
}


def _simulate_energy(robot: Robot, q0, qd0, method: str):
    """Integrate ``robot`` for ``N_PERIODS`` and return its energy time series."""
    diagram, system = robot.to_diagram(q0=q0, qd0=qd0, return_system=True)
    options = SimulatorOptions(
        math_backend="jax",
        ode_solver_method=method,
        buffer_length=8000,
        max_major_steps=8000,
        **_SOLVER_CONFIG[method],
    )
    result = jaxonomy.simulate(
        diagram,
        diagram.create_context(),
        (0.0, N_PERIODS * _PERIOD),
        options=options,
        recorded_signals={"state": system.output_ports[0]},
    )
    states = jnp.asarray(result.outputs["state"])
    return np.asarray(_energy_fn(robot)(states))


# --- tests -----------------------------------------------------------------

_ROBOTS = {
    "pendulum": (_pendulum, [0.5], [0.0]),
    "double_pendulum": (_double_pendulum, [0.5, 0.3], [0.0, 0.0]),
}


@pytest.mark.parametrize("robot_name", list(_ROBOTS))
@pytest.mark.parametrize("solver", tol.SOLVERS)
def test_energy_conserved(robot_name: str, solver: str):
    """Total mechanical energy is conserved over 10 periods (every solver)."""
    builder, q0, qd0 = _ROBOTS[robot_name]
    robot = builder()
    energy = _simulate_energy(robot, q0, qd0, solver)

    e0 = energy[0]
    drift = float(np.max(np.abs(energy - e0)))
    rel_drift = drift / abs(float(e0))
    bound = tol.rtol("joint_velocity", solver, "float64")

    assert rel_drift < bound, (
        f"energy not conserved for robot {robot_name!r} under solver {solver!r}: "
        f"relative drift {rel_drift:.3e} exceeds bound {bound:.3e} over "
        f"{N_PERIODS} periods ({len(energy)} samples); "
        f"absolute drift {drift:.3e} J on |E0|={abs(float(e0)):.4f} J."
    )


def _free_body() -> Robot:
    """A torque-free asymmetric rigid body on a floating (free) joint.

    Distinct principal inertias make it tumble (the Dzhanibekov / intermediate-
    axis regime) — a strong test of the free-joint quaternion integration.
    """
    body = Link(
        name="body",
        body_frame=Frame("body"),
        mass=jnp.asarray(1.0),
        com=jnp.zeros(3),
        inertia=jnp.diag(jnp.array([0.1, 0.2, 0.3])),
    )
    free = Joint(
        name="root",
        joint_type=JointType.FLOATING,
        parent_link="world",
        child_link="body",
        axis=FramedQuantity(Position(jnp.array([0.0, 0.0, 1.0])), Frame("world")),
        limits=JointLimits(lower=-1.0, upper=1.0, effort=1.0, velocity=1.0),
        origin=Transform.identity(Frame("world")),
        damping=jnp.asarray(0.0),
        friction=jnp.asarray(0.0),
    )
    return Robot(links=(body,), joints=(free,), name="free_body", gravity=0.0)


@pytest.mark.parametrize("solver", tol.SOLVERS)
def test_free_body_conserves_energy_and_momentum(solver):
    """The third case (floating base): a torque-free rigid body conserves
    kinetic energy *and* angular momentum over 10 tumbling periods, every solver.
    """
    from jaxterity.robot.mjcf import _quat_to_matrix

    robot = _free_body()
    qd0 = [0.0, 0.0, 0.0, 0.1, 2.0, 0.05]  # spin mainly about the intermediate axis
    diagram, system = robot.to_diagram(q0=None, qd0=qd0, return_system=True)
    options = SimulatorOptions(
        math_backend="jax",
        ode_solver_method=solver,
        buffer_length=8000,
        max_major_steps=8000,
        **_SOLVER_CONFIG[solver],
    )
    result = jaxonomy.simulate(
        diagram,
        diagram.create_context(),
        (0.0, N_PERIODS * _PERIOD),
        options=options,
        recorded_signals={"state": system.output_ports[0]},
    )
    states = np.asarray(result.outputs["state"])

    # kinetic energy (gravity is zero, so total energy is kinetic)
    energy = np.asarray(_energy_fn(robot)(jnp.asarray(states)))
    e_drift = float(np.max(np.abs(energy - energy[0])) / abs(energy[0]))

    # angular momentum in the world frame: L = R(quat) · I_body · ω_body
    inertia = np.diag([0.1, 0.2, 0.3])
    momenta = np.array([_quat_to_matrix(s[3:7]) @ (inertia @ s[10:13]) for s in states])
    l_drift = float(
        np.max(np.linalg.norm(momenta - momenta[0], axis=1))
        / np.linalg.norm(momenta[0])
    )

    bound = tol.rtol("joint_velocity", solver, "float64")
    assert e_drift < bound, (
        f"free body energy drift {e_drift:.3e} ≥ {bound:.3e} ({solver})"
    )
    assert l_drift < bound, (
        f"free body angular-momentum drift {l_drift:.3e} ≥ {bound:.3e} ({solver})"
    )
