# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Actuator/Sensor base + trivial-impl tests."""

from __future__ import annotations

import abc

import jax.numpy as jnp
import jaxonomy
import numpy as np
import pytest
from jaxonomy import SimulatorOptions

from jaxterity.actuators import Actuator, IdealTorqueActuator
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
from jaxterity.sensors import IdealEncoder, Sensor

pytest.importorskip("mujoco")

_OPTS = SimulatorOptions(
    math_backend="jax", ode_solver_method="rk4", max_minor_step_size=0.005
)


def _pendulum() -> Robot:
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
        inertia=jnp.eye(3) * 1e-3,
    )
    pivot = Joint(
        name="pivot",
        joint_type=JointType.CONTINUOUS,  # unlimited → no MJX constraint
        parent_link="base",
        child_link="bob",
        axis=FramedQuantity(Position(jnp.array([0.0, 1.0, 0.0])), Frame("base")),
        limits=JointLimits(lower=-3.14, upper=3.14, effort=100.0, velocity=100.0),
        origin=Transform.identity(Frame("base")),
        damping=jnp.asarray(0.0),
        friction=jnp.asarray(0.0),
    )
    return Robot(links=(base, bob), joints=(pivot,), name="pendulum")


# --- abstract bases --------------------------------------------------------


def test_bases_are_abstract():
    assert issubclass(Actuator, abc.ABC) and issubclass(Sensor, abc.ABC)
    with pytest.raises(TypeError):
        Actuator()  # abstract: parameter_spec / validate unimplemented
    with pytest.raises(TypeError):
        Sensor()


def test_trivial_metadata():
    act = IdealTorqueActuator(1)
    enc = IdealEncoder()
    assert act.sysid_protocol.name == "ideal"
    assert tuple(act.parameter_spec()) == ()  # ideal: nothing identifiable
    assert act.validate(None).ok
    assert enc.sysid_protocol.name == "ideal_encoder"
    assert tuple(enc.parameter_spec()) == ()
    assert enc.validate(None).ok


# --- composition with a Robot ---------------------------------------------


def _final_state_with_torque(tau: float) -> np.ndarray:
    robot = _pendulum()
    robot_sys = robot.build_system(q0=[0.3], qd0=[0.0], actuation=True)
    act = IdealTorqueActuator(1, tau=[tau])
    b = jaxonomy.DiagramBuilder()
    b.add(robot_sys)
    b.add(act)
    b.connect(act.output_ports[0], robot_sys.input_ports[0])  # tau → robot
    diagram = b.build()
    res = jaxonomy.simulate(
        diagram, diagram.create_context(), (0.0, 0.5), options=_OPTS
    )
    return np.asarray(res.context[robot_sys.system_id].continuous_state)


def test_ideal_actuator_drives_robot():
    """A nonzero commanded torque changes the robot's trajectory vs zero torque."""
    free = _final_state_with_torque(0.0)
    driven = _final_state_with_torque(3.0)
    assert np.all(np.isfinite(free)) and np.all(np.isfinite(driven))
    assert float(np.linalg.norm(driven - free)) > 1e-2


def test_ideal_encoder_reads_robot_state():
    """An IdealEncoder wired to the robot's state output reads it back exactly."""
    robot = _pendulum()
    robot_sys = robot.build_system(q0=[0.3], qd0=[0.1])
    enc = IdealEncoder()
    b = jaxonomy.DiagramBuilder()
    b.add(robot_sys)
    b.add(enc)
    b.connect(robot_sys.output_ports[0], enc.input_ports[0])  # state → encoder
    diagram = b.build()
    ctx = diagram.create_context()
    reading = np.asarray(enc.output_ports[0].eval(ctx))
    state = np.asarray(robot_sys.output_ports[0].eval(ctx))
    np.testing.assert_array_equal(reading, state)
