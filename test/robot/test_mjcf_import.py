# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""MJCF importer tests: parse, simulate, round-trip via Robot.to_mjcf."""

from __future__ import annotations

import jaxonomy
import numpy as np
import pytest
from jaxonomy import SimulatorOptions

from jaxterity.robot import JointType, Robot, from_mjcf

pytest.importorskip("mujoco")

_ARM_MJCF = """<mujoco model="arm">
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <body name="base" pos="0 0 0">
      <inertial pos="0 0 0" mass="0.5" diaginertia="0.01 0.01 0.01"/>
      <body name="link1" pos="0 0 0">
        <joint name="j1" type="hinge" axis="0 1 0" range="-2 2"
               damping="0.1" armature="0.007"/>
        <inertial pos="0 0 -0.5" mass="1.0" fullinertia="0.1 0.1 0.001 0 0 0"/>
        <body name="link2" pos="0 0 -1">
          <joint name="j2" type="hinge" axis="0 1 0" range="-2 2" damping="0.05"/>
          <inertial pos="0 0 -0.5" mass="0.8" fullinertia="0.08 0.08 0.001 0 0 0"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _write(tmp_path, text: str, name: str = "arm.xml"):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_from_mjcf_parses(tmp_path):
    robot = from_mjcf(_write(tmp_path, _ARM_MJCF))
    assert isinstance(robot, Robot)
    assert set(robot.link_map) == {"base", "link1", "link2"}
    assert set(robot.joint_map) == {"j1", "j2"}
    assert robot.joint_map["j1"].joint_type is JointType.REVOLUTE
    np.testing.assert_allclose(float(robot.link_map["link1"].mass), 1.0)
    np.testing.assert_allclose(np.asarray(robot.link_map["link1"].com), [0, 0, -0.5])
    np.testing.assert_allclose(float(robot.joint_map["j1"].damping), 0.1)
    np.testing.assert_allclose(float(robot.joint_map["j1"].armature), 0.007)
    np.testing.assert_allclose(float(robot.joint_map["j2"].armature), 0.0)
    np.testing.assert_allclose(
        np.diag(np.asarray(robot.link_map["link1"].inertia)), [0.1, 0.1, 0.001]
    )


def test_imported_mjcf_simulates(tmp_path):
    robot = from_mjcf(_write(tmp_path, _ARM_MJCF))
    diagram, sysm = robot.to_diagram(q0=[0.2, -0.3], qd0=[0.0, 0.0], return_system=True)
    res = jaxonomy.simulate(
        diagram,
        diagram.create_context(),
        (0.0, 1.0),
        options=SimulatorOptions(
            math_backend="jax", ode_solver_method="rk4", max_minor_step_size=0.005
        ),
    )
    x = np.asarray(res.context[sysm.system_id].continuous_state)
    assert x.shape == (4,) and np.all(np.isfinite(x))


def test_mjcf_robot_roundtrips_through_to_mjcf(tmp_path):
    """from_mjcf -> Robot -> to_mjcf reparses to the same masses/structure."""
    import mujoco

    robot = from_mjcf(_write(tmp_path, _ARM_MJCF))
    m2 = mujoco.MjModel.from_xml_string(robot.to_mjcf())
    assert m2.nq == 2 and m2.nv == 2
    masses = sorted(round(float(x), 3) for x in m2.body_mass[1:])
    assert masses == [0.5, 0.8, 1.0]
    # armature survives the round trip (previously silently dropped,
    # leaving MJCF-sourced robots with far lighter distal joints than upstream)
    dof = int(m2.jnt_dofadr[int(m2.joint("j1").id)])
    np.testing.assert_allclose(float(m2.dof_armature[dof]), 0.007)
