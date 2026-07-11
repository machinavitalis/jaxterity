# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""URDF importer tests: parse, warnings, round-trip, simulate."""

from __future__ import annotations

import jaxonomy
import numpy as np
import pytest
from jaxonomy import SimulatorOptions

from jaxterity.robot import JointType, Robot, UrdfWarning, from_urdf, to_urdf

pytest.importorskip("mujoco")

_ARM_URDF = """<?xml version="1.0"?>
<robot name="arm">
  <link name="base">
    <inertial><origin xyz="0 0 0"/><mass value="0.5"/>
      <inertia ixx="0.01" iyy="0.01" izz="0.01"/></inertial>
  </link>
  <link name="link1">
    <inertial><origin xyz="0 0 -0.5"/><mass value="1.0"/>
      <inertia ixx="0.1" iyy="0.1" izz="0.001"/></inertial>
  </link>
  <link name="link2">
    <inertial><origin xyz="0 0 -0.5"/><mass value="0.8"/>
      <inertia ixx="0.08" iyy="0.08" izz="0.001"/></inertial>
  </link>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="link1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" effort="50" velocity="10"/>
    <dynamics damping="0.1" friction="0.0"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="link1"/><child link="link2"/>
    <origin xyz="0 0 -1.0" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" effort="50" velocity="10"/>
    <dynamics damping="0.05" friction="0.0"/>
  </joint>
</robot>
"""


def _write(tmp_path, text: str, name: str = "arm.urdf"):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_from_urdf_parses_structure_and_params(tmp_path):
    robot = from_urdf(_write(tmp_path, _ARM_URDF))
    assert isinstance(robot, Robot)
    assert set(robot.link_map) == {"base", "link1", "link2"}
    assert set(robot.joint_map) == {"j1", "j2"}
    assert robot.joint_map["j1"].joint_type is JointType.REVOLUTE
    np.testing.assert_allclose(float(robot.link_map["link1"].mass), 1.0)
    np.testing.assert_allclose(np.asarray(robot.link_map["link1"].com), [0, 0, -0.5])
    np.testing.assert_allclose(float(robot.joint_map["j1"].damping), 0.1)
    # joint origin translation came from <origin xyz>
    np.testing.assert_allclose(
        np.asarray(robot.joint_map["j2"].origin.translation), [0, 0, -1.0]
    )


def test_unsupported_joint_type_warns(tmp_path):
    urdf = _ARM_URDF.replace('type="revolute"', 'type="floating"', 1)
    with pytest.warns(UrdfWarning, match="unsupported type 'floating'"):
        robot = from_urdf(_write(tmp_path, urdf))
    assert "j1" not in robot.joint_map  # skipped, not silently kept


def test_missing_inertial_warns(tmp_path):
    urdf = """<?xml version="1.0"?>
<robot name="r">
  <link name="base"/>
  <link name="bob"><inertial><origin xyz="0 0 -1"/><mass value="1"/>
    <inertia ixx="1e-6" iyy="1e-6" izz="1e-6"/></inertial></link>
  <joint name="j" type="continuous"><parent link="base"/><child link="bob"/>
    <origin xyz="0 0 0"/><axis xyz="0 1 0"/></joint>
</robot>"""
    with pytest.warns(UrdfWarning, match="no <inertial>"):
        from_urdf(_write(tmp_path, urdf))


def test_semantic_roundtrip(tmp_path):
    robot = from_urdf(_write(tmp_path, _ARM_URDF))
    out = tmp_path / "out.urdf"
    to_urdf(robot, out)
    robot2 = from_urdf(out)
    assert set(robot.link_map) == set(robot2.link_map)
    assert set(robot.joint_map) == set(robot2.joint_map)
    for name, link in robot.link_map.items():
        link2 = robot2.link_map[name]
        np.testing.assert_allclose(float(link.mass), float(link2.mass))
        np.testing.assert_allclose(np.asarray(link.com), np.asarray(link2.com))
        np.testing.assert_allclose(
            np.asarray(link.inertia), np.asarray(link2.inertia), atol=1e-12
        )
    for name, joint in robot.joint_map.items():
        j2 = robot2.joint_map[name]
        assert joint.joint_type is j2.joint_type
        np.testing.assert_allclose(float(joint.damping), float(j2.damping))
        np.testing.assert_allclose(
            np.asarray(joint.origin.translation),
            np.asarray(j2.origin.translation),
            atol=1e-9,
        )


def test_imported_robot_simulates(tmp_path):
    robot = from_urdf(_write(tmp_path, _ARM_URDF))
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
    assert x.shape == (4,)  # [q1, q2, qd1, qd2]
    assert np.all(np.isfinite(x))
