# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Robot Rerun logger — FK-driven 3D state + per-joint signal streaming.

The recording back-end (``rerun-sdk``) is optional and its on-disk ``.rrd``
format is version-specific, so these tests assert what's stable: the logger
builds the right kinematic/signal structure for each robot topology, streams
without error, writes a non-empty recording, and rejects malformed input.
"""

from __future__ import annotations

import numpy as np
import pytest

from jaxterity import zoo
from jaxterity.diagnostics.rerun_robot import (
    RobotRerunLogger,
    _build_kinematics,
    _require_rerun,
    log_robot_trajectory,
)
from jaxterity.errors import JaxterityError

pytest.importorskip("mujoco")


def test_require_rerun_raises_when_missing():
    """Without rerun-sdk, constructing a logger raises with install instructions."""
    try:
        import rerun  # noqa: F401
    except ImportError:
        with pytest.raises(JaxterityError, match="pip install"):
            _require_rerun()
    else:
        pytest.skip("rerun-sdk is installed; can't test the missing-dep path here")


def test_kinematics_structure_fixed_base_arm():
    """A fixed-base arm: one body per link, six 1-DOF joints, square nq==nv."""
    fk, body_names, parents, joints, nq, nv = _build_kinematics(zoo.load("so100"))
    assert (nq, nv) == (6, 6)
    assert "world" in body_names and len(body_names) > 6
    assert len(joints) == 6
    assert all(nq_j == 1 and nv_j == 1 for _, _, nq_j, _, nv_j in joints)
    xpos, xquat = fk(np.zeros(nq))
    assert np.asarray(xpos).shape == (len(body_names), 3)
    assert np.asarray(xquat).shape == (len(body_names), 4)


def test_kinematics_structure_floating_base():
    """A floating base shows up as one 7-qpos/6-dof joint (nq == nv + 1)."""
    fk, body_names, parents, joints, nq, nv = _build_kinematics(zoo.load("crazyflie"))
    assert (nq, nv) == (7, 6)
    free = [j for j in joints if j[2] == 7]
    assert len(free) == 1 and free[0][4] == 6  # one joint with nq=7, nv=6


def test_log_trajectory_writes_recording(tmp_path):
    """Streaming an arm trajectory writes a non-empty .rrd (no viewer spawned)."""
    pytest.importorskip("rerun")
    robot = zoo.load("so100")
    t = 8
    qs = np.linspace(0.0, 0.5, t)[:, None] * np.ones((t, 6))
    out = tmp_path / "arm.rrd"
    log_robot_trajectory(
        robot,
        qs,
        qds=np.zeros((t, 6)),
        taus=np.zeros((t, 6)),
        dt=0.05,
        save_path=out,
        namespace="arm",
    )
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.parametrize(
    "loader",
    [lambda: zoo.load("crazyflie"), lambda: zoo.unitree_g1.load(floating_base=True)],
    ids=["crazyflie", "g1_floating"],
)
def test_live_logger_streams_floating_base(tmp_path, loader):
    """The live per-step API streams a floating-base robot (quad + humanoid)."""
    pytest.importorskip("rerun")
    robot = loader()
    _, _, _, _, nq, nv = _build_kinematics(robot)
    q = np.zeros(nq)
    q[3] = 1.0  # unit quaternion (w=1) for the free joint
    out = tmp_path / "float.rrd"
    logger = RobotRerunLogger(robot, save_path=out, namespace="fb")
    for i in range(5):
        q[2] = 0.1 - 0.01 * i  # descend in z
        logger.log_step(q, qd=np.zeros(nv), tau=np.zeros(nv), time=i * 0.05)
    assert out.exists() and out.stat().st_size > 0


def test_log_step_rejects_wrong_nq(tmp_path):
    """A configuration of the wrong length is a typed error, not a silent log."""
    pytest.importorskip("rerun")
    logger = RobotRerunLogger(zoo.load("so100"), save_path=tmp_path / "x.rrd")
    with pytest.raises(JaxterityError, match="expected nq"):
        logger.log_step(np.zeros(5))


def test_trajectory_rejects_bad_shapes(tmp_path):
    """qs must be 2-D, and qds/taus row counts must match qs."""
    pytest.importorskip("rerun")
    robot = zoo.load("so100")
    with pytest.raises(JaxterityError, match="must be"):
        log_robot_trajectory(robot, np.zeros(6), save_path=tmp_path / "a.rrd")
    with pytest.raises(JaxterityError, match="match qs"):
        log_robot_trajectory(
            robot, np.zeros((4, 6)), qds=np.zeros((3, 6)), save_path=tmp_path / "b.rrd"
        )
