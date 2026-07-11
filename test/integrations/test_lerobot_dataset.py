# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""LeRobotDataset writer/reader round-trip and schema checks.

The dependency-light path (write + local read) needs only ``pyarrow`` and runs
in the normal test suite. A final test loads the same directory through the real
``lerobot`` package and self-skips when the ``[lerobot]`` extra is absent.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("pyarrow")

from jaxterity.integrations import lerobot as lr  # noqa: E402


def _make_episodes(rng, n=4, dim=6, include_goal=True):
    eps = []
    for k in range(n):
        T = 8 + k
        data = {
            "observation.state": rng.standard_normal((T, dim)).astype("f4"),
            "action": rng.standard_normal((T, dim)).astype("f4"),
        }
        if include_goal:
            data["observation.goal"] = rng.standard_normal((T, dim)).astype("f4")
        eps.append(lr.Episode(data=data, task="reach the goal joint configuration"))
    return eps


def test_features_from_robot_uses_so100_motor_order():
    from jaxterity import zoo

    robot = zoo.load("so100")
    feats = lr.features_from_robot(
        robot, joint_names=lr.SO100_JOINTS, include_goal=True
    )
    assert feats["observation.state"]["names"] == list(lr.SO100_JOINTS)
    assert feats["action"]["shape"] == [6]
    # bookkeeping columns LeRobot requires are present
    for col in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        assert col in feats


def test_write_then_local_read_round_trips(tmp_path):
    rng = np.random.default_rng(0)
    robot_feats = lr.features_from_robot(
        __import__("jaxterity").zoo.load("so100"),
        joint_names=lr.SO100_JOINTS,
        include_goal=True,
    )
    eps = _make_episodes(rng)
    root = tmp_path / "ds"
    lr.write_lerobot_dataset(root, eps, robot_feats, fps=30, robot_type="so100")

    ds = lr.read_local_lerobot_dataset(root)
    assert len(ds) == len(eps)
    assert ds.fps == 30
    # frame counts and values survive the parquet round-trip exactly
    total = sum(e.length for e in eps)
    assert ds.frames("action").shape == (total, 6)
    for got, want in zip(ds.episodes, eps):
        np.testing.assert_array_equal(
            got.data["observation.state"], want.data["observation.state"]
        )
        np.testing.assert_array_equal(got.data["action"], want.data["action"])


def test_info_json_matches_v21_schema(tmp_path):
    rng = np.random.default_rng(1)
    feats = lr.features_from_robot(
        __import__("jaxterity").zoo.load("so100"),
        joint_names=lr.SO100_JOINTS,
        include_goal=True,
    )
    eps = _make_episodes(rng, n=3)
    root = tmp_path / "ds"
    lr.write_lerobot_dataset(root, eps, feats, fps=30, robot_type="so100")

    info = json.loads((root / "meta" / "info.json").read_text())
    assert info["codebase_version"] == lr.CODEBASE_VERSION == "v2.1"
    assert info["robot_type"] == "so100"
    assert info["fps"] == 30
    assert info["total_episodes"] == 3
    assert info["total_frames"] == sum(e.length for e in eps)
    assert info["total_tasks"] == 1
    # the per-episode parquet files exist where data_path says they are
    for i in range(3):
        rel = info["data_path"].format(episode_chunk=0, episode_index=i)
        assert (root / rel).exists()
    # meta sidecars are present and line-counted
    for name, count in [
        ("tasks.jsonl", 1),
        ("episodes.jsonl", 3),
        ("episodes_stats.jsonl", 3),
    ]:
        lines = (root / "meta" / name).read_text().splitlines()
        assert len([ln for ln in lines if ln.strip()]) == count


def test_monotonic_global_index_across_episodes(tmp_path):
    import pyarrow.parquet as pq

    rng = np.random.default_rng(2)
    feats = lr.features_from_robot(
        __import__("jaxterity").zoo.load("so100"),
        joint_names=lr.SO100_JOINTS,
        include_goal=False,
    )
    eps = _make_episodes(rng, n=3, include_goal=False)
    root = tmp_path / "ds"
    lr.write_lerobot_dataset(root, eps, feats, fps=30, robot_type="so100")

    idx = []
    for i in range(3):
        t = pq.read_table(root / f"data/chunk-000/episode_{i:06d}.parquet")
        idx.extend(t.column("index").to_pylist())
    assert idx == list(range(sum(e.length for e in eps)))


def test_refuses_to_overwrite(tmp_path):
    rng = np.random.default_rng(3)
    feats = lr.features_from_robot(
        __import__("jaxterity").zoo.load("so100"),
        joint_names=lr.SO100_JOINTS,
        include_goal=False,
    )
    eps = _make_episodes(rng, n=2, include_goal=False)
    root = tmp_path / "ds"
    lr.write_lerobot_dataset(root, eps, feats, fps=30, robot_type="so100")
    with pytest.raises(FileExistsError):
        lr.write_lerobot_dataset(root, eps, feats, fps=30, robot_type="so100")


def test_ragged_episode_rejected(tmp_path):
    feats = lr.features_from_robot(
        __import__("jaxterity").zoo.load("so100"),
        joint_names=lr.SO100_JOINTS,
        include_goal=False,
    )
    bad = lr.Episode(
        data={
            "observation.state": np.zeros((5, 6), "f4"),
            "action": np.zeros((4, 6), "f4"),  # mismatched length
        },
        task="reach",
    )
    with pytest.raises(ValueError):
        lr.write_lerobot_dataset(
            tmp_path / "ds", [bad], feats, fps=30, robot_type="so100"
        )


@pytest.mark.skipif(not lr.HAS_LEROBOT, reason="needs the [lerobot] extra")
def test_real_lerobot_can_load_written_dataset(tmp_path):  # pragma: no cover
    """When ``lerobot`` is installed, it loads a dataset we wrote."""
    rng = np.random.default_rng(4)
    feats = lr.features_from_robot(
        __import__("jaxterity").zoo.load("so100"),
        joint_names=lr.SO100_JOINTS,
        include_goal=True,
    )
    eps = _make_episodes(rng, n=3)
    root = tmp_path / "ds"
    lr.write_lerobot_dataset(root, eps, feats, fps=30, robot_type="so100")
    ds = lr.load_lerobot_dataset(root, prefer_lerobot=True)
    # real LeRobotDataset exposes num_episodes / num_frames
    assert int(getattr(ds, "num_episodes", len(ds))) == 3
