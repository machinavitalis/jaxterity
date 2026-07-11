# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Read and write the LeRobot ``LeRobotDataset`` format from Jaxterity.

`LeRobot <https://github.com/huggingface/lerobot>`_ is HuggingFace's open-source
robot-learning stack, and the SO-100/SO-101 arm in the Jaxterity zoo is its
canonical low-cost manipulator. This module is the seam between the two: it
serialises rollouts collected in a Jaxterity simulation into the on-disk
``LeRobotDataset`` v2.1 layout (so a LeRobot policy — ACT, diffusion, … — can
train on Jaxterity-simulated demonstrations), and it reads that layout back,
either through the real ``lerobot`` package when it is installed or through a
small dependency-light reader so the round-trip works in CI.

Jaxterity owns the *robot model and the simulated data*; it does **not** ship a
trainer (CONTEXT.md). The contract here is purely the dataset format — the
open-source piece we lean on rather than reinvent.

On-disk layout written by :func:`write_lerobot_dataset` (LeRobotDataset v2.1)::

    <root>/
      meta/info.json            # schema: fps, robot_type, features, totals
      meta/tasks.jsonl          # {task_index, task}
      meta/episodes.jsonl       # {episode_index, tasks, length}
      meta/episodes_stats.jsonl # per-episode per-feature min/max/mean/std
      data/chunk-000/episode_000000.parquet   # one parquet per episode

Each parquet row is one control frame; vector features (``observation.state``,
``action``, …) are stored as fixed-length ``list<float32>`` columns, exactly as
LeRobot expects.

Writing and the dependency-light reader require ``pyarrow`` (the ``[lerobot]``
or ``[data]`` extra). Loading through the real library additionally needs the
``lerobot`` package (``pip install jaxterity[lerobot]``).
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

#: True when the real ``lerobot`` package is importable. When False, callers
#: still get a working local reader from :func:`load_lerobot_dataset`.
HAS_LEROBOT = importlib.util.find_spec("lerobot") is not None

#: The ``LeRobotDataset`` on-disk schema version this module reads and writes.
CODEBASE_VERSION = "v2.1"

#: Frames per chunk directory. LeRobot groups episodes into ``chunk-{n:03d}``
#: directories of this many episodes; the nano demos never approach it.
DEFAULT_CHUNK_SIZE = 1000

_DATA_PATH_TMPL = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"


def _require_pyarrow():
    """Import pyarrow or raise a structured, fix-oriented error."""
    try:
        import pyarrow  # noqa: F401
        import pyarrow.parquet  # noqa: F401
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "jaxterity.integrations.lerobot needs pyarrow to read/write parquet. "
            "Install it with `pip install jaxterity[lerobot]` (or `[data]`)."
        ) from e
    import pyarrow

    return pyarrow


# --------------------------------------------------------------------------- #
# Feature schema
# --------------------------------------------------------------------------- #


def vector_feature(names: Sequence[str], *, dtype: str = "float32") -> dict:
    """A LeRobot vector-feature spec (e.g. ``observation.state``).

    Args
    ----
    names : sequence of str
        Per-element names (one per joint/DoF). ``shape`` is ``[len(names)]``.
    dtype : str
        LeRobot dtype string. Defaults to ``"float32"``.

    Returns
    -------
    dict
        ``{"dtype", "shape", "names"}`` as stored under ``info.json["features"]``.
    """
    return {"dtype": dtype, "shape": [len(names)], "names": list(names)}


#: Joint order MuJoCo/MJX assigns the SO-100/SO-101 — identical to LeRobot's
#: canonical SO-100 motor order, so ``observation.state``/``action`` columns map
#: one-to-one onto a real teleoperation dataset's columns.
SO100_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def features_from_robot(
    robot, *, joint_names: Sequence[str] | None = None, include_goal: bool = False
) -> dict:
    """Build the LeRobot ``features`` schema for a Jaxterity robot.

    Emits ``observation.state`` and ``action`` as one vector feature per joint
    (both named with the robot's revolute-joint names), plus the bookkeeping
    columns LeRobot requires (``timestamp``, ``frame_index``, ``episode_index``,
    ``index``, ``task_index``). With ``include_goal`` it also adds an
    ``observation.goal`` vector feature — used by the goal-conditioned
    nanoLeRobot reaching demo.

    Args
    ----
    robot : Robot
        Source robot; its revolute joints define the feature width and names.
        Ignored when ``joint_names`` is given.
    joint_names : sequence of str, optional
        Override the joint names (e.g. :data:`SO100_JOINTS`). When omitted, the
        robot's revolute joints are used, in the robot's joint order.
    include_goal : bool
        Add an ``observation.goal`` feature (default False).

    Returns
    -------
    dict
        The ``features`` mapping for ``info.json``.
    """
    if joint_names is None:
        from ..robot.structure import JointType

        joint_names = [
            j.name for j in robot.joints if j.joint_type == JointType.REVOLUTE
        ]
    feats: dict = {
        "observation.state": vector_feature(joint_names),
        "action": vector_feature(joint_names),
    }
    if include_goal:
        feats["observation.goal"] = vector_feature(joint_names)
    feats.update(
        {
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        }
    )
    return feats


def _vector_feature_names(features: Mapping[str, dict]) -> list[str]:
    """Names of the user vector features (everything LeRobot stores as a list)."""
    book = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    return [k for k in features if k not in book]


# --------------------------------------------------------------------------- #
# Episode container
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class Episode:
    """One recorded episode: each vector feature as a ``(T, dim)`` array.

    Attributes
    ----------
    data : dict[str, np.ndarray]
        Maps a vector-feature name (``"observation.state"``, ``"action"``, …)
        to a ``(T, dim)`` float array. Every feature must share the same ``T``.
    task : str
        Natural-language task description for the episode (LeRobot indexes
        these in ``tasks.jsonl``).
    """

    data: dict[str, np.ndarray]
    task: str

    @property
    def length(self) -> int:
        """Number of control frames ``T`` in the episode."""
        return int(next(iter(self.data.values())).shape[0])

    def _validate(self, vector_features: Sequence[str]) -> None:
        lengths = {k: np.asarray(v).shape[0] for k, v in self.data.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"Episode features disagree on length T: {lengths}. Every feature "
                "must have one row per control frame."
            )
        missing = [k for k in vector_features if k not in self.data]
        if missing:
            raise ValueError(
                f"Episode is missing features declared in the schema: {missing}. "
                f"Provided: {sorted(self.data)}."
            )


# --------------------------------------------------------------------------- #
# Writer
# --------------------------------------------------------------------------- #


def _feature_stats(arr: np.ndarray) -> dict:
    """Per-channel min/max/mean/std/count for one ``(T, dim)`` feature array."""
    arr = np.asarray(arr, dtype=np.float64)
    return {
        "min": arr.min(axis=0).astype(np.float32).tolist(),
        "max": arr.max(axis=0).astype(np.float32).tolist(),
        "mean": arr.mean(axis=0).astype(np.float32).tolist(),
        "std": arr.std(axis=0).astype(np.float32).tolist(),
        "count": [int(arr.shape[0])],
    }


def write_lerobot_dataset(
    root: str | Path,
    episodes: Sequence[Episode],
    features: Mapping[str, dict],
    *,
    fps: int,
    robot_type: str,
    default_task: str = "reach",
) -> Path:
    """Write ``episodes`` to ``root`` in the LeRobotDataset v2.1 layout.

    Args
    ----
    root : str or Path
        Output directory (created; must not already contain a dataset).
    episodes : sequence of Episode
        Recorded episodes. Each must carry every vector feature in ``features``.
    features : mapping
        The schema from :func:`features_from_robot`.
    fps : int
        Control rate, written to ``info.json`` and used to fill ``timestamp``.
    robot_type : str
        Robot identifier (e.g. ``"so100"``), written to ``info.json``.
    default_task : str
        Task string used for any episode whose ``task`` is empty.

    Returns
    -------
    Path
        ``root`` as a :class:`pathlib.Path`.

    Raises
    ------
    FileExistsError
        If ``root`` already contains a ``meta/info.json``.
    ValueError
        If an episode is malformed (ragged or missing a declared feature).
    """
    pa = _require_pyarrow()
    import pyarrow.parquet as pq

    root = Path(root)
    if (root / "meta" / "info.json").exists():
        raise FileExistsError(
            f"{root} already holds a LeRobotDataset (meta/info.json). Refusing to "
            "overwrite; write to a fresh directory."
        )
    if not episodes:
        raise ValueError("write_lerobot_dataset: no episodes to write.")

    vector_features = _vector_feature_names(features)
    (root / "meta").mkdir(parents=True, exist_ok=True)

    # Resolve task strings -> stable task indices (insertion order).
    task_index: dict[str, int] = {}
    for ep in episodes:
        task = ep.task or default_task
        task_index.setdefault(task, len(task_index))

    episodes_meta: list[dict] = []
    episodes_stats: list[dict] = []
    global_index = 0  # monotonic frame index across the whole dataset

    for ep_idx, ep in enumerate(episodes):
        ep._validate(vector_features)
        task = ep.task or default_task
        t_idx = task_index[task]
        length = ep.length

        columns: dict[str, object] = {}
        for name in vector_features:
            rows = np.asarray(ep.data[name], dtype=np.float32)
            dim = int(features[name]["shape"][0])
            columns[name] = pa.array(list(rows), type=pa.list_(pa.float32(), dim))
        frame_idx = np.arange(length, dtype=np.int64)
        columns["timestamp"] = pa.array((frame_idx / float(fps)).astype(np.float32))
        columns["frame_index"] = pa.array(frame_idx)
        columns["episode_index"] = pa.array(np.full(length, ep_idx, dtype=np.int64))
        columns["index"] = pa.array(
            np.arange(global_index, global_index + length, dtype=np.int64)
        )
        columns["task_index"] = pa.array(np.full(length, t_idx, dtype=np.int64))
        global_index += length

        table = pa.table(columns)
        chunk = ep_idx // DEFAULT_CHUNK_SIZE
        rel = _DATA_PATH_TMPL.format(episode_chunk=chunk, episode_index=ep_idx)
        out = root / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out)

        episodes_meta.append(
            {"episode_index": ep_idx, "tasks": [task], "length": length}
        )
        episodes_stats.append(
            {
                "episode_index": ep_idx,
                "stats": {
                    name: _feature_stats(ep.data[name]) for name in vector_features
                },
            }
        )

    total_frames = global_index
    total_chunks = (len(episodes) - 1) // DEFAULT_CHUNK_SIZE + 1
    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": robot_type,
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_tasks": len(task_index),
        "total_videos": 0,
        "total_chunks": total_chunks,
        "chunks_size": DEFAULT_CHUNK_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": _DATA_PATH_TMPL,
        "video_path": None,
        "features": dict(features),
    }
    _write_json(root / "meta" / "info.json", info)
    _write_jsonl(
        root / "meta" / "tasks.jsonl",
        [{"task_index": i, "task": t} for t, i in task_index.items()],
    )
    _write_jsonl(root / "meta" / "episodes.jsonl", episodes_meta)
    _write_jsonl(root / "meta" / "episodes_stats.jsonl", episodes_stats)
    return root


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class LocalLeRobotDataset:
    """Dependency-light view over a ``LeRobotDataset`` directory.

    Reads only ``meta/info.json`` and the per-episode parquet files with
    ``pyarrow`` — no ``torch``/``datasets``/``lerobot`` required — so a dataset
    written by :func:`write_lerobot_dataset` round-trips in plain CI. For the
    full library (transforms, video decoding, ``torch`` tensors) install the
    ``[lerobot]`` extra and use :func:`load_lerobot_dataset`.

    Attributes
    ----------
    root : Path
        Dataset directory.
    info : dict
        Parsed ``meta/info.json``.
    episodes : list[Episode]
        One :class:`Episode` per parquet file, vector features only.
    """

    root: Path
    info: dict
    episodes: list[Episode]

    @property
    def fps(self) -> int:
        return int(self.info["fps"])

    @property
    def features(self) -> dict:
        return self.info["features"]

    def frames(self, feature: str) -> np.ndarray:
        """All frames of ``feature`` stacked across episodes, ``(N, dim)``."""
        return np.concatenate([ep.data[feature] for ep in self.episodes], axis=0)

    def __len__(self) -> int:
        return len(self.episodes)


def read_local_lerobot_dataset(root: str | Path) -> LocalLeRobotDataset:
    """Load a ``LeRobotDataset`` directory without the ``lerobot`` package."""
    pa = _require_pyarrow()
    import pyarrow.parquet as pq

    root = Path(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    vector_features = _vector_feature_names(info["features"])
    data_tmpl = info.get("data_path", _DATA_PATH_TMPL)

    episodes: list[Episode] = []
    ep_meta = {
        m["episode_index"]: m for m in _read_jsonl(root / "meta" / "episodes.jsonl")
    }
    for ep_idx in sorted(ep_meta):
        chunk = ep_idx // int(info.get("chunks_size", DEFAULT_CHUNK_SIZE))
        rel = data_tmpl.format(episode_chunk=chunk, episode_index=ep_idx)
        table = pq.read_table(root / rel)
        cols = table.to_pydict()
        data = {
            name: np.asarray(cols[name], dtype=np.float32) for name in vector_features
        }
        task = ep_meta[ep_idx]["tasks"][0]
        episodes.append(Episode(data=data, task=task))
    _ = pa  # keep the import referenced for clarity
    return LocalLeRobotDataset(root=root, info=info, episodes=episodes)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_lerobot_dataset(root: str | Path, *, prefer_lerobot: bool = True):
    """Load a dataset, preferring the real ``lerobot`` library when available.

    Args
    ----
    root : str or Path
        Dataset directory written by :func:`write_lerobot_dataset` (or any
        v2.1 ``LeRobotDataset``).
    prefer_lerobot : bool
        When True (default) and the ``lerobot`` package is installed, return a
        real ``lerobot.common.datasets.lerobot_dataset.LeRobotDataset`` rooted
        at ``root``. Otherwise return a :class:`LocalLeRobotDataset`.

    Returns
    -------
    LeRobotDataset or LocalLeRobotDataset
        The richest reader available for the current environment.
    """
    if prefer_lerobot and HAS_LEROBOT:  # pragma: no cover - needs the optional dep
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        return LeRobotDataset(repo_id=Path(root).name, root=str(root))
    return read_local_lerobot_dataset(root)


__all__ = [
    "HAS_LEROBOT",
    "CODEBASE_VERSION",
    "SO100_JOINTS",
    "Episode",
    "LocalLeRobotDataset",
    "vector_feature",
    "features_from_robot",
    "write_lerobot_dataset",
    "read_local_lerobot_dataset",
    "load_lerobot_dataset",
]
