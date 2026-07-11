# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Smoke test for the robot Rerun demo (state streaming).

Runs the example end-to-end on the cartpole (the cheapest zoo robot) headless,
so it catches rerun-API drift and rollout/logging breakage. Self-skips without
``rerun-sdk``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("rerun")
pytest.importorskip("mujoco")

_EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "rerun_robot_demo.py"


def _load():
    spec = importlib.util.spec_from_file_location("rerun_robot_demo", _EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_demo_writes_recording(tmp_path, monkeypatch) -> None:
    demo = _load()
    out = tmp_path / "cartpole.rrd"
    monkeypatch.setattr(
        sys, "argv", ["rerun_robot_demo.py", "--robot", "cartpole", "--save", str(out)]
    )
    demo.main()  # must not raise
    assert out.exists() and out.stat().st_size > 0
