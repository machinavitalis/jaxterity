# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Smoke test for the nanojaxterity Rerun overlay (launch viz).

Exercises the Rerun logging path with a tiny synthetic trajectory — no policy
or env needed — so it catches rerun-API drift (the kind that broke
``set_time_seconds`` / ``Scalar``). Self-skips without ``rerun-sdk``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rerun")

_EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "nanojaxterity_rerun.py"


def _load():
    spec = importlib.util.spec_from_file_location("nanojaxterity_rerun", _EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_log_trajectory_writes_recording(tmp_path) -> None:
    overlay = _load()
    n = 12
    states = np.zeros((n, 4))
    states[:, 1] = np.linspace(0.1, 0.0, n)  # pole tilt settling to upright
    states[:, 0] = np.linspace(0.0, 0.05, n)  # small cart move
    forces = np.linspace(-5.0, 0.0, n)

    out = tmp_path / "demo.rrd"
    overlay.log_trajectory(states, forces, save_path=str(out))  # must not raise

    assert out.exists()
