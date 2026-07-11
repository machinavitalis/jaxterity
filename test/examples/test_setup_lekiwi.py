# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Verifies the LeKiwi first-run setup example runs (shippable-surface rule).

Exercises ``examples/setup_lekiwi.py`` end to end against the simulator: the bus
is fully discovered, health passes, and all nine joints are calibrated.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

pytest.importorskip("mujoco")

_DEMO_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "examples" / "setup_lekiwi.py"
)
_spec = importlib.util.spec_from_file_location("setup_lekiwi_demo", _DEMO_PATH)
demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo)


def test_simulated_setup_calibrates_full_lekiwi():
    result = demo.run_simulated_setup("lekiwi")
    assert result.ok
    assert result.discovery.found_ids == tuple(range(1, 10))
    assert len(result.calibrations) == 9


def test_example_main_runs(capsys):
    demo.main()
    out = capsys.readouterr().out
    assert "all clear: True" in out
