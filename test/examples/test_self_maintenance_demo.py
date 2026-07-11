# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Self-maintenance demo — composition + reproducibility.

The point is that the self-maintenance stages actually compose into one
reproducible story: a unit ages and the twin tracks it, a fault is surfaced by
name, the envelope shrinks as it heats, evidence binds to the handle — all
hashing to a stable ``run_hash``. Full hardware validation needs physical
hardware (asserted nowhere here).
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

pytest.importorskip("mujoco")

_DEMO = (
    pathlib.Path(__file__).resolve().parents[2]
    / "examples"
    / "self_maintenance_demo_so101.py"
)
_spec = importlib.util.spec_from_file_location("self_maintenance_demo_so101", _DEMO)
demo = importlib.util.module_from_spec(_spec)
sys.modules["self_maintenance_demo_so101"] = demo
_spec.loader.exec_module(demo)


@pytest.fixture(scope="module")
def result():
    return demo.run()


def test_aging_is_tracked(result):
    """The windowed re-fit tracks the drifted unit, in an unbroken handle chain."""
    assert result.n_updates == 5
    assert result.chain_linked
    assert result.tracked_r_th == pytest.approx(result.true_final_r_th, rel=1e-2)
    assert result.final_handle != result.initial_handle


def test_fault_is_surfaced_by_channel(result):
    assert result.fault_channel == "offbody.R_th[servo_2]"


def test_envelope_shrinks_under_heat(result):
    """A command the cool envelope accepts is derated/vetoed under the hot one."""
    assert result.cool_action == "ok"
    assert result.hot_action in ("derate", "veto")


def test_evidence_binds_and_run_is_reproducible(result):
    assert len(result.evidence_hash) == 64
    assert len(result.run_hash) == 64
    # Deterministic: a second run reproduces the same run hash.
    assert demo.run().run_hash == result.run_hash
