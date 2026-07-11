# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""End-to-end real-LeKiwi predictive self-check chain.

Exercises ``examples/selfcheck_demo_lekiwi.py``: the whole chain wired on real
artifacts (real friction calibration + real battery fit) — active self-ID →
attested twin → co-processor (zero misses, injected clock) → thermal veto AND
brownout veto that a body-only model proceeds on → preflight rejects unsafe — and
asserts it is reproducible from the ``run_hash``.
"""

from __future__ import annotations

import importlib.util
import itertools
import pathlib
import sys

import pytest

pytest.importorskip("mujoco")

_DEMO = (
    pathlib.Path(__file__).resolve().parents[2]
    / "examples"
    / "selfcheck_demo_lekiwi.py"
)
_spec = importlib.util.spec_from_file_location("selfcheck_demo_lekiwi", _DEMO)
demo = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = demo  # dataclass + `X | None` needs the module registered
_spec.loader.exec_module(demo)


def _fast_clock():
    """A clock whose every tick is ~1 ms — deterministic zero deadline misses."""
    return itertools.count(0.0, 1e-3).__next__


def test_real_wedge_chain_composes_end_to_end():
    r = demo.run_demo(active=False, rate_hz=20.0, clock=_fast_clock())
    # 1: the calibration driving the twin is the real (non-synthetic) fit.
    assert r.calibration_synthetic is False
    # 2: attested twin + real battery fit (healthy ~12 V pack, sane R_int).
    assert len(r.twin_handle) == 64
    assert 11.5 < r.battery_v_oc < 12.6 and 0.0 < r.battery_r_int < 1.0
    # 3: co-processor held the rate.
    assert r.deadline_misses == 0
    # 4a: thermal check — twin vetoes servo temperature; body-only proceeds.
    assert r.thermal_action == "veto"
    assert r.thermal_constraint == "servo_temperature"
    assert r.thermal_time_to_breach_s is not None
    assert r.body_only_thermal_proceeds is True
    # 4b: brownout check — twin vetoes bus voltage; body-only (no battery) proceeds.
    assert r.brownout_action == "veto"
    assert r.brownout_constraint == "bus_voltage"
    assert r.brownout_v_min < 10.0
    assert r.body_only_brownout_proceeds is True
    # 5: pre-flight rejects the unsafe trajectory, passes the safe one.
    assert r.preflight_safe_passed is True
    assert r.preflight_unsafe_passed is False


def test_real_wedge_is_reproducible_from_hashes():
    a = demo.run_demo(active=False, rate_hz=20.0, clock=_fast_clock())
    b = demo.run_demo(active=False, rate_hz=20.0, clock=_fast_clock())
    assert a.run_hash == b.run_hash


@pytest.mark.nightly
def test_active_self_id_path_refits_from_real_capture():
    # The --active path re-runs the closed-loop fit from the committed real capture.
    r = demo.run_demo(active=True, rate_hz=20.0, clock=_fast_clock())
    assert r.thermal_action == "veto"
    assert r.brownout_action == "veto"
