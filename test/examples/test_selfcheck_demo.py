# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""End-to-end predictive self-check demo — composition + reproducibility.

The PR-tier checks run the fast (committed-calibration) path with an injected
clock so the zero-deadline-misses assertion is deterministic; a nightly check
runs the full ``--active`` self-ID path. The point is that the self-check stages
actually compose into one reproducible story — full hardware validation needs
physical hardware (asserted nowhere here).
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

pytest.importorskip("mujoco")

_DEMO = (
    pathlib.Path(__file__).resolve().parents[2] / "examples" / "selfcheck_demo_so101.py"
)
_spec = importlib.util.spec_from_file_location("selfcheck_demo_so101", _DEMO)
demo = importlib.util.module_from_spec(_spec)
# Register before exec: the module-level @dataclass looks itself up in
# sys.modules while it is being processed.
sys.modules["selfcheck_demo_so101"] = demo
_spec.loader.exec_module(demo)


def _fast_clock(n_ticks):
    """A clock whose every tick is ~1 ms — deterministic zero deadline misses."""
    vals = []
    t = 0.0
    for _ in range(n_ticks):
        vals.append(t)
        t += 0.001
        vals.append(t)
        t += 1.0
    it = iter(vals)
    return lambda: next(it)


def test_wedge_composes_end_to_end():
    n = 8
    r = demo.run_demo(active=False, n_ticks=n, rate_hz=20.0, clock=_fast_clock(n))
    # 1-2: attested twin handle present.
    assert len(r.twin_handle) == 64
    # 3: co-processor held the rate (injected fast clock → zero misses).
    assert r.deadline_misses == 0 and r.ticks_published == n
    # 4: the catch — body-only proceeds, twin vetoes on the servo-thermal limit.
    assert r.body_only_proceeds is True
    assert r.safety_action == "veto"
    assert r.safety_constraint == "servo_temperature"
    assert r.time_to_breach_s is not None
    # 5: pre-flight rejects the unsafe trajectory, passes the safe one.
    assert r.preflight_safe_passed is True
    assert r.preflight_unsafe_passed is False


def test_wedge_is_reproducible_from_hashes():
    n = 6
    a = demo.run_demo(active=False, n_ticks=n, rate_hz=20.0, clock=_fast_clock(n))
    b = demo.run_demo(active=False, n_ticks=n, rate_hz=20.0, clock=_fast_clock(n))
    assert a.run_hash == b.run_hash


def test_main_prints_story(capsys):
    demo.main(active=False)
    out = capsys.readouterr().out
    assert "attested twin handle" in out
    assert "run hash" in out


@pytest.mark.nightly
def test_wedge_with_active_self_id():
    """The full path including live loopback active sysid (slow: LM + jacfwd)."""
    r = demo.run_demo(active=True, n_ticks=4, rate_hz=10.0)
    assert r.safety_action == "veto"
    assert r.preflight_unsafe_passed is False
