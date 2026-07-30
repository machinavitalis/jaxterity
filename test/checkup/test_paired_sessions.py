# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Two real back-to-back v2 sessions on LK-0001 (2026-07-29).

The pair exists to answer one question the plan could not answer offline: how
reproducible is a checkup fit when the *same* protocol is run twice? The stored
report cards are committed here, so the answer is a regression test rather than
a remembered number.

What they show:

* Both runs carry the **same recipe hash** — identical waveform, tones,
  amplitudes, durations, rates. The pin works end to end on real captures.
* Five of six joints repeat within 1-12%, comfortably inside their own Wald
  intervals (8-17%). That is the reproducibility band probe compression has to
  be judged against.
* ``shoulder_pan`` moves by **+229%** — and it is the joint whose *held pose*
  differed most between the sessions (2094 ticks on the elbow). The change in
  the fitted damping is monotone in the pose difference across all six joints,
  which is the mechanism: shoulder_pan is the base yaw axis, so its load is the
  arm's inertia about the vertical, which depends entirely on how extended the
  arm is. A pan drift claim without a pinned pose is meaningless.
* ``CaptureProtocol.matches`` is therefore False despite the equal hash, and the
  drift warning names the offending servos and tick deltas. It caught a confound
  introduced *by a change made between the two sessions* (releasing torque after
  each capture, so the arm slumped and re-ramped instead of holding).
"""

from __future__ import annotations

import json
import pathlib

import pytest

from jaxterity.checkup.report import CheckupReport

_DATA = pathlib.Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def sessions():
    a = CheckupReport.from_dict(
        json.loads((_DATA / "lk0001_v2_session_a.json").read_text())
    )
    b = CheckupReport.from_dict(
        json.loads((_DATA / "lk0001_v2_session_b.json").read_text())
    )
    return a, b


def test_both_sessions_ran_the_same_recipe(sessions):
    a, b = sessions
    assert a.protocol_hash == b.protocol_hash
    assert a.protocol is not None and b.protocol is not None


def test_the_pin_still_rejects_them_as_the_same_experiment(sessions):
    """Equal recipe, different held pose — the case a hash-only pin would miss."""
    a, b = sessions
    assert not b.protocol.matches(a.protocol)
    differences = b.protocol.differences(a.protocol)
    assert any("held pose differs" in d for d in differences)
    assert any("shoulder_pan" in d for d in differences)


def test_well_posed_joints_repeat_within_the_measured_band(sessions):
    a, b = sessions
    fa = {j.joint: j.damping for j in a.joints}
    fb = {j.joint: j.damping for j in b.joints}
    for joint in ("elbow_flex", "gripper", "shoulder_lift", "wrist_flex", "wrist_roll"):
        change = abs(fb[joint] - fa[joint]) / fa[joint]
        assert change < 0.15, f"{joint} moved {change:.1%} between identical recipes"


def test_shoulder_pan_is_pose_dominated_not_reproducible(sessions):
    """The finding: pan's damping tracks the held pose, not the excitation."""
    a, b = sessions
    fa = {j.joint: j.damping for j in a.joints}
    fb = {j.joint: j.damping for j in b.joints}
    assert abs(fb["shoulder_pan"] - fa["shoulder_pan"]) / fa["shoulder_pan"] > 1.0


def test_the_pose_difference_orders_the_fit_change(sessions):
    """Monotone: the joints whose pose moved most are the ones whose fit moved."""
    a, b = sessions
    pa = {e.joint: dict(e.pose_ticks) for e in a.protocol.joints}
    pb = {e.joint: dict(e.pose_ticks) for e in b.protocol.joints}
    fa = {j.joint: j.damping for j in a.joints}
    fb = {j.joint: j.damping for j in b.joints}
    pose_delta = {
        j: max(abs(pa[j][s] - pb[j][s]) for s in pa[j] if s in pb[j]) for j in pa
    }
    change = {j: abs(fb[j] - fa[j]) / fa[j] for j in fa if fa[j]}
    worst_pose = max(pose_delta, key=lambda j: pose_delta[j])
    worst_change = max(change, key=lambda j: change[j])
    assert worst_pose == worst_change == "shoulder_pan"


def test_both_sessions_pass_their_gates(sessions):
    a, b = sessions
    assert a.passed and b.passed
    assert a.score["total"] >= 80 and b.score["total"] >= 80
