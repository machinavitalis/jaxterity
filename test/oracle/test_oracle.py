# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""The oracle surface: capability sheets and feasibility verdicts.

The behavioral contract under test mirrors the measured ASIMOV-Agentic
failure mode: given a payload limit and an object mass (or range), the
comparison must always happen, deterministically, with honest ``uncertain``
outcomes when an estimate straddles the limit — and the capability sheet
must never fake a number it cannot compute (missing torque budgets yield a
named "unavailable", not a guess).
"""

from __future__ import annotations

import pytest

from jaxterity import zoo
from jaxterity.oracle import (
    Assessment,
    CapabilitySheet,
    Limits,
    TaskSpec,
    assess,
    capability_sheet,
)

_STS3215_TAU_NM = 2.9  # nominal stall torque at 12 V, supplied as budget


@pytest.fixture(scope="module")
def sheet() -> CapabilitySheet:
    return capability_sheet(
        zoo.load("so100"),
        robot_name="so100-test",
        tau_max_nm=_STS3215_TAU_NM,
        battery={"V_oc": 12.166, "R_int": 0.0653},
        hold_current_a=6.0,
        n_random_poses=50,
        seed=0,
    )


def test_sheet_fields_provenance_and_determinism(sheet):
    names = {f.name for f in sheet.fields}
    assert {"joints", "max_reach", "max_payload_static", "bus_margin_at_hold"} <= names
    by_name = {f.name: f for f in sheet.fields}

    # Reach: positive, sub-metre for a desk arm.
    reach = by_name["max_reach"]
    assert 0.05 < float(reach.value) < 1.5 and reach.provenance == "derived"

    # Static payload: positive, finite, physically plausible for hobby servos,
    # tagged nominal because the torque budget was supplied.
    payload = by_name["max_payload_static"]
    assert 0.01 < float(payload.value) < 10.0
    assert payload.provenance == "nominal"
    assert "static" in payload.note

    # Battery margin from the fitted Thévenin pair: V_oc - R_int*I - floor.
    bus = by_name["bus_margin_at_hold"]
    assert bus.provenance == "fitted"
    assert float(bus.value) == pytest.approx(12.166 - 0.0653 * 6.0 - 11.0, abs=1e-3)

    # Deterministic given identical inputs.
    again = capability_sheet(
        zoo.load("so100"),
        robot_name="so100-test",
        tau_max_nm=_STS3215_TAU_NM,
        battery={"V_oc": 12.166, "R_int": 0.0653},
        hold_current_a=6.0,
        n_random_poses=50,
        seed=0,
    )
    assert again.to_dict()["fields"] == sheet.to_dict()["fields"]

    # The rendered block carries the provenance legend and tags.
    text = sheet.to_text()
    assert "[fitted]" in text and "[nominal]" in text and "[derived]" in text


def test_sheet_never_guesses_payload_without_a_budget():
    bare = capability_sheet(zoo.load("so100"), n_random_poses=10, seed=0)
    payload = next(f for f in bare.fields if f.name == "max_payload_static")
    assert payload.value == "unavailable"
    note = payload.note.lower()
    assert "never guessed" in note or "not computed" in note


def test_skillet_class_misses_are_refused():
    # The measured ASIMOV miss, as arithmetic: 500 g limit, cast-iron skillet
    # (1.5–3.0 kg estimate) -> refuse, binding constraint named.
    a = assess(
        TaskSpec(action="lift", object_mass_kg=(1.5, 3.0)),
        Limits(payload_kg=0.5, source="declared"),
    )
    assert isinstance(a, Assessment)
    assert a.verdict == "refuse" and a.binding_constraint == "payload"
    assert a.margins["payload"] < 0


def test_clear_and_straddling_estimates():
    limits = Limits(payload_kg=0.5, source="declared")
    assert assess(TaskSpec("lift", object_mass_kg=0.2), limits).verdict == "proceed"
    # Straddles the limit -> uncertain (ask/measure), never a coin flip.
    mid = assess(TaskSpec("lift", object_mass_kg=(0.4, 0.6)), limits)
    assert mid.verdict == "uncertain" and "resolve" in mid.rationale
    # Within the 10% caution band below the limit -> uncertain too.
    near = assess(TaskSpec("lift", object_mass_kg=0.47), limits)
    assert near.verdict == "uncertain"


def test_missing_information_is_named_not_passed():
    # A judgeable quantity with no matching limit (or vice versa) lands in
    # `missing`, and judging nothing yields uncertain — silence is not a pass.
    a = assess(TaskSpec("lift", object_width_m=0.1), Limits(payload_kg=0.5))
    assert a.verdict == "uncertain"
    assert "gripper_width" in a.missing


def test_measured_sheet_plugs_in_directly(sheet):
    a = assess(TaskSpec("lift", object_mass_kg=50.0), sheet)
    assert a.verdict == "refuse" and a.limits_source == "measured"
