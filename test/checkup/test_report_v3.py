# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""The v3 report card: total uncertainty and the pose a parameter was measured at.

Both were measured to matter on this hardware before they were surfaced — the
least-squares interval understated damping by ~4x, and shoulder_pan's damping
moved 3.29x with pose alone under an identical excitation recipe. A report that
omitted them was making a quantified claim that was wrong, not merely an
incomplete one.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import pytest

from jaxterity.checkup import CheckupReport, CheckupStore, run_checkup
from jaxterity.checkup.report import SCHEMA, SCHEMA_V1, SCHEMA_V2, _composition_gate
from jaxterity.errors import CalibrationError

_DATA = pathlib.Path(__file__).parents[1] / "calibrate" / "data" / "lekiwi"
_CAL = str(_DATA / "calibration_final.json")
_SESSIONS = pathlib.Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def report():
    captures = sorted(str(p) for p in _DATA.glob("commanded_*.json"))
    return run_checkup(captures, serial="LK-0001", calibration=_CAL)


# --- total uncertainty on the report --------------------------------------


def test_the_schema_is_v3(report):
    assert report.schema == SCHEMA == "jaxterity-checkup-v3"


def test_every_joint_carries_a_total_interval_wider_than_its_wald_one(report):
    for joint in report.joints:
        if joint.damping is None:
            continue
        assert joint.damping_ci95 is not None
        assert joint.damping_total_ci95 is not None
        wald = joint.damping_ci95[1] - joint.damping_ci95[0]
        total = joint.damping_total_ci95[1] - joint.damping_total_ci95[0]
        assert total >= wald, joint.joint


def test_the_real_unit_was_understating_every_channel_not_just_one(report):
    # The finding this wiring exists for. Earlier work measured it on
    # shoulder_pan; running it across the report shows it is every channel.
    ratios = [
        (j.damping_total_ci95[1] - j.damping_total_ci95[0])
        / (j.damping_ci95[1] - j.damping_ci95[0])
        for j in report.joints
        if j.damping_ci95 and j.damping_total_ci95
    ]
    assert len(ratios) == 6
    assert all(r > 1.5 for r in ratios), ratios


def test_the_bias_share_is_recorded_per_joint(report):
    for joint in report.joints:
        if joint.damping_total_ci95 is None:
            continue
        assert joint.damping_bias_fraction is not None
        assert 0.0 <= joint.damping_bias_fraction <= 1.0


def test_turning_the_ladder_off_reproduces_a_v2_style_joint(report):
    captures = sorted(str(p) for p in _DATA.glob("commanded_*.json"))
    plain = run_checkup(
        captures, serial="LK-0001", calibration=_CAL, total_uncertainty=False
    )
    for joint in plain.joints:
        assert joint.damping_total_ci95 is None
        assert joint.damping_bias_fraction is None
    # The parameters themselves are untouched by measuring their uncertainty.
    assert [j.damping for j in plain.joints] == [j.damping for j in report.joints]


# --- the composition gate --------------------------------------------------


def test_a_unit_wide_problem_produces_one_gate_not_six(report):
    gates = [g for g in report.gates if g.gate == "uncertainty_composition"]
    assert len(gates) == 1, [g.scope for g in gates]
    assert gates[0].scope == "unit"
    assert gates[0].level == "warn"
    # It must name the cause, since it is not a joint fault.
    assert "property of the capture protocol" in gates[0].detail
    assert "6 of 6" in gates[0].detail


def test_a_minority_of_affected_channels_stays_per_joint(report):
    # One odd channel out of six is a joint-level observation and is reported as
    # one; it is only the unit-wide case that collapses to a single gate.
    joints = list(report.joints)
    for i, joint in enumerate(joints):
        joints[i] = dataclasses.replace(joint, damping_bias_fraction=0.1)
    joints[0] = dataclasses.replace(joints[0], damping_bias_fraction=0.9)
    gates = _composition_gate(joints)
    assert [g.scope for g in gates] == [joints[0].joint]


def test_no_measured_channels_means_no_gate(report):
    joints = [dataclasses.replace(j, damping_bias_fraction=None) for j in report.joints]
    assert _composition_gate(joints) == []


def test_the_confidence_gate_judges_the_total_interval(report):
    # A channel whose Wald interval is tight but whose total is wide must warn:
    # gating on the least-squares interval alone passed exactly this case.
    joint = next(j for j in report.joints if j.damping)
    widened = dataclasses.replace(
        joint,
        damping_ci95=(joint.damping - 1e-6, joint.damping + 1e-6),
        # The gate fires above ~2x the value in relative half-width
        # (_CONFIDENCE_REL_SE * 1.96 * 2), so clear it decisively rather than
        # encoding the exact threshold here.
        damping_total_ci95=(joint.damping * -1.5, joint.damping * 3.5),
        damping_bias_fraction=0.99,
    )
    from jaxterity.checkup.report import _joint_gates

    gates = [g for g in _joint_gates(widened, None) if g.gate == "confidence"]
    assert len(gates) == 1
    assert "capture-length dependence" in gates[0].detail
    assert "Capture longer" in gates[0].remedy


# --- the pose a parameter was measured at ---------------------------------


def test_the_yaw_axis_is_scored_against_inertia_and_the_rest_against_gravity(report):
    features = {j.joint: j.pose_feature_name for j in report.joints}
    assert features["shoulder_pan"] == "axis_inertia_kg_m2"
    for joint, name in features.items():
        if joint != "shoulder_pan":
            assert name == "gravity_torque_Nm", joint


def test_every_joint_records_the_pose_it_was_measured_at(report):
    for joint in report.joints:
        assert joint.pose_feature is not None, joint.joint


def test_without_a_calibration_the_pose_feature_is_absent_not_invented():
    captures = sorted(str(p) for p in _DATA.glob("commanded_*.json"))[:1]
    plain = run_checkup(captures, serial="LK-0001", total_uncertainty=False)
    assert all(j.pose_feature is None for j in plain.joints)
    assert all(j.pose_feature_name == "" for j in plain.joints)


# --- round-tripping and back-compatibility --------------------------------


def test_the_report_round_trips_through_json(report):
    restored = CheckupReport.from_dict(json.loads(json.dumps(report.to_dict())))
    assert restored.schema == SCHEMA
    for before, after in zip(report.joints, restored.joints, strict=True):
        assert after.damping_total_ci95 == pytest.approx(before.damping_total_ci95)
        assert after.damping_bias_fraction == pytest.approx(
            before.damping_bias_fraction
        )
        assert after.pose_feature == pytest.approx(before.pose_feature)
        assert after.pose_feature_name == before.pose_feature_name


def test_a_v2_report_still_reads_back_with_the_new_fields_empty():
    doc = json.loads((_SESSIONS / "lk0001_v2_session_a.json").read_text())
    assert doc["schema"] == SCHEMA_V2
    restored = CheckupReport.from_dict(doc)
    assert restored.schema == SCHEMA_V2
    for joint in restored.joints:
        assert joint.damping_total_ci95 is None
        assert joint.pose_feature is None


def test_a_v1_report_is_still_accepted():
    doc = json.loads((_SESSIONS / "lk0001_v2_session_a.json").read_text())
    restored = CheckupReport.from_dict({**doc, "schema": SCHEMA_V1})
    assert restored.schema == SCHEMA_V1


def test_an_unknown_schema_is_refused_naming_all_three():
    doc = json.loads((_SESSIONS / "lk0001_v2_session_a.json").read_text())
    with pytest.raises(CalibrationError) as exc:
        CheckupReport.from_dict({**doc, "schema": "jaxterity-checkup-v99"})
    assert SCHEMA in str(exc.value) and SCHEMA_V1 in str(exc.value)


# --- pose attribution across runs -----------------------------------------


def test_pose_attribution_separates_geometry_from_wear(tmp_path, report):
    """The quantitative half of the canonical-pose contract.

    Two stored runs whose damping tripled *and* whose pose feature tripled must
    come back with a normalized ratio near 1.0 — that is geometry, not a joint
    falling apart. This is the LK-0001 case: 3.29x damping against 3.42x inertia.
    """
    store = CheckupStore(tmp_path)
    first = report
    scaled = dataclasses.replace(
        report,
        joints=tuple(
            dataclasses.replace(
                j,
                damping=(j.damping or 0.0) * 3.29,
                pose_feature=(j.pose_feature or 1.0) * 3.42,
            )
            for j in report.joints
        ),
        created_utc="2026-07-30T00:00:00Z",
    )
    store.save(first)
    store.save(scaled)
    attribution = store.pose_attribution(report.serial)
    assert attribution
    for joint, values in attribution.items():
        assert values["damping_ratio"] == pytest.approx(3.29, rel=1e-3), joint
        assert values["feature_ratio"] == pytest.approx(3.42, rel=1e-3), joint
        assert values["normalized_ratio"] == pytest.approx(0.962, rel=1e-2), joint


def test_real_wear_survives_the_pose_normalization(tmp_path, report):
    # The converse: damping doubles while the pose does not move. That is the
    # signal the normalization must not explain away.
    store = CheckupStore(tmp_path)
    store.save(report)
    store.save(
        dataclasses.replace(
            report,
            joints=tuple(
                dataclasses.replace(j, damping=(j.damping or 0.0) * 2.0)
                for j in report.joints
            ),
            created_utc="2026-07-30T00:00:00Z",
        )
    )
    for values in store.pose_attribution(report.serial).values():
        assert values["feature_ratio"] == pytest.approx(1.0)
        assert values["normalized_ratio"] == pytest.approx(2.0)


def test_pose_attribution_needs_two_runs(tmp_path, report):
    store = CheckupStore(tmp_path)
    store.save(report)
    assert store.pose_attribution(report.serial) == {}


def test_joints_without_a_recorded_feature_are_omitted_not_guessed(tmp_path, report):
    store = CheckupStore(tmp_path)
    blank = tuple(dataclasses.replace(j, pose_feature=None) for j in report.joints)
    store.save(dataclasses.replace(report, joints=blank))
    store.save(
        dataclasses.replace(report, joints=blank, created_utc="2026-07-30T00:00:00Z")
    )
    assert store.pose_attribution(report.serial) == {}


def test_html_quotes_the_total_interval(report):
    from jaxterity.checkup import render_report_html

    html = render_report_html(report)
    assert "damping_total_ci95" in html
    assert "capture-length" in html
