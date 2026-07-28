# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""The per-unit checkup pipeline on REAL LeKiwi telemetry.

Runs ``jaxterity.checkup`` end-to-end on the committed real captures
(``test/calibrate/data/lekiwi/``): every arm joint's commanded-excitation
capture plus the aggregate battery load trace. No hardware needed — the gate
this suite guards is that the composition (captures → per-joint fit → battery
fit → gates → report card → store → drift → HTML) stays reproducible and its
numbers stay consistent with the underlying fits it composes
(``test_closed_loop_sysid.py``, ``test_lekiwi_floor_derivation.py``).
"""

from __future__ import annotations

import json
import pathlib

import pytest

from jaxterity.checkup import (
    SCHEMA,
    CheckupReport,
    CheckupStore,
    render_report_html,
    run_checkup,
)
from jaxterity.cli import main as cli_main
from jaxterity.errors import CalibrationError

_DATA = pathlib.Path(__file__).parents[1] / "calibrate" / "data" / "lekiwi"
_CALIB = _DATA / "calibration_final.json"
_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
_NOW = "2026-07-28T00:00:00+00:00"


@pytest.fixture(scope="module")
def report() -> CheckupReport:
    return run_checkup(
        [_DATA / f"commanded_{j}.json" for j in _JOINTS],
        serial="LK-TEST-0001",
        calibration=_CALIB,
        battery_trace=_DATA / "aggregate_load_trace.json",
        now=_NOW,
    )


def test_report_composes_all_joints_and_battery(report):
    assert report.schema == SCHEMA
    assert {j.joint for j in report.joints} == set(_JOINTS)

    # The composed fit must agree with the standalone closed-loop fit gate
    # (test_closed_loop_sysid.py): shoulder_pan's known real-fit value.
    sp = next(j for j in report.joints if j.joint == "shoulder_pan")
    assert sp.damping == pytest.approx(0.4150, abs=2e-3)
    assert sp.held_out["reduction"] >= 0.45
    assert sp.damping_ci95 is not None and sp.damping_ci95[0] > 0.0
    assert sp.response_ratio is not None and 0.3 <= sp.response_ratio <= 2.5

    # Battery Thévenin values are the committed real-capture fit.
    assert report.battery is not None
    assert report.battery.V_oc == pytest.approx(12.166, abs=5e-3)
    assert report.battery.R_int == pytest.approx(0.0653, abs=5e-3)
    assert report.battery.margin_v == pytest.approx(
        report.battery.predicted_min_v - 11.0
    )

    # Score is bounded, carries every documented component, and the report
    # passes (real unit, real captures, no failing gate).
    assert set(report.score) == {
        "dynamics",
        "identification",
        "actuation",
        "power",
        "total",
    }
    assert all(0 <= v <= 100 for v in report.score.values())
    assert report.passed
    assert report.attestation_handle
    assert "shoulder_pan" in report.summary()


def test_report_is_deterministic_and_round_trips(report):
    again = run_checkup(
        [_DATA / f"commanded_{j}.json" for j in _JOINTS],
        serial="LK-TEST-0001",
        calibration=_CALIB,
        battery_trace=_DATA / "aggregate_load_trace.json",
        now=_NOW,
    )
    assert again.to_dict() == report.to_dict()
    rebuilt = CheckupReport.from_dict(json.loads(json.dumps(report.to_dict())))
    assert rebuilt.to_dict() == report.to_dict()


def test_unreasonable_floor_fails_the_margin_gate():
    small = run_checkup(
        [_DATA / "commanded_shoulder_pan.json"],
        serial="LK-TEST-0002",
        calibration=_CALIB,
        battery_trace=_DATA / "aggregate_load_trace.json",
        bus_floor_v=12.0,  # above the predicted bus minimum -> negative margin
        now=_NOW,
    )
    margin = next(g for g in small.gates if g.gate == "battery-margin")
    assert margin.level == "fail" and not small.passed


def test_empty_captures_are_rejected():
    with pytest.raises(CalibrationError, match="no capture"):
        run_checkup([], serial="LK-TEST-0003")


def test_store_orders_runs_and_reports_drift(tmp_path, report):
    store = CheckupStore(tmp_path)
    p1 = store.save(report)
    p2 = store.save(report)
    assert (p1.name, p2.name) == ("run-0001.json", "run-0002.json")
    runs = store.runs("LK-TEST-0001")
    assert len(runs) == 2 and runs[0].to_dict() == report.to_dict()
    drift = store.drift("LK-TEST-0001")
    assert drift["shoulder_pan.damping"] == [report.joints[0].damping] * 2
    assert drift["score.total"] == [float(report.score["total"])] * 2
    assert "battery.R_int" in drift
    assert store.runs("LK-NEVER-SEEN") == []


def test_html_report_embeds_the_run(report):
    html = render_report_html(report)
    assert "<!DOCTYPE html>" in html and "__CHECKUP_DATA_JSON__" not in html
    assert "LK-TEST-0001" in html and "shoulder_pan" in html
    # The inline JSON block must not terminate the script tag early.
    payload = html.split('type="application/json">')[1].split("</script>")[0]
    assert json.loads(payload.replace("<\\/", "</"))["runs"]


def test_cli_checkup_end_to_end(tmp_path, capsys):
    rc = cli_main(
        [
            "checkup",
            "--captures",
            str(_DATA),
            "--serial",
            "LK-CLI-0001",
            "--store",
            str(tmp_path),
            "--html",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "LK-CLI-0001" in out and "score" in out
    assert (tmp_path / "LK-CLI-0001" / "run-0001.json").exists()
    assert (tmp_path / "LK-CLI-0001" / "report.html").exists()
