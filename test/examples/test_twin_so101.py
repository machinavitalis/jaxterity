# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Verifies the SO-101 twin example runs (shippable-surface rule).

Exercises the example module end to end: builds the twin, checks the attestation
handle is a 64-hex digest and serial-sensitive, and confirms the holding-torque
forecast predicts a positive temperature rise (the off-body state the body model
never sees).
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

pytest.importorskip("mujoco")

_DEMO_PATH = pathlib.Path(__file__).resolve().parents[2] / "examples" / "twin_so101.py"
_spec = importlib.util.spec_from_file_location("twin_so101_demo", _DEMO_PATH)
demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo)


def test_demo_twin_handle_is_hex_and_serial_sensitive():
    twin = demo.build_demo_twin("SO101-0001")
    h = twin.attestation_handle
    assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)
    assert h != twin.with_serial("SO101-0002").attestation_handle


def test_holding_forecast_predicts_a_rise():
    twin = demo.build_demo_twin("SO101-0001")
    forecast = demo.predict_holding_temperature(twin, seconds=10.0, dt=0.05)
    assert forecast["rise"] > 0.0
    assert forecast["T_final"] > forecast["T0"]


def test_main_runs(capsys):
    demo.main()
    out = capsys.readouterr().out
    assert "attestation handle" in out
    assert "predicted T_final" in out
