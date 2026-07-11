# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Verifies the predictive-safety demo runs (shippable-surface rule).

Exercises the example end to end: the body-only check proceeds on every command,
and the twin self-check vetoes the sustained heavy hold (naming the binding
servo-thermal constraint).
"""

from __future__ import annotations

import importlib.util
import pathlib

import jax.numpy as jnp
import pytest

pytest.importorskip("mujoco")

_DEMO = pathlib.Path(__file__).resolve().parents[2] / "examples" / "selfcheck_so101.py"
_spec = importlib.util.spec_from_file_location("selfcheck_so101_demo", _DEMO)
demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo)


def test_body_only_always_proceeds():
    from jaxterity.twin import so101_twin

    twin = so101_twin("SO101-0001")
    # The MJX body model has no effort limit — any finite torque "proceeds".
    assert demo.body_only_proceeds(twin, jnp.full(twin.n_servos, 5.0))


def test_main_runs_and_contrasts_body_vs_twin(capsys):
    demo.main()
    out = capsys.readouterr().out
    assert "body-only" in out
    assert "veto" in out  # the sustained heavy hold is vetoed by the twin
