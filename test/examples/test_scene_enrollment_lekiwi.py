# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""The scene-enrollment demo composes end to end and stays reproducible.

The point of the demo is that the stages actually chain — a vision prior bounds
a probe, the probe yields a measurement, the measurement changes the unit's
attested identity, and the loaded unit decides the technique. This test asserts
the chain and the labels that keep the simulated heft honest; it asserts nothing
about hardware.
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
    / "scene_enrollment_lekiwi.py"
)
_spec = importlib.util.spec_from_file_location("scene_enrollment_lekiwi", _DEMO)
demo = importlib.util.module_from_spec(_spec)
# Register before exec: the module-level @dataclass looks itself up in
# sys.modules while it is being processed.
sys.modules["scene_enrollment_lekiwi"] = demo
_spec.loader.exec_module(demo)


@pytest.fixture(scope="module")
def result():
    return demo.run_demo(n_steps=30, dt=0.005, n_draws=2, speeds=(0.30, 0.10))


def test_demo_arc_composes(result):
    # 1. the guess is a bracket, wide and honest.
    lo, hi = result.prior_bracket_kg
    assert lo < demo.PLANTED_MASS_KG < hi
    # 2. the probe was vetted against the heavy end, not the guess's middle.
    assert result.worst_case_mass_kg == pytest.approx(hi)
    assert result.probe_margin_m >= demo.LEKIWI_BASE.margin_min_m
    # 3. the heft recovered the planted mass, with an interval and a declared
    #    prior in its provenance.
    assert result.identified_mass_kg == pytest.approx(demo.PLANTED_MASS_KG, abs=5e-3)
    ilo, ihi = result.mass_interval_kg
    assert ilo <= result.identified_mass_kg <= ihi
    assert result.recipe == "objects.payload+semantic_prior"
    # 4. holding the object is a different attested configuration.
    assert result.loaded_twin_handle != result.empty_twin_handle
    # 5. the coupled screen separated the two techniques on tip-over.
    assert result.technique == "gentle lift"
    assert result.rejected_technique == "brisk lift"
    assert result.rejected_outcome == "veto"
    assert result.binding_constraint == "tip_over"
    # 6. and the same pose with a heavier guess is refused outright.
    assert result.heavy_approved is False
    assert result.heavy_best_margin_m < demo.LEKIWI_BASE.margin_min_m


@pytest.mark.nightly
def test_demo_is_reproducible(result):
    """A second full run reproduces the first (nightly — it repays the arc)."""
    again = demo.run_demo(n_steps=30, dt=0.005, n_draws=2, speeds=(0.30, 0.10))
    assert again.run_hash == result.run_hash
