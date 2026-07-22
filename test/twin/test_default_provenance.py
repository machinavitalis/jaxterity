# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Twin-builder default provenance: baked constants match the committed fit.

The LeKiwi battery defaults in ``jaxterity/twin/__init__.py`` are fitted from
the real aggregate-load capture committed at
``test/calibrate/data/lekiwi/aggregate_load_trace.json`` by
``scripts/fit_twin_defaults.py`` (T-220). This test re-runs that fit from the
committed capture and asserts the baked defaults still match it — so the
defaults can never silently drift from the data they claim as provenance. The
thermal defaults are *representative* by measurement (the committed ``temp_C``
traces do not identify the thermal model — see the fit script), so only the
battery half is pinned here.
"""

from __future__ import annotations

import pathlib

import pytest

from jaxterity.calibrate import fit_battery_sag, load_bus_load_trace
from jaxterity.twin import lekiwi_twin

_TRACE = (
    pathlib.Path(__file__).parent.parent
    / "calibrate"
    / "data"
    / "lekiwi"
    / "aggregate_load_trace.json"
)

#: The baked defaults keep 4-5 significant figures of the fit, so the
#: fit-vs-constant relative error is a rounding bound, not a physics tolerance.
_ROUNDING_RTOL = 1e-3


@pytest.fixture(scope="module")
def fit():
    return fit_battery_sag(load_bus_load_trace(str(_TRACE)))


def test_lekiwi_battery_defaults_match_committed_fit(fit):
    from jaxterity.twin import _LEKIWI_R_INT, _LEKIWI_V_OC

    v_oc = float(fit.fitted["V_oc"])
    r_int = float(fit.fitted["R_int"])
    assert _LEKIWI_V_OC == pytest.approx(v_oc, rel=_ROUNDING_RTOL)
    assert _LEKIWI_R_INT == pytest.approx(r_int, rel=_ROUNDING_RTOL)
    # The fit the defaults come from is itself credible (same gate the
    # real-trace brownout test enforces).
    assert fit.residuals["r_squared"] > 0.5
    assert fit.residuals["sag_v"] > 0.3


def test_lekiwi_twin_carries_the_fitted_battery_defaults(fit):
    # The builder (not just the module constants) exposes the fitted values.
    twin = lekiwi_twin("LK-PROVENANCE")
    assert float(twin.V_oc) == pytest.approx(
        float(fit.fitted["V_oc"]), rel=_ROUNDING_RTOL
    )
    assert float(twin.R_int) == pytest.approx(
        float(fit.fitted["R_int"]), rel=_ROUNDING_RTOL
    )
