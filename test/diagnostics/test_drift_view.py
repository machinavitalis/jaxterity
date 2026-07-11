# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Local single-unit drift view.

The data-prep layer must turn a TwinHistory into reproducible drift
series; the Rerun logger has a smoke test (writes a non-empty .rrd, no viewer).
"""

from __future__ import annotations

import numpy as np
import pytest

from jaxterity.data import ThermalTelemetry
from jaxterity.diagnostics import log_drift_view, prepare_drift_view
from jaxterity.physics.thermal import thermal_response
from jaxterity.twin import so101_twin, track_thermal_drift

pytest.importorskip("mujoco")


@pytest.fixture(scope="module")
def twin():
    return so101_twin("SO101-DRIFT")


def _window(twin, *, r_th, k_pwr=6.0, tau_amp=0.8, fs=50.0, n=600):
    nv = twin.n_servos
    C_th = np.asarray(twin.C_th)
    T_amb = twin.T_amb
    t = np.arange(n, dtype=np.float64) / fs
    cols = [
        np.asarray(
            thermal_response(
                t,
                tau_amp,
                T0=T_amb,
                T_amb=T_amb,
                C_th=float(C_th[j]),
                R_th=r_th,
                k_pwr=k_pwr,
            )
        )
        for j in range(nv)
    ]
    telem = ThermalTelemetry(
        time=t,
        sample_rate_hz=fs,
        servo_names=twin.servo_names,
        temperature=np.stack(cols, axis=1),
    )
    return telem, np.full((n, nv), tau_amp)


@pytest.fixture(scope="module")
def history(twin):
    windows = [
        (*_window(twin, r_th=r), float(i)) for i, r in enumerate((3.0, 3.5, 4.0, 4.5))
    ]
    _, hist = track_thermal_drift(twin, windows)
    return hist


def test_prepare_is_reproducible_from_stored_records(history):
    a = prepare_drift_view(history)
    b = prepare_drift_view(history)
    for key in a.keys:
        assert np.array_equal(a.values[key], b.values[key])
        assert np.array_equal(a.times, b.times)


def test_drift_series_reflect_the_aging(history, twin):
    view = prepare_drift_view(history)
    assert "offbody.R_th" in view.keys
    assert view.times.shape == (4,)
    assert view.values["offbody.R_th"].shape == (4, twin.n_servos)
    # R_th was driven up across updates → the fitted value rises.
    r_series = view.values["offbody.R_th"][:, 0]
    assert r_series.tolist() == pytest.approx(sorted(r_series.tolist()))
    # Divergence from nominal (the first update) grows monotonically.
    dn = view.divergence_norm("offbody.R_th")
    assert dn[0] == pytest.approx(0.0, abs=1e-9)
    assert dn.tolist() == pytest.approx(sorted(dn.tolist()))


def test_explicit_nominal_baseline(history, twin):
    nominal = {"offbody.R_th": np.full(twin.n_servos, 3.0)}
    view = prepare_drift_view(history, nominal=nominal)
    # Divergence measured from 3.0 → first update (~3.0) is ~0, last (~4.5) ~1.5.
    div = view.divergence("offbody.R_th")
    assert np.all(div[0] < div[-1])
    np.testing.assert_allclose(view.nominal["offbody.R_th"], 3.0)


def test_std_series_carried_from_uncertainty(history, twin):
    view = prepare_drift_view(history)
    assert view.std["offbody.R_th"].shape == (4, twin.n_servos)
    assert np.all(view.std["offbody.R_th"] >= 0.0)


def test_empty_history_raises(twin):
    from jaxterity.twin import TwinHistory

    with pytest.raises(ValueError, match="non-empty"):
        prepare_drift_view(TwinHistory.empty(twin))


def test_log_drift_view_writes_recording(history, tmp_path):
    """Smoke test: streaming the drift view writes a non-empty .rrd (no viewer)."""
    pytest.importorskip("rerun")
    out = tmp_path / "drift.rrd"
    log_drift_view(history, save_path=out, namespace="drift/test")
    assert out.exists() and out.stat().st_size > 0
