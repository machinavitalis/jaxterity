# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Twin currency / aging — windowed re-fit tracks a drifting unit.

A unit's true thermal leaves are drifted across a stream of recalibration
windows; the windowed re-fit must follow the drift within tolerance, and each
re-fit must advance + link the attestation handle into an unbroken chain (the
provenance-chain integrity check).
"""

from __future__ import annotations

import numpy as np
import pytest

from jaxterity.data import ThermalTelemetry
from jaxterity.physics.thermal import thermal_response
from jaxterity.testing import tolerances as tol
from jaxterity.twin import (
    TwinHistory,
    so101_twin,
    track_thermal_drift,
    update_twin_thermal,
)

pytest.importorskip("mujoco")


@pytest.fixture(scope="module")
def twin():
    return so101_twin("SO101-AGING")


def _window(twin, *, r_th, k_pwr, tau_amp=0.8, fs=50.0, n=600):
    """A thermal window consistent with the given (uniform) true leaves."""
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
    tau = np.full((n, nv), tau_amp)
    return telem, tau


def test_windowed_refit_tracks_a_drifting_unit(twin):
    """As true R_th drifts up across windows, the tracked twin follows it."""
    r_drift = [3.0, 3.5, 4.0, 4.5]
    windows = [_window(twin, r_th=r, k_pwr=6.0) for r in r_drift]
    final, history = track_thermal_drift(twin, windows)
    assert len(history.updates) == len(r_drift)
    rtol = max(tol.rtol("motor_temperature", "rk4", "float64"), 1e-3)
    # Each re-fit recovers that window's true R_th.
    for r_true, update in zip(r_drift, history.updates):
        np.testing.assert_allclose(update.value("offbody.R_th"), r_true, rtol=rtol)
    # The final twin reflects the latest window.
    np.testing.assert_allclose(np.asarray(final.R_th), r_drift[-1], rtol=rtol)


def test_each_refit_links_to_its_predecessor(twin):
    """The version history is an unbroken handle chain, all handles distinct."""
    windows = [_window(twin, r_th=r, k_pwr=6.0) for r in (3.0, 3.6, 4.2)]
    _, history = track_thermal_drift(twin, windows)
    assert history.is_linked_chain
    handles = history.handles
    assert len(set(handles)) == len(handles)  # every version distinct
    # The first update links to the initial handle; each next to its predecessor.
    assert history.updates[0].prior_handle == history.initial_handle
    for prev, cur in zip(history.updates, history.updates[1:]):
        assert cur.prior_handle == prev.handle


def test_update_carries_t210_uncertainty(twin):
    """A re-fit carries the fit's per-leaf uncertainty onto the update."""
    telem, tau = _window(twin, r_th=4.0, k_pwr=6.0)
    refit, update = update_twin_thermal(twin, telem, tau, index=0)
    keys = {k for k, _ in update.uncertainty}
    assert keys == {"offbody.R_th", "offbody.k_pwr"}
    # The twin itself also exposes it.
    assert set(refit.uncertainty()) == {"offbody.R_th", "offbody.k_pwr"}


def test_parameter_series_for_drift_view(twin):
    """parameter_series returns the per-update timeseries the drift view needs."""
    windows = [
        (*_window(twin, r_th=r, k_pwr=6.0), float(i))
        for i, r in enumerate((3.0, 3.5, 4.0))
    ]
    _, history = track_thermal_drift(twin, windows)
    times, values = history.parameter_series("offbody.R_th")
    assert times.shape == (3,)
    assert values.shape == (3, twin.n_servos)
    np.testing.assert_allclose(times, [0.0, 1.0, 2.0])
    # Values increase across updates (the drift is visible in the series).
    assert values[:, 0].tolist() == pytest.approx(sorted(values[:, 0].tolist()))


def test_empty_history_is_a_linked_chain(twin):
    history = TwinHistory.empty(twin)
    assert history.is_linked_chain
    assert history.handles == (twin.attestation_handle,)
    times, values = history.parameter_series("offbody.R_th")
    assert times.shape == (0,)
