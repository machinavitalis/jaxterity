# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Per-unit off-body thermal identification.

Plants per-servo R_th/k_pwr, synthesizes a model-consistent temperature trace via
the analytic first-order response, and asserts the C_th-known fit recovers
them and that `calibrate_twin_thermal` updates the twin (new handle + provenance).
"""

from __future__ import annotations

import numpy as np
import pytest

from jaxterity.calibrate import calibrate_twin_thermal, fit_servo_thermal
from jaxterity.data import ThermalTelemetry
from jaxterity.physics.thermal import thermal_response
from jaxterity.testing import tolerances as tol
from jaxterity.twin import so101_twin

pytest.importorskip("mujoco")

_R_TRUE = np.array([3.0, 4.0, 5.0, 3.5, 4.5, 2.5])
_K_TRUE = np.array([5.0, 6.0, 7.0, 5.5, 6.5, 4.5])
_TAU = np.array([0.8, 0.9, 1.0, 0.7, 0.6, 0.5])


@pytest.fixture(scope="module")
def twin():
    return so101_twin("SO101-THERMAL")


def _synthetic_trace(twin, *, fs=50.0, n=1000):
    """A per-servo temperature trace consistent with the planted params."""
    C_th = np.asarray(twin.C_th)
    T_amb = twin.T_amb
    t = np.arange(n, dtype=np.float64) / fs
    cols = [
        np.asarray(
            thermal_response(
                t,
                _TAU[j],
                T0=T_amb,
                T_amb=T_amb,
                C_th=float(C_th[j]),
                R_th=_R_TRUE[j],
                k_pwr=_K_TRUE[j],
            )
        )
        for j in range(twin.n_servos)
    ]
    T = np.stack(cols, axis=1)
    tau = np.tile(_TAU, (n, 1))
    telem = ThermalTelemetry(
        time=t,
        sample_rate_hz=fs,
        servo_names=twin.servo_names,
        temperature=T,
    )
    return telem, tau


def test_fit_recovers_per_servo_r_th_and_k_pwr(twin):
    telem, tau = _synthetic_trace(twin)
    fit = fit_servo_thermal(telem, tau, C_th=np.asarray(twin.C_th), T_amb=twin.T_amb)
    r = tol.rtol("motor_temperature", "rk4", "float64")
    np.testing.assert_allclose(fit["R_th"], _R_TRUE, rtol=max(r, 1e-3))
    np.testing.assert_allclose(fit["k_pwr"], _K_TRUE, rtol=max(r, 1e-3))


def test_fit_rejects_nonpositive_c_th(twin):
    telem, tau = _synthetic_trace(twin)
    with pytest.raises(ValueError, match="C_th"):
        fit_servo_thermal(telem, tau, C_th=0.0, T_amb=twin.T_amb)


def test_fit_rejects_shape_mismatch(twin):
    telem, tau = _synthetic_trace(twin)
    with pytest.raises(ValueError, match="same 2-D"):
        fit_servo_thermal(
            telem, tau[:, :3], C_th=np.asarray(twin.C_th), T_amb=twin.T_amb
        )


def test_calibrate_twin_thermal_updates_twin(twin):
    telem, tau = _synthetic_trace(twin)
    fitted = calibrate_twin_thermal(twin, telem, tau)
    # the fitted leaves replace the representative defaults.
    np.testing.assert_allclose(np.asarray(fitted.R_th), _R_TRUE, rtol=1e-3)
    np.testing.assert_allclose(np.asarray(fitted.k_pwr), _K_TRUE, rtol=1e-3)
    # the unit's identity moved, with a thermal-sysid provenance record.
    assert fitted.attestation_handle != twin.attestation_handle
    assert any(rec[0] == "thermal.sysid" for rec in fitted.provenance)


def test_with_thermal_keeps_unset_leaves(twin):
    # only k_pwr replaced → C_th and R_th unchanged, handle still moves.
    new_k = np.asarray(twin.k_pwr) + 1.0
    updated = twin.with_thermal(k_pwr=new_k)
    np.testing.assert_array_equal(np.asarray(updated.C_th), np.asarray(twin.C_th))
    np.testing.assert_array_equal(np.asarray(updated.R_th), np.asarray(twin.R_th))
    np.testing.assert_allclose(np.asarray(updated.k_pwr), new_k)
    assert updated.attestation_handle != twin.attestation_handle
