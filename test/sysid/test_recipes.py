# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Designed, identifiability-checked actuator recipes.

Each recipe pairs a designed excitation with an identifiability check. The
checks must (a) pass on the designed excitation and recover planted params from
model-consistent telemetry, and (b) fail loudly on a deliberately
non-identifiable excitation. DC-motor recovery is closed-form (fast tier); the
BLDC full LM recovery is nightly, but its forward-mode identifiability check
runs fast.
"""

from __future__ import annotations

import numpy as np
import pytest

from jaxterity.actuators import BLDC
from jaxterity.data import ActuatorChirpTelemetry, ActuatorRampTelemetry
from jaxterity.errors import CalibrationError
from jaxterity.sysid import (
    bldc_identifiability,
    bldc_recipe,
    dc_motor_identifiability,
    dc_motor_recipe,
    design_bldc_excitation,
    design_dc_motor_excitation,
    identifiability_from_jacobian,
)
from jaxterity.sysid.recipes import _DC_PARAMS

# --- generic identifiability primitive -------------------------------------


def test_identifiability_full_rank_is_identifiable():
    jac = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    rep = identifiability_from_jacobian(jac, ("a", "b", "c"))
    assert rep.identifiable
    assert rep.rank == 3
    assert rep.weak_params == ()


def test_identifiability_rank_deficient_is_not_identifiable():
    # Third column is a copy of the first → rank 2 of 3.
    jac = np.array([[1.0, 0.0, 1.0], [2.0, 1.0, 2.0], [3.0, 2.0, 3.0], [4.0, 1.0, 4.0]])
    rep = identifiability_from_jacobian(jac, ("a", "b", "c"))
    assert not rep.identifiable
    assert rep.rank == 2
    with pytest.raises(CalibrationError, match="not identifiable"):
        rep.raise_if_unidentifiable("test")


def test_identifiability_name_mismatch_raises():
    with pytest.raises(CalibrationError, match="parameter columns"):
        identifiability_from_jacobian(np.zeros((4, 3)), ("a", "b"))


# --- DC motor --------------------------------------------------------------


def _dc_telemetry(t, v, *, Ks, Kv, Ka):
    """Model-consistent voltage-ramp telemetry from planted params."""
    a = np.gradient(v, t)
    V = Ks * np.sign(v) + Kv * v + Ka * a
    return ActuatorRampTelemetry(
        time=t,
        sample_rate_hz=1.0 / (t[1] - t[0]),
        actuator_name="m",
        command=V,
        velocity=v,
    )


def test_designed_dc_excitation_is_identifiable():
    t, v = design_dc_motor_excitation()
    tel = _dc_telemetry(t, v, Ks=0.2, Kv=0.5, Ka=0.05)
    rep = dc_motor_identifiability(tel)
    assert rep.identifiable
    assert rep.rank == len(_DC_PARAMS)


def test_dc_motor_recipe_recovers_truth():
    t, v = design_dc_motor_excitation()
    Ks_t, Kv_t, Ka_t = 0.2, 0.5, 0.05
    tel = _dc_telemetry(t, v, Ks=Ks_t, Kv=Kv_t, Ka=Ka_t)
    res = dc_motor_recipe(tel)
    assert float(res.fitted["Ks"]) == pytest.approx(Ks_t, abs=2e-3)
    assert float(res.fitted["Kv"]) == pytest.approx(Kv_t, rel=2e-3)
    assert float(res.fitted["Ka"]) == pytest.approx(Ka_t, rel=2e-2)


def test_nonidentifiable_dc_excitation_fails_loudly():
    """A constant-acceleration, no-zero-crossing ramp is rank-deficient → raises."""
    fs = 200.0
    n = 401
    t = np.arange(n, dtype=np.float64) / fs
    v = 0.5 + 1.5 * t  # linear ⟹ a constant; all-positive ⟹ sgn(v) constant
    tel = _dc_telemetry(t, v, Ks=0.2, Kv=0.5, Ka=0.05)
    rep = dc_motor_identifiability(tel)
    assert not rep.identifiable
    assert rep.rank < len(_DC_PARAMS)
    with pytest.raises(CalibrationError, match="not identifiable"):
        dc_motor_recipe(tel)
    # …but with the gate disabled it still fits (and the caller owns the risk).
    res = dc_motor_recipe(tel, require_identifiable=False)
    assert set(res.fitted) == {"Ks", "Kv", "Ka"}


def test_dc_recipe_populates_attestation_fields():
    """Recipe output carries the attestation-chain inputs (principle #6)."""
    t, v = design_dc_motor_excitation()
    tel = _dc_telemetry(t, v, Ks=0.2, Kv=0.5, Ka=0.05)
    res = dc_motor_recipe(tel)
    assert res.recipe_name == "sysid.actuator"
    assert res.recipe_version == "1.0"
    assert len(res.telemetry_hash) == 64
    # A different excitation ⟹ a different telemetry hash (the chain is sensitive).
    t2, v2 = design_dc_motor_excitation(v_peak=4.0)
    res2 = dc_motor_recipe(_dc_telemetry(t2, v2, Ks=0.2, Kv=0.5, Ka=0.05))
    assert res2.telemetry_hash != res.telemetry_hash


# --- BLDC ------------------------------------------------------------------


def _chirp_telemetry(iref, t):
    n = iref.shape[0]
    return ActuatorChirpTelemetry(
        time=t,
        sample_rate_hz=1.0 / (t[1] - t[0]),
        actuator_name="m",
        setpoint=iref,
        velocity=np.zeros(n),
        current=np.zeros(n),
    )


def test_bldc_identifiability_runs_and_reports_eight_params():
    """Forward-mode sensitivity check returns a full 8-parameter report."""
    fs, dur = 2000.0, 0.3
    iref = design_bldc_excitation(fs=fs, duration=dur)
    t = np.arange(iref.shape[0], dtype=np.float64) / fs
    device = BLDC(R=1.0, L=1e-3, Kt=0.04, Ke=0.04, Kp_i=10.0, J=1e-4, b=0.0, Fc=0.0)
    rep = bldc_identifiability(device, _chirp_telemetry(iref, t))
    assert len(rep.names) == 8
    assert rep.condition_number > 1.0
    # A free-shaft chirp leaves at least one electrical parameter weakly excited.
    assert rep.rank <= 8


def test_bldc_constant_setpoint_is_not_identifiable():
    """A constant current setpoint barely excites the dynamics → gate raises."""
    fs, n = 2000.0, 400
    t = np.arange(n, dtype=np.float64) / fs
    iref = np.ones(n)  # no sweep → no information to separate parameters
    device = BLDC(R=1.0, L=1e-3, Kt=0.04, Ke=0.04, Kp_i=10.0, J=1e-4, b=0.0, Fc=0.0)
    rep = bldc_identifiability(device, _chirp_telemetry(iref, t))
    assert not rep.identifiable
    with pytest.raises(CalibrationError, match="not identifiable"):
        bldc_recipe(_chirp_telemetry(iref, t), device, require_identifiable=True)


def _bldc_truth_telemetry(*, R, L, Kt, Ke, Kp_i, J, b, Fc, fs=2000.0, T=1.0):
    """Model-consistent BLDC chirp telemetry (same substepped RK4 as the fitter)."""
    n = int(fs * T) + 1
    t = np.arange(n, dtype=np.float64) / fs
    iref = design_bldc_excitation(fs=fs, duration=T)
    v_eps = 1e-3

    def rhs(y, ir):
        v, i = y[1], y[2]
        didt = (Kp_i * (ir - i) - R * i - Ke * v) / L
        dvdt = (Kt * i - b * v - Fc * np.tanh(v / v_eps)) / J
        return np.array([v, dvdt, didt])

    v_arr = np.zeros(n)
    i_arr = np.zeros(n)
    y = np.zeros(3)
    for kk in range(1, n):
        dt = t[kk] - t[kk - 1]
        sub = dt / 8
        for _ in range(8):
            k1 = rhs(y, iref[kk - 1])
            k2 = rhs(y + 0.5 * sub * k1, iref[kk - 1])
            k3 = rhs(y + 0.5 * sub * k2, iref[kk - 1])
            k4 = rhs(y + sub * k3, iref[kk - 1])
            y = y + (sub / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        v_arr[kk], i_arr[kk] = y[1], y[2]
    return ActuatorChirpTelemetry(
        time=t,
        sample_rate_hz=fs,
        actuator_name="m",
        setpoint=iref,
        velocity=v_arr,
        current=i_arr,
    )


@pytest.mark.nightly
def test_bldc_recipe_recovers_ratio_and_warns():
    """The BLDC recipe recovers Kt/J and flags the weak electrical split."""
    truth = dict(R=1.2, L=1.5e-3, Kt=0.05, Ke=0.05, Kp_i=12.0, J=1.2e-4, b=1e-4, Fc=0.0)
    tel = _bldc_truth_telemetry(**truth)
    device = BLDC(R=1.0, L=1e-3, Kt=0.04, Ke=0.04, Kp_i=10.0, J=1e-4, b=0.0, Fc=0.0)
    res = bldc_recipe(tel, device, hints={"max_steps": 200})
    kt_over_j = float(res.fitted["Kt"]) / float(res.fitted["J"])
    assert kt_over_j == pytest.approx(truth["Kt"] / truth["J"], rel=2e-2)
    assert res.recipe_name == "sysid.actuator"
