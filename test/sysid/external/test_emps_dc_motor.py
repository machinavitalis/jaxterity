# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""External-hardware validation of the DC-motor recipe against EMPS.

This is the only sysid check whose ground truth is a **real physical rig
external to the jax*y stack** — the EMPS electro-mechanical positioning system
(Janot, Gautier & Brunot 2019). Every other recipe oracle (round-trip, Radau,
jaxonomy) is in-stack, so a physics error shared across the stack could survive
it; this one cannot, because the data comes from real hardware.

The rig's physics ``F = M*a + Fv*v + Fc*sgn(v) + offset`` is exactly the
DC-motor recipe's ``V = Ks*sgn(v) + Kv*v + Ka*a`` form (Ka↔M, Kv↔Fv, Ks↔Fc; the
recipe has no offset term). We check two things on the real data:

1. our recipe agrees with the paper's own documented identification method
   (IDIM-LS, reproduced here independently of jaxterity's code), and
2. its fit predicts a held-out validation trace (the pulses set) it never saw.

**Opt-in / Mode B.** The EMPS data is not vendored (unclear license — see
``fetch_emps.py``), so this test *skips* unless the files are present locally.
Fetch them with ``python test/sysid/external/fetch_emps.py`` then run
``pytest -m external_data``. Recorded outcomes live in ``RESULTS.md``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.external_data

# Where fetch_emps.py deposits the (uncommitted) data; kept in sync with it.
DATA_DIR = Path(__file__).resolve().parent / "_emps_data"

pytest.importorskip("scipy", reason="EMPS validation needs scipy")

# Published EMPS reference parameters (Janot/Gautier/Brunot 2019), reproduced by
# `_reference_idim_ls` below — the external anchor our recipe must land near.
_REF_M, _REF_FV, _REF_FC = 95.1, 204.7, 20.3  # kg, N*s/m, N


def _require_data():
    if not (DATA_DIR / "DATA_EMPS.mat").exists():
        pytest.skip(
            "EMPS data absent — run `python test/sysid/external/fetch_emps.py` "
            "to enable this external-hardware validation."
        )


def _load(name):
    from scipy.io import loadmat

    d = loadmat(str(DATA_DIR / name))
    t = d["t"].ravel()
    qm = d["qm"].ravel()
    vir = d["vir"].ravel()
    gtau = float(d["gtau"].ravel()[0])
    return t, gtau * vir, qm  # time, motor force (N), measured position (m)


def _velocity(t, qm):
    """Butterworth-filter position and central-difference it — the paper's route."""
    from scipy.signal import butter, filtfilt

    fs = 1.0 / (t[1] - t[0])
    b, a = butter(4, (5 * 20.0) / (fs / 2))  # 100 Hz, 4th-order, as in Script_IDIM_LS.m
    return np.gradient(filtfilt(b, a, qm), t), fs


def _reference_idim_ls(t, force, v):
    """Paper's IDIM-LS: F = M*a + Fv*v + Fc*sgn(v) + offset (not our code)."""
    a = np.gradient(v, t)
    x = np.stack([a, v, np.sign(v), np.ones_like(v)], axis=1)
    beta, *_ = np.linalg.lstsq(x, force, rcond=None)
    return beta  # M, Fv, Fc, offset


def test_dc_motor_recipe_matches_emps_hardware():
    """Our DC-motor fit recovers EMPS mass+friction and predicts the held-out set."""
    _require_data()
    import jaxterity.sysid as sysid
    from jaxterity.actuators import DCMotorWithFriction
    from jaxterity.data import ActuatorRampTelemetry

    # --- estimation set: fit + compare to the paper's reference method ---
    t, force, qm = _load("DATA_EMPS.mat")
    v, fs = _velocity(t, qm)
    n = 60  # drop filter/diff edge transients
    t, force, v = t[n:-n], force[n:-n], v[n:-n]

    m_ref, fv_ref, fc_ref, _off = _reference_idim_ls(t, force, v)
    # Guard: the reproduced reference must match the published EMPS parameters,
    # else the data/preprocessing is wrong and the comparison is meaningless.
    assert m_ref == pytest.approx(_REF_M, rel=0.05)
    assert fv_ref == pytest.approx(_REF_FV, rel=0.05)
    assert fc_ref == pytest.approx(_REF_FC, rel=0.05)

    tel = ActuatorRampTelemetry(
        time=t, sample_rate_hz=fs, actuator_name="emps", command=force, velocity=v
    )
    res = sysid.actuator(tel, DCMotorWithFriction.uncalibrated())
    ka, kv, ks = (float(res.fitted[k]) for k in ("Ka", "Kv", "Ks"))

    # Our recipe (no offset term) must agree with the reference fit on the real
    # rig's inertia and friction. A model-form/sign error would blow these up.
    assert ka == pytest.approx(m_ref, rel=0.02)
    assert kv == pytest.approx(fv_ref, rel=0.02)
    assert ks == pytest.approx(fc_ref, rel=0.02)

    # --- held-out: predict the pulses validation trace (never seen in the fit) ---
    t2, force2, qm2 = _load("DATA_EMPS_PULSES.mat")
    v2, _ = _velocity(t2, qm2)
    t2, force2, v2 = t2[n:-n], force2[n:-n], v2[n:-n]
    a2 = np.gradient(v2, t2)
    pred = ks * np.sign(v2) + kv * v2 + ka * a2
    rel = float(np.linalg.norm(force2 - pred) / np.linalg.norm(force2))
    assert rel < 0.20  # ~13.6% observed; the gap is the recipe's missing offset term
