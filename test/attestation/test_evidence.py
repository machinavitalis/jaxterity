# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Structured calibration evidence record — unsigned.

The record must build from a real calibration, hash stably, round-trip through
its schema, and **bind to the twin handle** — a parameter change must move both
the recorded handle and the content hash.
"""

from __future__ import annotations

import numpy as np
import pytest

from jaxterity.calibrate import calibrate_twin_thermal
from jaxterity.data import ThermalTelemetry
from jaxterity.diagnostics import EvidenceRecord, build_evidence
from jaxterity.physics.thermal import thermal_response
from jaxterity.twin import so101_twin

pytest.importorskip("mujoco")


def _calibrated_twin(serial, *, r_true=4.0, k_true=6.0):
    """A twin with a real off-body thermal fit (provenance + uncertainty)."""
    twin = so101_twin(serial)
    n = twin.n_servos
    C_th = np.asarray(twin.C_th)
    T_amb = twin.T_amb
    fs, n_samp = 50.0, 600
    t = np.arange(n_samp, dtype=np.float64) / fs
    cols = [
        np.asarray(
            thermal_response(
                t,
                0.8,
                T0=T_amb,
                T_amb=T_amb,
                C_th=float(C_th[j]),
                R_th=r_true,
                k_pwr=k_true,
            )
        )
        for j in range(n)
    ]
    telem = ThermalTelemetry(
        time=t,
        sample_rate_hz=fs,
        servo_names=twin.servo_names,
        temperature=np.stack(cols, axis=1),
    )
    tau = np.full((n_samp, n), 0.8)
    return calibrate_twin_thermal(twin, telem, tau)


def _evidence(twin):
    return build_evidence(
        twin,
        held_out_error={"temperature_rms": 0.05},
        recovery={"R_th_rel": 1e-3, "k_pwr_rel": 2e-3},
        identifiability={"rank": 2.0, "condition_number": 12.0},
    )


def test_built_from_a_real_calibration():
    twin = _calibrated_twin("SO101-EV1")
    rec = _evidence(twin)
    assert rec.twin_handle == twin.attestation_handle
    assert rec.recipe_name == "thermal.sysid"
    assert dict(rec.uncertainty).keys() == {"offbody.R_th", "offbody.k_pwr"}
    assert dict(rec.held_out_error)["temperature_rms"] == 0.05


def test_content_hash_is_stable_and_manifest_referenceable():
    twin = _calibrated_twin("SO101-EV2")
    rec = _evidence(twin)
    h = rec.content_hash()
    # Deterministic across rebuilds of an identical record.
    assert _evidence(twin).content_hash() == h
    # A 64-hex BLAKE3 digest — the value a Jaxility manifest references.
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_binds_to_the_twin_handle():
    """A parameter change moves both the recorded handle and the content hash."""
    twin = _calibrated_twin("SO101-EV3")
    rec = _evidence(twin)
    # Perturb an off-body leaf → a new twin version (new handle).
    bumped = twin.with_thermal(R_th=np.asarray(twin.R_th) * 1.05)
    rec2 = _evidence(bumped)
    assert rec2.twin_handle != rec.twin_handle
    assert rec2.content_hash() != rec.content_hash()


def test_schema_round_trips_and_preserves_hash():
    twin = _calibrated_twin("SO101-EV4")
    rec = _evidence(twin)
    # dict round-trip
    back = EvidenceRecord.from_dict(rec.to_dict())
    assert back == rec
    assert back.content_hash() == rec.content_hash()
    # json round-trip
    back_j = EvidenceRecord.from_json(rec.to_json())
    assert back_j == rec
    assert back_j.content_hash() == rec.content_hash()


def test_unsigned_record_has_no_signature_surface():
    """Honest scope: this is the unsigned primitive — no signing fields here."""
    fields = set(EvidenceRecord.__dataclass_fields__)
    assert not (fields & {"signature", "public_key", "cert", "signed_by"})
