# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""The rolling attestation log."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from jaxterity.errors import JaxterityError

pytest.importorskip("mujoco")

from jaxterity import zoo  # noqa: E402
from jaxterity.calibrate import WindowedFitResult  # noqa: E402
from jaxterity.diagnostics import AttestationLog  # noqa: E402
from jaxterity.sysid import SysidResult  # noqa: E402
from jaxterity.twin import Twin  # noqa: E402


@pytest.fixture(scope="module")
def twin() -> Twin:
    return Twin(
        zoo.load("cartpole"),
        serial="UNIT-7",
        C_th=jnp.full(2, 15.0),
        R_th=jnp.full(2, 4.0),
        k_pwr=jnp.full(2, 6.0),
        K_t_scale=jnp.ones(2),
    )


def _fit(twin: Twin, kt, telemetry_hash: str) -> WindowedFitResult:
    """A re-fit commit (the log only reads the result's surface)."""
    refit = twin.with_actuation(
        jnp.asarray(kt),
        provenance_record=("sysid.windowed_drift", "1.0", telemetry_hash),
    )
    return WindowedFitResult(
        twin=refit,
        sysid_result=SysidResult(
            fitted={"offbody.K_t_scale": jnp.asarray(kt)},
            telemetry_hash=telemetry_hash,
            recipe_name="sysid.windowed_drift",
            recipe_version="1.0",
        ),
        residual_rms_before=1.0,
        residual_rms_after=0.42,
    )


def test_chain_grows_and_renders(twin):
    log = AttestationLog(twin, t=0.0)
    assert log.serial == "UNIT-7"
    assert len(log.entries) == 1
    assert log.entries[0].recipe == "baseline"
    baseline_handle = log.entries[0].handle
    assert baseline_handle == twin.attestation_handle

    first = _fit(twin, [0.98, 1.01], "aa11")
    record = log.commit(3600.0, first)
    second = _fit(first.twin, [0.97, 1.02], "bb22")
    log.commit(7200.0, second)

    assert len(log.entries) == 3
    handles = {e.handle for e in log.entries}
    assert len(handles) == 3  # every commit rolled the handle
    assert log.last_verified.t == 7200.0
    assert record.short_handle == f"{record.handle[:4]}…{record.handle[-4:]}"

    lines = log.render(last=2, time_unit="s")
    assert len(lines) == 2
    assert "7200" in lines[0] and "sysid.windowed_drift" in lines[0]
    assert "rms 0.42" in lines[0]


def test_commit_validation(twin):
    log = AttestationLog(twin, t=0.0)
    fit = _fit(twin, [0.98, 1.01], "aa11")

    # Wrong unit.
    other = fit.twin.with_serial("UNIT-8")
    wrong = WindowedFitResult(
        twin=other,
        sysid_result=fit.sysid_result,
        residual_rms_before=1.0,
        residual_rms_after=0.4,
    )
    with pytest.raises(JaxterityError, match="belongs to unit"):
        log.commit(10.0, wrong)

    log.commit(10.0, fit)
    # Time must advance.
    with pytest.raises(JaxterityError, match="append-only"):
        log.commit(5.0, _fit(fit.twin, [0.9, 1.1], "cc33"))
    # A commit must change the model: re-committing the same twin raises.
    same = WindowedFitResult(
        twin=fit.twin,
        sysid_result=fit.sysid_result,
        residual_rms_before=1.0,
        residual_rms_after=0.4,
    )
    with pytest.raises(JaxterityError, match="must change the model"):
        log.commit(20.0, same)
