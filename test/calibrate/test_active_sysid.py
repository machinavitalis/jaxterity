# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Active sysid — loopback recovery, determinism, limit-bounding, CLI.

The loopback link is a seeded simulated arm: it runs the excitation through the
robot's free dynamics with *planted* damping, so the forward-mode fit must
recover it. The recovery, determinism, and limit-rejection checks need no
hardware; the real serial-bus path is ``hardware``-marked and skipped here.
"""

from __future__ import annotations

import os

import jax.numpy as jnp
import numpy as np
import pytest

from jaxterity import zoo
from jaxterity.calibrate.active import (
    Excitation,
    LoopbackLink,
    ReplayLink,
    SerialBusLink,
    active_identify,
    design_coast_excitation,
    dof_joint_names,
    excitation_identifiability,
    validate_excitation,
)
from jaxterity.errors import CalibrationError

pytest.importorskip("mujoco")

_TARGETS = ("shoulder_lift.damping", "elbow_flex.damping")
_TRUE = 0.12


@pytest.fixture(scope="module")
def robot():
    return zoo.load("so100")


@pytest.fixture(scope="module")
def excitation(robot):
    return design_coast_excitation(robot, speed=0.1)


# --- excitation design + limit-bounding ------------------------------------


def test_default_excitation_is_within_limits(robot, excitation):
    """The designed coast probe passes validation (no raise)."""
    validate_excitation(robot, excitation)  # must not raise


def test_out_of_limit_excitation_is_rejected(robot):
    """A too-fast probe that would slam a joint stop is rejected before sending."""
    reckless = design_coast_excitation(robot, speed=50.0)
    with pytest.raises(CalibrationError, match="limit"):
        validate_excitation(robot, reckless)


def test_active_identify_rejects_before_touching_the_link(robot):
    """active_identify validates first: a bad excitation never reaches the link."""

    class _ExplodingLink(LoopbackLink):
        def run_excitation(self, excitation):  # pragma: no cover - must not run
            raise AssertionError("link was touched despite an invalid excitation")

    reckless = design_coast_excitation(robot, speed=50.0)
    link = _ExplodingLink(robot, {p: _TRUE for p in _TARGETS})
    with pytest.raises(CalibrationError, match="limit"):
        active_identify(robot, link, _TARGETS, excitation=reckless)


# --- loopback recovery -----------------------------------------------------


def test_loopback_recovers_planted_damping(robot, excitation):
    """One end-to-end call recovers the planted damping within tolerance."""
    link = LoopbackLink(robot, {p: _TRUE for p in _TARGETS})
    result = active_identify(robot, link, _TARGETS, excitation=excitation)
    assert result.recipe_name == "active.sysid"
    for p in _TARGETS:
        assert float(result.fitted[p]) == pytest.approx(_TRUE, abs=1e-3)


def test_seeded_fake_gives_bit_identical_fits(robot, excitation):
    """A seeded loopback → bit-identical fitted parameters across runs."""
    a = active_identify(
        robot,
        LoopbackLink(robot, {p: _TRUE for p in _TARGETS}),
        _TARGETS,
        excitation=excitation,
    )
    b = active_identify(
        robot,
        LoopbackLink(robot, {p: _TRUE for p in _TARGETS}),
        _TARGETS,
        excitation=excitation,
    )
    for p in _TARGETS:
        assert float(a.fitted[p]) == float(b.fitted[p])


def test_result_applies_to_robot_with_provenance(robot, excitation):
    """The fit packages as a SysidResult that calibrates the robot + records it."""
    result = active_identify(
        robot,
        LoopbackLink(robot, {p: _TRUE for p in _TARGETS}),
        _TARGETS,
        excitation=excitation,
    )
    calibrated = result.apply_to(robot)
    assert calibrated.attestation_handle != robot.attestation_handle
    assert any(r[0] == "active.sysid" for r in calibrated.provenance)


# --- design-time identifiability -------------------------------------------


def test_designed_excitation_is_identifiable(robot, excitation):
    """The default coast probe carries information about every target damping."""
    report = excitation_identifiability(robot, _TARGETS, excitation)
    assert report.names == _TARGETS
    assert report.rank == len(_TARGETS)
    assert report.identifiable
    assert report.weak_params == ()


def test_degenerate_excitation_is_flagged_before_any_fit(robot, excitation):
    """A zero-motion probe (sampled at t=0) identifies nothing — flagged up front."""
    still = Excitation(
        q0=tuple(0.0 for _ in range(excitation.n_dof)),
        qd0=tuple(0.0 for _ in range(excitation.n_dof)),
        sample_times=(0.0,),
    )
    report = excitation_identifiability(robot, _TARGETS, still)
    assert not report.identifiable
    assert report.rank < len(_TARGETS)
    with pytest.raises(CalibrationError, match="not identifiable"):
        report.raise_if_unidentifiable("test")


# --- physical-bounds guards + warnings (T-219 regression) -------------------

#: A physically impossible planted value: the loopback then produces telemetry
#: whose best-fit damping is genuinely negative — the deterministic analog of
#: the G1 tutorial's noise-driven failure (the SO-100 coast probe is too
#: well-conditioned for realistic noise alone to flip a sign; see
#: `examples/tutorial_unitree_g1_sysid.json` for the noise-driven case).
_UNPHYSICAL = {"shoulder_lift.damping": _TRUE, "elbow_flex.damping": -0.05}


def test_clean_fit_with_bounds_still_recovers_exactly(robot, excitation):
    """The non-negativity projection does not perturb a well-posed clean fit."""
    link = LoopbackLink(robot, {p: _TRUE for p in _TARGETS})
    result = active_identify(robot, link, _TARGETS, excitation=excitation)
    for p in _TARGETS:
        assert float(result.fitted[p]) == pytest.approx(_TRUE, abs=1e-3)
    assert result.warnings == ()


def test_unphysical_best_fit_is_bounded_and_flagged(robot, excitation):
    """Telemetry whose best fit is negative → pinned at 0 and named, not silent.

    The pre-T-219 behavior was to return the negative damping with an empty
    warnings list. Now the projected fit stops at the physical bound, the
    bound-pin is named, and the covariance flags the (mismatch-degraded)
    estimates as unresolved.
    """
    link = LoopbackLink(robot, _UNPHYSICAL)
    result = active_identify(robot, link, _TARGETS, excitation=excitation)
    for p in _TARGETS:
        assert float(result.fitted[p]) >= 0.0
    assert float(result.fitted["elbow_flex.damping"]) == 0.0
    assert any("lower bound" in w for w in result.warnings)
    assert any("unresolved" in w for w in result.warnings)


def test_bounds_opt_out_reproduces_the_unconstrained_fit(robot, excitation):
    """bounds=False recovers the old behavior (the fit goes negative) but the
    covariance-based warning still fires — nonsense is never silent."""
    link = LoopbackLink(robot, _UNPHYSICAL)
    result = active_identify(robot, link, _TARGETS, excitation=excitation, bounds=False)
    assert float(result.fitted["elbow_flex.damping"]) < 0.0
    assert result.warnings != ()


# --- transport contract ----------------------------------------------------


def test_loopback_telemetry_shape_and_labels(robot, excitation):
    link = LoopbackLink(robot, {p: _TRUE for p in _TARGETS})
    telem = link.run_excitation(excitation)
    n = link.n_dof
    assert telem.joint_names == dof_joint_names(robot)
    assert telem.q.shape == (len(excitation.sample_times), n)
    assert telem.qd.shape == (len(excitation.sample_times), n)


def test_excitation_dof_mismatch_raises(robot):
    bad = Excitation(q0=(0.0, 0.0), qd0=(0.1, 0.1), sample_times=(0.1, 0.2))
    link = LoopbackLink(robot, {p: _TRUE for p in _TARGETS})
    with pytest.raises(CalibrationError, match="joints"):
        active_identify(robot, link, _TARGETS, excitation=bad)


# --- ReplayLink (split capture/fit: device captures, workstation fits) ------


def test_replay_link_fits_identically_to_its_source(robot, excitation):
    """Capturing telemetry then replaying it gives the *same* fit as live.

    This is the split-architecture contract: a device streams the excitation and
    records telemetry; replaying that telemetry through `active_identify` on the
    workstation must reproduce the live-link fit bit-for-bit (same observer, same
    LM), so moving capture off-box changes nothing about the result.
    """
    live = LoopbackLink(robot, {p: _TRUE for p in _TARGETS})
    captured = live.run_excitation(excitation)

    fit_live = active_identify(robot, live, _TARGETS, excitation=excitation)
    fit_replay = active_identify(
        robot, ReplayLink(captured), _TARGETS, excitation=excitation
    )
    for p in _TARGETS:
        assert float(fit_replay.fitted[p]) == float(fit_live.fitted[p])
    assert fit_replay.telemetry_hash == fit_live.telemetry_hash


def test_replay_link_rejects_sample_count_mismatch(robot, excitation):
    """Telemetry that wasn't sampled at the excitation's times is a loud error."""
    captured = LoopbackLink(robot, {p: _TRUE for p in _TARGETS}).run_excitation(
        excitation
    )
    shorter = Excitation(
        q0=captured.joint_names and tuple(0.0 for _ in range(captured.q.shape[1])),
        qd0=tuple(0.1 for _ in range(captured.q.shape[1])),
        sample_times=(0.1,),  # one time vs the capture's two samples
    )
    with pytest.raises(CalibrationError, match="sample"):
        active_identify(robot, ReplayLink(captured), _TARGETS, excitation=shorter)


# --- CLI (one-invocation, no manual log handoff) ---------------------------


def test_cli_loopback_runs_end_to_end(capsys):
    from jaxterity.cli import main

    rc = main(["calibrate", "--active", "--port", "loopback", "--true-damping", "0.1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "attestation handle" in out
    assert "fitted" in out


# --- hardware (real serial bus) — skipped by scripts/check.sh --------------


@pytest.mark.hardware
@pytest.mark.skipif(
    not os.environ.get("JAXTERITY_ACTIVE_PORT"),
    reason="needs a real servo bus (set JAXTERITY_ACTIVE_PORT to the device path)",
)
def test_serial_bus_link_against_hardware():  # pragma: no cover - hardware
    robot = zoo.load("so100")
    names = dof_joint_names(robot)
    link = SerialBusLink(
        os.environ["JAXTERITY_ACTIVE_PORT"],
        names,
        tuple(range(1, len(names) + 1)),
        tuple(4096.0 / (2.0 * np.pi) for _ in names),
    )
    result = active_identify(robot, link, _TARGETS)
    link.close()
    for p in _TARGETS:
        assert jnp.isfinite(result.fitted[p])
