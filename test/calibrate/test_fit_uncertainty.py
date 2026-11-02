# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Fit uncertainty bounds carried through the calibrate + twin path.

The estimator math is unit-tested in ``test/sysid/test_uncertainty.py``; here we
check the MJX-backed integration: ``active_identify`` attaches a covariance the
interval reads, a noisy fit's interval covers the planted truth, the thermal fit
propagates variance into the twin, and — because uncertainty is a *derived*
quantity — attaching it does **not** move the twin's attestation handle.
"""

from __future__ import annotations

import numpy as np
import pytest

from jaxterity import zoo
from jaxterity.calibrate import calibrate_twin_thermal, fit_servo_thermal
from jaxterity.calibrate.active import (
    LoopbackLink,
    active_identify,
    design_coast_excitation,
)
from jaxterity.data import ThermalTelemetry
from jaxterity.errors import CalibrationError
from jaxterity.physics.thermal import thermal_response
from jaxterity.testing import tolerances as tol
from jaxterity.twin import so101_twin

pytest.importorskip("mujoco")

_TARGETS = ("shoulder_lift.damping", "elbow_flex.damping")
_TRUE = 0.12

# The bdf lane of the solver grid runs on the cartpole instead of the so100 —
# see test_active_uncertainty_under_bdf for why.
_CARTPOLE_TARGETS = ("slider_to_cart.damping", "cart_to_pole.damping")
_GRID_SOLVERS = tuple(s for s in tol.SOLVERS if s != "bdf")


@pytest.fixture(scope="module")
def robot():
    return zoo.load("so100")


@pytest.fixture(scope="module")
def excitation(robot):
    return design_coast_excitation(robot, speed=0.1)


@pytest.fixture(scope="module")
def cartpole():
    return zoo.load("cartpole")


@pytest.fixture(scope="module")
def cartpole_coast(cartpole):
    """A longer, brisker coast than the so100's.

    Sample times cost nothing in graph size (the bounded step loop is not
    unrolled, so the HLO is the same for one sample time as for two), so the
    horizon is free — and this one identifies both dampings at condition
    number ~1.1, where the so100 default coast leaves the cartpole fit short
    of the planted value.
    """
    return design_coast_excitation(cartpole, speed=0.5, sample_times=(0.4, 0.8))


# --- active fit: covariance attached + interval contract -------------------


def test_laplace_covariance_is_attached_by_default(robot, excitation):
    """A default active fit reports a per-parameter covariance + finite interval."""
    link = LoopbackLink(robot, {p: _TRUE for p in _TARGETS})
    result = active_identify(robot, link, _TARGETS, excitation=excitation)
    assert result.covariance is not None
    assert set(result.covariance) == set(_TARGETS)
    intervals = result.interval(0.95)
    for p in _TARGETS:
        lo, hi = intervals[p]
        assert lo <= float(result.fitted[p]) <= hi
        assert np.isfinite(lo) and np.isfinite(hi)


def test_none_uncertainty_skips_covariance(robot, excitation):
    link = LoopbackLink(robot, {p: _TRUE for p in _TARGETS})
    result = active_identify(
        robot, link, _TARGETS, excitation=excitation, uncertainty=None
    )
    assert result.covariance is None
    assert result.interval() is None


def test_unknown_uncertainty_raises(robot, excitation):
    link = LoopbackLink(robot, {p: _TRUE for p in _TARGETS})
    with pytest.raises(CalibrationError, match="uncertainty"):
        active_identify(
            robot, link, _TARGETS, excitation=excitation, uncertainty="quantum"
        )


def test_noisy_fit_interval_covers_truth(robot, excitation):
    """On noisy telemetry the 99% interval brackets the planted damping."""
    link = LoopbackLink(robot, {p: _TRUE for p in _TARGETS}, seed=0, noise_std=1e-3)
    result = active_identify(robot, link, _TARGETS, excitation=excitation)
    intervals = result.interval(0.99)
    for p in _TARGETS:
        lo, hi = intervals[p]
        assert lo <= _TRUE <= hi, f"{p}: truth {_TRUE} outside [{lo}, {hi}]"
    # A noisy fit must report a strictly positive standard error.
    se = result.std_error()
    assert all(se[p] > 0.0 for p in _TARGETS)


# --- thermal fit: variance + carried into the twin -------------------------


def _thermal_trace(twin, *, r_true, k_true, tau_amp, fs=50.0, n=600):
    C_th = np.asarray(twin.C_th)
    T_amb = twin.T_amb
    t = np.arange(n, dtype=np.float64) / fs
    cols = [
        np.asarray(
            thermal_response(
                t,
                tau_amp[j],
                T0=T_amb,
                T_amb=T_amb,
                C_th=float(C_th[j]),
                R_th=r_true[j],
                k_pwr=k_true[j],
            )
        )
        for j in range(twin.n_servos)
    ]
    telem = ThermalTelemetry(
        time=t,
        sample_rate_hz=fs,
        servo_names=twin.servo_names,
        temperature=np.stack(cols, axis=1),
    )
    return telem, np.tile(tau_amp, (n, 1))


def test_thermal_fit_reports_variance():
    twin = so101_twin("SO101-UNC")
    n = twin.n_servos
    r_true = np.full(n, 4.0)
    k_true = np.full(n, 6.0)
    tau_amp = np.full(n, 0.8)
    telem, tau = _thermal_trace(twin, r_true=r_true, k_true=k_true, tau_amp=tau_amp)
    fit = fit_servo_thermal(telem, tau, C_th=np.asarray(twin.C_th), T_amb=twin.T_amb)
    for key in ("R_th_var", "k_pwr_var"):
        assert fit[key].shape == (n,)
        assert np.all(fit[key] >= 0.0)


def test_calibrate_twin_thermal_carries_uncertainty():
    twin = so101_twin("SO101-UNC2")
    n = twin.n_servos
    telem, tau = _thermal_trace(
        twin, r_true=np.full(n, 4.0), k_true=np.full(n, 6.0), tau_amp=np.full(n, 0.8)
    )
    fitted = calibrate_twin_thermal(twin, telem, tau)
    unc = fitted.uncertainty()
    assert set(unc) == {"offbody.R_th", "offbody.k_pwr"}
    assert unc["offbody.R_th"].shape == (n,)
    assert np.all(unc["offbody.R_th"] >= 0.0)


def test_uncertainty_does_not_change_the_attestation_handle():
    """Attaching fit uncertainty is a derived quantity — the handle is invariant."""
    twin = so101_twin("SO101-INV")
    n = twin.n_servos
    r_new = np.full(n, 4.2)
    k_new = np.full(n, 6.1)
    bare = twin.with_thermal(R_th=r_new, k_pwr=k_new)
    with_unc = twin.with_thermal(
        R_th=r_new,
        k_pwr=k_new,
        param_uncertainty=(
            ("offbody.R_th", tuple(0.01 for _ in range(n))),
            ("offbody.k_pwr", tuple(0.02 for _ in range(n))),
        ),
    )
    assert with_unc.attestation_handle == bare.attestation_handle
    # …but the thermal re-fit itself (new leaf values) does move the handle.
    assert bare.attestation_handle != twin.attestation_handle


def test_thermal_fit_recovers_within_tolerance_with_variance():
    """The variance addition does not disturb the planted-param recovery."""
    twin = so101_twin("SO101-REC")
    n = twin.n_servos
    r_true = np.array([3.0, 4.0, 5.0, 3.5, 4.5, 2.5])[:n]
    k_true = np.array([5.0, 6.0, 7.0, 5.5, 6.5, 4.5])[:n]
    tau_amp = np.array([0.8, 0.9, 1.0, 0.7, 0.6, 0.5])[:n]
    telem, tau = _thermal_trace(twin, r_true=r_true, k_true=k_true, tau_amp=tau_amp)
    fit = fit_servo_thermal(telem, tau, C_th=np.asarray(twin.C_th), T_amb=twin.T_amb)
    r = tol.rtol("motor_temperature", "rk4", "float64")
    np.testing.assert_allclose(fit["R_th"], r_true, rtol=max(r, 1e-3))
    np.testing.assert_allclose(fit["k_pwr"], k_true, rtol=max(r, 1e-3))


# --- nightly: solver grid + bootstrap on the MJX path ----------------------


def test_the_solver_grid_and_the_bdf_case_cover_every_solver():
    """The split below must not silently drop a solver from the claim."""
    assert set(_GRID_SOLVERS) | {"bdf"} == set(tol.SOLVERS)


@pytest.mark.nightly
@pytest.mark.parametrize("solver", _GRID_SOLVERS)
def test_active_uncertainty_under_solver_grid(robot, excitation, solver):
    """Covariance is attached and finite under every supported solver.

    Every solver but bdf, on the so100; bdf is the test below, on a cheaper
    model, for the reason given there.
    """
    from jaxonomy import SimulatorOptions

    options = SimulatorOptions(
        math_backend="jax",
        ode_solver_method=solver,
        enable_autodiff=False,
        max_minor_step_size=0.005,
        max_major_steps=200,
    )
    link = LoopbackLink(robot, {p: _TRUE for p in _TARGETS}, seed=0, noise_std=1e-3)
    result = active_identify(
        robot, link, _TARGETS, excitation=excitation, options=options
    )
    intervals = result.interval(0.95)
    for p in _TARGETS:
        lo, hi = intervals[p]
        assert np.isfinite(lo) and np.isfinite(hi) and hi > lo


@pytest.mark.nightly
def test_active_uncertainty_under_bdf(cartpole, cartpole_coast):
    """The bdf lane of the claim above, on a model that fits in a CI runner.

    bdf is the only supported solver that forms its *own* Jacobian and
    LU-factorises it inside nested Newton and step-control loops, so
    differentiating it forward-mode is second-order AD through implicit control
    flow: ~87k HLO lines, and on the so100 roughly 15 GB to **compile** —
    more than a 16 GB hosted runner has, which is what killed this shard. The
    cartpole carries 4 states against the so100's 12 and costs ~6.5 GB while
    exercising the identical covariance path. Keep this case off the so100.
    """
    from jaxonomy import SimulatorOptions

    options = SimulatorOptions(
        math_backend="jax",
        ode_solver_method="bdf",
        enable_autodiff=False,
        max_minor_step_size=0.005,
        max_major_steps=200,
    )
    link = LoopbackLink(
        cartpole, {p: _TRUE for p in _CARTPOLE_TARGETS}, seed=0, noise_std=1e-3
    )
    result = active_identify(
        cartpole,
        link,
        _CARTPOLE_TARGETS,
        excitation=cartpole_coast,
        options=options,
    )
    intervals = result.interval(0.95)
    for p in _CARTPOLE_TARGETS:
        lo, hi = intervals[p]
        assert np.isfinite(lo) and np.isfinite(hi) and hi > lo
        # A covariance attached to a fit that never found the planted damping
        # would prove nothing, so pin the fixture's quality too.
        assert float(result.fitted[p]) == pytest.approx(_TRUE, rel=0.05)


@pytest.mark.nightly
def test_bootstrap_uncertainty_on_active_fit(robot, excitation):
    """The bootstrap estimator runs end-to-end on the MJX active fit."""
    link = LoopbackLink(robot, {p: _TRUE for p in _TARGETS}, seed=0, noise_std=1e-3)
    result = active_identify(
        robot,
        link,
        _TARGETS,
        excitation=excitation,
        uncertainty="bootstrap",
        n_boot=16,
        seed=1,
    )
    assert result.covariance is not None
    se = result.std_error()
    assert all(se[p] > 0.0 for p in _TARGETS)
