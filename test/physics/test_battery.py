# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""BatterySag (LeKiwi) — helpers, feedthrough, fit recovery, validate, grads.

The battery model is purely algebraic, so the "forward sim" check evaluates the
``V_bus`` output port through a Jaxonomy context (no integration). The recovery
test synthesizes a known-load voltage trace and asserts the OLS fitter recovers
both identifiable leaves (``V_oc``, ``R_int``) within the ``bus_voltage`` policy
tolerance.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jaxonomy
import numpy as np
import pytest
from jaxonomy.library import Constant

from jaxterity.data import BusVoltageTelemetry
from jaxterity.physics import BatterySag, bus_voltage
from jaxterity.physics.battery import _fit_battery
from jaxterity.testing import tolerances as tol

# --- metadata --------------------------------------------------------------


def test_uncalibrated_constructs_with_neutral_params():
    blk = BatterySag.uncalibrated(3)
    assert float(blk.V_oc) == 12.0
    assert float(blk.R_int) == 0.0  # no sag identified yet
    assert blk.n_servos == 3
    np.testing.assert_array_equal(np.asarray(blk.k_i), np.ones(3))


def test_parameter_spec_lists_identifiable_leaves():
    assert tuple(BatterySag.uncalibrated(2).parameter_spec()) == ("V_oc", "R_int")


# --- analytic helper -------------------------------------------------------


def test_no_load_voltage_is_open_circuit():
    # zero torque → zero current → V_bus = V_oc.
    v = bus_voltage(jnp.zeros(3), V_oc=12.0, R_int=0.5, k_i=jnp.ones(3))
    assert float(v) == pytest.approx(12.0, abs=1e-9)


def test_loaded_voltage_sags_by_ir_drop():
    # I_total = Σ k_i |tau_i| = 2*1 + 3*2 = 8 A (signs ignored); drop = 0.5*8 = 4 V.
    tau = jnp.asarray([1.0, -2.0])
    v = bus_voltage(tau, V_oc=12.0, R_int=0.5, k_i=jnp.asarray([2.0, 3.0]))
    assert float(v) == pytest.approx(12.0 - 4.0, abs=1e-6)


def test_current_draw_is_sign_independent():
    # +tau and -tau draw the same current (smoothed |tau|).
    k_i = jnp.asarray([2.0, 3.0])
    v_pos = bus_voltage(jnp.asarray([1.0, 2.0]), V_oc=12.0, R_int=0.5, k_i=k_i)
    v_neg = bus_voltage(jnp.asarray([-1.0, -2.0]), V_oc=12.0, R_int=0.5, k_i=k_i)
    assert float(v_pos) == pytest.approx(float(v_neg), abs=1e-9)


# --- feedthrough through the Jaxonomy diagram ------------------------------


def test_output_port_evaluates_sagged_voltage():
    """The V_bus output port returns the analytic sag for a constant torque load."""
    k_i = jnp.asarray([2.0, 3.0])
    V_oc, R_int = 12.0, 0.5
    tau = jnp.asarray([1.0, 2.0])
    blk = BatterySag(k_i=k_i, V_oc=V_oc, R_int=R_int)
    src = Constant(tau, name="tau_src")
    b = jaxonomy.DiagramBuilder()
    b.add(src)
    b.add(blk)
    b.connect(src.output_ports[0], blk.input_ports[0])
    diagram = b.build()
    ctx = diagram.create_context()
    v_port = float(np.asarray(blk.output_ports[0].eval(ctx)))
    v_exact = float(bus_voltage(tau, V_oc=V_oc, R_int=R_int, k_i=k_i))
    assert v_port == pytest.approx(
        v_exact,
        abs=tol.atol("bus_voltage", "rk4", "float64"),
        rel=tol.rtol("bus_voltage", "rk4", "float64"),
    )


# --- recovery (round-trip on analytically consistent telemetry) -----------


def _synthetic_bus_telemetry(
    *,
    V_oc: float,
    R_int: float,
    k_i: np.ndarray,
    sample_rate_hz: float = 100.0,
    n: int = 500,
    seed: int = 0,
) -> tuple[BusVoltageTelemetry, np.ndarray]:
    """Synthesize a known-load (voltage, torque) trace spanning a range of loads."""
    rng = np.random.default_rng(seed)
    n_servos = k_i.size
    t = np.arange(n, dtype=np.float64) / sample_rate_hz
    # torques span a wide range so R_int is well-conditioned in the fit.
    tau = rng.uniform(-3.0, 3.0, size=(n, n_servos))
    I_total = np.sum(k_i * np.abs(tau), axis=1)
    V = V_oc - R_int * I_total
    telem = BusVoltageTelemetry(
        time=t,
        sample_rate_hz=sample_rate_hz,
        voltage=V,
        servo_names=tuple(f"servo_{i}" for i in range(n_servos)),
    )
    return telem, tau


def test_fit_recovers_v_oc_and_r_int():
    """OLS recovers V_oc and R_int within the bus_voltage rtol on clean data."""
    V_oc_true, R_int_true = 12.0, 0.5
    k_i = np.asarray([2.0, 3.0])
    telem, tau = _synthetic_bus_telemetry(V_oc=V_oc_true, R_int=R_int_true, k_i=k_i)
    fitted = _fit_battery(telem, tau, k_i)
    r = tol.rtol("bus_voltage", "rk4", "float64")
    assert float(fitted["V_oc"]) == pytest.approx(V_oc_true, rel=r)
    assert float(fitted["R_int"]) == pytest.approx(R_int_true, rel=r)


# --- validate --------------------------------------------------------------


def test_validate_on_truth_telemetry_is_ok():
    V_oc, R_int = 12.0, 0.5
    k_i = np.asarray([2.0, 3.0])
    telem, tau = _synthetic_bus_telemetry(V_oc=V_oc, R_int=R_int, k_i=k_i)
    report = BatterySag(k_i=k_i, V_oc=V_oc, R_int=R_int).validate(telem, tau)
    assert report.ok, report.summary()


def test_validate_on_wrong_params_flags_residuals():
    V_oc, R_int = 12.0, 0.5
    k_i = np.asarray([2.0, 3.0])
    telem, tau = _synthetic_bus_telemetry(V_oc=V_oc, R_int=R_int, k_i=k_i)
    # R_int off by 4× → sag wrong by several volts → large residual.
    report = BatterySag(k_i=k_i, V_oc=V_oc, R_int=4.0 * R_int).validate(telem, tau)
    assert not report.ok


# --- optional current channel (schema behavior at this boundary) -----------


def test_optional_current_channel_changes_hash_when_present():
    """An omitted current channel hashes differently from a populated one."""
    k_i = np.asarray([2.0, 3.0])
    telem, tau = _synthetic_bus_telemetry(V_oc=12.0, R_int=0.5, k_i=k_i)
    with_current = BusVoltageTelemetry(
        time=telem.time,
        sample_rate_hz=telem.sample_rate_hz,
        voltage=telem.voltage,
        servo_names=telem.servo_names,
        current=np.abs(tau) * k_i,
    )
    assert telem.current is None
    assert telem.hash() != with_current.hash()


# --- gradients (forward- and reverse-mode vs central finite difference) -----


def test_gradients_match_finite_difference():
    """jax.grad / jax.jacfwd of bus_voltage agree with central differences.

    Differentiates V_bus wrt (V_oc, R_int) — both identifiable leaves — through
    the smoothed-|tau| current draw.
    """
    k_i = jnp.asarray([2.0, 3.0])
    tau = jnp.asarray([1.0, -2.0])

    def f(args):
        V_oc, R_int = args
        return bus_voltage(tau, V_oc=V_oc, R_int=R_int, k_i=k_i)

    x = jnp.asarray([12.0, 0.5])
    grad_rev = jax.grad(f)(x)
    grad_fwd = jax.jacfwd(f)(x)
    np.testing.assert_allclose(np.asarray(grad_fwd), np.asarray(grad_rev), rtol=1e-9)

    rtol = tol.gradient_rtol("float64")
    atol = tol.gradient_atol("float64")
    for i in range(2):

        def f_i(v, i=i):
            return float(f(x.at[i].set(v)))

        fd = (f_i(float(x[i]) + 1e-5) - f_i(float(x[i]) - 1e-5)) / 2e-5
        assert float(grad_rev[i]) == pytest.approx(fd, rel=rtol, abs=atol)
