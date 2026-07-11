# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""ServoThermal — helpers, forward sim, fit recovery, validate, gradients.

The forward-sim test checks the Jaxonomy-integrated ODE against the analytic
first-order response :func:`thermal_response` (closed-form solution of the
lumped model). The recovery test is necessarily self-consistent: we pick truth
parameters, synthesize a model-consistent ``(T, tau)`` trace, and assert the
``C_th``-known fitter recovers ``R_th`` and ``k_pwr`` — the two leaves a single
temperature trace identifies (see ``thermal.py`` for why ``C_th`` is held).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jaxonomy
import numpy as np
import pytest
from jaxonomy import SimulatorOptions
from jaxonomy.library import Constant

from jaxterity.data import ThermalTelemetry
from jaxterity.physics import (
    ServoThermal,
    dissipated_power,
    thermal_response,
    thermal_steady_state,
)
from jaxterity.physics.thermal import _T_AMB_DEFAULT, _fit_thermal
from jaxterity.testing import tolerances as tol

# --- metadata --------------------------------------------------------------


def test_uncalibrated_constructs_with_neutral_params():
    blk = ServoThermal.uncalibrated()
    # C_th, R_th non-zero (ODE denominators); k_pwr=0 (no heating identified).
    assert float(blk.C_th) == 1.0
    assert float(blk.R_th) == 1.0
    assert float(blk.k_pwr) == 0.0
    assert float(blk.T_amb) == pytest.approx(_T_AMB_DEFAULT)


def test_parameter_spec_lists_identifiable_leaves():
    assert tuple(ServoThermal.uncalibrated().parameter_spec()) == (
        "C_th",
        "R_th",
        "k_pwr",
    )


def test_initial_temperature_defaults_to_ambient():
    """With T0 unset, the block starts at T_amb (the T output reads ambient at t=0)."""
    T_amb = 300.0
    blk = ServoThermal(C_th=2.0, R_th=2.0, k_pwr=3.0, T_amb=T_amb)
    src = Constant(jnp.asarray(1.0), name="tau_src")
    b = jaxonomy.DiagramBuilder()
    b.add(src)
    b.add(blk)
    b.connect(src.output_ports[0], blk.input_ports[0])
    diagram = b.build()
    ctx = diagram.create_context()
    assert float(np.asarray(blk.output_ports[0].eval(ctx))) == pytest.approx(T_amb)


def test_explicit_initial_temperature_is_honored():
    """A supplied T0 overrides the ambient default."""
    blk = ServoThermal(C_th=2.0, R_th=2.0, k_pwr=3.0, T_amb=300.0, T0=320.0)
    src = Constant(jnp.asarray(1.0), name="tau_src")
    b = jaxonomy.DiagramBuilder()
    b.add(src)
    b.add(blk)
    b.connect(src.output_ports[0], blk.input_ports[0])
    diagram = b.build()
    ctx = diagram.create_context()
    assert float(np.asarray(blk.output_ports[0].eval(ctx))) == pytest.approx(320.0)


# --- analytic helpers ------------------------------------------------------


def test_dissipated_power_is_smooth_square():
    # P = k_pwr * tau^2; smooth (no |tau| kink), zero at tau=0.
    assert float(dissipated_power(jnp.asarray(0.0), 3.0)) == 0.0
    assert float(dissipated_power(jnp.asarray(2.0), 3.0)) == pytest.approx(12.0)
    # even in tau (sign-independent heating).
    assert float(dissipated_power(jnp.asarray(-2.0), 3.0)) == pytest.approx(12.0)


def test_steady_state_is_ambient_plus_gain_times_power():
    # T_ss = T_amb + k_pwr*R_th*tau^2.
    T_ss = thermal_steady_state(2.0, T_amb=300.0, R_th=2.0, k_pwr=3.0)
    assert float(T_ss) == pytest.approx(300.0 + 3.0 * 2.0 * 4.0)


def test_thermal_response_endpoints():
    # at t=0 returns T0; as t→∞ approaches T_ss.
    kw = dict(tau=1.0, T0=300.0, T_amb=300.0, C_th=2.0, R_th=2.0, k_pwr=3.0)
    assert float(thermal_response(0.0, **kw)) == pytest.approx(300.0)
    T_ss = float(thermal_steady_state(1.0, T_amb=300.0, R_th=2.0, k_pwr=3.0))
    assert float(thermal_response(1e3, **kw)) == pytest.approx(T_ss, abs=1e-6)


# --- forward sim (exercises the ODE through the Jaxonomy diagram) ----------

_RK4_OPTS = SimulatorOptions(
    math_backend="jax", ode_solver_method="rk4", max_minor_step_size=0.001
)


def _simulate_constant_torque(
    tau: float,
    *,
    C_th: float,
    R_th: float,
    k_pwr: float,
    T_amb: float,
    t_end: float,
    solver: str = "rk4",
    dtype=jnp.float64,
) -> float:
    """Drive ServoThermal with a constant torque; return its terminal temperature."""
    blk = ServoThermal(C_th=C_th, R_th=R_th, k_pwr=k_pwr, T_amb=T_amb)
    src = Constant(jnp.asarray(tau, dtype=dtype), name="tau_src")
    b = jaxonomy.DiagramBuilder()
    b.add(src)
    b.add(blk)
    b.connect(src.output_ports[0], blk.input_ports[0])
    diagram = b.build()
    opts = SimulatorOptions(
        math_backend="jax", ode_solver_method=solver, max_minor_step_size=0.001
    )
    res = jaxonomy.simulate(
        diagram, diagram.create_context(), (0.0, t_end), options=opts
    )
    return float(np.asarray(res.context[blk.system_id].continuous_state))


def test_forward_sim_matches_analytic_first_order_response():
    """The integrated ODE reproduces the closed-form first-order rise.

    With C_th=2, R_th=2 (τ=4 s), k_pwr=3, tau=1, the block starts at ambient and
    rises toward T_ss=T_amb+6. At t=5 s the simulated terminal temperature must
    match :func:`thermal_response` to the ``motor_temperature`` policy bound.
    """
    C_th, R_th, k_pwr, T_amb, tau = 2.0, 2.0, 3.0, 300.0, 1.0
    t_end = 5.0
    T_sim = _simulate_constant_torque(
        tau, C_th=C_th, R_th=R_th, k_pwr=k_pwr, T_amb=T_amb, t_end=t_end
    )
    T_exact = float(
        thermal_response(
            t_end, tau, T0=T_amb, T_amb=T_amb, C_th=C_th, R_th=R_th, k_pwr=k_pwr
        )
    )
    assert T_sim == pytest.approx(
        T_exact,
        abs=tol.atol("motor_temperature", "rk4", "float64"),
        rel=tol.rtol("motor_temperature", "rk4", "float64"),
    )


# --- recovery (round-trip on analytically consistent telemetry) -----------


def _synthetic_thermal_telemetry(
    *,
    C_th: float,
    R_th: float,
    k_pwr: float,
    T_amb: float,
    tau_value: float,
    sample_rate_hz: float = 100.0,
    duration_s: float = 20.0,
) -> tuple[ThermalTelemetry, np.ndarray]:
    """Synthesize a model-consistent (temperature, torque) trace.

    Constant torque step from ambient; temperature follows the analytic
    first-order response. Returns the telemetry and the aligned torque array.
    """
    n = int(duration_s * sample_rate_hz) + 1
    t = np.arange(n, dtype=np.float64) / sample_rate_hz
    T = np.asarray(
        thermal_response(
            t, tau_value, T0=T_amb, T_amb=T_amb, C_th=C_th, R_th=R_th, k_pwr=k_pwr
        )
    )
    tau = np.full(n, tau_value, dtype=np.float64)
    telem = ThermalTelemetry(
        time=t,
        sample_rate_hz=sample_rate_hz,
        servo_names=("servo_0",),
        temperature=T.reshape(-1, 1),
    )
    return telem, tau


def test_fit_recovers_r_th_and_k_pwr_with_known_c_th():
    """C_th-known OLS recovers R_th and k_pwr within the motor_temperature rtol."""
    C_th, R_th_true, k_pwr_true, T_amb = 2.0, 2.0, 3.0, 300.0
    telem, tau = _synthetic_thermal_telemetry(
        C_th=C_th, R_th=R_th_true, k_pwr=k_pwr_true, T_amb=T_amb, tau_value=1.0
    )
    fitted = _fit_thermal(telem, tau, T_amb=T_amb, C_th=C_th)
    r = tol.rtol("motor_temperature", "rk4", "float64")
    assert float(fitted["R_th"]) == pytest.approx(R_th_true, rel=r)
    assert float(fitted["k_pwr"]) == pytest.approx(k_pwr_true, rel=r)


def test_fit_rejects_nonpositive_c_th():
    telem, tau = _synthetic_thermal_telemetry(
        C_th=2.0, R_th=2.0, k_pwr=3.0, T_amb=300.0, tau_value=1.0
    )
    with pytest.raises(ValueError, match="C_th"):
        _fit_thermal(telem, tau, T_amb=300.0, C_th=0.0)


# --- validate --------------------------------------------------------------


def test_validate_on_truth_telemetry_is_ok():
    C_th, R_th, k_pwr, T_amb = 2.0, 2.0, 3.0, 300.0
    telem, tau = _synthetic_thermal_telemetry(
        C_th=C_th, R_th=R_th, k_pwr=k_pwr, T_amb=T_amb, tau_value=1.0
    )
    report = ServoThermal(C_th=C_th, R_th=R_th, k_pwr=k_pwr, T_amb=T_amb).validate(
        telem, tau
    )
    assert report.ok, report.summary()


def test_validate_on_wrong_params_flags_residuals():
    C_th, R_th, k_pwr, T_amb = 2.0, 2.0, 3.0, 300.0
    telem, tau = _synthetic_thermal_telemetry(
        C_th=C_th, R_th=R_th, k_pwr=k_pwr, T_amb=T_amb, tau_value=1.0
    )
    # k_pwr off by 3× → steady-state rise wrong by 3× → large residual.
    report = ServoThermal(
        C_th=C_th, R_th=R_th, k_pwr=3.0 * k_pwr, T_amb=T_amb
    ).validate(telem, tau)
    assert not report.ok


# --- gradients (forward- and reverse-mode vs central finite difference) -----


def _jax_rollout_final_temp(C_th, R_th, k_pwr, *, tau, T0, T_amb, dt, n_steps):
    """Pure-JAX fixed-step RK4 rollout of the thermal ODE; return final T.

    A differentiable scalar function of the three identifiable leaves, used to
    cross-check autodiff against a central finite difference.
    """

    def rhs(T):
        return (k_pwr * tau * tau - (T - T_amb) / R_th) / C_th

    def step(T, _):
        k1 = rhs(T)
        k2 = rhs(T + 0.5 * dt * k1)
        k3 = rhs(T + 0.5 * dt * k2)
        k4 = rhs(T + dt * k3)
        return T + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), None

    T_final, _ = jax.lax.scan(step, T0, None, length=n_steps)
    return T_final


def _central_diff(f, x, h):
    return (f(x + h) - f(x - h)) / (2.0 * h)


def test_gradients_match_finite_difference_forward_and_reverse():
    """jax.grad / jax.jacfwd of the rollout agree with central differences.

    Differentiates the final temperature wrt each of C_th, R_th, k_pwr. Both
    autodiff modes are checked: forward-mode is the mode that composes through
    the Servo's smoothed saturation in the twin, reverse-mode is the systematic
    sweep mode. Tolerance is the FD-truncation-bounded ``gradient_*``.
    """
    C_th, R_th, k_pwr = 2.0, 2.0, 3.0
    kw = dict(tau=1.0, T0=300.0, T_amb=300.0, dt=0.02, n_steps=200)

    def f(args):
        c, r, k = args
        return _jax_rollout_final_temp(c, r, k, **kw)

    x = jnp.asarray([C_th, R_th, k_pwr])
    grad_rev = jax.grad(f)(x)
    grad_fwd = jax.jacfwd(f)(x)

    rtol = tol.gradient_rtol("float64")
    atol = tol.gradient_atol("float64")
    # forward- and reverse-mode agree with each other tightly.
    np.testing.assert_allclose(np.asarray(grad_fwd), np.asarray(grad_rev), rtol=1e-9)
    # and both agree with a central finite difference.
    for i, h in enumerate((1e-5, 1e-5, 1e-5)):

        def f_i(v, i=i):
            xx = x.at[i].set(v)
            return float(f(xx))

        fd = _central_diff(f_i, float(x[i]), h)
        assert float(grad_rev[i]) == pytest.approx(fd, rel=rtol, abs=atol)


# --- nightly ---------------------------------------------------------------


@pytest.mark.nightly
@pytest.mark.parametrize("solver", tol.SOLVERS)
@pytest.mark.parametrize("dtype", tol.DTYPES)
def test_forward_sim_grid_matches_analytic(solver, dtype):
    """Forward sim reproduces the analytic response under every (solver, dtype)."""
    C_th, R_th, k_pwr, T_amb, tau = 2.0, 2.0, 3.0, 300.0, 1.0
    t_end = 5.0
    _DT = {"float64": jnp.float64, "float32": jnp.float32}
    T_sim = _simulate_constant_torque(
        tau,
        C_th=C_th,
        R_th=R_th,
        k_pwr=k_pwr,
        T_amb=T_amb,
        t_end=t_end,
        solver=solver,
        dtype=_DT[dtype],
    )
    T_exact = float(
        thermal_response(
            t_end, tau, T0=T_amb, T_amb=T_amb, C_th=C_th, R_th=R_th, k_pwr=k_pwr
        )
    )
    assert T_sim == pytest.approx(
        T_exact,
        abs=tol.atol("motor_temperature", solver, dtype),
        rel=tol.rtol("motor_temperature", solver, dtype),
    )
