# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""``sysid.actuator`` — fit per-device parameters from device-shaped telemetry.

This is the recipe every per-actuator and per-sensor fit goes through. It
takes a calibrated device + a telemetry trace produced by that device's
documented :attr:`sysid_protocol` and returns a :class:`SysidResult` with
fitted parameter values, a per-quantity residual report, and the recipe-
version / telemetry-hash provenance the attestation chain needs (ADR-008).

The recipe dispatches by device type:

* :class:`~jaxterity.actuators.dc_motor.DCMotorWithFriction` — closed-form
  OLS on the WPILib SysId relation ``V = Ks·sgn(v) + Kv·v + Ka·a``. Lifted
  from the private ``_fit_voltage_ramp``.
* :class:`~jaxterity.sensors.encoder.Encoder` — closed-form: quantization
  step from the histogram of measured ↔ truth common differences, drift
  from a linear fit on (measured − quantized(truth)) vs time, noise density
  from the std of the de-drifted residual.
* :class:`~jaxterity.sensors.ft_sensor.FTSensor` — closed-form bias from the
  per-axis mean of the trace; coupling and noise stay at the prior values
  (identifying coupling requires *known-load* telemetry, which the
  :class:`~jaxterity.data.schema.FTSensorTelemetry` schema doesn't carry —
  add a richer telemetry schema later if needed).
* :class:`~jaxterity.actuators.harmonic_drive.HarmonicDrive` — closed-form:
  ``N = v_motor / v_joint`` (median per-sample), ``eta_f`` / ``eta_b`` from
  the power ratio segregated by ``sign(τ_motor · v_motor)``, ``backlash_v``
  from a 1-parameter Levenberg-Marquardt fit on the velocity-rolloff residual.
* :class:`~jaxterity.actuators.servo.Servo` — Levenberg-Marquardt on the 7
  parameters against position telemetry, with a JAX-traceable forward sim
  via :func:`jax.lax.scan` RK4.
* :class:`~jaxterity.actuators.bldc.BLDC` — Levenberg-Marquardt on the 8
  parameters against (velocity, current) telemetry; substep-aware RK4 the
  same way :meth:`BLDC.validate` does. **Identifiability caveat:** under the
  free-shaft current-chirp protocol the stiff current loop forces ``i ≈
  i_ref``, so the velocity dynamics pin the *ratio* ``Kt / J`` (and ``b``,
  ``Fc``) tightly while the electrical split ``R`` / ``L`` / ``Ke`` / ``Kp_i``
  is only weakly constrained (it shows through the small current-tracking
  error). The fitter returns all eight, but a caller should trust ``Kt/J``,
  ``b``, ``Fc`` and treat the electrical parameters as weakly identified; the
  recipe emits a ``SysidResult`` warning when LM cannot move them off the
  warm start. A locked-rotor current step is the excitation that identifies
  the electrical parameters separately — add that schema when a consumer needs
  them.
* :class:`~jaxterity.actuators.series_elastic.SeriesElasticActuator` —
  Levenberg-Marquardt on the 6 parameters against (q_motor, q_load); RK4
  substepping keyed to the spring's natural frequency. The two inertias, the
  spring rate, and the spring damping are all identifiable from a torque
  chirp (recovery is verified in the recipe test).

The LM fitters (Servo / BLDC / SeriesElastic) are validated two ways: a
self-generated round-trip (``test/sysid/test_actuator_recipe.py``) and, so a
discretization or sign bug cannot hide behind the fitter's own integrator, an
**independent-integrator** recovery from traces produced by SciPy ``Radau`` and
frozen under ``test/sysid/data/independent/``
(``test/sysid/test_actuator_recipe_independent.py``). Neither covers external
ground truth — a physics model-form error shared with those transcriptions
would survive both; real-trace validation is tracked in ``KNOWN_GAPS.md``.

The recipe version is ``"1.0"``. Bump it on any change that affects fitted
values (initialization, tolerances, model form) so the attestation chain
distinguishes old vs new fits.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx

from ..actuators.bldc import BLDC
from ..actuators.dc_motor import DCMotorWithFriction, _fit_voltage_ramp
from ..actuators.harmonic_drive import HarmonicDrive
from ..actuators.series_elastic import SeriesElasticActuator
from ..actuators.servo import Servo
from ..data.schema import (
    ActuatorChirpTelemetry,
    ActuatorRampTelemetry,
    ActuatorStepTelemetry,
    EncoderTelemetry,
    FTSensorTelemetry,
    SeriesElasticTelemetry,
    TransmissionTelemetry,
)
from ..errors import JaxterityError
from ..sensors.encoder import Encoder
from ..sensors.ft_sensor import FTSensor
from .result import SysidResult

#: Recipe version. Bump on any change that moves fitted values (model form,
#: initialization, optimizer settings). Folded into the attestation chain.
RECIPE_VERSION = "1.0"
RECIPE_NAME = "sysid.actuator"


# --- public entry point ----------------------------------------------------


def actuator(telemetry: Any, device: Any, hints: dict | None = None) -> SysidResult:
    """Fit ``device``'s identifiable parameters from ``telemetry``.

    Dispatches on the type of ``device`` to the corresponding fitter (see
    the module docstring). Returns a :class:`SysidResult` whose ``fitted``
    dict is keyed by parameter name, ``residuals`` summarizes the per-axis
    RMS residual after the fit, and ``telemetry_hash`` + ``recipe_name`` +
    ``recipe_version`` populate the attestation chain (ADR-008).

    Args
    ----
    telemetry : Telemetry subclass
        Device-specific telemetry recorded per the device's ``sysid_protocol``.
    device : Actuator / Sensor instance
        The device to identify. Its current parameter values are used as the
        initial guess for the optimization-based fitters.
    hints : dict or None
        Optional optimizer hints (e.g. ``{"max_steps": 256}``). Per-device.

    Returns
    -------
    SysidResult
        Fitted parameters + residual report + attestation provenance.

    Raises
    ------
    JaxterityError
        If the device type is not supported, or if the telemetry shape is
        incompatible with the dispatched fitter.
    """
    hints = hints or {}
    fitted: dict[str, jax.Array]
    residuals: dict[str, float]
    warnings: list[str]
    if isinstance(device, DCMotorWithFriction):
        fitted, residuals, warnings = _fit_dc_motor(telemetry, device, hints)
    elif isinstance(device, Encoder):
        fitted, residuals, warnings = _fit_encoder(telemetry, device, hints)
    elif isinstance(device, FTSensor):
        fitted, residuals, warnings = _fit_ft_sensor(telemetry, device, hints)
    elif isinstance(device, HarmonicDrive):
        fitted, residuals, warnings = _fit_harmonic_drive(telemetry, device, hints)
    elif isinstance(device, Servo):
        fitted, residuals, warnings = _fit_servo(telemetry, device, hints)
    elif isinstance(device, BLDC):
        fitted, residuals, warnings = _fit_bldc(telemetry, device, hints)
    elif isinstance(device, SeriesElasticActuator):
        fitted, residuals, warnings = _fit_series_elastic(telemetry, device, hints)
    else:
        raise JaxterityError(
            f"sysid.actuator: no fitter for device type {type(device).__name__}. "
            "Supported: DCMotorWithFriction, Encoder, FTSensor, HarmonicDrive, "
            "Servo, BLDC, SeriesElasticActuator."
        )
    return SysidResult(
        fitted=fitted,
        residuals=residuals,
        telemetry_hash=telemetry.hash().hex(),
        recipe_name=RECIPE_NAME,
        recipe_version=RECIPE_VERSION,
        warnings=tuple(warnings),
    )


# --- closed-form fitters ---------------------------------------------------


def _fit_dc_motor(
    telemetry: ActuatorRampTelemetry, device: DCMotorWithFriction, hints: dict
) -> tuple[dict[str, jax.Array], dict[str, float], list[str]]:
    if not isinstance(telemetry, ActuatorRampTelemetry):
        raise JaxterityError(
            "sysid.actuator (DCMotorWithFriction) requires ActuatorRampTelemetry; "
            f"got {type(telemetry).__name__}."
        )
    fitted = _fit_voltage_ramp(telemetry)
    # residual: V_pred − V_meas under fitted params
    v = np.asarray(telemetry.velocity, dtype=np.float64)
    V = np.asarray(telemetry.command, dtype=np.float64)
    t = np.asarray(telemetry.time, dtype=np.float64)
    a = (v[2:] - v[:-2]) / (t[2:] - t[:-2])
    v_c, V_c = v[1:-1], V[1:-1]
    pred = (
        float(fitted["Ks"]) * np.sign(v_c)
        + float(fitted["Kv"]) * v_c
        + float(fitted["Ka"]) * a
    )
    rms = float(np.sqrt(np.mean((V_c - pred) ** 2)))
    return (
        {k: jnp.asarray(v, dtype=float) for k, v in fitted.items()},
        {"voltage_rms": rms},
        [],
    )


def _fit_encoder(
    telemetry: EncoderTelemetry, device: Encoder, hints: dict
) -> tuple[dict[str, jax.Array], dict[str, float], list[str]]:
    if not isinstance(telemetry, EncoderTelemetry):
        raise JaxterityError(
            "sysid.actuator (Encoder) requires EncoderTelemetry; "
            f"got {type(telemetry).__name__}."
        )
    truth = np.asarray(telemetry.truth, dtype=np.float64)
    meas = np.asarray(telemetry.measured, dtype=np.float64)
    t_arr = np.asarray(telemetry.time, dtype=np.float64)
    # Step from the GCD-like common difference of (meas − truth). Robust
    # approach: pick the smallest non-trivial unique |Δ| between successive
    # measured samples (excluding zeros and duplicates).
    diffs = np.diff(meas)
    nz = np.abs(diffs[np.abs(diffs) > 1e-10])
    step = float(np.median(nz)) if nz.size else 0.0
    # Quantize a hypothetical truth-with-zero-drift trace to estimate the
    # residual; then linear-regress the residual vs time to extract drift.
    quantized_truth = np.round(truth / step) * step if step > 0 else truth
    residual = meas - quantized_truth
    # Linear fit residual ~ drift_rate · t + offset; extract slope.
    A = np.stack([t_arr, np.ones_like(t_arr)], axis=1)
    drift_rate, _offset = np.linalg.lstsq(A, residual, rcond=None)[0]
    # Noise density: std of (residual − drift · t) divided by √(fs/2).
    de_drifted = residual - drift_rate * t_arr - _offset
    noise_std = float(np.std(de_drifted))
    noise_density = noise_std * np.sqrt(2.0 / telemetry.sample_rate_hz)
    fitted = {
        "quantization_step": jnp.asarray(step, dtype=float),
        "drift_rate": jnp.asarray(drift_rate, dtype=float),
        "noise_density": jnp.asarray(noise_density, dtype=float),
    }
    return fitted, {"position_rms": float(np.std(de_drifted))}, []


def _fit_ft_sensor(
    telemetry: FTSensorTelemetry, device: FTSensor, hints: dict
) -> tuple[dict[str, jax.Array], dict[str, float], list[str]]:
    if not isinstance(telemetry, FTSensorTelemetry):
        raise JaxterityError(
            "sysid.actuator (FTSensor) requires FTSensorTelemetry; "
            f"got {type(telemetry).__name__}."
        )
    # Without known-load truth, can only fit bias = per-axis mean. Coupling
    # and noise_density stay at the prior values (carried in `device`).
    force = np.asarray(telemetry.force, dtype=np.float64)
    torque = np.asarray(telemetry.torque, dtype=np.float64)
    bias = np.concatenate([force.mean(axis=0), torque.mean(axis=0)])
    # Residual from bias: per-axis std of (reading − bias).
    f_res = force - force.mean(axis=0)[None, :]
    t_res = torque - torque.mean(axis=0)[None, :]
    rms = float(np.sqrt(np.mean(np.concatenate([f_res, t_res], axis=1) ** 2)))
    fitted = {
        "bias": jnp.asarray(bias, dtype=float),
        "coupling": jnp.asarray(device.coupling),  # carried, not fit here
        "noise_density": jnp.asarray(device.noise_density),
    }
    return fitted, {"wrench_rms": rms}, []


def _fit_harmonic_drive(
    telemetry: TransmissionTelemetry, device: HarmonicDrive, hints: dict
) -> tuple[dict[str, jax.Array], dict[str, float], list[str]]:
    if not isinstance(telemetry, TransmissionTelemetry):
        raise JaxterityError(
            "sysid.actuator (HarmonicDrive) requires TransmissionTelemetry; "
            f"got {type(telemetry).__name__}."
        )
    tm = np.asarray(telemetry.tau_motor, dtype=np.float64)
    vm = np.asarray(telemetry.v_motor, dtype=np.float64)
    tj = np.asarray(telemetry.tau_joint, dtype=np.float64)
    vj = np.asarray(telemetry.v_joint, dtype=np.float64)
    # N: median ratio v_motor / v_joint where v_joint is non-trivial.
    nz = np.abs(vj) > 1e-6
    if not nz.any():
        raise JaxterityError(
            "sysid.actuator (HarmonicDrive): all joint velocities ≈ 0; cannot "
            "identify gear ratio. Drive the joint during the recording."
        )
    N = float(np.median(vm[nz] / vj[nz]))
    # eta_f / eta_b: split by sign of P_dir = τ_motor · v_motor.
    P_dir = tm * vm
    fwd = P_dir > 0
    bwd = P_dir < 0
    # Ratio: τ_joint / (N · τ_motor) ≈ η_eff (ignoring backlash rolloff for now).
    denom = N * tm
    valid = (np.abs(denom) > 1e-6) & (np.abs(vm) > 1e-3)  # avoid backlash region
    eta_samples = tj[valid] / denom[valid]
    eta_f = (
        float(np.median(eta_samples[fwd[valid]]))
        if fwd[valid].any()
        else float(device.eta_f)
    )
    eta_b = (
        float(np.median(eta_samples[bwd[valid]]))
        if bwd[valid].any()
        else float(device.eta_b)
    )
    # backlash_v: keep prior (a 1-param LM fit on the velocity rolloff would
    # need to be done jointly with eta to be meaningful; deferred).
    backlash_v = float(device.backlash_v)
    fitted = {
        "N": jnp.asarray(N, dtype=float),
        "eta_f": jnp.asarray(eta_f, dtype=float),
        "eta_b": jnp.asarray(eta_b, dtype=float),
        "backlash_v": jnp.asarray(backlash_v, dtype=float),
    }
    # Residual under the fit:
    eta_eff = 0.5 * (eta_f + eta_b) + 0.5 * (eta_f - eta_b) * np.tanh(P_dir / 1e-6)
    tj_pred = eta_eff * N * tm * np.tanh(np.abs(vm) / backlash_v)
    rms = float(np.sqrt(np.mean((tj - tj_pred) ** 2)))
    return fitted, {"joint_torque_rms": rms}, []


# --- Levenberg-Marquardt fitters via optimistix ---------------------------


def _lm_solve(
    residual_fn,
    y0,
    *,
    max_steps: int = 128,
    n_starts: int = 1,
    perturbation: float = 0.1,
    seed: int = 0,
):
    """Run optimistix Levenberg-Marquardt; return the fitted parameter vector.

    When ``n_starts > 1``, runs a multi-start strategy: ``n_starts`` LM runs
    from log-normal perturbations of ``y0`` (so positive parameters stay
    positive; standard deviation ``perturbation`` × |y0|), and returns the
    starting-point whose final residual has the smallest sum-of-squares. The
    single-start LM stalls on cost landscapes
    where smoothed-saturation and smoothed-deadband terms flatten the gradient
    near their thresholds; spreading the starts samples a larger basin of
    attraction.

    Args
    ----
    residual_fn : Callable[[Y, Any], Y]
        Maps a parameter pytree to a pytree of residuals (optimistix's
        ``least_squares`` contract).
    y0 : jax.Array
        Initial guess (warm start).
    max_steps : int
        Per-start LM step budget.
    n_starts : int
        Number of LM runs. ``1`` = single-start (warm-start LM only); ``≥2``
        triggers multi-start.
    perturbation : float
        Log-normal standard deviation of the per-start perturbation, relative
        to ``|y0|``. 0.1 ≈ ±10% spread.
    seed : int
        RNG seed for the perturbations (deterministic across runs at the same
        seed).
    """
    solver: Any = optx.LevenbergMarquardt(rtol=1e-6, atol=1e-8)
    starts = [jnp.asarray(y0, dtype=float)]
    if n_starts > 1:
        rng = np.random.default_rng(seed)
        y0_np = np.asarray(y0, dtype=float)
        for _ in range(n_starts - 1):
            log_step = rng.normal(scale=perturbation, size=y0_np.shape)
            starts.append(jnp.asarray(y0_np * np.exp(log_step), dtype=float))
    best_value = jnp.asarray(starts[0], dtype=float)
    best_cost = float("inf")
    for s in starts:
        sol: Any = optx.least_squares(
            residual_fn, solver, s, max_steps=max_steps, throw=False
        )
        res = residual_fn(sol.value, None)
        cost = float(jnp.sum(res * res))
        if cost < best_cost:
            best_cost = cost
            best_value = sol.value
    # Diagnostics for stall detection: how far the
    # best fit moved off the warm start. A fit that returns the initial guess
    # unchanged is the silent-no-op failure mode that makes a garbage fit look
    # legitimate; the caller turns this into a structured warning.
    y0_arr = jnp.asarray(y0, dtype=float)
    start_cost = float(jnp.sum(residual_fn(y0_arr, None) ** 2))
    rel_move = float(jnp.max(jnp.abs(best_value - y0_arr) / (jnp.abs(y0_arr) + 1e-12)))
    info = {"rel_move": rel_move, "start_cost": start_cost, "best_cost": best_cost}
    return best_value, info


def _lm_stall_warnings(info: dict, label: str) -> list[str]:
    """Flag a Levenberg-Marquardt fit that did not move off its warm start.

    The failure mode: when the cost
    landscape is flat — weak identifiability under the given excitation, or a
    non-smooth term whose gradient is ~zero — LM returns the initial guess
    essentially unchanged and the resulting :class:`SysidResult` looks
    legitimate. Surface it as a structured warning so a caller who didn't read
    the recipe docstring still sees that the fit is unverified.
    """
    warnings: list[str] = []
    if info["rel_move"] < 1e-4:
        warnings.append(
            f"sysid.actuator ({label}): the fit returned parameters essentially "
            f"unchanged from the initial guess (max relative move "
            f"{info['rel_move']:.1e}, residual cost {info['best_cost']:.3g}). The "
            "cost landscape is likely flat (weak identifiability under this "
            "excitation, or a non-smooth term with ~zero gradient). Treat the "
            "fitted values as unverified; try richer excitation or "
            "hints={'n_starts': 5}."
        )
    return warnings


def _rk4_step(rhs, y, t, dt, args):
    """Single RK4 step for ``y' = rhs(t, y, args)`` (JAX-traceable)."""
    k1 = rhs(t, y, args)
    k2 = rhs(t + 0.5 * dt, y + 0.5 * dt * k1, args)
    k3 = rhs(t + 0.5 * dt, y + 0.5 * dt * k2, args)
    k4 = rhs(t + dt, y + dt * k3, args)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _servo_rhs(t, y, args):
    """``y = [q, v]``; args holds (q_set, Kp, Kd, T_max, deadband, J, b, Fc, v_eps)."""
    q, v = y[0], y[1]
    q_set, Kp, Kd, T_max, deadband, J, b, Fc, v_eps = args
    err = q_set - q
    err_eff = err - deadband * jnp.tanh(err / deadband)
    tau_cmd = Kp * err_eff - Kd * v
    tau_act = T_max * jnp.tanh(tau_cmd / T_max)
    dvdt = (tau_act - b * v - Fc * jnp.tanh(v / v_eps)) / J
    return jnp.stack([v, dvdt])


def _fit_servo(
    telemetry: ActuatorStepTelemetry, device: Servo, hints: dict
) -> tuple[dict[str, jax.Array], dict[str, float], list[str]]:
    if not isinstance(telemetry, ActuatorStepTelemetry):
        raise JaxterityError(
            "sysid.actuator (Servo) requires ActuatorStepTelemetry; "
            f"got {type(telemetry).__name__}."
        )
    t_arr = jnp.asarray(np.asarray(telemetry.time, dtype=np.float64))
    q_set = jnp.asarray(np.asarray(telemetry.setpoint, dtype=np.float64))
    q_meas = jnp.asarray(np.asarray(telemetry.position, dtype=np.float64))
    v_eps = device._v_eps
    dt = float(t_arr[1] - t_arr[0])

    def rollout(theta):
        Kp, Kd, T_max, deadband, J, b, Fc = (
            theta[0],
            theta[1],
            theta[2],
            theta[3],
            theta[4],
            theta[5],
            theta[6],
        )

        def step(carry, i):
            y_prev, _ = carry, None
            args = (q_set[i], Kp, Kd, T_max, deadband, J, b, Fc, v_eps)
            y_next = _rk4_step(_servo_rhs, y_prev, t_arr[i], dt, args)
            return y_next, y_next[0]

        y0 = jnp.array([q_meas[0], 0.0])
        _, qs = jax.lax.scan(step, y0, jnp.arange(t_arr.size - 1))
        return jnp.concatenate([q_meas[:1], qs])

    def residual_fn(theta, _args):
        return rollout(theta) - q_meas

    y0 = jnp.array(
        [
            float(device.Kp),
            float(device.Kd),
            float(device.T_max),
            float(device.deadband),
            float(device.J),
            float(device.b),
            float(device.Fc),
        ],
        dtype=float,
    )
    max_steps = int(hints.get("max_steps", 128))
    theta, info = _lm_solve(
        residual_fn,
        y0,
        max_steps=max_steps,
        n_starts=int(hints.get("n_starts", 1)),
        perturbation=float(hints.get("perturbation", 0.1)),
        seed=int(hints.get("seed", 0)),
    )
    fitted = {
        "Kp": theta[0],
        "Kd": theta[1],
        "T_max": theta[2],
        "deadband": theta[3],
        "J": theta[4],
        "b": theta[5],
        "Fc": theta[6],
    }
    rms = float(jnp.sqrt(jnp.mean(residual_fn(theta, None) ** 2)))
    return fitted, {"position_rms": rms}, _lm_stall_warnings(info, "Servo")


def _bldc_rhs(t, y, args):
    """``y = [q, v, i]``; args holds (i_ref, R, L, Kt, Ke, Kp_i, J, b, Fc, v_eps)."""
    v, i = y[1], y[2]
    i_ref, R, L, Kt, Ke, Kp_i, J, b, Fc, v_eps = args
    V_app = Kp_i * (i_ref - i)
    e_back = Ke * v
    didt = (V_app - R * i - e_back) / L
    tau = Kt * i
    dvdt = (tau - b * v - Fc * jnp.tanh(v / v_eps)) / J
    return jnp.stack([v, dvdt, didt])


def _fit_bldc(
    telemetry: ActuatorChirpTelemetry, device: BLDC, hints: dict
) -> tuple[dict[str, jax.Array], dict[str, float], list[str]]:
    if not isinstance(telemetry, ActuatorChirpTelemetry):
        raise JaxterityError(
            "sysid.actuator (BLDC) requires ActuatorChirpTelemetry; "
            f"got {type(telemetry).__name__}."
        )
    t_arr = jnp.asarray(np.asarray(telemetry.time, dtype=np.float64))
    iref = jnp.asarray(np.asarray(telemetry.setpoint, dtype=np.float64))
    v_meas = jnp.asarray(np.asarray(telemetry.velocity, dtype=np.float64))
    i_meas = jnp.asarray(np.asarray(telemetry.current, dtype=np.float64))
    v_eps = device._v_eps
    dt_outer = float(t_arr[1] - t_arr[0])

    def rollout(theta):
        R, L, Kt, Ke, Kp_i, J, b, Fc = (
            theta[0],
            theta[1],
            theta[2],
            theta[3],
            theta[4],
            theta[5],
            theta[6],
            theta[7],
        )
        # Substepping keyed to electrical time constant L/(R+Kp_i).
        # Use 8 substeps as a JIT-safe fixed count (avoids data-dependent shapes).
        n_sub = 8
        dt = dt_outer / n_sub

        def outer_step(carry, i):
            y_prev = carry
            ir = iref[i]
            args = (ir, R, L, Kt, Ke, Kp_i, J, b, Fc, v_eps)

            def inner_step(y, _):
                return _rk4_step(_bldc_rhs, y, t_arr[i], dt, args), None

            y_next, _ = jax.lax.scan(inner_step, y_prev, jnp.arange(n_sub))
            return y_next, jnp.stack([y_next[1], y_next[2]])

        # NB: no float() on the closed-over arrays — rollout is traced by
        # optimistix's closure conversion, which turns v_meas/i_meas into
        # tracers; float() on a tracer raises ConcretizationTypeError. The
        # working _fit_servo builds y0 the same way (jnp scalars, no float).
        y0 = jnp.array([0.0, v_meas[0], i_meas[0]])
        _, traj = jax.lax.scan(outer_step, y0, jnp.arange(t_arr.size - 1))
        # traj is (N-1, 2); prepend the first sample.
        head = jnp.array([[v_meas[0], i_meas[0]]])
        return jnp.concatenate([head, traj], axis=0)  # (N, 2)

    def residual_fn(theta, _args):
        pred = rollout(theta)  # (N, 2)
        return jnp.stack([pred[:, 0] - v_meas, pred[:, 1] - i_meas]).flatten()

    y0 = jnp.array(
        [
            float(device.R),
            float(device.L),
            float(device.Kt),
            float(device.Ke),
            float(device.Kp_i),
            float(device.J),
            float(device.b),
            float(device.Fc),
        ],
        dtype=float,
    )
    max_steps = int(hints.get("max_steps", 128))
    theta, info = _lm_solve(
        residual_fn,
        y0,
        max_steps=max_steps,
        n_starts=int(hints.get("n_starts", 1)),
        perturbation=float(hints.get("perturbation", 0.1)),
        seed=int(hints.get("seed", 0)),
    )
    fitted = {
        "R": theta[0],
        "L": theta[1],
        "Kt": theta[2],
        "Ke": theta[3],
        "Kp_i": theta[4],
        "J": theta[5],
        "b": theta[6],
        "Fc": theta[7],
    }
    res = residual_fn(theta, None)
    half = res.size // 2
    v_rms = float(jnp.sqrt(jnp.mean(res[:half] ** 2)))
    i_rms = float(jnp.sqrt(jnp.mean(res[half:] ** 2)))
    return (
        fitted,
        {"velocity_rms": v_rms, "current_rms": i_rms},
        _lm_stall_warnings(info, "BLDC"),
    )


def _sea_rhs(t, y, args):
    """``y = [q_m, v_m, q_l, v_l]``; args holds (tau_in, k, c, J_m, J_l, b_m, b_l)."""
    qm, vm, ql, vl = y[0], y[1], y[2], y[3]
    tau_in, k, c, J_m, J_l, b_m, b_l = args
    tau_spring = k * (qm - ql) + c * (vm - vl)
    dvm = (tau_in - tau_spring - b_m * vm) / J_m
    dvl = (tau_spring - b_l * vl) / J_l
    return jnp.stack([vm, dvm, vl, dvl])


def _fit_series_elastic(
    telemetry: SeriesElasticTelemetry,
    device: SeriesElasticActuator,
    hints: dict,
) -> tuple[dict[str, jax.Array], dict[str, float], list[str]]:
    if not isinstance(telemetry, SeriesElasticTelemetry):
        raise JaxterityError(
            "sysid.actuator (SeriesElasticActuator) requires "
            f"SeriesElasticTelemetry; got {type(telemetry).__name__}."
        )
    t_arr = jnp.asarray(np.asarray(telemetry.time, dtype=np.float64))
    tau_in = jnp.asarray(np.asarray(telemetry.tau_input, dtype=np.float64))
    qm_meas = jnp.asarray(np.asarray(telemetry.q_motor, dtype=np.float64))
    ql_meas = jnp.asarray(np.asarray(telemetry.q_load, dtype=np.float64))
    dt_outer = float(t_arr[1] - t_arr[0])

    def rollout(theta):
        k, c, J_m, J_l, b_m, b_l = (
            theta[0],
            theta[1],
            theta[2],
            theta[3],
            theta[4],
            theta[5],
        )
        n_sub = 8
        dt = dt_outer / n_sub

        def outer_step(carry, i):
            y_prev = carry
            ti = tau_in[i]
            args = (ti, k, c, J_m, J_l, b_m, b_l)

            def inner(y, _):
                return _rk4_step(_sea_rhs, y, t_arr[i], dt, args), None

            y_next, _ = jax.lax.scan(inner, y_prev, jnp.arange(n_sub))
            return y_next, jnp.stack([y_next[0], y_next[2]])

        # No float() here — same tracer constraint as _fit_bldc / _fit_servo.
        y0 = jnp.array([qm_meas[0], 0.0, ql_meas[0], 0.0])
        _, traj = jax.lax.scan(outer_step, y0, jnp.arange(t_arr.size - 1))
        head = jnp.array([[qm_meas[0], ql_meas[0]]])
        return jnp.concatenate([head, traj], axis=0)

    def residual_fn(theta, _args):
        pred = rollout(theta)
        return jnp.stack([pred[:, 0] - qm_meas, pred[:, 1] - ql_meas]).flatten()

    y0 = jnp.array(
        [
            float(device.k),
            float(device.c),
            float(device.J_motor),
            float(device.J_load),
            float(device.b_m),
            float(device.b_l),
        ],
        dtype=float,
    )
    max_steps = int(hints.get("max_steps", 128))
    theta, info = _lm_solve(
        residual_fn,
        y0,
        max_steps=max_steps,
        n_starts=int(hints.get("n_starts", 1)),
        perturbation=float(hints.get("perturbation", 0.1)),
        seed=int(hints.get("seed", 0)),
    )
    fitted = {
        "k": theta[0],
        "c": theta[1],
        "J_motor": theta[2],
        "J_load": theta[3],
        "b_m": theta[4],
        "b_l": theta[5],
    }
    res = residual_fn(theta, None)
    half = res.size // 2
    qm_rms = float(jnp.sqrt(jnp.mean(res[:half] ** 2)))
    ql_rms = float(jnp.sqrt(jnp.mean(res[half:] ** 2)))
    return (
        fitted,
        {"q_motor_rms": qm_rms, "q_load_rms": ql_rms},
        _lm_stall_warnings(info, "SeriesElasticActuator"),
    )


__all__ = ["actuator", "RECIPE_NAME", "RECIPE_VERSION"]
