# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Lumped first-order servo-thermal model.

``ServoThermal`` is the off-body physics the body (MJX) model never sees: a
servo's winding/case heats up under load and cools toward ambient. It is a
single-node lumped thermal block — a Jaxonomy ``LeafSystem`` owning one scalar
state, the temperature ``T`` (K) — driven by the actuator torque ``tau`` (N·m)
through a smooth torque²-proxy of dissipated power::

    C_th · dT/dt = k_pwr · tau²  −  (T − T_amb) / R_th

    ⇒  dT/dt = (k_pwr · tau²  −  (T − T_amb) / R_th) / C_th

Three identifiable parameters are PyTree leaves: ``C_th`` (J/K, lumped thermal
capacitance), ``R_th`` (K/W, thermal resistance to ambient), ``k_pwr``
(W per (N·m)², the torque²→dissipated-power gain). ``T_amb`` (K) is a static
(non-identifiable) ambient constant. The heating proxy is ``tau²`` rather than
``|tau|`` precisely because the square is smooth (C∞) everywhere — no kink at
``tau=0`` — so it composes through forward-mode autodiff, including the case
where ``tau`` arrives from the :class:`~jaxterity.actuators.servo.Servo`'s
``tanh`` saturation (the twin wires that path).

Identifiability — read before fitting. From temperature + torque telemetry
**alone**, the three leaves are *not* independently identifiable. The forward
ODE depends on only two combinations,

    a = k_pwr / C_th        (rate gain on tau²)
    b = 1 / (R_th · C_th)   (Newton-cooling rate)

so a step response reveals exactly one time constant ``τ = R_th·C_th = 1/b``
and one steady-state gain ``K = ΔT_ss/tau² = k_pwr·R_th = a/b`` — two numbers,
three unknowns. ``C_th`` is the free direction. The fitter
(:func:`_fit_thermal`) therefore takes ``C_th`` as **known** (it is the lumped
thermal mass, obtained from the servo's geometry/datasheet, not the trace) and
recovers ``R_th`` and ``k_pwr``; with ``C_th`` fixed the recovery is exact on
model-consistent data. Pretending all three fall out of one trace would be a
hollow claim; this is the honest contract.

This module ships the model, the analytic helpers
(:func:`thermal_steady_state`, :func:`thermal_response`), the ``C_th``-known
fitter, and a forward-simulation :meth:`ServoThermal.validate`. Full
composition onto the SO-101 ``Servo`` and the per-unit twin.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jaxonomy.framework.leaf_system import LeafSystem

from ..data.schema import ThermalTelemetry, ValidationReport
from ..parameter import ParameterSpec

#: Default ambient temperature (K) — 25 °C, the usual bench/datasheet reference.
_T_AMB_DEFAULT = 298.15


def dissipated_power(tau, k_pwr):
    """Smooth torque²-proxy of dissipated power.

    Implements ``P = k_pwr · tau²`` — the C∞ heating proxy the thermal ODE
    integrates. Squaring (not ``|tau|``) keeps the proxy differentiable at
    ``tau=0`` so it composes through forward-mode autodiff.

    Args
    ----
    tau : float or jax.Array
        Actuator torque, N·m.
    k_pwr : float or jax.Array
        Torque²→power gain, W per (N·m)².

    Returns
    -------
    jax.Array
        Dissipated power, W.

    Units
    -----
    tau: N·m; k_pwr: W/(N·m)²; returns W.
    """
    return k_pwr * tau * tau


def thermal_steady_state(tau, *, T_amb, R_th, k_pwr):
    """Steady-state temperature under a constant torque.

    Sets ``dT/dt = 0`` in the lumped ODE: ``T_ss = T_amb + k_pwr·R_th·tau²``.
    The rise ``ΔT_ss = k_pwr·R_th·tau²`` depends only on the product
    ``k_pwr·R_th`` (the steady-state gain), not on ``C_th``.

    Args
    ----
    tau : float or jax.Array
        Constant applied torque, N·m.
    T_amb : float or jax.Array
        Ambient temperature, K.
    R_th : float or jax.Array
        Thermal resistance to ambient, K/W.
    k_pwr : float or jax.Array
        Torque²→power gain, W per (N·m)².

    Returns
    -------
    jax.Array
        Steady-state temperature, K.

    Units
    -----
    tau: N·m; T_amb: K; R_th: K/W; k_pwr: W/(N·m)²; returns K.
    """
    return T_amb + dissipated_power(tau, k_pwr) * R_th


def thermal_response(t, tau, *, T0, T_amb, C_th, R_th, k_pwr):
    """Analytic first-order temperature response to a constant torque.

    Closed-form solution of the lumped ODE for a constant ``tau`` held from
    ``T(0) = T0``::

        T(t) = T_ss + (T0 − T_ss) · exp(−t / τ),   τ = R_th · C_th

    where ``T_ss`` is :func:`thermal_steady_state`. This is the reference the
    forward simulation is checked against.

    Args
    ----
    t : float or jax.Array
        Time since the torque step, s.
    tau : float or jax.Array
        Constant applied torque, N·m.
    T0 : float or jax.Array
        Initial temperature, K.
    T_amb : float or jax.Array
        Ambient temperature, K.
    C_th : float or jax.Array
        Lumped thermal capacitance, J/K.
    R_th : float or jax.Array
        Thermal resistance to ambient, K/W.
    k_pwr : float or jax.Array
        Torque²→power gain, W per (N·m)².

    Returns
    -------
    jax.Array
        Temperature at time ``t``, K.

    Units
    -----
    t: s; tau: N·m; T0/T_amb: K; C_th: J/K; R_th: K/W; k_pwr: W/(N·m)²; returns K.
    """
    T_ss = thermal_steady_state(tau, T_amb=T_amb, R_th=R_th, k_pwr=k_pwr)
    tau_time = R_th * C_th
    return T_ss + (T0 - T_ss) * jnp.exp(-t / tau_time)


class ServoThermal(LeafSystem):
    """Single-node lumped servo-thermal model as a stateful LeafSystem.

    See the module docstring for the model, the smooth-proxy rationale, and the
    (important) identifiability contract. State is the scalar temperature ``T``
    (K); input port ``tau`` is the actuator torque (N·m); output port ``T`` is
    the temperature.

    Args
    ----
    C_th : float or jax.Array
        Lumped thermal capacitance, J/K. Must be > 0 (ODE denominator).
    R_th : float or jax.Array
        Thermal resistance to ambient, K/W. Must be > 0 (sets cooling rate).
    k_pwr : float or jax.Array
        Torque²→dissipated-power gain, W per (N·m)². Must be ≥ 0.
    T_amb : float
        Ambient temperature, K (static; not identifiable). Default 298.15 K.
    T0 : float or None
        Initial temperature, K. Defaults to ``T_amb`` (starts at ambient).
    name : str
        Block name.
    """

    def __init__(
        self,
        *,
        C_th=1.0,
        R_th=1.0,
        k_pwr=0.0,
        T_amb: float = _T_AMB_DEFAULT,
        T0: float | None = None,
        name: str = "servo_thermal",
    ):
        super().__init__(name=name)
        self._T_amb = float(T_amb)
        for n, x in (("C_th", C_th), ("R_th", R_th), ("k_pwr", k_pwr)):
            self.declare_dynamic_parameter(n, jnp.asarray(x, dtype=float))
        self.declare_input_port(name="tau")
        T_init = self._T_amb if T0 is None else float(T0)
        self.declare_continuous_state(
            default_value=jnp.asarray(T_init, dtype=float), ode=self._ode
        )
        self.declare_continuous_state_output(name="T")

    @property
    def C_th(self):
        return self.dynamic_parameters["C_th"].get()

    @property
    def R_th(self):
        return self.dynamic_parameters["R_th"].get()

    @property
    def k_pwr(self):
        return self.dynamic_parameters["k_pwr"].get()

    @property
    def T_amb(self) -> float:
        """Static ambient temperature, K (not an identifiable leaf)."""
        return self._T_amb

    @classmethod
    def uncalibrated(cls, *, name: str = "servo_thermal") -> "ServoThermal":
        """Return an instance with identifiability-neutral defaults.

        ``C_th=R_th=1`` (non-zero so the ODE is well-defined) and ``k_pwr=0``
        (no heating identified yet). Use this when constructing a thermal block
        whose parameters a sysid recipe will fit.
        """
        return cls(name=name)

    def _ode(self, time, state, *inputs, **params):
        T = state.continuous_state
        tau = inputs[0]
        C_th, R_th, k_pwr = params["C_th"], params["R_th"], params["k_pwr"]
        power_in = dissipated_power(tau, k_pwr)
        power_out = (T - self._T_amb) / R_th
        return (power_in - power_out) / C_th

    def parameter_spec(self) -> ParameterSpec:
        return ParameterSpec(("C_th", "R_th", "k_pwr"))

    def validate(self, telemetry: ThermalTelemetry, tau) -> ValidationReport:
        """Forward-simulate the thermal ODE against a temperature trace.

        Integrates the scalar ODE with a fixed-step NumPy RK4 between telemetry
        samples (the torque is held constant on each interval — zero-order hold,
        matching a real recording). Initial temperature is seeded from the first
        telemetry sample. A warning is appended if the RMS temperature residual
        exceeds the ``motor_temperature`` policy bound (RK4 / float64, scaled
        ×100 to allow for the modeling gap between this clean lumped model and a
        real recording).

        Args
        ----
        telemetry : ThermalTelemetry
            Per-servo temperature trace. The first servo column is validated
            (single-node model); pass a single-servo trace for an unambiguous
            check.
        tau : numpy.ndarray, shape ``(n,)``, units N·m
            The driving torque, one sample per telemetry timestamp. Carried
            separately because the heating input is the actuator torque, not a
            thermal channel.

        Returns
        -------
        ValidationReport
            ``ok`` iff the RMS residual is within the policy bound.

        Units
        -----
        telemetry temperature: K; tau: N·m.
        """
        from ..testing import tolerances as tol

        t_arr = np.asarray(telemetry.time, dtype=np.float64)
        T_meas = np.asarray(telemetry.temperature, dtype=np.float64)[:, 0]
        tau_arr = np.asarray(tau, dtype=np.float64)

        C_th, R_th, k_pwr = float(self.C_th), float(self.R_th), float(self.k_pwr)
        T_amb = self._T_amb

        def rhs(T, tau_i):
            return (k_pwr * tau_i * tau_i - (T - T_amb) / R_th) / C_th

        T_sim = np.empty_like(T_meas)
        T_sim[0] = T_meas[0]
        T = T_meas[0]
        for i in range(1, t_arr.size):
            dt = float(t_arr[i] - t_arr[i - 1])
            u = tau_arr[i - 1]  # ZOH: torque applied on the leading edge
            k1 = rhs(T, u)
            k2 = rhs(T + 0.5 * dt * k1, u)
            k3 = rhs(T + 0.5 * dt * k2, u)
            k4 = rhs(T + dt * k3, u)
            T = T + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            T_sim[i] = T

        residual = T_sim - T_meas
        rms = float(np.sqrt(np.mean(residual * residual)))
        peak = float(np.max(np.abs(residual)))
        report = ValidationReport()
        bound = tol.atol("motor_temperature", "rk4", "float64") * 100.0
        if rms > bound:
            report.warnings.append(
                f"temperature residual RMS={rms:.3g} K (peak={peak:.3g} K) "
                f"exceeds policy bound {bound:.3g} K — thermal parameters may "
                "not match this trace."
            )
        return report


def _fit_thermal(
    telemetry: ThermalTelemetry,
    tau,
    *,
    T_amb: float,
    C_th: float,
) -> dict[str, float]:
    """Recover ``R_th`` and ``k_pwr`` from a temperature trace, ``C_th`` known.

    Linear-least-squares fit of the lumped ODE in its identifiable form. With
    measured ``T(t)`` and torque ``tau(t)``, the derivative ``Ṫ`` is computed by
    central difference and regressed as

        Ṫ = a · tau²  −  b · (T − T_amb),   a = k_pwr/C_th, b = 1/(R_th·C_th)

    The two OLS coefficients ``(a, b)`` are all the trace identifies (see the
    module docstring). With ``C_th`` supplied, ``k_pwr = a·C_th`` and
    ``R_th = 1/(b·C_th)``.

    Private to ``thermal.py`` (to make the recovery test runnable).
    The formal ``sysid`` recipe home is later work.

    Args
    ----
    telemetry : ThermalTelemetry
        Per-servo temperature trace; the first servo column is fitted.
    tau : numpy.ndarray, shape ``(n,)``, units N·m
        Driving torque, aligned with the telemetry timestamps.
    T_amb : float
        Ambient temperature, K (the model's static constant).
    C_th : float
        Known lumped thermal capacitance, J/K — fixes the otherwise
        unidentifiable scale (see module docstring).

    Returns
    -------
    dict[str, float]
        ``{"R_th": ..., "k_pwr": ...}``.

    Raises
    ------
    ValueError
        If ``C_th`` is not strictly positive.
    """
    if not (C_th > 0):
        raise ValueError(
            f"_fit_thermal needs a strictly positive known C_th (J/K), got "
            f"{C_th!r}. C_th fixes the unidentifiable scale of the lumped model "
            "— see thermal.py module docstring."
        )
    t = np.asarray(telemetry.time, dtype=np.float64)
    T = np.asarray(telemetry.temperature, dtype=np.float64)[:, 0]
    tau_arr = np.asarray(tau, dtype=np.float64)
    # central difference for Ṫ; boundary samples dropped to match
    Tdot = (T[2:] - T[:-2]) / (t[2:] - t[:-2])
    T_c = T[1:-1]
    tau_c = tau_arr[1:-1]
    X = np.stack([tau_c * tau_c, -(T_c - T_amb)], axis=1)
    coef, *_ = np.linalg.lstsq(X, Tdot, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    return {"k_pwr": a * C_th, "R_th": 1.0 / (b * C_th)}


__all__ = [
    "ServoThermal",
    "dissipated_power",
    "thermal_steady_state",
    "thermal_response",
]
