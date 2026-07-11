# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Bus-voltage sag under aggregate servo load (LeKiwi).

``BatterySag`` is the electrical off-body block the body model never sees: a
shared battery whose terminal (bus) voltage droops as the servos draw current.
It is a purely algebraic (feedthrough) Jaxonomy ``LeafSystem`` — no state. A
Thévenin source with open-circuit voltage ``V_oc`` (V) and internal resistance
``R_int`` (Ω) drops under the aggregate current drawn by ``n`` servos::

    I_total = Σ_i  k_i · |tau_i|        (per-servo torque→current map k_i)
    V_bus   = V_oc  −  R_int · I_total

A torque-controlled servo's current draw scales with torque magnitude (current
flows for either direction of torque), so the per-servo map is on ``|tau_i|``.
The absolute value is smoothed as ``tau_i · tanh(tau_i / i_eps)`` (the same
idiom the :class:`~jaxterity.actuators.servo.Servo` uses for Coulomb friction)
so the block composes through forward-mode autodiff with no kink at ``tau_i=0``:
the proxy is exactly zero at zero torque and exponentially close to ``|tau_i|``
for ``|tau_i| ≫ i_eps`` (no constant bias).

Two identifiable parameters are PyTree leaves: ``V_oc`` (V) and ``R_int`` (Ω) —
both linear in the model, so a known-load trace identifies them exactly by OLS
(:func:`_fit_battery`). The per-servo gains ``k_i`` (A per N·m) are static
configuration (the servo line's datasheet torque constant), not fitted here.

LeKiwi-only — the bench SO-101 path needs only
:class:`~jaxterity.physics.thermal.ServoThermal`. This module ships the model,
the :func:`bus_voltage` helper, the OLS fitter, and a :meth:`BatterySag.validate`
residual report. It composes onto the LeKiwi twin.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jaxonomy.framework.leaf_system import LeafSystem

from ..data.schema import BusVoltageTelemetry, ValidationReport
from ..parameter import ParameterSpec

#: Smoothing width for the current-draw |tau| (N·m). Small enough that the
#: smoothed magnitude matches |tau| well outside a narrow band around zero.
_I_EPS_DEFAULT = 1e-3


def _smooth_abs(x, eps: float):
    """Smooth absolute value: ``x · tanh(x / eps)``.

    Exactly zero at ``x=0``, even in ``x``, and exponentially close to ``|x|``
    for ``|x| ≫ eps`` (no constant bias). C∞ everywhere — the same smoothing
    idiom the servo uses for Coulomb friction.
    """
    return x * jnp.tanh(x / eps)


def bus_voltage(tau, *, V_oc, R_int, k_i, i_eps: float = _I_EPS_DEFAULT):
    """Bus voltage under an aggregate servo torque load.

    Implements ``V_bus = V_oc − R_int · Σ_i k_i·|tau_i|`` with a smoothed
    ``|·|``. Pure function — used by the feedthrough output, by
    :meth:`BatterySag.validate`, and (with raw ``np.abs``) by
    :func:`_fit_battery`.

    Args
    ----
    tau : jax.Array, shape ``(n,)``
        Per-servo torque, N·m.
    V_oc : float or jax.Array
        Open-circuit (no-load) bus voltage, V.
    R_int : float or jax.Array
        Battery internal resistance, Ω.
    k_i : jax.Array, shape ``(n,)``
        Per-servo torque→current gains, A per N·m.
    i_eps : float
        Smoothing width for the current-draw ``|tau|``, N·m.

    Returns
    -------
    jax.Array
        Bus voltage, V (scalar).

    Units
    -----
    tau: N·m; V_oc/V_bus: V; R_int: Ω; k_i: A/(N·m); i_eps: N·m.
    """
    current = jnp.sum(k_i * _smooth_abs(tau, i_eps))
    return V_oc - R_int * current


class BatterySag(LeafSystem):
    """Thévenin bus-voltage sag under aggregate servo load (algebraic block).

    See the module docstring for the model and the smooth-``|·|`` rationale.
    Input port ``tau`` is the per-servo torque vector (N·m, shape ``(n,)``);
    output port ``V_bus`` is the sagged bus voltage (V). Stateless — a direct
    feedthrough from ``tau`` to ``V_bus``.

    Args
    ----
    k_i : array-like, shape ``(n,)``
        Per-servo torque→current gains, A per N·m (static configuration).
    V_oc : float or jax.Array
        Open-circuit bus voltage, V. Must be > 0.
    R_int : float or jax.Array
        Battery internal resistance, Ω. Must be ≥ 0.
    i_eps : float
        Smoothing width for the current-draw ``|tau|``, N·m.
    name : str
        Block name.
    """

    def __init__(
        self,
        *,
        k_i,
        V_oc=12.0,
        R_int=0.0,
        i_eps: float = _I_EPS_DEFAULT,
        name: str = "battery_sag",
    ):
        super().__init__(name=name)
        self._k_i = jnp.asarray(k_i, dtype=float)
        self._n = int(self._k_i.shape[0])
        self._i_eps = float(i_eps)
        self.declare_dynamic_parameter("V_oc", jnp.asarray(V_oc, dtype=float))
        self.declare_dynamic_parameter("R_int", jnp.asarray(R_int, dtype=float))
        self.declare_input_port(name="tau")
        self.declare_output_port(self._v_bus_out, name="V_bus")

    @property
    def V_oc(self):
        return self.dynamic_parameters["V_oc"].get()

    @property
    def R_int(self):
        return self.dynamic_parameters["R_int"].get()

    @property
    def k_i(self):
        """Static per-servo torque→current gains, A per N·m (not a leaf)."""
        return self._k_i

    @property
    def n_servos(self) -> int:
        """Number of servos sharing the bus."""
        return self._n

    @classmethod
    def uncalibrated(cls, n_servos: int, *, name: str = "battery_sag") -> "BatterySag":
        """Return an instance with identifiability-neutral defaults.

        ``V_oc=12`` V (a benign nominal), ``R_int=0`` (no sag identified yet),
        and unit current gains ``k_i=1``. Use this when constructing a battery
        block whose parameters a sysid recipe will fit.
        """
        return cls(k_i=jnp.ones(n_servos), V_oc=12.0, R_int=0.0, name=name)

    def _v_bus_out(self, time, state, *inputs, **params):
        tau = inputs[0]
        return bus_voltage(
            tau,
            V_oc=params["V_oc"],
            R_int=params["R_int"],
            k_i=self._k_i,
            i_eps=self._i_eps,
        )

    def parameter_spec(self) -> ParameterSpec:
        return ParameterSpec(("V_oc", "R_int"))

    def validate(self, telemetry: BusVoltageTelemetry, tau) -> ValidationReport:
        """Per-sample bus-voltage residual report against a known-load trace.

        Computes ``V_pred = bus_voltage(tau; V_oc, R_int, k_i)`` for the current
        parameters and compares to the recorded ``voltage`` channel. The report
        is non-fatal: a warning is emitted if the RMS residual exceeds the policy
        ``bus_voltage`` (rk4 / float64) bound.

        Args
        ----
        telemetry : BusVoltageTelemetry
            Bus-voltage trace under load.
        tau : numpy.ndarray, shape ``(n, n_servos)``, units N·m
            Per-servo torque, one row per telemetry timestamp.

        Returns
        -------
        ValidationReport
            ``ok`` iff the RMS residual is within the policy bound.

        Units
        -----
        telemetry voltage: V; tau: N·m.
        """
        from ..testing import tolerances as tol

        V_meas = np.asarray(telemetry.voltage, dtype=np.float64)
        tau_arr = np.asarray(tau, dtype=np.float64)
        k_i = np.asarray(self._k_i, dtype=np.float64)
        V_oc, R_int = float(self.V_oc), float(self.R_int)
        I_total = np.sum(k_i * np.abs(tau_arr), axis=1)
        V_pred = V_oc - R_int * I_total
        residual = V_pred - V_meas
        rms = float(np.sqrt(np.mean(residual * residual)))
        peak = float(np.max(np.abs(residual)))
        report = ValidationReport()
        bound = tol.atol("bus_voltage", "rk4", "float64") * 100.0
        if rms > bound:
            report.warnings.append(
                f"bus-voltage residual RMS={rms:.3g} V (peak={peak:.3g} V) "
                f"exceeds policy bound {bound:.3g} V — battery parameters may "
                "not match this trace."
            )
        return report


def _fit_battery(
    telemetry: BusVoltageTelemetry,
    tau,
    k_i,
) -> dict[str, float]:
    """Closed-form OLS fit of ``V_oc`` and ``R_int`` from a known-load trace.

    Solves ``V_bus ≈ V_oc − R_int · I_total`` (linear in both parameters) over
    the trace, with ``I_total = Σ_i k_i·|tau_i|`` (raw ``np.abs`` — no gradient
    flows through telemetry). Both parameters are identifiable from a trace that
    spans a range of loads.

    Private to ``battery.py`` (to make the recovery test runnable);
    the formal ``sysid`` recipe home is later work.

    Args
    ----
    telemetry : BusVoltageTelemetry
        Bus-voltage trace under load.
    tau : numpy.ndarray, shape ``(n, n_servos)``, units N·m
        Per-servo torque, one row per timestamp.
    k_i : array-like, shape ``(n_servos,)``
        Per-servo torque→current gains, A per N·m.

    Returns
    -------
    dict[str, float]
        ``{"V_oc": ..., "R_int": ...}``.
    """
    V = np.asarray(telemetry.voltage, dtype=np.float64)
    tau_arr = np.asarray(tau, dtype=np.float64)
    k_arr = np.asarray(k_i, dtype=np.float64)
    I_total = np.sum(k_arr * np.abs(tau_arr), axis=1)
    X = np.stack([np.ones_like(I_total), -I_total], axis=1)
    coef, *_ = np.linalg.lstsq(X, V, rcond=None)
    return {"V_oc": float(coef[0]), "R_int": float(coef[1])}


__all__ = ["BatterySag", "bus_voltage"]
