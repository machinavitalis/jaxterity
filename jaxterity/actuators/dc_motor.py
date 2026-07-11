# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Calibrated DC motor with the WPILib SysId 1-DOF identification model.

``DCMotorWithFriction`` is a Jaxonomy ``LeafSystem`` implementing the WPILib
SysId model

    V(t) = Ks·sgn(v) + Kv·v + Ka·a

with three identifiable parameters: ``Ks`` (V), ``Kv`` (V/(rad/s)) and ``Ka``
(V/(rad/s²)). ``Ks`` folds in static / Coulomb friction projected to the
voltage axis; ``Kv`` folds in viscous friction and back-EMF; ``Ka`` folds in
rotor + reflected-load inertia and electrical effects. This is the canonical
"DC motor with friction" of FRC practice and the simplest realistic actuator
the rest of the library composes (the servo and BLDC build from the same
identification pattern).

Because the WPILib model identifies the *whole* drive assembly's dynamics
(load + motor + electrical), this block is a **stateful drive system**, not a
torque source: it owns a scalar velocity state ``v`` and integrates its own
ODE. It is therefore **not interchangeable** with
:class:`~jaxterity.actuators.IdealTorqueActuator` in a Robot diagram — it
outputs ``v``, not ``tau``. Composing a DC motor with a transmission and an
arbitrary Robot joint belongs to the harmonic-drive transmission and the
``sysid.actuator`` recipe.

The ``sgn(v)`` discontinuity is smoothed to ``tanh(v / v_eps)`` for
differentiability (PATTERNS invariant #6, the same idiom the harmonic-drive
transmission uses for backlash). Forward simulation uses the smooth form; the
closed-form fitter on telemetry uses raw ``np.sign`` (no gradient flows through
telemetry), so the smoothing only affects forward dynamics.

The closed-form OLS fitter :func:`_fit_voltage_ramp` ships privately here to
make the recovery test runnable; the formal ``sysid.actuator`` recipe will
lift it from private to public and wrap it in a
:class:`~jaxterity.sysid.SysidResult`.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..data.schema import ActuatorRampTelemetry, ValidationReport
from ..parameter import ParameterSpec
from ..sysid.protocol import SysidProtocol
from ._base import Actuator

#: Width of the smoothed sign (``tanh(v / v_eps)``). 1e-3 rad/s ≈ 0.01 rpm; small
#: enough that the model matches WPILib's discontinuous form well outside a
#: narrow band around zero velocity.
_V_EPS_DEFAULT = 1e-3


_SYSID_PROTOCOL = SysidProtocol(
    name="voltage-ramp",
    excitation="ramp",
    required_columns=("command", "velocity"),
    duration_s=2.0,
    sample_rate_hz=200.0,
    setup_notes=(
        "Free-spinning actuator (no external load beyond the rotor + reflected "
        "inertia identified into Ka). Apply a *quadratic-in-time* voltage profile "
        "so the design matrix [sgn(v), v, a] has full column rank — a pure linear "
        "ramp gives near-constant acceleration and a rank-deficient fit. Record "
        "applied voltage and rotor velocity at 200 Hz (WPILib SysId default). "
        "Acceleration is computed at fit time via central difference."
    ),
)


def predict_voltage(v, a, *, Ks, Kv, Ka, v_eps: float = _V_EPS_DEFAULT):
    """Predict applied voltage from velocity, acceleration, and motor parameters.

    Implements ``V = Ks · tanh(v/v_eps) + Kv · v + Ka · a``. Pure function — used
    by the forward ODE, by :meth:`DCMotorWithFriction.validate`, and (with raw
    ``np.sign`` instead of ``tanh``) by :func:`_fit_voltage_ramp`.

    Args
    ----
    v : float or jax.Array
        Rotor velocity, rad/s.
    a : float or jax.Array
        Rotor acceleration, rad/s².
    Ks, Kv, Ka : float or jax.Array
        Identified WPILib SysId parameters.
    v_eps : float
        Width of the smoothed sign, rad/s.

    Returns
    -------
    jax.Array
        Predicted voltage, V.

    Units
    -----
    V, Ks: V; v: rad/s; a: rad/s²; Kv: V/(rad/s); Ka: V/(rad/s²); v_eps: rad/s.
    """
    return Ks * jnp.tanh(v / v_eps) + Kv * v + Ka * a


class DCMotorWithFriction(Actuator):
    """WPILib SysId 1-DOF motor + friction model as a stateful drive block.

    See the module docstring for the model, identification protocol, and the
    "drive block, not torque source" boundary.

    Args
    ----
    Ks, Kv, Ka : float or jax.Array
        WPILib SysId parameters (see module docstring for units and physical
        interpretation). All three are identifiable; ``Ka`` must be > 0.
    v_eps : float
        Width of the smoothed sign function, rad/s. Default 1e-3 rad/s.
    v0 : float
        Initial rotor velocity, rad/s. Default 0.
    name : str
        Block name.
    """

    sysid_protocol = _SYSID_PROTOCOL

    def __init__(
        self,
        *,
        Ks=0.0,
        Kv=0.0,
        Ka=1.0,
        v_eps: float = _V_EPS_DEFAULT,
        v0: float = 0.0,
        name: str = "dc_motor",
    ):
        super().__init__(name=name)
        self._v_eps = float(v_eps)
        self.declare_dynamic_parameter("Ks", jnp.asarray(Ks, dtype=float))
        self.declare_dynamic_parameter("Kv", jnp.asarray(Kv, dtype=float))
        self.declare_dynamic_parameter("Ka", jnp.asarray(Ka, dtype=float))
        self.declare_input_port(name="V_cmd")
        self.declare_continuous_state(
            default_value=jnp.asarray(v0, dtype=float), ode=self._ode
        )
        self.declare_continuous_state_output(name="v")

    @property
    def Ks(self):
        return self.dynamic_parameters["Ks"].get()

    @property
    def Kv(self):
        return self.dynamic_parameters["Kv"].get()

    @property
    def Ka(self):
        return self.dynamic_parameters["Ka"].get()

    @classmethod
    def uncalibrated(cls, *, name: str = "dc_motor") -> "DCMotorWithFriction":
        """Return an instance with identifiability-neutral defaults.

        ``Ks=Kv=0`` (nothing identified yet) and ``Ka=1`` (a non-zero placeholder
        — ``Ka`` is in the ODE denominator so 0 is unsafe). Use this when
        constructing an actuator whose parameters a sysid recipe will fit.
        """
        return cls(Ks=0.0, Kv=0.0, Ka=1.0, name=name)

    def _ode(self, time, state, *inputs, **params):
        v = state.continuous_state
        V_cmd = inputs[0]
        Ks, Kv, Ka = params["Ks"], params["Kv"], params["Ka"]
        return (V_cmd - Ks * jnp.tanh(v / self._v_eps) - Kv * v) / Ka

    def parameter_spec(self) -> ParameterSpec:
        return ParameterSpec(("Ks", "Kv", "Ka"))

    def validate(self, telemetry: ActuatorRampTelemetry) -> ValidationReport:
        """Per-sample voltage residual report against a ramp telemetry trace.

        Computes ``V_pred = predict_voltage(v, a; Ks, Kv, Ka)`` for the current
        parameters and compares to the recorded ``command`` channel. Acceleration
        is computed from ``velocity`` via central difference (boundary points
        dropped to keep the comparison honest). The report is non-fatal: a
        ``warning`` line is emitted if the RMS residual exceeds the policy
        ``atol("actuator_voltage", "rk4", "float64")`` bound.

        Args
        ----
        telemetry : ActuatorRampTelemetry
            Voltage/velocity trace for this actuator.

        Returns
        -------
        ValidationReport
            ``ok`` iff the RMS residual is within the policy bound; a single
            warning line carries the RMS and max residual numbers.
        """
        from ..testing import tolerances as tol

        v = np.asarray(telemetry.velocity, dtype=np.float64)
        V_meas = np.asarray(telemetry.command, dtype=np.float64)
        t = np.asarray(telemetry.time, dtype=np.float64)
        # central difference; boundary samples dropped
        a = (v[2:] - v[:-2]) / (t[2:] - t[:-2])
        v_c, V_c = v[1:-1], V_meas[1:-1]
        Ks, Kv, Ka = float(self.Ks), float(self.Kv), float(self.Ka)
        V_pred = Ks * np.sign(v_c) + Kv * v_c + Ka * a
        residual = V_c - V_pred
        rms = float(np.sqrt(np.mean(residual * residual)))
        peak = float(np.max(np.abs(residual)))
        report = ValidationReport()
        bound = tol.atol("actuator_voltage", "rk4", "float64")
        if rms > bound:
            report.warnings.append(
                f"voltage residual RMS={rms:.3g} V (peak={peak:.3g} V) "
                f"exceeds policy bound {bound:.3g} V — model parameters "
                "may not match this trace."
            )
        return report


def _fit_voltage_ramp(telemetry: ActuatorRampTelemetry) -> dict[str, np.ndarray]:
    """Closed-form OLS fit of the WPILib SysId parameters from a ramp trace.

    Solves the linear least-squares problem ``V ≈ Ks · sgn(v) + Kv · v + Ka · a``
    over the trace samples (interior only — boundary samples are dropped to
    avoid the central-difference boundary effect on ``a``). Returns the fitted
    ``{"Ks", "Kv", "Ka"}`` as float64 arrays.

    Private to ``dc_motor.py``, where it is needed to make the recovery test
    runnable. The ``sysid.actuator`` recipe is the public home — when that
    lands, this function lifts to public and is wrapped in a
    :class:`~jaxterity.sysid.SysidResult`.

    Args
    ----
    telemetry : ActuatorRampTelemetry
        Voltage and velocity samples for one actuator under a quasi-static
        ramp (see :data:`DCMotorWithFriction.sysid_protocol` for the
        recommended excitation).

    Returns
    -------
    dict[str, numpy.ndarray]
        ``{"Ks": ..., "Kv": ..., "Ka": ...}`` — scalar float64 arrays.
    """
    v = np.asarray(telemetry.velocity, dtype=np.float64)
    V = np.asarray(telemetry.command, dtype=np.float64)
    t = np.asarray(telemetry.time, dtype=np.float64)
    a = (v[2:] - v[:-2]) / (t[2:] - t[:-2])
    v_c, V_c = v[1:-1], V[1:-1]
    X = np.stack([np.sign(v_c), v_c, a], axis=1)
    coef, *_ = np.linalg.lstsq(X, V_c, rcond=None)
    return {"Ks": coef[0], "Kv": coef[1], "Ka": coef[2]}


__all__ = ["DCMotorWithFriction", "predict_voltage"]
