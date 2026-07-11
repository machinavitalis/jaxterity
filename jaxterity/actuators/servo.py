# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Position-controlled servo with internal PD loop, saturation, and deadband.

``Servo`` models the kind of hobby/manipulator actuator the SO-100 uses: an
external position setpoint, an internal torque loop, and the joint dynamics it
drives, baked into a single stateful Jaxonomy ``LeafSystem``.

Model — state is ``[q, v]`` (rad, rad/s); input is the position setpoint
``q_set`` (rad); output ports are ``q`` and ``v``::

    err       = q_set − q
    err_eff   = err − deadband · tanh(err / deadband)      (smooth deadband)
    τ_cmd     = Kp · err_eff − Kd · v                       (PD controller)
    τ_act     = T_max · tanh(τ_cmd / T_max)                 (smooth saturation)
    q̇          = v
    v̇          = (τ_act − b · v − Fc · tanh(v / v_eps)) / J  (joint dynamics)

Seven identifiable parameters: ``Kp`` (N·m/rad), ``Kd`` (N·m·s/rad), ``T_max``
(N·m holding torque ≈ saturation), ``deadband`` (rad), ``J`` (kg·m² effective
inertia), ``b`` (N·m·s/rad viscous friction), ``Fc`` (N·m Coulomb friction).
Identification is from step-response telemetry (see :attr:`Servo.sysid_protocol`
and :class:`~jaxterity.data.schema.ActuatorStepTelemetry`).

The fitter — step-response identification through saturation + deadband is not
closed-form — is deferred to the ``sysid.actuator`` recipe (an iterative
nonlinear least squares). This module ships the model, the schema, the
``sysid_protocol``, and a forward-simulation ``validate`` that reports position
residuals against telemetry; that is the surface the recipe will consume.

Like :class:`~jaxterity.actuators.dc_motor.DCMotorWithFriction`, ``Servo`` is a
**stateful drive block** — it owns its joint state and is *not* a torque source
that drops into a Robot's ``tau`` input. Composing a servo through a transmission
to an arbitrary Robot joint belongs to the transmission and sysid recipe layers.

The PI term is intentionally omitted (PD only): hobby servos rarely use I, and
the integrator state interacts badly with saturation (windup) — when the recipe
demonstrably needs I, add it then with the standard anti-windup clamp.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..data.schema import ActuatorStepTelemetry, ValidationReport
from ..parameter import ParameterSpec
from ..sysid.protocol import SysidProtocol
from ._base import Actuator

#: Smoothing width for the Coulomb-friction sign (rad/s). Same idiom as the DC motor.
_V_EPS_DEFAULT = 1e-3


_SYSID_PROTOCOL = SysidProtocol(
    name="step-response",
    excitation="step",
    required_columns=("setpoint", "position"),
    duration_s=3.0,
    sample_rate_hz=200.0,
    setup_notes=(
        "Command a sequence of position step changes wide enough to engage the "
        "saturation (T_max) and small enough that some steps stay inside the "
        "deadband. Record commanded setpoint and measured position at 200 Hz. "
        "The fitter infers Kp/Kd/T_max/deadband/J/b/Fc by simulation "
        "matching — step response with saturation and deadband is not "
        "closed-form."
    ),
)


def _soft_deadband(err, deadband):
    """Smooth approximation of ``sign(err) · max(|err| − deadband, 0)``.

    Returns ``err − deadband · tanh(err / deadband)``: zero at err=0, smoothly
    approaches ``err − sign(err) · deadband`` for |err| ≫ deadband. Continuous
    and differentiable everywhere — preserved through forward-mode autodiff.
    """
    return err - deadband * jnp.tanh(err / deadband)


def _soft_saturate(tau, tau_max):
    """Smooth tanh saturation: ``tau_max · tanh(tau / tau_max)``."""
    return tau_max * jnp.tanh(tau / tau_max)


class Servo(Actuator):
    """Position-controlled servo + load as a stateful drive block.

    See the module docstring for the model, port shapes, and identification
    protocol.

    Args
    ----
    Kp, Kd : float or jax.Array
        Position / velocity gains of the internal PD loop. Units:
        N·m/rad and N·m·s/rad respectively.
    T_max : float or jax.Array
        Saturation / holding torque, N·m. Must be > 0.
    deadband : float or jax.Array
        Position-error half-width below which the PD output is zero, rad.
        Must be > 0; use a small value (~1e-4 rad) to effectively disable.
    J : float or jax.Array
        Effective inertia at the joint, kg·m². Must be > 0.
    b : float or jax.Array
        Viscous friction coefficient, N·m·s/rad.
    Fc : float or jax.Array
        Coulomb friction torque, N·m.
    v_eps : float
        Smoothing width for the Coulomb-friction sign (rad/s).
    q0, v0 : float
        Initial position and velocity (rad, rad/s).
    name : str
        Block name.
    """

    sysid_protocol = _SYSID_PROTOCOL

    def __init__(
        self,
        *,
        Kp=1.0,
        Kd=0.1,
        T_max=5.0,
        deadband=1e-4,
        J=1e-3,
        b=0.0,
        Fc=0.0,
        v_eps: float = _V_EPS_DEFAULT,
        q0: float = 0.0,
        v0: float = 0.0,
        name: str = "servo",
    ):
        super().__init__(name=name)
        self._v_eps = float(v_eps)
        for n, x in (
            ("Kp", Kp),
            ("Kd", Kd),
            ("T_max", T_max),
            ("deadband", deadband),
            ("J", J),
            ("b", b),
            ("Fc", Fc),
        ):
            self.declare_dynamic_parameter(n, jnp.asarray(x, dtype=float))
        self.declare_input_port(name="q_set")
        self.declare_continuous_state(
            default_value=jnp.asarray([q0, v0], dtype=float), ode=self._ode
        )
        # Custom output ports: q and v individually (not the full state vector),
        # since downstream consumers (sensors, controllers) typically want one or
        # the other, not the concatenated [q, v].
        self.declare_output_port(self._q_out, name="q", requires_inputs=False)
        self.declare_output_port(self._v_out, name="v", requires_inputs=False)

    @property
    def Kp(self):
        return self.dynamic_parameters["Kp"].get()

    @property
    def Kd(self):
        return self.dynamic_parameters["Kd"].get()

    @property
    def T_max(self):
        return self.dynamic_parameters["T_max"].get()

    @property
    def deadband(self):
        return self.dynamic_parameters["deadband"].get()

    @property
    def J(self):
        return self.dynamic_parameters["J"].get()

    @property
    def b(self):
        return self.dynamic_parameters["b"].get()

    @property
    def Fc(self):
        return self.dynamic_parameters["Fc"].get()

    @classmethod
    def uncalibrated(cls, *, name: str = "servo") -> "Servo":
        """Return an instance with identifiability-neutral defaults.

        Kp=1, Kd=0.1 (a benign PD baseline); T_max=1 N·m, deadband=1e-4 rad,
        J=1e-3 kg·m², b=Fc=0 (no friction). All denominator params (T_max, J,
        deadband) are non-zero so the ODE is well-defined out of the box.
        """
        return cls(name=name)

    def _q_out(self, time, state, *inputs, **params):
        return state.continuous_state[0]

    def _v_out(self, time, state, *inputs, **params):
        return state.continuous_state[1]

    def _ode(self, time, state, *inputs, **params):
        q, v = state.continuous_state[0], state.continuous_state[1]
        q_set = inputs[0]
        Kp, Kd, T_max, deadband = (
            params["Kp"],
            params["Kd"],
            params["T_max"],
            params["deadband"],
        )
        J, b, Fc = params["J"], params["b"], params["Fc"]
        err = q_set - q
        err_eff = _soft_deadband(err, deadband)
        tau_cmd = Kp * err_eff - Kd * v
        tau_act = _soft_saturate(tau_cmd, T_max)
        dvdt = (tau_act - b * v - Fc * jnp.tanh(v / self._v_eps)) / J
        return jnp.stack([v, dvdt])

    def parameter_spec(self) -> ParameterSpec:
        return ParameterSpec(("Kp", "Kd", "T_max", "deadband", "J", "b", "Fc"))

    def validate(self, telemetry: ActuatorStepTelemetry) -> ValidationReport:
        """Simulate the servo against the telemetry's setpoint and report residuals.

        Integrates the servo's 2-D ODE with a fixed-step NumPy RK4 between each
        pair of telemetry samples (the setpoint is held constant on each
        interval — zero-order hold matches a real recording's piecewise
        nature). Initial position is seeded from the first telemetry sample.
        A warning is appended if the RMS position residual exceeds the
        ``joint_position`` policy bound (RK4 / float64 scaled by × 100 to
        allow for the modeling gap between a clean Servo and a real recording).
        """
        from ..testing import tolerances as tol

        t_arr = np.asarray(telemetry.time, dtype=np.float64)
        q_set_arr = np.asarray(telemetry.setpoint, dtype=np.float64)
        q_meas = np.asarray(telemetry.position, dtype=np.float64)

        Kp, Kd = float(self.Kp), float(self.Kd)
        T_max, deadband = float(self.T_max), float(self.deadband)
        J, b, Fc = float(self.J), float(self.b), float(self.Fc)
        v_eps = self._v_eps

        def rhs(q, v, q_set):
            err = q_set - q
            err_eff = err - deadband * np.tanh(err / deadband)
            tau_cmd = Kp * err_eff - Kd * v
            tau_act = T_max * np.tanh(tau_cmd / T_max)
            return v, (tau_act - b * v - Fc * np.tanh(v / v_eps)) / J

        q_sim = np.empty_like(q_meas)
        q_sim[0] = q_meas[0]
        q, v = q_meas[0], 0.0
        for i in range(1, t_arr.size):
            dt = float(t_arr[i] - t_arr[i - 1])
            qs = q_set_arr[i - 1]  # ZOH: setpoint applied on the leading edge
            k1q, k1v = rhs(q, v, qs)
            k2q, k2v = rhs(q + 0.5 * dt * k1q, v + 0.5 * dt * k1v, qs)
            k3q, k3v = rhs(q + 0.5 * dt * k2q, v + 0.5 * dt * k2v, qs)
            k4q, k4v = rhs(q + dt * k3q, v + dt * k3v, qs)
            q = q + (dt / 6.0) * (k1q + 2 * k2q + 2 * k3q + k4q)
            v = v + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
            q_sim[i] = q

        residual = q_sim - q_meas
        rms = float(np.sqrt(np.mean(residual * residual)))
        peak = float(np.max(np.abs(residual)))
        report = ValidationReport()
        bound = tol.atol("joint_position", "rk4", "float64") * 100.0
        if rms > bound:
            report.warnings.append(
                f"position residual RMS={rms:.3g} rad (peak={peak:.3g} rad) "
                f"exceeds policy bound {bound:.3g} rad — servo parameters may "
                "not match this trace."
            )
        return report


__all__ = ["Servo"]
