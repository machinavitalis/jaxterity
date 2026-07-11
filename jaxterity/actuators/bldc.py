# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""BLDC motor with current control.

``BLDC`` models a brushless DC motor with a proportional current loop and the
joint dynamics it drives, baked into a single stateful Jaxonomy ``LeafSystem``.
This is the actuator family humanoids and quadrupeds use (K-Scale, Berkeley
Humanoid Lite).

Model — state is ``[q, v, i]`` (rad, rad/s, A); input is the current setpoint
``i_ref`` (A); output ports are ``q``, ``v``, ``i``::

    e_back   = Ke · v                                         (back-EMF)
    V_app    = Kp_i · (i_ref − i)                             (current-loop P)
    i̇        = (V_app − R · i − e_back) / L                   (electrical)
    τ        = Kt · i                                         (torque constant)
    q̇        = v
    v̇        = (τ − b · v − Fc · tanh(v / v_eps)) / J          (joint dynamics)

Eight identifiable parameters: ``R`` (Ω winding resistance), ``L`` (H winding
inductance), ``Kt`` (N·m/A torque constant), ``Ke`` (V/(rad/s) back-EMF
constant), ``Kp_i`` (V/A current-loop P gain), ``J`` (kg·m² effective inertia),
``b`` (N·m·s/rad viscous friction), ``Fc`` (N·m Coulomb friction). For SI
fundamental units ``Kt = Ke`` (a physical identity), but they are exposed
separately so a calibration recipe can flag deviations as a model-mismatch
warning rather than silently constraining them.

Identification is from chirp excitation on the current setpoint (see
:attr:`BLDC.sysid_protocol` and
:class:`~jaxterity.data.schema.ActuatorChirpTelemetry`). The fitter — frequency-
response identification through the current loop is not closed-form — is
deferred to the ``sysid.actuator`` recipe. This module ships the model, the
schema, the protocol, and a forward-simulation ``validate``.

Like :class:`~jaxterity.actuators.dc_motor.DCMotorWithFriction` and
:class:`~jaxterity.actuators.servo.Servo`, ``BLDC`` is a **stateful drive
block** — it owns its joint state and is *not* a torque source that drops into
a Robot's ``tau`` input. Composing through a transmission to a Robot
joint torque port is handled by the sysid recipe layer.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..data.schema import ActuatorChirpTelemetry, ValidationReport
from ..parameter import ParameterSpec
from ..sysid.protocol import SysidProtocol
from ._base import Actuator

#: Smoothing width for the Coulomb-friction sign (rad/s). Same idiom as the
#: DC motor and servo.
_V_EPS_DEFAULT = 1e-3


_SYSID_PROTOCOL = SysidProtocol(
    name="current-chirp",
    excitation="chirp",
    required_columns=("setpoint", "velocity", "current"),
    duration_s=5.0,
    sample_rate_hz=1000.0,
    setup_notes=(
        "Free-spinning or low-load actuator. Apply a linear chirp on the current "
        "setpoint sweeping from ~0.1 Hz to a few × the expected electrical "
        "bandwidth (1/(2π·L/R)) over 5 s. Record commanded setpoint, measured "
        "rotor velocity, and measured motor current at >= 1 kHz (the electrical "
        "time constant L/R is typically 0.1-10 ms; the sample rate must resolve "
        "it). The fitter infers R/L/Kt/Ke/Kp_i/J/b/Fc by frequency-domain "
        "matching — chirp identification through the current loop is not "
        "closed-form."
    ),
)


class BLDC(Actuator):
    """Brushless DC motor with proportional current loop as a stateful drive block.

    See the module docstring for the model, port shapes, and identification
    protocol.

    Args
    ----
    R : float or jax.Array
        Winding resistance, Ω. Must be > 0.
    L : float or jax.Array
        Winding inductance, H. Must be > 0.
    Kt : float or jax.Array
        Torque constant, N·m/A.
    Ke : float or jax.Array
        Back-EMF constant, V/(rad/s). Equal to Kt in SI fundamental units.
    Kp_i : float or jax.Array
        Current-loop P gain, V/A. Must be > 0.
    J : float or jax.Array
        Effective inertia at the joint, kg·m². Must be > 0.
    b : float or jax.Array
        Viscous friction coefficient, N·m·s/rad.
    Fc : float or jax.Array
        Coulomb friction torque, N·m.
    v_eps : float
        Smoothing width for the Coulomb-friction sign (rad/s).
    q0, v0, i0 : float
        Initial position (rad), velocity (rad/s), and current (A).
    name : str
        Block name.
    """

    sysid_protocol = _SYSID_PROTOCOL

    def __init__(
        self,
        *,
        R=1.0,
        L=1e-3,
        Kt=0.05,
        Ke=0.05,
        Kp_i=10.0,
        J=1e-4,
        b=0.0,
        Fc=0.0,
        v_eps: float = _V_EPS_DEFAULT,
        q0: float = 0.0,
        v0: float = 0.0,
        i0: float = 0.0,
        name: str = "bldc",
    ):
        super().__init__(name=name)
        self._v_eps = float(v_eps)
        for n, x in (
            ("R", R),
            ("L", L),
            ("Kt", Kt),
            ("Ke", Ke),
            ("Kp_i", Kp_i),
            ("J", J),
            ("b", b),
            ("Fc", Fc),
        ):
            self.declare_dynamic_parameter(n, jnp.asarray(x, dtype=float))
        self.declare_input_port(name="i_ref")
        self.declare_continuous_state(
            default_value=jnp.asarray([q0, v0, i0], dtype=float), ode=self._ode
        )
        self.declare_output_port(self._q_out, name="q", requires_inputs=False)
        self.declare_output_port(self._v_out, name="v", requires_inputs=False)
        self.declare_output_port(self._i_out, name="i", requires_inputs=False)

    @property
    def R(self):
        return self.dynamic_parameters["R"].get()

    @property
    def L(self):
        return self.dynamic_parameters["L"].get()

    @property
    def Kt(self):
        return self.dynamic_parameters["Kt"].get()

    @property
    def Ke(self):
        return self.dynamic_parameters["Ke"].get()

    @property
    def Kp_i(self):
        return self.dynamic_parameters["Kp_i"].get()

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
    def uncalibrated(cls, *, name: str = "bldc") -> "BLDC":
        """Return an instance with identifiability-neutral defaults.

        Defaults are loosely representative of a small-bore servo motor: R=1 Ω,
        L=1 mH, Kt=Ke=0.05, Kp_i=10 V/A, J=1e-4 kg·m², no friction. All
        denominator parameters (R, L, J, Kp_i) are non-zero so the ODE is
        well-defined out of the box.
        """
        return cls(name=name)

    def _q_out(self, time, state, *inputs, **params):
        return state.continuous_state[0]

    def _v_out(self, time, state, *inputs, **params):
        return state.continuous_state[1]

    def _i_out(self, time, state, *inputs, **params):
        return state.continuous_state[2]

    def _ode(self, time, state, *inputs, **params):
        v = state.continuous_state[1]
        i = state.continuous_state[2]
        i_ref = inputs[0]
        R, L, Kt, Ke, Kp_i = (
            params["R"],
            params["L"],
            params["Kt"],
            params["Ke"],
            params["Kp_i"],
        )
        J, b, Fc = params["J"], params["b"], params["Fc"]
        V_app = Kp_i * (i_ref - i)
        e_back = Ke * v
        didt = (V_app - R * i - e_back) / L
        tau = Kt * i
        dvdt = (tau - b * v - Fc * jnp.tanh(v / self._v_eps)) / J
        return jnp.stack([v, dvdt, didt])

    def parameter_spec(self) -> ParameterSpec:
        return ParameterSpec(("R", "L", "Kt", "Ke", "Kp_i", "J", "b", "Fc"))

    def validate(self, telemetry: ActuatorChirpTelemetry) -> ValidationReport:
        """Simulate the BLDC against the telemetry's setpoint and report residuals.

        Integrates the 3-D ODE with a fixed-step NumPy RK4 between each pair of
        telemetry samples (zero-order hold on ``setpoint``). Initial state is
        seeded from the first telemetry sample (position 0, velocity / current
        from the first row). Reports the RMS velocity and current residuals; a
        warning is appended if either exceeds its policy bound (joint_velocity
        for v, actuator_torque scaled by 1/Kt for i — current is the lever, so
        we report it against the same physical scale as torque).
        """
        from ..testing import tolerances as tol

        t_arr = np.asarray(telemetry.time, dtype=np.float64)
        sp = np.asarray(telemetry.setpoint, dtype=np.float64)
        v_meas = np.asarray(telemetry.velocity, dtype=np.float64)
        i_meas = np.asarray(telemetry.current, dtype=np.float64)

        R, L, Kt, Ke, Kp_i = (
            float(self.R),
            float(self.L),
            float(self.Kt),
            float(self.Ke),
            float(self.Kp_i),
        )
        J, b, Fc = float(self.J), float(self.b), float(self.Fc)
        v_eps = self._v_eps

        def rhs(q, v, i, i_ref):
            V_app = Kp_i * (i_ref - i)
            e_back = Ke * v
            didt = (V_app - R * i - e_back) / L
            tau = Kt * i
            dvdt = (tau - b * v - Fc * np.tanh(v / v_eps)) / J
            return v, dvdt, didt

        # Pick an internal substep small enough for RK4 stability on the electrical
        # subsystem (time constant L/(R+Kp_i)). RK4 is stable when dt · |λ| < ~2.785;
        # use a 10× safety margin → dt_internal ≤ 0.25 · τ_e.
        tau_e = L / max(R + Kp_i, 1e-12)
        dt_internal_max = 0.25 * tau_e

        v_sim = np.empty_like(v_meas)
        i_sim = np.empty_like(i_meas)
        v_sim[0], i_sim[0] = v_meas[0], i_meas[0]
        q, v, i = 0.0, float(v_meas[0]), float(i_meas[0])
        for k in range(1, t_arr.size):
            dt_outer = float(t_arr[k] - t_arr[k - 1])
            i_ref = float(sp[k - 1])
            n_sub = max(1, int(np.ceil(dt_outer / dt_internal_max)))
            dt = dt_outer / n_sub
            for _ in range(n_sub):
                k1q, k1v, k1i = rhs(q, v, i, i_ref)
                k2q, k2v, k2i = rhs(
                    q + 0.5 * dt * k1q, v + 0.5 * dt * k1v, i + 0.5 * dt * k1i, i_ref
                )
                k3q, k3v, k3i = rhs(
                    q + 0.5 * dt * k2q, v + 0.5 * dt * k2v, i + 0.5 * dt * k2i, i_ref
                )
                k4q, k4v, k4i = rhs(q + dt * k3q, v + dt * k3v, i + dt * k3i, i_ref)
                q = q + (dt / 6.0) * (k1q + 2 * k2q + 2 * k3q + k4q)
                v = v + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
                i = i + (dt / 6.0) * (k1i + 2 * k2i + 2 * k3i + k4i)
            v_sim[k], i_sim[k] = v, i

        report = ValidationReport()
        if not (np.all(np.isfinite(v_sim)) and np.all(np.isfinite(i_sim))):
            report.warnings.append(
                "simulation produced non-finite values — BLDC parameters likely "
                "drive the ODE unstable at the telemetry's sample rate "
                "(electrical time constant L/R smaller than ~3× dt typically "
                "diverges under explicit RK4)."
            )
            return report
        v_res = v_sim - v_meas
        i_res = i_sim - i_meas
        v_rms = float(np.sqrt(np.mean(v_res * v_res)))
        i_rms = float(np.sqrt(np.mean(i_res * i_res)))
        v_bound = tol.atol("joint_velocity", "rk4", "float64") * 1e4
        i_bound = (tol.atol("actuator_torque", "rk4", "float64") / max(Kt, 1e-6)) * 1e4
        if v_rms > v_bound:
            report.warnings.append(
                f"velocity residual RMS={v_rms:.3g} rad/s exceeds bound "
                f"{v_bound:.3g} rad/s — BLDC mechanical params may not match."
            )
        if i_rms > i_bound:
            report.warnings.append(
                f"current residual RMS={i_rms:.3g} A exceeds bound "
                f"{i_bound:.3g} A — BLDC electrical params (R/L/Kp_i) may not match."
            )
        return report


__all__ = ["BLDC"]
