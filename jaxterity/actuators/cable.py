# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Cable-driven actuator with asymmetric spring law.

``CableActuator`` is the asymmetric twin of
:class:`~jaxterity.actuators.series_elastic.SeriesElasticActuator`: cables only
*pull* — they cannot push. The spring law is

    τ_spring = softplus(k · ΔΘ)         where ΔΘ = q_motor − q_load

(``softplus(x) = log(1 + exp(x))``: a smooth approximation of ``max(0, x)``
that preserves the going-slack mode while keeping the forward sim
differentiable for sysid).

Use when modelling a K-Scale humanoid's tendon drive or any other cable-driven
chain; use the symmetric :class:`SeriesElasticActuator` for two-sided springs.

Six identifiable parameters (same as SEA): ``k``, ``c``, ``J_motor``,
``J_load``, ``b_m``, ``b_l``. The damping ``c`` acts on the spring-velocity
difference and is symmetric (a cable's longitudinal damping doesn't care about
pull/push direction); only the elastic term is asymmetric.

State, input, output ports, telemetry schema, and sysid_protocol are identical
to :class:`SeriesElasticActuator` — reuse
:class:`~jaxterity.data.schema.SeriesElasticTelemetry`.

**System identification is not yet wired for ``CableActuator``.**
``sysid.actuator`` dispatches by exact device type and has no
``CableActuator`` branch, so it raises a clear "no fitter for device type
CableActuator" error rather than silently mis-fitting. Reusing the SEA fitter
would be *incorrect*, not merely missing: the SEA fitter's rollout RHS uses
the **symmetric** spring law ``τ = k·ΔΘ + c·Δω``, whereas the cable's law is
the one-sided ``softplus(k·ΔΘ)`` — fitting cable telemetry through the
symmetric model would converge to wrong parameters whenever the cable goes
slack. The follow-up is a dedicated cable fitter (the same JAX-traceable RK4
rollout, but with this module's ``softplus`` RHS); ``softplus`` is smooth
everywhere, so forward-mode autodiff handles it once that fitter exists.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..data.schema import SeriesElasticTelemetry, ValidationReport
from ..parameter import ParameterSpec
from ..sysid.protocol import SysidProtocol
from ._base import Actuator

_SYSID_PROTOCOL = SysidProtocol(
    name="cable-step",
    excitation="step",
    required_columns=("tau_input", "q_motor", "q_load"),
    duration_s=2.0,
    sample_rate_hz=500.0,
    setup_notes=(
        "Same as the SEA step-response protocol but with a one-sided cable: "
        "drive the motor in both directions; the cable transmits force when "
        "deflection is positive (motor leads load) and goes slack when "
        "negative. The recipe regresses against ``softplus(k·ΔΘ)`` instead of "
        "the symmetric ``k·ΔΘ``."
    ),
)


class CableActuator(Actuator):
    """Cable-driven (pull-only) series-elastic drive.

    See the module docstring for the asymmetric spring law and the
    boundary with :class:`SeriesElasticActuator`.

    Args (and units) match :class:`SeriesElasticActuator`. ``k`` is the
    cable's longitudinal stiffness when taut.
    """

    sysid_protocol = _SYSID_PROTOCOL

    def __init__(
        self,
        *,
        k=100.0,
        c=0.1,
        J_motor=1e-3,
        J_load=1e-2,
        b_m=0.0,
        b_l=0.0,
        q0_motor: float = 0.0,
        v0_motor: float = 0.0,
        q0_load: float = 0.0,
        v0_load: float = 0.0,
        name: str = "cable",
    ):
        super().__init__(name=name)
        for n, x in (
            ("k", k),
            ("c", c),
            ("J_motor", J_motor),
            ("J_load", J_load),
            ("b_m", b_m),
            ("b_l", b_l),
        ):
            self.declare_dynamic_parameter(n, jnp.asarray(x, dtype=float))
        self.declare_input_port(name="tau_input")
        self.declare_continuous_state(
            default_value=jnp.asarray(
                [q0_motor, v0_motor, q0_load, v0_load], dtype=float
            ),
            ode=self._ode,
        )
        self.declare_output_port(
            self._q_motor_out, name="q_motor", requires_inputs=False
        )
        self.declare_output_port(
            self._v_motor_out, name="v_motor", requires_inputs=False
        )
        self.declare_output_port(self._q_load_out, name="q_load", requires_inputs=False)
        self.declare_output_port(self._v_load_out, name="v_load", requires_inputs=False)

    @property
    def k(self):
        return self.dynamic_parameters["k"].get()

    @property
    def c(self):
        return self.dynamic_parameters["c"].get()

    @property
    def J_motor(self):
        return self.dynamic_parameters["J_motor"].get()

    @property
    def J_load(self):
        return self.dynamic_parameters["J_load"].get()

    @property
    def b_m(self):
        return self.dynamic_parameters["b_m"].get()

    @property
    def b_l(self):
        return self.dynamic_parameters["b_l"].get()

    @classmethod
    def uncalibrated(cls, *, name: str = "cable") -> "CableActuator":
        """Return an instance with identifiability-neutral defaults."""
        return cls(name=name)

    def _q_motor_out(self, time, state, *inputs, **params):
        return state.continuous_state[0]

    def _v_motor_out(self, time, state, *inputs, **params):
        return state.continuous_state[1]

    def _q_load_out(self, time, state, *inputs, **params):
        return state.continuous_state[2]

    def _v_load_out(self, time, state, *inputs, **params):
        return state.continuous_state[3]

    def _ode(self, time, state, *inputs, **params):
        q_m = state.continuous_state[0]
        v_m = state.continuous_state[1]
        q_l = state.continuous_state[2]
        v_l = state.continuous_state[3]
        tau_in = inputs[0]
        k, c, J_m, J_l, b_m, b_l = (
            params["k"],
            params["c"],
            params["J_motor"],
            params["J_load"],
            params["b_m"],
            params["b_l"],
        )
        deflection = q_m - q_l
        # Cable: softplus(k·ΔΘ) — zero in slack (ΔΘ ≤ 0), ~k·ΔΘ when taut.
        # Damping is gated by the same taut-indicator: a slack cable has no
        # longitudinal damping (no medium to transmit it). sigmoid is the
        # smooth approximation of the Heaviside step on deflection sign.
        tension = jax.nn.softplus(k * deflection)
        gate = jax.nn.sigmoid(k * deflection)
        tau_spring = tension + gate * c * (v_m - v_l)
        dvm = (tau_in - tau_spring - b_m * v_m) / J_m
        dvl = (tau_spring - b_l * v_l) / J_l
        return jnp.stack([v_m, dvm, v_l, dvl])

    def parameter_spec(self) -> ParameterSpec:
        return ParameterSpec(("k", "c", "J_motor", "J_load", "b_m", "b_l"))

    def validate(self, telemetry: SeriesElasticTelemetry) -> ValidationReport:
        """NumPy RK4 forward-sim residuals (mirrors SEA.validate, asymmetric law)."""
        from ..testing import tolerances as tol

        t_arr = np.asarray(telemetry.time, dtype=np.float64)
        tau_in_arr = np.asarray(telemetry.tau_input, dtype=np.float64)
        qm_meas = np.asarray(telemetry.q_motor, dtype=np.float64)
        ql_meas = np.asarray(telemetry.q_load, dtype=np.float64)
        k, c, J_m, J_l, b_m, b_l = (
            float(self.k),
            float(self.c),
            float(self.J_motor),
            float(self.J_load),
            float(self.b_m),
            float(self.b_l),
        )
        omega = np.sqrt(max(k / J_l, 1e-12))
        dt_internal_max = 0.1 * (2 * np.pi / omega)

        def softplus_np(x):
            return np.logaddexp(0.0, x)

        def sigmoid_np(x):
            return 1.0 / (1.0 + np.exp(-x))

        def rhs(qm, vm, ql, vl, ti):
            deflection = qm - ql
            tension = softplus_np(k * deflection)
            gate = sigmoid_np(k * deflection)
            ts = tension + gate * c * (vm - vl)
            return vm, (ti - ts - b_m * vm) / J_m, vl, (ts - b_l * vl) / J_l

        qm_sim = np.empty_like(qm_meas)
        ql_sim = np.empty_like(ql_meas)
        qm_sim[0], ql_sim[0] = qm_meas[0], ql_meas[0]
        qm, vm, ql, vl = float(qm_meas[0]), 0.0, float(ql_meas[0]), 0.0
        for kk in range(1, t_arr.size):
            dt_outer = float(t_arr[kk] - t_arr[kk - 1])
            n_sub = max(1, int(np.ceil(dt_outer / dt_internal_max)))
            dt = dt_outer / n_sub
            ti = float(tau_in_arr[kk - 1])
            for _ in range(n_sub):
                k1qm, k1vm, k1ql, k1vl = rhs(qm, vm, ql, vl, ti)
                k2qm, k2vm, k2ql, k2vl = rhs(
                    qm + 0.5 * dt * k1qm,
                    vm + 0.5 * dt * k1vm,
                    ql + 0.5 * dt * k1ql,
                    vl + 0.5 * dt * k1vl,
                    ti,
                )
                k3qm, k3vm, k3ql, k3vl = rhs(
                    qm + 0.5 * dt * k2qm,
                    vm + 0.5 * dt * k2vm,
                    ql + 0.5 * dt * k2ql,
                    vl + 0.5 * dt * k2vl,
                    ti,
                )
                k4qm, k4vm, k4ql, k4vl = rhs(
                    qm + dt * k3qm,
                    vm + dt * k3vm,
                    ql + dt * k3ql,
                    vl + dt * k3vl,
                    ti,
                )
                qm = qm + (dt / 6.0) * (k1qm + 2 * k2qm + 2 * k3qm + k4qm)
                vm = vm + (dt / 6.0) * (k1vm + 2 * k2vm + 2 * k3vm + k4vm)
                ql = ql + (dt / 6.0) * (k1ql + 2 * k2ql + 2 * k3ql + k4ql)
                vl = vl + (dt / 6.0) * (k1vl + 2 * k2vl + 2 * k3vl + k4vl)
            qm_sim[kk], ql_sim[kk] = qm, ql
        report = ValidationReport()
        if not (np.all(np.isfinite(qm_sim)) and np.all(np.isfinite(ql_sim))):
            report.warnings.append(
                "simulation produced non-finite values — cable parameters "
                "likely drive the ODE unstable."
            )
            return report
        qm_rms = float(np.sqrt(np.mean((qm_sim - qm_meas) ** 2)))
        ql_rms = float(np.sqrt(np.mean((ql_sim - ql_meas) ** 2)))
        bound = tol.atol("joint_position", "rk4", "float64") * 1e3
        if qm_rms > bound:
            report.warnings.append(
                f"motor-position residual RMS={qm_rms:.3g} rad exceeds bound "
                f"{bound:.3g} rad."
            )
        if ql_rms > bound:
            report.warnings.append(
                f"load-position residual RMS={ql_rms:.3g} rad exceeds bound "
                f"{bound:.3g} rad."
            )
        return report


__all__ = ["CableActuator"]
