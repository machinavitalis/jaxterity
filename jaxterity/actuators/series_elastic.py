# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Series-elastic actuator.

``SeriesElasticActuator`` (SEA) models the kind of compliant drive most cable-
driven humanoids and many quadrupeds use: a torque source (motor) feeds a
spring (the compliant element) that drives the load. The compliance decouples
load-side bandwidth from motor-side reflected inertia and gives a high-fidelity
torque-sensing signal (the spring deflection times stiffness *is* the torque).

Model — state is ``[q_motor, v_motor, q_load, v_load]`` (rad, rad/s, rad,
rad/s); input is the motor-side torque ``tau_input`` (N·m); output ports are
``q_motor``, ``v_motor``, ``q_load``, ``v_load``::

    τ_spring     = k · (q_motor − q_load) + c · (v_motor − v_load)
    q̇_motor     = v_motor
    v̇_motor     = (τ_input − τ_spring − b_m · v_motor) / J_motor
    q̇_load      = v_load
    v̇_load      = (τ_spring − b_l · v_load) / J_load

Six identifiable parameters: ``k`` (N·m/rad spring stiffness), ``c``
(N·m·s/rad spring damping), ``J_motor`` (kg·m² motor-side inertia), ``J_load``
(kg·m² load-side inertia), ``b_m`` (N·m·s/rad motor-side viscous friction),
``b_l`` (N·m·s/rad load-side viscous friction). All four denominator
parameters (``k``, ``J_motor``, ``J_load``) must be > 0.

Identification is from synchronized (τ_input, q_motor, q_load) recordings (see
:class:`~jaxterity.data.schema.SeriesElasticTelemetry`). The fitter is the
``sysid.actuator`` recipe (joint nonlinear least squares on the linear
6-parameter model). ``validate(telemetry)`` ships here as a forward-simulation
NumPy RK4 comparison.

Scope — what this model does *not* include (call-outs, not omissions):

* **Tendon-driven actuators.** Cables only *pull*, never push, which makes
  the spring law asymmetric (``τ_spring = max(0, k · ΔΘ)``) and introduces a
  going-slack mode that the symmetric SEA model misses. This deserves its own
  class (``CableActuator``) and is deferred — a single symmetric model would
  mis-fit the cable case silently. Add the asymmetric model when the K-Scale
  humanoid actually needs it.
* **Nonlinear spring (cubic / hardening).** Real elastomeric SEAs have
  measurable nonlinearity at large deflections. The linear ``k · ΔΘ`` model
  is good to first order and identifiable from any modest deflection range.
* **Composition with a motor block.** ``SeriesElasticActuator``
  takes a torque input, not a current or voltage — the motor blocks
  (DC motor, servo, BLDC) bundle their own load, so wiring them through an SEA
  needs the motors refactored to expose a τ-out port.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..data.schema import SeriesElasticTelemetry, ValidationReport
from ..parameter import ParameterSpec
from ..sysid.protocol import SysidProtocol
from ._base import Actuator

_SYSID_PROTOCOL = SysidProtocol(
    name="sea-step",
    excitation="step",
    required_columns=("tau_input", "q_motor", "q_load"),
    duration_s=2.0,
    sample_rate_hz=500.0,
    setup_notes=(
        "Apply a sequence of step changes in motor torque with the load free to "
        "respond (no external load beyond J_load + b_l). Record motor torque, "
        "motor-side position, and load-side position. The spring deflection "
        "(q_motor − q_load) is the lever the recipe regresses against to "
        "identify k, c, J_motor, J_load, b_m, b_l. 500 Hz captures the load's "
        "natural frequency for typical humanoid SEAs (sqrt(k/J_load) ≈ 20-50 Hz)."
    ),
)


class SeriesElasticActuator(Actuator):
    """Two-mass series-elastic drive: motor → spring → load.

    See the module docstring for the model, the parameter set, and the scope
    boundary (the tendon-driven asymmetric variant is *not* this class).

    Args
    ----
    k : float or jax.Array
        Spring stiffness, N·m/rad. Must be > 0.
    c : float or jax.Array
        Spring damping, N·m·s/rad.
    J_motor, J_load : float or jax.Array
        Motor-side and load-side inertias, kg·m². Both must be > 0.
    b_m, b_l : float or jax.Array
        Motor-side and load-side viscous friction, N·m·s/rad.
    q0_motor, v0_motor, q0_load, v0_load : float
        Initial state.
    name : str
        Block name.
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
        name: str = "series_elastic",
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
    def uncalibrated(cls, *, name: str = "series_elastic") -> "SeriesElasticActuator":
        """Return an instance with identifiability-neutral defaults.

        k=100 N·m/rad, c=0.1 N·m·s/rad, J_motor=1 g·m², J_load=10 g·m²,
        no friction. All denominator params (k, J_motor, J_load) > 0.
        """
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
        tau_spring = k * (q_m - q_l) + c * (v_m - v_l)
        dvm = (tau_in - tau_spring - b_m * v_m) / J_m
        dvl = (tau_spring - b_l * v_l) / J_l
        return jnp.stack([v_m, dvm, v_l, dvl])

    def parameter_spec(self) -> ParameterSpec:
        return ParameterSpec(("k", "c", "J_motor", "J_load", "b_m", "b_l"))

    def validate(self, telemetry: SeriesElasticTelemetry) -> ValidationReport:
        """Forward-sim under the telemetry's torque trace; compare q_motor and q_load.

        Integrates the 4-D ODE with a NumPy RK4 (zero-order hold on tau_input).
        Initial state is seeded from the first telemetry sample. Reports
        per-channel position residuals and warns if either RMS exceeds the
        joint_position policy bound (× 100 for the modeling gap between a
        clean SEA and a real recording).
        """
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
        # Substepping for SEA stiffness: spring's natural frequency sqrt(k/J_load);
        # use dt_internal ≤ 0.1 · period to keep RK4 well inside stability.
        omega = np.sqrt(max(k / J_l, 1e-12))
        dt_internal_max = 0.1 * (2 * np.pi / omega)

        def rhs(qm, vm, ql, vl, tau_in):
            tau_spring = k * (qm - ql) + c * (vm - vl)
            return (
                vm,
                (tau_in - tau_spring - b_m * vm) / J_m,
                vl,
                (tau_spring - b_l * vl) / J_l,
            )

        qm_sim = np.empty_like(qm_meas)
        ql_sim = np.empty_like(ql_meas)
        qm_sim[0], ql_sim[0] = qm_meas[0], ql_meas[0]
        qm, vm, ql, vl = float(qm_meas[0]), 0.0, float(ql_meas[0]), 0.0
        for kk in range(1, t_arr.size):
            dt_outer = float(t_arr[kk] - t_arr[kk - 1])
            n_sub = max(1, int(np.ceil(dt_outer / dt_internal_max)))
            dt = dt_outer / n_sub
            tau_in = float(tau_in_arr[kk - 1])
            for _ in range(n_sub):
                k1qm, k1vm, k1ql, k1vl = rhs(qm, vm, ql, vl, tau_in)
                k2qm, k2vm, k2ql, k2vl = rhs(
                    qm + 0.5 * dt * k1qm,
                    vm + 0.5 * dt * k1vm,
                    ql + 0.5 * dt * k1ql,
                    vl + 0.5 * dt * k1vl,
                    tau_in,
                )
                k3qm, k3vm, k3ql, k3vl = rhs(
                    qm + 0.5 * dt * k2qm,
                    vm + 0.5 * dt * k2vm,
                    ql + 0.5 * dt * k2ql,
                    vl + 0.5 * dt * k2vl,
                    tau_in,
                )
                k4qm, k4vm, k4ql, k4vl = rhs(
                    qm + dt * k3qm,
                    vm + dt * k3vm,
                    ql + dt * k3ql,
                    vl + dt * k3vl,
                    tau_in,
                )
                qm = qm + (dt / 6.0) * (k1qm + 2 * k2qm + 2 * k3qm + k4qm)
                vm = vm + (dt / 6.0) * (k1vm + 2 * k2vm + 2 * k3vm + k4vm)
                ql = ql + (dt / 6.0) * (k1ql + 2 * k2ql + 2 * k3ql + k4ql)
                vl = vl + (dt / 6.0) * (k1vl + 2 * k2vl + 2 * k3vl + k4vl)
            qm_sim[kk], ql_sim[kk] = qm, ql

        report = ValidationReport()
        if not (np.all(np.isfinite(qm_sim)) and np.all(np.isfinite(ql_sim))):
            report.warnings.append(
                "simulation produced non-finite values — SEA parameters likely "
                "drive the ODE unstable (k/J_load ratio incompatible with the "
                "telemetry sample rate)."
            )
            return report
        qm_rms = float(np.sqrt(np.mean((qm_sim - qm_meas) ** 2)))
        ql_rms = float(np.sqrt(np.mean((ql_sim - ql_meas) ** 2)))
        bound = tol.atol("joint_position", "rk4", "float64") * 1e3
        if qm_rms > bound:
            report.warnings.append(
                f"motor-position residual RMS={qm_rms:.3g} rad exceeds bound "
                f"{bound:.3g} rad — motor-side params may not match."
            )
        if ql_rms > bound:
            report.warnings.append(
                f"load-position residual RMS={ql_rms:.3g} rad exceeds bound "
                f"{bound:.3g} rad — spring or load-side params may not match."
            )
        return report


__all__ = ["SeriesElasticActuator"]
