# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Harmonic-drive / cycloidal gearbox transmission.

A transmission is a two-port device: motor side ↔ joint side. ``HarmonicDrive``
ships the simplest physically meaningful model — a kinematic gear ratio plus a
direction-dependent efficiency, with a velocity-domain smoothing that
approximates the *effect* of mechanical backlash on transmitted torque
(without modelling the angular-play state explicitly; see "scope" below).

Model — stateless two-port::

    v_joint = v_motor / N
    η_eff   = ½(η_f + η_b) + ½(η_f − η_b) · tanh(P_dir / P_eps)   (smooth switch)
    P_dir   = τ_motor · v_motor                                      (power flow)
    τ_joint = η_eff · N · τ_motor · tanh(|v_motor| / backlash_v)    (smoothed backlash)

Four identifiable parameters: ``N`` (gear ratio, dimensionless),
``eta_f`` (forward efficiency, motor → joint), ``eta_b`` (backward efficiency,
joint → motor / regenerative), ``backlash_v`` (rad/s; the velocity-domain
smoothing width that approximates the angular-play dead-zone).

Scope — what this model does *not* include:

* **True angular-play backlash.** Real backlash is a *positional* phenomenon:
  the joint angle lags the motor angle by up to ``backlash_rad`` until the
  motor "takes up the slack." That is a stateful relation between motor and
  joint angles. This model collapses this to a velocity-domain smoothing
  — torque transmission rolls off near zero motor velocity. This captures the
  dominant identifiable effect (lost torque near zero crossings) without
  carrying an angular-play state. The stateful model can be added later
  when a recipe demonstrably needs it.
* **Compliance.** Harmonic drives have measurable torsional stiffness. Adding
  a series-elastic compliance is the series-elastic actuator's job.
* **Composition with a motor.** ``HarmonicDrive`` is a separate block from the
  upstream motor — it expects motor-side torque and velocity as inputs. The
  motor family (DC motor, servo, BLDC) bundles motor + load into a
  single block; composing a motor *through* a transmission to a Robot joint
  needs the motors refactored to expose a τ-out port.

Identification is from synchronized (τ_motor, v_motor, τ_joint, v_joint)
recordings — see :class:`~jaxterity.data.schema.TransmissionTelemetry` and
:attr:`HarmonicDrive.sysid_protocol`. The fitter is closed-form (least squares
on the linearized log-power balance) when no direction reversals occur and
nonlinear otherwise; both variants belong to the ``sysid.actuator`` recipe.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..data.schema import TransmissionTelemetry, ValidationReport
from ..parameter import ParameterSpec
from ..sysid.protocol import SysidProtocol
from ._base import Actuator

#: Smoothing width for the power-direction sign (W). Below this, the
#: forward/backward efficiency switch is interpolated linearly rather than
#: switching abruptly. 1e-6 W is well below any realistic transmission's noise
#: floor; it exists purely for differentiability.
_P_EPS_DEFAULT = 1e-6


_SYSID_PROTOCOL = SysidProtocol(
    name="transmission-load-sweep",
    excitation="step",
    required_columns=("tau_motor", "v_motor", "tau_joint", "v_joint"),
    duration_s=10.0,
    sample_rate_hz=200.0,
    setup_notes=(
        "Mount the transmission with a known reaction load on the joint side. "
        "Drive the motor through a sequence of torque steps in BOTH directions "
        "(positive and negative) at a few speed settings, with at least one "
        "direction reversal so eta_f and eta_b are independently identifiable. "
        "Record motor-side and joint-side torques and velocities. The fitter "
        "infers N from the velocity ratio, eta_f / eta_b from the power "
        "ratio segregated by direction, and backlash_v from the torque roll-off "
        "near v_motor = 0."
    ),
)


class HarmonicDrive(Actuator):
    """Stateless two-port transmission with smooth direction-dependent efficiency.

    See the module docstring for the model, the parameter set, and the scope
    (in particular, what this first-cut does *not* model: true angular
    backlash, compliance, and motor composition).

    Args
    ----
    N : float or jax.Array
        Gear ratio, dimensionless. Must be > 0.
    eta_f : float or jax.Array
        Forward efficiency (motor → joint), in [0, 1].
    eta_b : float or jax.Array
        Backward efficiency (joint → motor / regenerative), in [0, 1].
    backlash_v : float or jax.Array
        Velocity-domain smoothing width approximating the backlash effect on
        torque transmission (rad/s). Must be > 0; use a small value (~1e-3
        rad/s) to effectively disable.
    p_eps : float
        Smoothing width for the power-direction sign (W).
    name : str
        Block name.
    """

    sysid_protocol = _SYSID_PROTOCOL

    def __init__(
        self,
        *,
        N=50.0,
        eta_f=0.9,
        eta_b=0.85,
        backlash_v=1e-3,
        p_eps: float = _P_EPS_DEFAULT,
        name: str = "harmonic_drive",
    ):
        super().__init__(name=name)
        self._p_eps = float(p_eps)
        for n, x in (
            ("N", N),
            ("eta_f", eta_f),
            ("eta_b", eta_b),
            ("backlash_v", backlash_v),
        ):
            self.declare_dynamic_parameter(n, jnp.asarray(x, dtype=float))
        self.declare_input_port(name="tau_motor")
        self.declare_input_port(name="v_motor")
        self.declare_output_port(self._tau_joint_out, name="tau_joint")
        self.declare_output_port(self._v_joint_out, name="v_joint")

    @property
    def N(self):
        return self.dynamic_parameters["N"].get()

    @property
    def eta_f(self):
        return self.dynamic_parameters["eta_f"].get()

    @property
    def eta_b(self):
        return self.dynamic_parameters["eta_b"].get()

    @property
    def backlash_v(self):
        return self.dynamic_parameters["backlash_v"].get()

    @classmethod
    def uncalibrated(cls, *, name: str = "harmonic_drive") -> "HarmonicDrive":
        """Return an instance with identifiability-neutral defaults.

        N=50 (a typical harmonic-drive ratio), eta_f=0.9, eta_b=0.85
        (loose-load efficiency on the order published for SHD/HDUC series),
        backlash_v=1e-3 rad/s (small enough that healthy traces never see it).
        """
        return cls(name=name)

    @staticmethod
    def _torque_map(tau_motor, v_motor, *, N, eta_f, eta_b, backlash_v, p_eps):
        """Joint-side torque from motor-side torque and velocity — pure function."""
        P_dir = tau_motor * v_motor
        eta_eff = 0.5 * (eta_f + eta_b) + 0.5 * (eta_f - eta_b) * jnp.tanh(
            P_dir / p_eps
        )
        return eta_eff * N * tau_motor * jnp.tanh(jnp.abs(v_motor) / backlash_v)

    def _tau_joint_out(self, time, state, *inputs, **params):
        return self._torque_map(
            inputs[0],
            inputs[1],
            N=params["N"],
            eta_f=params["eta_f"],
            eta_b=params["eta_b"],
            backlash_v=params["backlash_v"],
            p_eps=self._p_eps,
        )

    def _v_joint_out(self, time, state, *inputs, **params):
        return inputs[1] / params["N"]

    def parameter_spec(self) -> ParameterSpec:
        return ParameterSpec(("N", "eta_f", "eta_b", "backlash_v"))

    def validate(self, telemetry: TransmissionTelemetry) -> ValidationReport:
        """Per-sample torque + velocity residuals against the model.

        Closed-form: applies the model to each sample's (τ_motor, v_motor) and
        compares to the telemetry's (τ_joint, v_joint). RMS-residual warnings
        on either channel exceeding the policy bound.
        """
        from ..testing import tolerances as tol

        tm = np.asarray(telemetry.tau_motor, dtype=np.float64)
        vm = np.asarray(telemetry.v_motor, dtype=np.float64)
        tj_meas = np.asarray(telemetry.tau_joint, dtype=np.float64)
        vj_meas = np.asarray(telemetry.v_joint, dtype=np.float64)

        N, eta_f, eta_b, bv = (
            float(self.N),
            float(self.eta_f),
            float(self.eta_b),
            float(self.backlash_v),
        )
        p_eps = self._p_eps
        # Vectorized: NumPy versions of the pure model.
        P_dir = tm * vm
        eta_eff = 0.5 * (eta_f + eta_b) + 0.5 * (eta_f - eta_b) * np.tanh(P_dir / p_eps)
        tj_pred = eta_eff * N * tm * np.tanh(np.abs(vm) / bv)
        vj_pred = vm / N

        tj_res = tj_pred - tj_meas
        vj_res = vj_pred - vj_meas
        tj_rms = float(np.sqrt(np.mean(tj_res * tj_res)))
        vj_rms = float(np.sqrt(np.mean(vj_res * vj_res)))
        report = ValidationReport()
        tau_bound = tol.atol("actuator_torque", "rk4", "float64") * 1e4
        vel_bound = tol.atol("joint_velocity", "rk4", "float64") * 1e4
        if tj_rms > tau_bound:
            report.warnings.append(
                f"joint-torque residual RMS={tj_rms:.3g} N·m exceeds bound "
                f"{tau_bound:.3g} N·m — N / eta / backlash_v may not match."
            )
        if vj_rms > vel_bound:
            report.warnings.append(
                f"joint-velocity residual RMS={vj_rms:.3g} rad/s exceeds bound "
                f"{vel_bound:.3g} rad/s — N may not match."
            )
        return report


__all__ = ["HarmonicDrive"]
