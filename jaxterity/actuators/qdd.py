# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Quasi-direct-drive actuator: the humanoid/quadruped case.

A quasi-direct drive is a large-gap BLDC behind a *low* reduction (typically
6:1 to 10:1). That single choice — low ratio instead of the 100:1 a classic
servo uses — is what makes modern legged robots possible, and it changes which
model errors matter:

* **Reflected inertia scales as N².** At 6:1 the rotor contributes 36× its own
  inertia to the joint; at 100:1 it would contribute 10 000×. This is *the*
  reason a QDD leg can be back-driven by an impact instead of shattering, and
  it means the actuator's own inertia is a first-order term in the leg
  dynamics, not a rounding error. :attr:`QuasiDirectDrive.reflected_inertia`
  exposes it.
* **The torque limit is a function of speed.** With little gearing there is
  little torque multiplication, so a QDD runs near its electrical limit, and
  the limit itself collapses as back-EMF eats the bus:
  ``τ_max(ω) = N·η·K_t·min(i_max, (V_bus − K_e·N·|ω|)/R)``. A model with a
  constant torque limit will happily plan a jump the hardware cannot make.
  :func:`qdd_torque_envelope` is that curve.
* **Friction is small enough to see through.** The low ratio makes the joint
  transparent, which is why a QDD leg can estimate contact from current — and
  why an over-fitted friction term destroys that estimate.

Model — stateful drive block, state ``[q, v]`` (rad, rad/s)::

    ω_motor  = N·v
    i_avail  = clip((V_bus − K_e·|ω_motor|)/R, 0, i_max)
    τ_max    = N·η·K_t·i_avail                       (speed-dependent envelope)
    τ_applied= τ_max·tanh(τ_ref/τ_max)               (smooth saturation)
    q̇        = v
    v̇        = (τ_applied − τ_ext − b·v − F_c·tanh(v/v_eps)) / J_total
    J_total  = J_load + N²·J_rotor

Input ports are ``tau_ref`` (N·m, commanded joint torque) and ``tau_ext``
(N·m, the external/load torque on the joint); outputs are ``q``, ``v``, and
``tau`` (the torque actually delivered after saturation — the channel a
current-based contact estimator consumes).

The saturation is a ``tanh``, named and exposed as such: it is smooth (so the
block differentiates), and it rolls off *before* the hard limit rather than at
it, which is conservative in the right direction for a planner. Field
weakening, thermal derating (see
:mod:`jaxterity.physics.thermal_network`), and the current loop's own dynamics
(see :class:`~jaxterity.actuators.bldc.BLDC`, which models it explicitly) are
out of scope here — this block is about the envelope and the reflected
inertia.

Units
-----
N·m, rad, rad/s, A, V, Ω, kg·m².
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..data.schema import TorqueStepTelemetry, ValidationReport
from ..parameter import ParameterSpec
from ..sysid.protocol import SysidProtocol
from ._base import Actuator

#: Smoothing width for the Coulomb-friction sign (rad/s) — same idiom as the
#: DC motor, servo and BLDC blocks.
_V_EPS_DEFAULT = 1e-3
#: Floor on the available torque so the smooth saturation never divides by
#: zero at the no-load speed (N·m).
_TAU_FLOOR = 1e-9

_SYSID_PROTOCOL = SysidProtocol(
    name="qdd-envelope-and-inertia",
    excitation="step",
    required_columns=("tau_command", "position"),
    duration_s=20.0,
    sample_rate_hz=1000.0,
    setup_notes=(
        "Two segments, because the envelope and the inertia are identified by "
        "different experiments. (1) FREE SPIN, no load: step the torque "
        "command and record the acceleration — the slope identifies J_total, "
        "from which N^2*J_rotor follows once the fixture inertia is known. "
        "(2) LOCKED-ROTOR then SPEED SWEEP: hold the output against a torque "
        "sensor and step the current to read N*eta*K_t directly, then run "
        "constant-command sweeps at increasing speed and record where the "
        "delivered torque starts to fall — that knee identifies K_e and R "
        "against the measured bus voltage. Log the bus voltage: a sagging "
        "pack moves the whole envelope down and looks exactly like a lower "
        "K_t (see jaxterity.physics.power_tree)."
    ),
)


def qdd_torque_envelope(omega, *, N, K_t, K_e, R, V_bus, i_max, eta=1.0):
    """Peak joint torque available at a given joint speed.

    The electrical envelope: back-EMF ``K_e·N·|ω|`` eats the bus voltage, what
    is left drives current through the winding resistance, and the current
    limit caps it. At zero speed this is the stall torque; it falls linearly
    with speed and reaches zero at the no-load speed. Always non-negative.

    Args
    ----
    omega : float or jax.Array
        Joint speed, rad/s.
    N : float or jax.Array
        Gear ratio (motor turns per joint turn), dimensionless.
    K_t : float or jax.Array
        Motor torque constant, N·m/A.
    K_e : float or jax.Array
        Back-EMF constant, V/(rad/s) at the *motor*.
    R : float or jax.Array
        Winding resistance, Ω.
    V_bus : float or jax.Array
        Available bus voltage, V.
    i_max : float or jax.Array
        Drive current limit, A.
    eta : float or jax.Array
        Reduction-stage efficiency in (0, 1].

    Returns
    -------
    jax.Array
        Peak joint torque, N·m (>= 0).

    Units
    -----
    rad/s, N·m/A, V/(rad/s), Ω, V, A → N·m.

    Examples
    --------
    >>> stall = qdd_torque_envelope(
    ...     0.0, N=6.0, K_t=0.1, K_e=0.1, R=0.5, V_bus=24.0, i_max=30.0
    ... )
    >>> float(stall)  # current-limited at standstill: 6 * 0.1 * 30
    18.0
    """
    headroom = V_bus - K_e * N * jnp.abs(omega)
    available = jnp.clip(headroom / R, 0.0, i_max)
    return N * eta * K_t * available


class QuasiDirectDrive(Actuator):
    """Low-ratio BLDC drive with a speed-dependent torque envelope.

    See the module docstring for the model and why the envelope and the
    reflected inertia are the two things this block exists to get right.
    Input ports are ``tau_ref`` and ``tau_ext`` (N·m); output ports are ``q``
    (rad), ``v`` (rad/s) and ``tau`` (N·m, delivered).

    Args
    ----
    N : float or jax.Array
        Gear ratio. Must be > 0; QDD practice is 4–10.
    K_t : float or jax.Array
        Motor torque constant, N·m/A. Must be > 0.
    K_e : float or jax.Array
        Back-EMF constant, V/(rad/s). Equals ``K_t`` in SI for an ideal
        machine; kept separate so a fit can flag a discrepancy instead of
        assuming one away.
    R : float or jax.Array
        Winding resistance, Ω. Must be > 0.
    V_bus : float or jax.Array
        Bus voltage, V. Not a fitted parameter by default — wire the measured
        rail (see :mod:`jaxterity.physics.power_tree`) if it sags.
    i_max : float or jax.Array
        Drive current limit, A.
    eta : float or jax.Array
        Reduction efficiency in (0, 1].
    J_rotor : float or jax.Array
        Rotor inertia, kg·m² (motor side — it is reflected by ``N²``).
    J_load : float or jax.Array
        Joint-side inertia, kg·m².
    b : float or jax.Array
        Viscous friction at the joint, N·m·s/rad.
    F_c : float or jax.Array
        Coulomb friction at the joint, N·m.
    v_eps : float
        Coulomb-sign smoothing width, rad/s.
    q0, v0 : float
        Initial joint position (rad) and velocity (rad/s).
    name : str
        Block name.

    Examples
    --------
    >>> qdd = QuasiDirectDrive(N=6.0, J_rotor=5e-5, J_load=1e-3)
    >>> float(qdd.reflected_inertia)  # N^2 * J_rotor
    0.0018
    """

    sysid_protocol = _SYSID_PROTOCOL

    def __init__(
        self,
        *,
        N=6.0,
        K_t=0.1,
        K_e=0.1,
        R=0.5,
        V_bus=24.0,
        i_max=30.0,
        eta=0.95,
        J_rotor=5.0e-5,
        J_load=1.0e-3,
        b=0.0,
        F_c=0.0,
        v_eps: float = _V_EPS_DEFAULT,
        q0: float = 0.0,
        v0: float = 0.0,
        name: str = "qdd",
    ):
        super().__init__(name=name)
        self._v_eps = float(v_eps)
        for n, x in (
            ("N", N),
            ("K_t", K_t),
            ("K_e", K_e),
            ("R", R),
            ("V_bus", V_bus),
            ("i_max", i_max),
            ("eta", eta),
            ("J_rotor", J_rotor),
            ("J_load", J_load),
            ("b", b),
            ("F_c", F_c),
        ):
            self.declare_dynamic_parameter(n, jnp.asarray(x, dtype=float))
        self.declare_input_port(name="tau_ref")
        self.declare_input_port(name="tau_ext")
        self.declare_continuous_state(
            default_value=jnp.asarray([q0, v0], dtype=float), ode=self._ode
        )
        self.declare_output_port(self._q_out, name="q", requires_inputs=False)
        self.declare_output_port(self._v_out, name="v", requires_inputs=False)
        self.declare_output_port(self._tau_out, name="tau")

    @property
    def N(self):
        return self.dynamic_parameters["N"].get()

    @property
    def K_t(self):
        return self.dynamic_parameters["K_t"].get()

    @property
    def K_e(self):
        return self.dynamic_parameters["K_e"].get()

    @property
    def R(self):
        return self.dynamic_parameters["R"].get()

    @property
    def V_bus(self):
        return self.dynamic_parameters["V_bus"].get()

    @property
    def i_max(self):
        return self.dynamic_parameters["i_max"].get()

    @property
    def eta(self):
        return self.dynamic_parameters["eta"].get()

    @property
    def J_rotor(self):
        return self.dynamic_parameters["J_rotor"].get()

    @property
    def J_load(self):
        return self.dynamic_parameters["J_load"].get()

    @property
    def b(self):
        return self.dynamic_parameters["b"].get()

    @property
    def F_c(self):
        return self.dynamic_parameters["F_c"].get()

    @property
    def reflected_inertia(self):
        """Rotor inertia seen at the joint, ``N²·J_rotor`` (kg·m²).

        The number that separates a QDD from a geared servo: at 6:1 it is 36×
        the rotor's own inertia, at 100:1 it would be 10 000×.
        """
        return self.N**2 * self.J_rotor

    @property
    def total_inertia(self):
        """Joint inertia including the reflected rotor, kg·m²."""
        return self.J_load + self.reflected_inertia

    @property
    def no_load_speed(self):
        """Joint speed at which the envelope reaches zero torque, rad/s."""
        return self.V_bus / (self.K_e * self.N)

    @classmethod
    def uncalibrated(cls, *, name: str = "qdd") -> "QuasiDirectDrive":
        """Return an instance with identifiability-neutral defaults.

        A 6:1 stage with unit efficiency and zero friction, so a fit can only
        add measured loss. The electrical parameters keep their
        order-of-magnitude defaults (a zero ``R`` or ``K_t`` would make the
        envelope degenerate rather than neutral).
        """
        return cls(eta=1.0, b=0.0, F_c=0.0, name=name)

    def _envelope(self, v, params):
        return jnp.maximum(
            qdd_torque_envelope(
                v,
                N=params["N"],
                K_t=params["K_t"],
                K_e=params["K_e"],
                R=params["R"],
                V_bus=params["V_bus"],
                i_max=params["i_max"],
                eta=params["eta"],
            ),
            _TAU_FLOOR,
        )

    def _delivered(self, v, tau_ref, params):
        envelope = self._envelope(v, params)
        return envelope * jnp.tanh(tau_ref / envelope)

    def _q_out(self, time, state, *inputs, **params):
        return state.continuous_state[0]

    def _v_out(self, time, state, *inputs, **params):
        return state.continuous_state[1]

    def _tau_out(self, time, state, *inputs, **params):
        return self._delivered(state.continuous_state[1], inputs[0], params)

    def _ode(self, time, state, *inputs, **params):
        q, v = state.continuous_state
        tau_ref, tau_ext = inputs[0], inputs[1]
        tau = self._delivered(v, tau_ref, params)
        friction = params["b"] * v + params["F_c"] * jnp.tanh(v / self._v_eps)
        inertia = params["J_load"] + params["N"] ** 2 * params["J_rotor"]
        return jnp.array([v, (tau - tau_ext - friction) / inertia])

    def parameter_spec(self) -> ParameterSpec:
        return ParameterSpec(
            ("N", "K_t", "K_e", "R", "i_max", "eta", "J_rotor", "J_load", "b", "F_c")
        )

    def validate(self, telemetry: TorqueStepTelemetry) -> ValidationReport:
        """Forward-simulate a torque-step response against a recorded trace.

        Integrates the block's own ODE with a fixed-step NumPy RK4 (zero-order
        hold on the command, matching a real recording), seeded from the
        trace's first position sample and a zero initial velocity, with no
        external load.

        Args
        ----
        telemetry : TorqueStepTelemetry
            Torque-step response trace.

        Returns
        -------
        ValidationReport
            ``ok`` iff the RMS position residual is within the policy bound.

        Units
        -----
        N·m, rad.
        """
        from ..testing import tolerances as tol

        t = np.asarray(telemetry.time, dtype=np.float64)
        command = np.asarray(telemetry.tau_command, dtype=np.float64)
        measured = np.asarray(telemetry.position, dtype=np.float64)
        params = {
            name: float(getattr(self, name))
            for name in (
                "N",
                "K_t",
                "K_e",
                "R",
                "V_bus",
                "i_max",
                "eta",
                "J_rotor",
                "J_load",
                "b",
                "F_c",
            )
        }
        inertia = params["J_load"] + params["N"] ** 2 * params["J_rotor"]

        def rhs(q, v, u):
            envelope = max(
                float(
                    qdd_torque_envelope(
                        v,
                        N=params["N"],
                        K_t=params["K_t"],
                        K_e=params["K_e"],
                        R=params["R"],
                        V_bus=params["V_bus"],
                        i_max=params["i_max"],
                        eta=params["eta"],
                    )
                ),
                _TAU_FLOOR,
            )
            tau = envelope * np.tanh(u / envelope)
            friction = params["b"] * v + params["F_c"] * np.tanh(v / self._v_eps)
            return v, (tau - friction) / inertia

        simulated = np.empty_like(measured)
        simulated[0] = measured[0]
        q, v = float(measured[0]), 0.0
        for i in range(1, t.size):
            dt = float(t[i] - t[i - 1])
            u = float(command[i - 1])
            k1 = np.array(rhs(q, v, u))
            k2 = np.array(rhs(q + 0.5 * dt * k1[0], v + 0.5 * dt * k1[1], u))
            k3 = np.array(rhs(q + 0.5 * dt * k2[0], v + 0.5 * dt * k2[1], u))
            k4 = np.array(rhs(q + dt * k3[0], v + dt * k3[1], u))
            q = q + (dt / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            v = v + (dt / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            simulated[i] = q

        residual = simulated - measured
        rms = float(np.sqrt(np.mean(residual * residual)))
        report = ValidationReport()
        bound = tol.atol("joint_position", "rk4", "float64") * 1e6
        if rms > bound:
            report.warnings.append(
                f"QDD position residual RMS={rms:.3g} rad exceeds bound "
                f"{bound:.3g} rad — the inertia, the envelope, or the friction "
                "may not match this trace."
            )
        return report


__all__ = ["QuasiDirectDrive", "qdd_torque_envelope"]
