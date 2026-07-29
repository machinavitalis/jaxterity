# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Belt / cable-pulley drive: compliance first, ratio second.

The gear family (:class:`~jaxterity.actuators.gearbox.Gearbox`,
:class:`~jaxterity.actuators.harmonic_drive.HarmonicDrive`) treats a
transmission as stiff: torque in, torque out, ratio fixed. A belt is the
opposite. Its defining property on a real arm is that it *stretches* — the
pulley ratio is exact only in steady state, and the first structural resonance
of many belt-driven arms (SO-100 wrist, gantry axes, most 3D-printer-derived
kinematics) is the belt, not the links. A model without belt compliance
predicts a bandwidth the hardware does not have.

Model — a spring-damper in the belt span, stateless in the block::

    stretch   = r_motor·θ_motor − r_load·θ_load          (m)
    stretcḣ   = r_motor·θ̇_motor − r_load·θ̇_load          (m/s)
    F_ideal   = k_belt·stretch + d_belt·stretcḣ          (N)
    F         = f_slip · tanh(F_ideal / f_slip)          (smooth slip limit)
    τ_load    = r_load  · F
    τ_reaction= r_motor · F                              (what the motor feels)

Two named approximations, both exposed:

* **Slip is a saturation, not a state.** A real belt that slips (or a toothed
  belt that jumps a tooth) *permanently* loses registration — the zero of
  ``stretch`` moves. This block saturates the transmitted force at ``f_slip``
  and does not track the lost registration, so it models the torque ceiling
  correctly and the position error after a slip event not at all. When a
  toothed belt is running below its jump threshold — the design case — the
  distinction never arises. ``f_slip = inf`` disables the limit.
* **The stretch is a function of the two angles**, not an integrated state,
  for the same reason as
  :class:`~jaxterity.actuators.backlash.PositionalBacklash`: both angles are
  states of the body model, so the belt stays a pure map and composes.

Identifiability. ``r_motor`` and ``r_load`` enter only through the ratio
``r_load/r_motor`` and the force scale, so a torque-only trace identifies the
ratio and ``k_belt·r²``; measure the pulley radii (they are geometry, not a
fit) and the trace then identifies ``k_belt`` and ``d_belt`` outright. The
cleanest ``k_belt`` comes from the resonance: with a load inertia ``J`` on the
output, ``ω_n = r_load·√(k_belt/J)`` — see :func:`belt_resonance_hz`.

Units
-----
m, rad, rad/s, N, N/m, N·s/m, N·m.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..data.schema import TransmissionTelemetry, ValidationReport
from ..errors import CalibrationError
from ..parameter import ParameterSpec
from ..sysid.protocol import SysidProtocol
from ._base import Actuator

_SYSID_PROTOCOL = SysidProtocol(
    name="belt-chirp-and-ratio",
    excitation="chirp",
    required_columns=("tau_motor", "v_motor", "tau_joint", "v_joint"),
    duration_s=20.0,
    sample_rate_hz=1000.0,
    setup_notes=(
        "Measure the pulley pitch radii with calipers first — they are "
        "geometry and fitting them wastes identifiability. Then: (1) a slow "
        "constant-speed run identifies the ratio and any steady loss; (2) a "
        "logarithmic chirp from below to well above the expected belt "
        "resonance (start at 1 Hz, end at 5x the estimate from "
        "belt_resonance_hz) with a KNOWN inertia on the output identifies "
        "k_belt from the resonant peak and d_belt from its width. Sample at "
        "least 10x the highest chirp frequency. Do not exceed the belt's "
        "rated tension: a slipping run identifies f_slip but invalidates the "
        "registration for everything after it."
    ),
)


def belt_force(
    stretch,
    stretch_rate,
    *,
    k_belt,
    d_belt,
    f_slip=float("inf"),
):
    """Belt tension from span stretch and its rate, with a smooth slip ceiling.

    Args
    ----
    stretch : float or jax.Array
        Span stretch ``r_motor·θ_motor − r_load·θ_load``, m.
    stretch_rate : float or jax.Array
        Its time derivative, m/s.
    k_belt : float or jax.Array
        Belt span stiffness, N/m.
    d_belt : float or jax.Array
        Belt span damping, N·s/m.
    f_slip : float or jax.Array
        Force at which the belt slips (the ``tanh`` saturation scale), N.
        ``inf`` disables the limit.

    Returns
    -------
    jax.Array
        Transmitted belt force, N.

    Units
    -----
    m, m/s, N/m, N·s/m, N → N.
    """
    ideal = k_belt * stretch + d_belt * stretch_rate
    limited = jnp.isfinite(f_slip)
    scale = jnp.where(limited, f_slip, 1.0)
    return jnp.where(limited, scale * jnp.tanh(ideal / scale), ideal)


def belt_resonance_hz(k_belt, r_load, inertia):
    """Natural frequency of a load inertia on a compliant belt, Hz.

    ``ω_n = r_load·√(k_belt/J)`` — the mode a stiff-transmission model misses
    entirely, and the cheapest way to measure ``k_belt`` (drive a chirp, read
    the peak, invert this).

    Args
    ----
    k_belt : float or jax.Array
        Belt span stiffness, N/m.
    r_load : float or jax.Array
        Load pulley pitch radius, m.
    inertia : float or jax.Array
        Load-side inertia about the pulley axis, kg·m².

    Returns
    -------
    jax.Array
        Undamped natural frequency, Hz.

    Units
    -----
    N/m, m, kg·m² → Hz.

    Examples
    --------
    >>> round(float(belt_resonance_hz(2.0e5, 0.02, 4e-4)), 1)
    71.2
    """
    return r_load * jnp.sqrt(k_belt / inertia) / (2.0 * jnp.pi)


class BeltDrive(Actuator):
    """Compliant belt/pulley drive with a smooth slip limit.

    See the module docstring for the model and its two named approximations.
    Input ports are ``theta_motor``, ``theta_load`` (rad) and ``v_motor``,
    ``v_load`` (rad/s); output ports are ``tau_load`` (N·m, applied to the
    load) and ``tau_reaction`` (N·m, the reaction on the motor pulley).

    Args
    ----
    r_motor : float or jax.Array
        Motor pulley pitch radius, m. Must be > 0.
    r_load : float or jax.Array
        Load pulley pitch radius, m. Must be > 0.
    k_belt : float or jax.Array
        Belt span stiffness, N/m. Must be > 0.
    d_belt : float or jax.Array
        Belt span damping, N·s/m. Must be >= 0.
    f_slip : float or jax.Array
        Slip force limit, N (``inf`` for none).
    name : str
        Block name.

    Examples
    --------
    >>> belt = BeltDrive(r_motor=0.01, r_load=0.02, k_belt=2e5)
    >>> round(float(belt.ratio), 3)
    2.0
    """

    sysid_protocol = _SYSID_PROTOCOL

    def __init__(
        self,
        *,
        r_motor=0.01,
        r_load=0.02,
        k_belt=1.0e5,
        d_belt=0.0,
        f_slip=float("inf"),
        name: str = "belt_drive",
    ):
        super().__init__(name=name)
        for n, x in (
            ("r_motor", r_motor),
            ("r_load", r_load),
            ("k_belt", k_belt),
            ("d_belt", d_belt),
            ("f_slip", f_slip),
        ):
            self.declare_dynamic_parameter(n, jnp.asarray(x, dtype=float))
        for port in ("theta_motor", "theta_load", "v_motor", "v_load"):
            self.declare_input_port(name=port)
        self.declare_output_port(self._tau_load_out, name="tau_load")
        self.declare_output_port(self._tau_reaction_out, name="tau_reaction")

    @property
    def r_motor(self):
        return self.dynamic_parameters["r_motor"].get()

    @property
    def r_load(self):
        return self.dynamic_parameters["r_load"].get()

    @property
    def k_belt(self):
        return self.dynamic_parameters["k_belt"].get()

    @property
    def d_belt(self):
        return self.dynamic_parameters["d_belt"].get()

    @property
    def f_slip(self):
        return self.dynamic_parameters["f_slip"].get()

    @property
    def ratio(self):
        """Kinematic ratio ``r_load / r_motor`` (motor turns per load turn)."""
        return self.r_load / self.r_motor

    @classmethod
    def uncalibrated(cls, *, name: str = "belt_drive") -> "BeltDrive":
        """Return an instance with identifiability-neutral defaults.

        A 1:2 pulley pair with a stiff (1e5 N/m) undamped belt and no slip
        limit — the "measure the radii, fit the rest" starting point.
        """
        return cls(name=name)

    def _force(self, inputs, params):
        theta_motor, theta_load, v_motor, v_load = inputs
        r_m, r_l = params["r_motor"], params["r_load"]
        return belt_force(
            r_m * theta_motor - r_l * theta_load,
            r_m * v_motor - r_l * v_load,
            k_belt=params["k_belt"],
            d_belt=params["d_belt"],
            f_slip=params["f_slip"],
        )

    def _tau_load_out(self, time, state, *inputs, **params):
        return params["r_load"] * self._force(inputs, params)

    def _tau_reaction_out(self, time, state, *inputs, **params):
        return params["r_motor"] * self._force(inputs, params)

    def parameter_spec(self) -> ParameterSpec:
        return ParameterSpec(("r_motor", "r_load", "k_belt", "d_belt", "f_slip"))

    def validate(self, telemetry: TransmissionTelemetry) -> ValidationReport:
        """Score the *steady-state* ratio and torque scaling against a trace.

        A two-port torque/velocity trace constrains the belt's kinematics
        (``v_motor/v_load ≈ ratio``) and its torque scaling
        (``τ_load/τ_motor ≈ ratio``) but says nothing about ``k_belt`` — the
        compliance is only visible in the transient, which
        :attr:`sysid_protocol`'s chirp segment is for. This report checks what
        the trace can check and stays quiet about the rest.

        Args
        ----
        telemetry : TransmissionTelemetry
            Motor-side and load-side torque/velocity (``*_joint`` channels
            carry the load side).

        Returns
        -------
        ValidationReport
            ``ok`` iff the velocity ratio matches within policy tolerance.

        Units
        -----
        N·m, rad/s.
        """
        from ..testing import tolerances as tol

        v_motor = np.asarray(telemetry.v_motor, dtype=np.float64)
        v_load = np.asarray(telemetry.v_joint, dtype=np.float64)
        ratio = float(self.ratio)
        residual = v_motor / ratio - v_load
        rms = float(np.sqrt(np.mean(residual * residual)))
        report = ValidationReport()
        bound = tol.atol("joint_velocity", "rk4", "float64") * 1e4
        if rms > bound:
            report.warnings.append(
                f"belt velocity-ratio residual RMS={rms:.3g} rad/s exceeds "
                f"bound {bound:.3g} rad/s — the pulley radii may not match, or "
                "the belt slipped during this trace."
            )
        return report


def _fit_belt(telemetry: TransmissionTelemetry) -> dict[str, float]:
    """Recover the pulley ratio and the quasi-static force scale from a trace.

    Two closed-form estimates:

    * ``ratio = r_load/r_motor`` from the velocity regression (independent of
      every force parameter).
    * ``torque_ratio = τ_load/τ_motor``, which equals the same ratio for an
      ideal belt; its *departure* from the velocity ratio is the belt's
      steady-state loss and is reported so a caller can see it rather than
      have it absorbed silently.

    ``k_belt``/``d_belt`` are deliberately **not** fitted here: they need the
    chirp segment and a known output inertia (see :func:`belt_resonance_hz`),
    and a quasi-static trace that appears to identify them is fitting noise.

    Private to ``belt.py`` (to make the recovery test runnable); the formal
    ``sysid`` recipe home is later work.

    Args
    ----
    telemetry : TransmissionTelemetry
        Two-port trace; the ``*_joint`` channels carry the load side.

    Returns
    -------
    dict[str, float]
        ``{"ratio": ..., "torque_ratio": ...}``, both dimensionless.

    Raises
    ------
    CalibrationError
        If the load never moves or the motor never applies torque.

    Units
    -----
    rad/s, N·m in; dimensionless out.
    """
    v_motor = np.asarray(telemetry.v_motor, dtype=np.float64)
    v_load = np.asarray(telemetry.v_joint, dtype=np.float64)
    tau_motor = np.asarray(telemetry.tau_motor, dtype=np.float64)
    tau_load = np.asarray(telemetry.tau_joint, dtype=np.float64)
    if float(np.sum(v_load * v_load)) <= 0.0:
        raise CalibrationError(
            "_fit_belt: the load pulley never turns, so the ratio is "
            "unobservable. Fix: record a motion sweep."
        )
    if float(np.sum(tau_motor * tau_motor)) <= 0.0:
        raise CalibrationError(
            "_fit_belt: the motor applies no torque in this trace, so the "
            "force scaling is unobservable."
        )
    ratio = float(np.sum(v_motor * v_load) / np.sum(v_load * v_load))
    torque_ratio = float(np.sum(tau_motor * tau_load) / np.sum(tau_motor * tau_motor))
    return {"ratio": ratio, "torque_ratio": torque_ratio}


__all__ = ["BeltDrive", "belt_force", "belt_resonance_hz"]
