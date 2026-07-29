# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Hybrid stepper motor: open-loop until it isn't.

Every other actuator in this package is commanded in torque or closes a loop
around position. A stepper does neither: the driver commands a *magnetic
field angle*, and the rotor follows it — as long as the load torque stays
inside the holding torque. Past that, the rotor slips a pole, the commanded
and actual angles part company permanently, and the only symptom is that the
machine is now wrong about where it is. A model that treats a stepper as an
ideal position source (the usual shortcut, and the one behind most "it works
in sim" printer/gantry code) cannot exhibit the single failure mode that
defines the family.

Model — stateless torque map, driven by the commanded field angle::

    θ_cmd_q = microstep · round(θ_cmd / microstep)        (driver quantization)
    error   = N_r · (θ_cmd_q − θ)                         (electrical angle)
    T_h(ω)  = T_hold / √(1 + (N_r·ω/ω_corner)²)           (inductance droop)
    τ       = T_h(ω)·sin(error) − T_detent·sin(4·N_r·θ) − b·ω

Four effects, each one a thing a step/direction abstraction hides:

* **The sinusoidal torque-angle curve.** Torque is proportional to
  ``sin(N_r·Δθ)``, not to ``Δθ``: it *peaks* at a quarter electrical cycle and
  falls beyond it. That falling branch is the instability — once past the
  peak, more error means less torque.
* **Pull-out.** Synchronism is lost when the load exceeds ``T_h(ω)``, i.e.
  when the electrical error passes ±π/2. :func:`pull_out_margin` reports the
  distance to that cliff, and it is the number a stepper-driven axis should
  be planned against.
* **Detent torque.** The unpowered cogging torque, at four times the
  electrical frequency: the reason a stepper holds position with the driver
  off, and a measurable ripple at low speed.
* **Speed droop.** Winding inductance limits how fast current commutes, so
  the available holding torque falls with speed — the classic
  speed/torque curve, here as a first-order corner at ``ω_corner``
  (electrical rad/s).

Scope. The rotor's own angle is a state of whatever this block drives (the
body model or a load inertia), not of the block, so slipping is *represented*
(the torque map is periodic in ``θ`` and the margin goes negative) but the
resulting permanent step loss is a property of the simulated trajectory rather
than a counter this block keeps. Current-loop dynamics, microstep current
quantization, and mid-band resonance are not modeled.

Units
-----
rad, rad/s, N·m; ``rotor_teeth`` is dimensionless (50 for a 1.8° motor).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..data.schema import ActuatorStepTelemetry, ValidationReport
from ..parameter import ParameterSpec
from ..sysid.protocol import SysidProtocol
from ._base import Actuator

#: Rotor teeth of the ubiquitous 1.8°/full-step hybrid stepper.
_ROTOR_TEETH_1_8_DEG = 50

_SYSID_PROTOCOL = SysidProtocol(
    name="stepper-pullout-and-detent",
    excitation="ramp",
    required_columns=("setpoint", "position"),
    duration_s=30.0,
    sample_rate_hz=1000.0,
    setup_notes=(
        "Three segments. (1) DETENT: with the driver disabled, back-drive the "
        "shaft slowly through several full revolutions against a torque "
        "sensor — the 4*N_r-per-revolution ripple amplitude is T_detent. "
        "(2) HOLDING: energize at a fixed commanded angle and apply an "
        "increasing external torque until the rotor breaks away; the peak is "
        "T_hold. (3) PULL-OUT CURVE: repeat (2) at increasing step rates and "
        "record the breakaway torque vs speed — the knee identifies "
        "omega_corner. Do NOT try to identify T_hold from a step response: "
        "an open-loop stepper that keeps synchronism tracks the command "
        "regardless of T_hold, so the trace is insensitive to the very "
        "parameter you want."
    ),
)


def stepper_torque(
    theta,
    omega,
    theta_command,
    *,
    T_hold,
    T_detent=0.0,
    rotor_teeth: int = _ROTOR_TEETH_1_8_DEG,
    b=0.0,
    omega_corner=float("inf"),
    microstep=0.0,
):
    """Torque a hybrid stepper produces at a given rotor and command angle.

    Pure function behind :class:`Stepper` — see the module docstring for the
    model and what each term is for.

    Args
    ----
    theta : float or jax.Array
        Rotor angle, rad.
    omega : float or jax.Array
        Rotor speed, rad/s.
    theta_command : float or jax.Array
        Commanded (field) angle, rad.
    T_hold : float or jax.Array
        Holding torque at zero speed, N·m.
    T_detent : float or jax.Array
        Detent (unpowered cogging) torque amplitude, N·m.
    rotor_teeth : int
        Rotor tooth count (50 for a 1.8°/step motor), dimensionless.
    b : float or jax.Array
        Viscous friction, N·m·s/rad.
    omega_corner : float or jax.Array
        Electrical speed at which the holding torque has fallen to
        ``T_hold/√2``, rad/s. ``inf`` disables the droop.
    microstep : float or jax.Array
        Driver command quantization, rad. 0 disables it.

    Returns
    -------
    jax.Array
        Torque on the rotor, N·m.

    Units
    -----
    rad, rad/s, N·m → N·m.

    Examples
    --------
    >>> # A quarter electrical cycle of error gives peak torque.
    >>> import math
    >>> peak = stepper_torque(0.0, 0.0, math.pi / (2 * 50), T_hold=2.0,
    ...                       rotor_teeth=50)
    >>> round(float(peak), 6)
    2.0
    """
    quantum = jnp.asarray(microstep, dtype=float)
    commanded = jnp.where(
        quantum > 0,
        jnp.round(theta_command / jnp.where(quantum > 0, quantum, 1.0)) * quantum,
        theta_command,
    )
    electrical_error = rotor_teeth * (commanded - theta)
    droop = jnp.sqrt(1.0 + (rotor_teeth * omega / omega_corner) ** 2)
    holding = T_hold / droop
    return (
        holding * jnp.sin(electrical_error)
        - T_detent * jnp.sin(4.0 * rotor_teeth * theta)
        - b * omega
    )


def pull_out_margin(theta, theta_command, *, rotor_teeth: int = _ROTOR_TEETH_1_8_DEG):
    """How far a stepper is from losing synchronism, in [1, −∞).

    ``1 − |N_r·(θ_cmd − θ)| / (π/2)``: 1 when the rotor is exactly on the
    commanded angle, 0 at the torque peak (the cliff edge), negative once the
    rotor has fallen past it and the torque is *decreasing* with error. This
    is the quantity to plan a stepper axis against — not the torque itself,
    which looks healthy right up to the peak.

    Args
    ----
    theta : float or jax.Array
        Rotor angle, rad.
    theta_command : float or jax.Array
        Commanded (field) angle, rad.
    rotor_teeth : int
        Rotor tooth count, dimensionless.

    Returns
    -------
    jax.Array
        Margin, dimensionless (1 = on target, <= 0 = past pull-out).

    Units
    -----
    rad in, dimensionless out.

    Examples
    --------
    >>> import math
    >>> float(pull_out_margin(0.0, 0.0))
    1.0
    >>> round(float(pull_out_margin(0.0, math.pi / (2 * 50))), 6)
    0.0
    """
    error = rotor_teeth * jnp.abs(jnp.asarray(theta_command) - jnp.asarray(theta))
    return 1.0 - error / (0.5 * jnp.pi)


class Stepper(Actuator):
    """Hybrid stepper torque map with detent, speed droop, and microstepping.

    See the module docstring for the model and the scope. Input ports are
    ``theta`` (rad), ``omega`` (rad/s) and ``theta_command`` (rad); the output
    port ``tau`` is the torque on the rotor (N·m).

    Args
    ----
    T_hold : float or jax.Array
        Holding torque at zero speed, N·m. Must be > 0.
    T_detent : float or jax.Array
        Detent torque amplitude, N·m. Must be >= 0.
    rotor_teeth : int
        Rotor tooth count (static structure, not a fitted parameter): 50 for a
        1.8°/step motor, 100 for 0.9°.
    b : float or jax.Array
        Viscous friction, N·m·s/rad.
    omega_corner : float or jax.Array
        Electrical corner speed for the torque droop, rad/s.
    microstep : float or jax.Array
        Driver command quantization, rad (e.g. ``2π/(200·16)`` for 16×
        microstepping on a 1.8° motor). 0 disables it.
    name : str
        Block name.

    Examples
    --------
    >>> motor = Stepper(T_hold=2.0, rotor_teeth=50)
    >>> motor.full_step_rad  # 1.8 degrees
    0.031415926535897934
    """

    sysid_protocol = _SYSID_PROTOCOL

    def __init__(
        self,
        *,
        T_hold=1.0,
        T_detent=0.0,
        rotor_teeth: int = _ROTOR_TEETH_1_8_DEG,
        b=0.0,
        omega_corner=float("inf"),
        microstep=0.0,
        name: str = "stepper",
    ):
        super().__init__(name=name)
        self._rotor_teeth = int(rotor_teeth)
        for n, x in (
            ("T_hold", T_hold),
            ("T_detent", T_detent),
            ("b", b),
            ("omega_corner", omega_corner),
            ("microstep", microstep),
        ):
            self.declare_dynamic_parameter(n, jnp.asarray(x, dtype=float))
        for port in ("theta", "omega", "theta_command"):
            self.declare_input_port(name=port)
        self.declare_output_port(self._tau_out, name="tau")

    @property
    def T_hold(self):
        return self.dynamic_parameters["T_hold"].get()

    @property
    def T_detent(self):
        return self.dynamic_parameters["T_detent"].get()

    @property
    def b(self):
        return self.dynamic_parameters["b"].get()

    @property
    def omega_corner(self):
        return self.dynamic_parameters["omega_corner"].get()

    @property
    def microstep(self):
        return self.dynamic_parameters["microstep"].get()

    @property
    def rotor_teeth(self) -> int:
        """Rotor tooth count (static structure, not an identifiable leaf)."""
        return self._rotor_teeth

    @property
    def full_step_rad(self) -> float:
        """Mechanical angle of one full step, rad (``2π/(4·N_r)``)."""
        return 2.0 * np.pi / (4.0 * self._rotor_teeth)

    @classmethod
    def uncalibrated(cls, *, name: str = "stepper") -> "Stepper":
        """Return an instance with identifiability-neutral defaults.

        A 1.8° motor with 1 N·m holding torque, no detent, no droop and no
        microstep quantization — every *loss* term starts at zero so a fit can
        only add what it measures.
        """
        return cls(name=name)

    def _tau_out(self, time, state, *inputs, **params):
        theta, omega, theta_command = inputs
        return stepper_torque(
            theta,
            omega,
            theta_command,
            T_hold=params["T_hold"],
            T_detent=params["T_detent"],
            rotor_teeth=self._rotor_teeth,
            b=params["b"],
            omega_corner=params["omega_corner"],
            microstep=params["microstep"],
        )

    def parameter_spec(self) -> ParameterSpec:
        return ParameterSpec(("T_hold", "T_detent", "b", "omega_corner", "microstep"))

    def validate(self, telemetry: ActuatorStepTelemetry) -> ValidationReport:
        """Check a commanded-vs-measured angle trace for lost synchronism.

        A stepper that keeps synchronism tracks its command to within a
        fraction of an electrical cycle; a stepper that slipped is off by a
        multiple of a full step, forever. Rather than score a torque residual
        the trace cannot constrain (see :attr:`sysid_protocol`), this report
        answers the question the trace *can* answer: did the rotor stay inside
        the pull-out margin, and if not, by how many steps did it slip?

        Args
        ----
        telemetry : ActuatorStepTelemetry
            ``setpoint`` is the commanded angle (rad), ``position`` the
            measured rotor angle (rad).

        Returns
        -------
        ValidationReport
            ``ok`` iff every sample stayed inside the pull-out margin.

        Units
        -----
        rad.
        """
        commanded = np.asarray(telemetry.setpoint, dtype=np.float64)
        measured = np.asarray(telemetry.position, dtype=np.float64)
        margin = np.asarray(
            pull_out_margin(measured, commanded, rotor_teeth=self._rotor_teeth)
        )
        report = ValidationReport()
        worst = float(np.min(margin))
        if worst <= 0.0:
            final_error = float(commanded[-1] - measured[-1])
            steps = final_error / self.full_step_rad
            report.warnings.append(
                f"stepper lost synchronism: worst pull-out margin {worst:.3g} "
                f"(<= 0 means past the torque peak), ending "
                f"{steps:.1f} full step(s) behind the command. The load "
                "exceeded the holding torque at this speed — reduce "
                "acceleration, lower the step rate, or raise the current."
            )
        return report


__all__ = ["Stepper", "pull_out_margin", "stepper_torque"]
