# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Gear stage with direction-dependent efficiency and load-independent loss.

A gearbox is not a multiplier. Two effects break that idealization on any real
robot, and both show up in a calibration as *torque you commanded and never
got*:

1. **Direction-dependent efficiency.** Driving the load and being driven by it
   are different loss paths. A worm or high-ratio harmonic stage can be ~80 %
   efficient forward and effectively non-backdrivable in reverse.
2. **Load-independent loss.** Seal drag, grease churn, and preload cost torque
   even with nothing on the output — the term that makes a gearbox's
   *efficiency* rise with load rather than being a constant.

Model — stateless two-port, motor side in, joint side out::

    τ_net = τ_motor − τ_loss·tanh(v_motor/v_eps) − b·v_motor    (losses first)
    P_dir = τ_net · v_motor                                     (power flow)
    η_eff = ½(η_f + 1/η_b) + ½(η_f − 1/η_b)·tanh(P_dir/p_eps)   (smooth switch)
    τ_joint = N · η_eff · τ_net
    v_joint = v_motor / N

The efficiency switch deserves a word. In the driving quadrant the joint gets
``η_f`` of the motor's power, so ``τ_joint = N·η_f·τ_net``. In the
*regenerative* quadrant the load is the source, the motor is the sink, and
power balance runs the other way: ``η_b·(joint power) = (motor power)``, giving
``τ_joint = N·τ_net/η_b``. The reciprocal is not a typo — back-driving a lossy
stage costs the *load* more torque, not less. (Contrast
:class:`~jaxterity.actuators.harmonic_drive.HarmonicDrive`, which applies
``η_b`` as a plain multiplier and folds backlash into a velocity-domain
roll-off: it is the single-encoder first cut, this is the two-quadrant model.
Compose this with
:class:`~jaxterity.actuators.backlash.PositionalBacklash` when a joint-side
encoder makes the play observable.)

Both smoothings are named and exposed: ``v_eps`` (rad/s) for the Coulomb-loss
sign and ``p_eps`` (W) for the power-direction switch. Nothing is hidden.

Identifiability. ``η_f`` and ``τ_loss`` are separable only from a trace that
spans a *range of load torques* at each direction — at a single operating point
they trade off exactly. ``η_b`` needs the regenerative quadrant, which means
back-driving the joint (lower the load, or drive it externally); a trace with
only positive power leaves ``η_b`` at its prior, and :func:`_fit_gearbox` says
so rather than reporting a number it did not measure.

Units
-----
N·m, rad/s, W; ratios and efficiencies dimensionless.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..data.schema import TransmissionTelemetry, ValidationReport
from ..errors import CalibrationError
from ..parameter import ParameterSpec
from ..sysid.protocol import SysidProtocol
from ._base import Actuator

#: Smoothing width for the Coulomb-loss sign (rad/s).
_V_EPS_DEFAULT = 1e-3
#: Smoothing width for the power-direction switch (W).
_P_EPS_DEFAULT = 1e-6

_SYSID_PROTOCOL = SysidProtocol(
    name="gearbox-two-quadrant-sweep",
    excitation="ramp",
    required_columns=("tau_motor", "v_motor", "tau_joint", "v_joint"),
    duration_s=30.0,
    sample_rate_hz=200.0,
    setup_notes=(
        "Instrument both sides (motor torque + speed, joint torque + speed). "
        "Sweep the load torque from near-zero to rated at several speeds in "
        "BOTH power directions: first driving the load (motor accelerating a "
        "brake/dyno), then back-driving (the load overhauling the motor, e.g. "
        "lowering a mass or driving the output with a second motor). The "
        "load sweep separates eta_f from tau_loss (a single load point cannot); "
        "the regenerative segment is the only thing that identifies eta_b."
    ),
)


def gearbox_torque(
    tau_motor,
    v_motor,
    *,
    N,
    eta_f,
    eta_b,
    tau_loss,
    b,
    v_eps: float = _V_EPS_DEFAULT,
    p_eps: float = _P_EPS_DEFAULT,
):
    """Joint-side torque from motor-side torque and velocity.

    Pure function behind :class:`Gearbox` — see the module docstring for the
    model and the two-quadrant efficiency convention.

    Args
    ----
    tau_motor : float or jax.Array
        Motor-side torque, N·m.
    v_motor : float or jax.Array
        Motor-side angular velocity, rad/s.
    N : float or jax.Array
        Gear ratio (motor turns per joint turn), dimensionless, > 0.
    eta_f : float or jax.Array
        Forward (motor → joint) efficiency in (0, 1].
    eta_b : float or jax.Array
        Backward (joint → motor) efficiency in (0, 1].
    tau_loss : float or jax.Array
        Load-independent Coulomb loss torque referred to the motor side, N·m.
    b : float or jax.Array
        Viscous loss coefficient referred to the motor side, N·m·s/rad.
    v_eps : float
        Smoothing width for the Coulomb-loss sign, rad/s.
    p_eps : float
        Smoothing width for the power-direction switch, W.

    Returns
    -------
    jax.Array
        Joint-side torque, N·m.

    Units
    -----
    N·m, rad/s in; N·m out.
    """
    tau_net = tau_motor - tau_loss * jnp.tanh(v_motor / v_eps) - b * v_motor
    power = tau_net * v_motor
    eta_eff = 0.5 * (eta_f + 1.0 / eta_b) + 0.5 * (eta_f - 1.0 / eta_b) * jnp.tanh(
        power / p_eps
    )
    return N * eta_eff * tau_net


class Gearbox(Actuator):
    """Gear stage: ratio, two-quadrant efficiency, and load-independent loss.

    See the module docstring for the model and its identifiability. Input
    ports are ``tau_motor`` (N·m) and ``v_motor`` (rad/s); output ports are
    ``tau_joint`` (N·m) and ``v_joint`` (rad/s).

    Args
    ----
    N : float or jax.Array
        Gear ratio (motor turns per joint turn). Must be > 0.
    eta_f : float or jax.Array
        Forward efficiency in (0, 1].
    eta_b : float or jax.Array
        Backward efficiency in (0, 1]. Small values model a nearly
        non-backdrivable stage; exactly 0 is not representable (the model
        divides by it) — use a small positive value.
    tau_loss : float or jax.Array
        Load-independent Coulomb loss torque, motor side, N·m. Must be >= 0.
    b : float or jax.Array
        Viscous loss coefficient, motor side, N·m·s/rad. Must be >= 0.
    v_eps : float
        Coulomb-sign smoothing width, rad/s.
    p_eps : float
        Power-direction smoothing width, W.
    name : str
        Block name.

    Examples
    --------
    >>> box = Gearbox(N=30.0, eta_f=0.8, eta_b=0.6)
    >>> tuple(box.parameter_spec())
    ('N', 'eta_f', 'eta_b', 'tau_loss', 'b')
    """

    sysid_protocol = _SYSID_PROTOCOL

    def __init__(
        self,
        *,
        N=30.0,
        eta_f=0.9,
        eta_b=0.8,
        tau_loss=0.0,
        b=0.0,
        v_eps: float = _V_EPS_DEFAULT,
        p_eps: float = _P_EPS_DEFAULT,
        name: str = "gearbox",
    ):
        super().__init__(name=name)
        self._v_eps = float(v_eps)
        self._p_eps = float(p_eps)
        for n, x in (
            ("N", N),
            ("eta_f", eta_f),
            ("eta_b", eta_b),
            ("tau_loss", tau_loss),
            ("b", b),
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
    def tau_loss(self):
        return self.dynamic_parameters["tau_loss"].get()

    @property
    def b(self):
        return self.dynamic_parameters["b"].get()

    @classmethod
    def uncalibrated(cls, *, name: str = "gearbox") -> "Gearbox":
        """Return an instance with identifiability-neutral defaults.

        ``N=30`` (a mid-range robot stage), ``eta_f=eta_b=1`` and
        ``tau_loss=b=0`` — a lossless gear, so a fit can only *add* measured
        loss rather than start from an assumed one.
        """
        return cls(eta_f=1.0, eta_b=1.0, tau_loss=0.0, b=0.0, name=name)

    def _tau_joint_out(self, time, state, *inputs, **params):
        return gearbox_torque(
            inputs[0],
            inputs[1],
            N=params["N"],
            eta_f=params["eta_f"],
            eta_b=params["eta_b"],
            tau_loss=params["tau_loss"],
            b=params["b"],
            v_eps=self._v_eps,
            p_eps=self._p_eps,
        )

    def _v_joint_out(self, time, state, *inputs, **params):
        return inputs[1] / params["N"]

    def parameter_spec(self) -> ParameterSpec:
        return ParameterSpec(("N", "eta_f", "eta_b", "tau_loss", "b"))

    def validate(self, telemetry: TransmissionTelemetry) -> ValidationReport:
        """Per-sample torque and velocity residuals against a two-port trace.

        Args
        ----
        telemetry : TransmissionTelemetry
            Synchronized motor-side and joint-side torque/velocity.

        Returns
        -------
        ValidationReport
            ``ok`` iff both RMS residuals are within their policy bounds.

        Units
        -----
        N·m, rad/s.
        """
        from ..testing import tolerances as tol

        tau_motor = np.asarray(telemetry.tau_motor, dtype=np.float64)
        v_motor = np.asarray(telemetry.v_motor, dtype=np.float64)
        predicted_tau = np.asarray(
            gearbox_torque(
                jnp.asarray(tau_motor),
                jnp.asarray(v_motor),
                N=float(self.N),
                eta_f=float(self.eta_f),
                eta_b=float(self.eta_b),
                tau_loss=float(self.tau_loss),
                b=float(self.b),
                v_eps=self._v_eps,
                p_eps=self._p_eps,
            )
        )
        tau_residual = predicted_tau - np.asarray(telemetry.tau_joint, dtype=np.float64)
        v_residual = v_motor / float(self.N) - np.asarray(
            telemetry.v_joint, dtype=np.float64
        )
        report = ValidationReport()
        tau_rms = float(np.sqrt(np.mean(tau_residual**2)))
        v_rms = float(np.sqrt(np.mean(v_residual**2)))
        tau_bound = tol.atol("actuator_torque", "rk4", "float64") * 1e4
        v_bound = tol.atol("joint_velocity", "rk4", "float64") * 1e4
        if tau_rms > tau_bound:
            report.warnings.append(
                f"joint-torque residual RMS={tau_rms:.3g} N·m exceeds bound "
                f"{tau_bound:.3g} N·m — N / efficiency / loss may not match."
            )
        if v_rms > v_bound:
            report.warnings.append(
                f"joint-velocity residual RMS={v_rms:.3g} rad/s exceeds bound "
                f"{v_bound:.3g} rad/s — N may not match."
            )
        return report


def _fit_gearbox(
    telemetry: TransmissionTelemetry, *, v_eps: float = _V_EPS_DEFAULT
) -> dict[str, float]:
    """Recover ``N``, ``eta_f``, ``eta_b``, ``tau_loss`` and ``b`` from a trace.

    Three stages, each closed form:

    1. ``N`` from the velocity ratio (least squares of ``v_motor`` on
       ``v_joint``) — independent of every loss parameter.
    2. The driving quadrant (``τ_net·v_motor > 0``): with
       ``τ_joint/N = η_f·(τ_motor − τ_loss·sgn(v) − b·v)``, an OLS on
       ``[τ_motor, −sgn(v), −v]`` returns ``(η_f, η_f·τ_loss, η_f·b)``, so all
       three fall out of one regression.
    3. The regenerative quadrant, same design matrix, returns ``1/η_b`` as its
       leading coefficient.

    Both quadrants must be present. When the trace never back-drives, the
    function raises rather than reporting an ``eta_b`` the data never
    constrained.

    Private to ``gearbox.py`` (to make the recovery test runnable); the formal
    ``sysid`` recipe home is later work.

    Args
    ----
    telemetry : TransmissionTelemetry
        Two-quadrant load sweep per :attr:`Gearbox.sysid_protocol`.
    v_eps : float
        Velocity below which a sample is treated as stationary and dropped
        (its power direction is meaningless), rad/s.

    Returns
    -------
    dict[str, float]
        ``{"N", "eta_f", "eta_b", "tau_loss", "b"}``.

    Raises
    ------
    CalibrationError
        If the joint never turns, or either power quadrant is missing or too
        thin to regress.

    Units
    -----
    N·m, rad/s.
    """
    tau_motor = np.asarray(telemetry.tau_motor, dtype=np.float64)
    v_motor = np.asarray(telemetry.v_motor, dtype=np.float64)
    tau_joint = np.asarray(telemetry.tau_joint, dtype=np.float64)
    v_joint = np.asarray(telemetry.v_joint, dtype=np.float64)

    denominator = float(np.sum(v_joint * v_joint))
    if denominator <= 0.0:
        raise CalibrationError(
            "_fit_gearbox: the joint never moves in this trace, so the gear "
            "ratio is unobservable. Fix: record a motion sweep, not a stall."
        )
    N = float(np.sum(v_motor * v_joint) / denominator)

    moving = np.abs(v_motor) > v_eps
    power = tau_motor * v_motor
    quadrants = {
        "forward": moving & (power > 0),
        "regenerative": moving & (power < 0),
    }
    coefficients: dict[str, np.ndarray] = {}
    for label, mask in quadrants.items():
        design = np.stack(
            [tau_motor[mask], -np.sign(v_motor[mask]), -v_motor[mask]], axis=1
        )
        if design.shape[0] < 3 or np.linalg.matrix_rank(design, tol=1e-12) < 3:
            raise CalibrationError(
                f"_fit_gearbox: the {label} quadrant has too little independent "
                f"data ({design.shape[0]} sample(s)) to separate efficiency "
                "from the load-independent loss. Fix: sweep the load torque "
                "over a range at more than one speed, in both power "
                "directions (see Gearbox.sysid_protocol)."
            )
        solution, *_ = np.linalg.lstsq(design, tau_joint[mask] / N, rcond=None)
        coefficients[label] = solution

    eta_f = float(coefficients["forward"][0])
    eta_b = 1.0 / float(coefficients["regenerative"][0])
    # The loss terms are shared; take them from the forward quadrant, where the
    # regression is best conditioned (efficiency multiplies them directly).
    tau_loss = float(coefficients["forward"][1]) / eta_f
    b = float(coefficients["forward"][2]) / eta_f
    return {"N": N, "eta_f": eta_f, "eta_b": eta_b, "tau_loss": tau_loss, "b": b}


__all__ = ["Gearbox", "gearbox_torque"]
