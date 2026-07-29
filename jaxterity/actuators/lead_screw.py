# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Lead screw / ball screw: rotary in, linear out, often one-way.

The rotary transmissions in this package (gearbox, harmonic drive, belt) keep
the same units on both ports. A screw does not: it converts torque and angular
velocity into force and linear velocity, and it is the family where
*non-backdrivability* stops being an approximation and becomes the design
intent — a lead screw holding a vertical axis against gravity with the motor
unpowered is doing its job.

Model — stateless two-port, motor side in, linear side out::

    v_load   = ω_motor · lead / 2π                        (kinematics, exact)
    τ_net    = τ_motor − τ_loss·tanh(ω_motor/ω_eps)       (drag at the bearings)
    P_dir    = τ_net · ω_motor
    η_eff    = ½(η_f + 1/η_b) + ½(η_f − 1/η_b)·tanh(P_dir/p_eps)
    F_load   = η_eff · 2π · τ_net / lead  −  F_preload·tanh(v_load/v_eps)

with the same two-quadrant efficiency convention as
:class:`~jaxterity.actuators.gearbox.Gearbox` (see its module docstring for the
reciprocal in the regenerative quadrant), plus a nut-preload Coulomb force on
the linear side.

Backdrivability is geometry, not a fit. For a screw of lead ``L``, pitch
diameter ``d`` and friction coefficient ``µ``, with lead angle
``λ = atan(L/(π·d))``:

    η_f = (1 − µ·tanλ) / (1 + µ/tanλ)        driving
    η_b = (1 − µ/tanλ) / (1 + µ·tanλ)        back-driving

and ``η_b ≤ 0`` — i.e. ``tanλ ≤ µ`` — means the screw *cannot* be back-driven
at all. :func:`screw_efficiency` computes both and
:func:`is_backdrivable` answers the question directly; use them to seed a fit
rather than guessing, and note that a non-backdrivable screw makes ``η_b``
unidentifiable from any trace (there is no regenerative quadrant to measure).
The block clamps ``η_b`` away from zero because the model divides by it; a
truly self-locking screw is the ``η_b → 0⁺`` limit, where back-driving takes
unbounded load force — which is the physically right answer.

Units
-----
N·m, rad/s, N, m/s, m (lead, per revolution).
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from ..data.schema import LinearTransmissionTelemetry, ValidationReport
from ..errors import CalibrationError
from ..parameter import ParameterSpec
from ..sysid.protocol import SysidProtocol
from ._base import Actuator

#: Smoothing widths: motor speed (rad/s) for bearing drag, linear speed (m/s)
#: for nut preload, power (W) for the direction switch.
_W_EPS_DEFAULT = 1e-3
_V_EPS_DEFAULT = 1e-5
_P_EPS_DEFAULT = 1e-6
#: Floor on the fitted/declared backward efficiency (the model divides by it).
_ETA_B_FLOOR = 1e-3

_SYSID_PROTOCOL = SysidProtocol(
    name="screw-two-quadrant-sweep",
    excitation="ramp",
    required_columns=("tau_motor", "v_motor", "force_load", "v_load"),
    duration_s=30.0,
    sample_rate_hz=200.0,
    setup_notes=(
        "Mount the screw with an inline load cell on the moving element and "
        "measure motor torque and speed. Take the lead from the hardware "
        "spec — it is geometry, and fitting it wastes identifiability. Sweep "
        "the load force from near-zero to rated while driving (motor pushing "
        "the load), then let the load overhaul the screw (lower a mass, or "
        "drive the nut externally) for the regenerative segment. If the screw "
        "is self-locking, the second segment is impossible and eta_b is not "
        "identifiable from ANY trace — use screw_efficiency() from the "
        "geometry instead and say so in the report."
    ),
)


def screw_efficiency(lead: float, pitch_diameter: float, mu: float) -> dict[str, float]:
    """Forward and backward efficiency of a screw from its geometry.

    The classical square-thread result (see the module docstring for the
    formulas). Returns both efficiencies; ``eta_b <= 0`` means the screw is
    self-locking, which is reported as-is rather than clamped — the sign is
    the answer to "can this be back-driven".

    Args
    ----
    lead : float
        Axial travel per revolution, m. Must be > 0.
    pitch_diameter : float
        Thread pitch diameter, m. Must be > 0.
    mu : float
        Coulomb friction coefficient at the thread, dimensionless. Must be
        >= 0 (≈0.15 for a dry steel/bronze lead screw, ≈0.01 for a ball screw).

    Returns
    -------
    dict[str, float]
        ``{"eta_f": ..., "eta_b": ..., "lead_angle_rad": ...}``.

    Raises
    ------
    CalibrationError
        If ``lead`` or ``pitch_diameter`` is not positive, or ``mu`` < 0.

    Units
    -----
    m, m, dimensionless.

    Examples
    --------
    >>> e = screw_efficiency(lead=0.008, pitch_diameter=0.012, mu=0.15)
    >>> round(e["eta_f"], 3)
    0.492
    >>> e["eta_b"] > 0  # a 8 mm lead on 12 mm is backdrivable at mu=0.15
    True
    """
    if lead <= 0 or pitch_diameter <= 0 or mu < 0:
        raise CalibrationError(
            f"screw_efficiency needs lead > 0, pitch_diameter > 0 and mu >= 0; "
            f"got {lead}, {pitch_diameter}, {mu}."
        )
    tan_lambda = lead / (math.pi * pitch_diameter)
    eta_f = (1.0 - mu * tan_lambda) / (1.0 + mu / tan_lambda)
    eta_b = (1.0 - mu / tan_lambda) / (1.0 + mu * tan_lambda)
    return {
        "eta_f": eta_f,
        "eta_b": eta_b,
        "lead_angle_rad": math.atan(tan_lambda),
    }


def is_backdrivable(lead: float, pitch_diameter: float, mu: float) -> bool:
    """Whether a screw can be back-driven (``tanλ > µ``).

    Args
    ----
    lead : float
        Axial travel per revolution, m.
    pitch_diameter : float
        Thread pitch diameter, m.
    mu : float
        Thread friction coefficient, dimensionless.

    Returns
    -------
    bool
        ``True`` if a load force can turn the screw, ``False`` if it
        self-locks.

    Units
    -----
    m, m, dimensionless.

    Examples
    --------
    >>> is_backdrivable(lead=0.002, pitch_diameter=0.012, mu=0.15)
    False
    """
    return screw_efficiency(lead, pitch_diameter, mu)["eta_b"] > 0.0


def screw_force(
    tau_motor,
    v_motor,
    *,
    lead,
    eta_f,
    eta_b,
    tau_loss=0.0,
    f_preload=0.0,
    w_eps: float = _W_EPS_DEFAULT,
    v_eps: float = _V_EPS_DEFAULT,
    p_eps: float = _P_EPS_DEFAULT,
):
    """Axial load force from motor torque and speed.

    Pure function behind :class:`LeadScrew`; see the module docstring.

    Args
    ----
    tau_motor : float or jax.Array
        Motor-side torque, N·m.
    v_motor : float or jax.Array
        Motor-side angular velocity, rad/s.
    lead : float or jax.Array
        Axial travel per revolution, m.
    eta_f, eta_b : float or jax.Array
        Forward and backward efficiency in (0, 1].
    tau_loss : float or jax.Array
        Bearing/seal drag torque on the motor side, N·m.
    f_preload : float or jax.Array
        Nut-preload Coulomb force on the linear side, N.
    w_eps, v_eps, p_eps : float
        Smoothing widths (rad/s, m/s, W).

    Returns
    -------
    jax.Array
        Axial force delivered to the load, N.

    Units
    -----
    N·m, rad/s → N.
    """
    tau_net = tau_motor - tau_loss * jnp.tanh(v_motor / w_eps)
    power = tau_net * v_motor
    eta_eff = 0.5 * (eta_f + 1.0 / eta_b) + 0.5 * (eta_f - 1.0 / eta_b) * jnp.tanh(
        power / p_eps
    )
    v_load = v_motor * lead / (2.0 * jnp.pi)
    ideal = eta_eff * 2.0 * jnp.pi * tau_net / lead
    return ideal - f_preload * jnp.tanh(v_load / v_eps)


class LeadScrew(Actuator):
    """Rotary→linear screw with two-quadrant efficiency and nut preload.

    See the module docstring for the model and the backdrivability contract.
    Input ports are ``tau_motor`` (N·m) and ``v_motor`` (rad/s); output ports
    are ``force_load`` (N) and ``v_load`` (m/s).

    Args
    ----
    lead : float or jax.Array
        Axial travel per revolution, m. Must be > 0.
    eta_f : float or jax.Array
        Forward efficiency in (0, 1].
    eta_b : float or jax.Array
        Backward efficiency in (0, 1]; floored at 1e-3 (a self-locking screw
        is the limit — see the module docstring).
    tau_loss : float or jax.Array
        Motor-side drag torque, N·m. Must be >= 0.
    f_preload : float or jax.Array
        Nut-preload Coulomb force, N. Must be >= 0.
    w_eps, v_eps, p_eps : float
        Smoothing widths (rad/s, m/s, W).
    name : str
        Block name.

    Examples
    --------
    >>> screw = LeadScrew(lead=0.008, eta_f=0.5, eta_b=0.3)
    >>> tuple(screw.parameter_spec())
    ('lead', 'eta_f', 'eta_b', 'tau_loss', 'f_preload')
    """

    sysid_protocol = _SYSID_PROTOCOL

    def __init__(
        self,
        *,
        lead=0.008,
        eta_f=0.9,
        eta_b=0.8,
        tau_loss=0.0,
        f_preload=0.0,
        w_eps: float = _W_EPS_DEFAULT,
        v_eps: float = _V_EPS_DEFAULT,
        p_eps: float = _P_EPS_DEFAULT,
        name: str = "lead_screw",
    ):
        super().__init__(name=name)
        self._w_eps = float(w_eps)
        self._v_eps = float(v_eps)
        self._p_eps = float(p_eps)
        for n, x in (
            ("lead", lead),
            ("eta_f", eta_f),
            ("eta_b", jnp.maximum(jnp.asarray(eta_b, dtype=float), _ETA_B_FLOOR)),
            ("tau_loss", tau_loss),
            ("f_preload", f_preload),
        ):
            self.declare_dynamic_parameter(n, jnp.asarray(x, dtype=float))
        self.declare_input_port(name="tau_motor")
        self.declare_input_port(name="v_motor")
        self.declare_output_port(self._force_out, name="force_load")
        self.declare_output_port(self._v_load_out, name="v_load")

    @property
    def lead(self):
        return self.dynamic_parameters["lead"].get()

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
    def f_preload(self):
        return self.dynamic_parameters["f_preload"].get()

    @classmethod
    def uncalibrated(cls, *, name: str = "lead_screw") -> "LeadScrew":
        """Return an instance with identifiability-neutral defaults.

        An 8 mm lead, lossless in both directions and no preload — so a fit
        can only add measured loss.
        """
        return cls(eta_f=1.0, eta_b=1.0, tau_loss=0.0, f_preload=0.0, name=name)

    @classmethod
    def from_geometry(
        cls,
        *,
        lead: float,
        pitch_diameter: float,
        mu: float,
        name: str = "lead_screw",
    ) -> "LeadScrew":
        """Build a screw whose efficiencies come from its geometry.

        Uses :func:`screw_efficiency`; a self-locking screw (``eta_b <= 0``)
        is instantiated at the floor, which is the ``η_b → 0⁺`` limit — back
        driving it takes unbounded force, as it should.

        Args
        ----
        lead : float
            Axial travel per revolution, m.
        pitch_diameter : float
            Thread pitch diameter, m.
        mu : float
            Thread friction coefficient, dimensionless.
        name : str
            Block name.

        Returns
        -------
        LeadScrew
            The block with ``eta_f``/``eta_b`` from the geometry.

        Units
        -----
        m, m, dimensionless.
        """
        efficiency = screw_efficiency(lead, pitch_diameter, mu)
        return cls(
            lead=lead,
            eta_f=max(efficiency["eta_f"], _ETA_B_FLOOR),
            eta_b=max(efficiency["eta_b"], _ETA_B_FLOOR),
            name=name,
        )

    def _force_out(self, time, state, *inputs, **params):
        return screw_force(
            inputs[0],
            inputs[1],
            lead=params["lead"],
            eta_f=params["eta_f"],
            eta_b=params["eta_b"],
            tau_loss=params["tau_loss"],
            f_preload=params["f_preload"],
            w_eps=self._w_eps,
            v_eps=self._v_eps,
            p_eps=self._p_eps,
        )

    def _v_load_out(self, time, state, *inputs, **params):
        return inputs[1] * params["lead"] / (2.0 * jnp.pi)

    def parameter_spec(self) -> ParameterSpec:
        return ParameterSpec(("lead", "eta_f", "eta_b", "tau_loss", "f_preload"))

    def validate(self, telemetry: LinearTransmissionTelemetry) -> ValidationReport:
        """Per-sample force and linear-velocity residuals against a trace.

        Args
        ----
        telemetry : LinearTransmissionTelemetry
            Synchronized motor torque/speed and load force/velocity.

        Returns
        -------
        ValidationReport
            ``ok`` iff both RMS residuals are within their policy bounds.

        Units
        -----
        N·m, rad/s, N, m/s.
        """
        from ..testing import tolerances as tol

        tau_motor = np.asarray(telemetry.tau_motor, dtype=np.float64)
        v_motor = np.asarray(telemetry.v_motor, dtype=np.float64)
        predicted = np.asarray(
            screw_force(
                jnp.asarray(tau_motor),
                jnp.asarray(v_motor),
                lead=float(self.lead),
                eta_f=float(self.eta_f),
                eta_b=float(self.eta_b),
                tau_loss=float(self.tau_loss),
                f_preload=float(self.f_preload),
                w_eps=self._w_eps,
                v_eps=self._v_eps,
                p_eps=self._p_eps,
            )
        )
        force_residual = predicted - np.asarray(telemetry.force_load, dtype=np.float64)
        v_residual = v_motor * float(self.lead) / (2.0 * np.pi) - np.asarray(
            telemetry.v_load, dtype=np.float64
        )
        report = ValidationReport()
        force_rms = float(np.sqrt(np.mean(force_residual**2)))
        v_rms = float(np.sqrt(np.mean(v_residual**2)))
        force_bound = tol.atol("contact_force", "rk4", "float64") * 1e4
        v_bound = tol.atol("joint_velocity", "rk4", "float64") * 1e4
        if force_rms > force_bound:
            report.warnings.append(
                f"screw force residual RMS={force_rms:.3g} N exceeds bound "
                f"{force_bound:.3g} N — lead / efficiency / preload may not "
                "match this trace."
            )
        if v_rms > v_bound:
            report.warnings.append(
                f"screw linear-velocity residual RMS={v_rms:.3g} m/s exceeds "
                f"bound {v_bound:.3g} m/s — the lead may not match."
            )
        return report


def _fit_lead_screw(
    telemetry: LinearTransmissionTelemetry,
    *,
    lead: float,
    w_eps: float = _W_EPS_DEFAULT,
) -> dict[str, float]:
    """Recover the screw's efficiencies and drag from a two-quadrant trace.

    With the lead known (it is geometry), each power quadrant is linear:

        F·lead/2π = η_eff·(τ_motor − τ_loss·sgn(ω))

    so an OLS on ``[τ_motor, −sgn(ω)]`` returns ``(η_f, η_f·τ_loss)`` in the
    driving quadrant and ``(1/η_b, ·)`` in the regenerative one. The nut
    preload is absorbed into ``τ_loss`` here (both are Coulomb terms that
    switch on the same sign); separating them needs a trace at two different
    preloads, which no single capture provides — hence one reported number,
    not two invented ones.

    Private to ``lead_screw.py`` (to make the recovery test runnable); the
    formal ``sysid`` recipe home is later work.

    Args
    ----
    telemetry : LinearTransmissionTelemetry
        Two-quadrant sweep per :attr:`LeadScrew.sysid_protocol`.
    lead : float
        Known axial travel per revolution, m.
    w_eps : float
        Speed below which a sample is dropped as stationary, rad/s.

    Returns
    -------
    dict[str, float]
        ``{"eta_f", "eta_b", "tau_loss"}``.

    Raises
    ------
    CalibrationError
        If ``lead`` is not positive, or a power quadrant is missing (a
        self-locking screw simply has no regenerative data — see the module
        docstring).

    Units
    -----
    N·m, rad/s, N, m.
    """
    if lead <= 0:
        raise CalibrationError(
            f"_fit_lead_screw needs a known positive lead (m), got {lead}."
        )
    tau_motor = np.asarray(telemetry.tau_motor, dtype=np.float64)
    v_motor = np.asarray(telemetry.v_motor, dtype=np.float64)
    force = np.asarray(telemetry.force_load, dtype=np.float64)
    target = force * lead / (2.0 * np.pi)

    moving = np.abs(v_motor) > w_eps
    power = tau_motor * v_motor
    results: dict[str, np.ndarray] = {}
    for label, mask in (
        ("forward", moving & (power > 0)),
        ("regenerative", moving & (power < 0)),
    ):
        design = np.stack([tau_motor[mask], -np.sign(v_motor[mask])], axis=1)
        if design.shape[0] < 2 or np.linalg.matrix_rank(design, tol=1e-12) < 2:
            raise CalibrationError(
                f"_fit_lead_screw: the {label} quadrant has too little "
                f"independent data ({design.shape[0]} sample(s)). A "
                "self-locking screw has no regenerative quadrant at all — in "
                "that case take eta_b from screw_efficiency() and report it as "
                "geometry, not as a fit."
            )
        solution, *_ = np.linalg.lstsq(design, target[mask], rcond=None)
        results[label] = solution

    eta_f = float(results["forward"][0])
    return {
        "eta_f": eta_f,
        "eta_b": 1.0 / float(results["regenerative"][0]),
        "tau_loss": float(results["forward"][1]) / eta_f,
    }


__all__ = [
    "LeadScrew",
    "is_backdrivable",
    "screw_efficiency",
    "screw_force",
]
