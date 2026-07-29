# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""True positional backlash: the angular play between motor and joint.

:class:`~jaxterity.actuators.harmonic_drive.HarmonicDrive` approximates
backlash in the *velocity* domain — torque transmission rolls off near zero
motor velocity. That captures the dominant symptom (lost torque through a
reversal) with no extra state, and it is the right first cut when only one
encoder exists. It is not backlash. Real backlash is *positional*: the gear
teeth are separated by a gap, the joint angle lags the motor angle by up to the
play, and no torque crosses the mesh until the slack is taken up. Two
consequences the velocity-domain proxy cannot produce: the lost motion is a
function of *reversal history*, not of speed, and a slow reversal loses exactly
as much angle as a fast one.

Model — a compliant mesh behind a dead zone, stateless in the block::

    Δθ  = θ_motor − θ_joint                    (both referred to the joint side)
    τ   = k_mesh · deadzone(Δθ, play)  +  d_mesh · engagement(Δθ, play) · Δθ̇

with

    deadzone(x, b)   = x − b·tanh(x/b)     ≈ x − b·sign(x) for |x| ≫ b
    engagement(x, b) = tanh(x/b)²          ∈ [0, 1], the smooth d(deadzone)/dx

The dead zone is *smoothed* (``tanh``, not ``clip``), and both the smoothing
and its width are named and exposed: ``play`` is the parameter, and the
approximation is a cubic ``x³/(3b²)`` inside the gap instead of an exact zero.
Nothing is hidden (CONTEXT.md invariant 6). Damping acts only through the
engagement factor, so a joint coasting inside its own backlash is free.

Where the state went. The block is a *torque map*, not an integrator: the
joint angle is a state of the body (MJX) model, not of the transmission, so
the positional history lives where it belongs and this block stays a pure
function of the two angles. That is what makes it compose — the same map is
valid inside a diagram, inside a fit, and inside a hysteresis plot.

Identification needs **two encoders**. ``play`` is the offset between the two
loading branches of the ``(Δθ, τ)`` hysteresis loop; with only a motor encoder
there is no Δθ, and no amount of torque data recovers it —
:class:`~jaxterity.data.schema.BacklashTelemetry` makes that requirement
explicit. :func:`lost_motion` reads the loop width straight off a trace as a
model-free cross-check on the fit.

Units
-----
rad, N·m, N·m/rad, N·m·s/rad.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..data.schema import BacklashTelemetry, ValidationReport
from ..errors import CalibrationError
from ..parameter import ParameterSpec
from ..sysid.protocol import SysidProtocol
from ._base import Actuator

#: Fraction of the peak |torque| below which a sample counts as "inside the
#: dead zone" for the fit's engaged-sample gate and for :func:`lost_motion`.
_ENGAGED_FRACTION = 0.1

_SYSID_PROTOCOL = SysidProtocol(
    name="backlash-reversal-sweep",
    excitation="ramp",
    required_columns=("theta_motor", "theta_joint", "tau_joint"),
    duration_s=20.0,
    sample_rate_hz=200.0,
    setup_notes=(
        "Instrument BOTH sides: a motor encoder and a joint-side (output) "
        "encoder, sampled on one clock, with the motor angle referred to the "
        "joint side (divided by the gear ratio). Hold the joint against a "
        "known compliant load and drive the motor slowly (well below any "
        "structural resonance) through several full direction reversals, "
        "covering positive and negative torque. Slow is essential: the fit "
        "must see the dead zone as a *position* effect, and speed-dependent "
        "friction masquerades as one. play is the half-width offset between "
        "the two loading branches of the (theta_motor - theta_joint, "
        "tau_joint) hysteresis loop; k_mesh is their common slope."
    ),
)


def deadzone(x, play):
    """Smooth dead zone: ``x − play·tanh(x/play)``.

    Zero (to third order) inside ``|x| < play`` and asymptotically
    ``x − play·sign(x)`` outside it — the take-up-the-slack map. C∞
    everywhere, so it composes through forward-mode autodiff, unlike the
    ``clip``-based dead zone whose kinks land exactly where a fit is most
    sensitive. ``play <= 0`` passes ``x`` through unchanged (no backlash).

    Args
    ----
    x : float or jax.Array
        Mesh deflection, rad.
    play : float or jax.Array
        Backlash half-width, rad. Total lost motion is ``2·play``.

    Returns
    -------
    jax.Array
        The engaged deflection, rad.

    Units
    -----
    rad in, rad out.

    Examples
    --------
    >>> float(deadzone(0.0, 0.01))
    0.0
    >>> bool(abs(float(deadzone(0.5, 0.01)) - 0.49) < 1e-6)
    True
    """
    safe = jnp.where(play > 0, play, 1.0)
    return jnp.where(play > 0, x - safe * jnp.tanh(x / safe), x)


def engagement(x, play):
    """How engaged the mesh is: ``tanh(x/play)²`` in [0, 1].

    The derivative of :func:`deadzone` with respect to ``x`` — 0 in the middle
    of the gap, 1 once the teeth are loaded. Used to gate mesh damping so a
    joint coasting inside its backlash feels no drag.

    Args
    ----
    x : float or jax.Array
        Mesh deflection, rad.
    play : float or jax.Array
        Backlash half-width, rad.

    Returns
    -------
    jax.Array
        Engagement fraction, dimensionless in [0, 1].

    Units
    -----
    rad in, dimensionless out.
    """
    safe = jnp.where(play > 0, play, 1.0)
    return jnp.where(play > 0, jnp.tanh(x / safe) ** 2, jnp.ones_like(jnp.asarray(x)))


def backlash_torque(delta_theta, delta_velocity, *, play, k_mesh, d_mesh):
    """Transmitted torque through a compliant mesh with positional play.

    Pure function behind :class:`PositionalBacklash` — usable directly for
    hysteresis plots and residual scoring without building a diagram.

    Args
    ----
    delta_theta : float or jax.Array
        ``θ_motor − θ_joint``, both referred to the joint side, rad.
    delta_velocity : float or jax.Array
        ``θ̇_motor − θ̇_joint``, rad/s.
    play : float or jax.Array
        Backlash half-width, rad.
    k_mesh : float or jax.Array
        Mesh stiffness once engaged, N·m/rad.
    d_mesh : float or jax.Array
        Mesh damping once engaged, N·m·s/rad.

    Returns
    -------
    jax.Array
        Transmitted joint torque, N·m.

    Units
    -----
    rad, rad/s, N·m/rad, N·m·s/rad → N·m.
    """
    return (
        k_mesh * deadzone(delta_theta, play)
        + d_mesh * engagement(delta_theta, play) * delta_velocity
    )


class PositionalBacklash(Actuator):
    """Angular-play backlash with a compliant mesh (stateless torque map).

    See the module docstring for the model, the smoothing contract, and why
    identification needs two encoders. Input ports are ``theta_motor``,
    ``theta_joint`` (rad, both referred to the joint side) and ``v_motor``,
    ``v_joint`` (rad/s); the output port ``tau_joint`` is the transmitted
    torque (N·m).

    Args
    ----
    play : float or jax.Array
        Backlash half-width, rad (total lost motion ``2·play``). Must be >= 0;
        0 disables the dead zone.
    k_mesh : float or jax.Array
        Engaged mesh stiffness, N·m/rad. Must be > 0.
    d_mesh : float or jax.Array
        Engaged mesh damping, N·m·s/rad. Must be >= 0.
    name : str
        Block name.

    Examples
    --------
    >>> blk = PositionalBacklash(play=0.01, k_mesh=200.0)
    >>> tuple(blk.parameter_spec())
    ('play', 'k_mesh', 'd_mesh')
    """

    sysid_protocol = _SYSID_PROTOCOL

    def __init__(
        self,
        *,
        play=0.0,
        k_mesh=100.0,
        d_mesh=0.0,
        name: str = "backlash",
    ):
        super().__init__(name=name)
        for n, x in (("play", play), ("k_mesh", k_mesh), ("d_mesh", d_mesh)):
            self.declare_dynamic_parameter(n, jnp.asarray(x, dtype=float))
        for port in ("theta_motor", "theta_joint", "v_motor", "v_joint"):
            self.declare_input_port(name=port)
        self.declare_output_port(self._tau_out, name="tau_joint")

    @property
    def play(self):
        return self.dynamic_parameters["play"].get()

    @property
    def k_mesh(self):
        return self.dynamic_parameters["k_mesh"].get()

    @property
    def d_mesh(self):
        return self.dynamic_parameters["d_mesh"].get()

    @classmethod
    def uncalibrated(cls, *, name: str = "backlash") -> "PositionalBacklash":
        """Return an instance with identifiability-neutral defaults.

        ``play=0`` (no backlash identified yet — a fit can only add lost
        motion, never assume it), ``k_mesh=100`` N·m/rad, ``d_mesh=0``.
        """
        return cls(name=name)

    def _tau_out(self, time, state, *inputs, **params):
        theta_motor, theta_joint, v_motor, v_joint = inputs
        return backlash_torque(
            theta_motor - theta_joint,
            v_motor - v_joint,
            play=params["play"],
            k_mesh=params["k_mesh"],
            d_mesh=params["d_mesh"],
        )

    def parameter_spec(self) -> ParameterSpec:
        return ParameterSpec(("play", "k_mesh", "d_mesh"))

    def validate(self, telemetry: BacklashTelemetry) -> ValidationReport:
        """Per-sample transmitted-torque residual against a dual-encoder trace.

        Applies the model to each sample's ``(θ_motor − θ_joint)`` and compares
        to the recorded ``tau_joint``. Mesh damping is evaluated from a central
        difference of the two angles, so a noisy encoder inflates the residual
        only when ``d_mesh`` is non-zero.

        Args
        ----
        telemetry : BacklashTelemetry
            Dual-encoder trace.

        Returns
        -------
        ValidationReport
            ``ok`` iff the RMS torque residual is within the policy bound.

        Units
        -----
        rad, N·m.
        """
        from ..testing import tolerances as tol

        t = np.asarray(telemetry.time, dtype=np.float64)
        delta = np.asarray(telemetry.theta_motor, dtype=np.float64) - np.asarray(
            telemetry.theta_joint, dtype=np.float64
        )
        measured = np.asarray(telemetry.tau_joint, dtype=np.float64)
        delta_v = np.gradient(delta, t)
        predicted = np.asarray(
            backlash_torque(
                jnp.asarray(delta),
                jnp.asarray(delta_v),
                play=float(self.play),
                k_mesh=float(self.k_mesh),
                d_mesh=float(self.d_mesh),
            )
        )
        residual = predicted - measured
        rms = float(np.sqrt(np.mean(residual * residual)))
        peak = float(np.max(np.abs(residual)))
        report = ValidationReport()
        bound = tol.atol("actuator_torque", "rk4", "float64") * 1e4
        if rms > bound:
            report.warnings.append(
                f"backlash torque residual RMS={rms:.3g} N·m (peak={peak:.3g} "
                f"N·m) exceeds bound {bound:.3g} N·m — play / k_mesh may not "
                "match this trace."
            )
        return report


def lost_motion(
    telemetry: BacklashTelemetry, *, engaged_fraction: float = _ENGAGED_FRACTION
) -> float:
    """Measure total lost motion off a trace, with no model in the way.

    The width of the ``(Δθ, τ)`` hysteresis loop at (near) zero torque: the
    span of ``θ_motor − θ_joint`` over the samples whose transmitted torque is
    below ``engaged_fraction`` of the trace's peak. The cross-check to run
    *before* trusting a fitted ``play``, because it assumes no stiffness and no
    model at all.

    What it measures exactly, so the number can be reported honestly: with a
    torque gate ``τ* = engaged_fraction · peak|τ|``, the width is

        width = 2 · deadzone⁻¹(τ*/k_mesh)

    which is *not* ``2·play`` unless the gate happens to sit at the knee. For a
    **hard** dead zone — real gear teeth — ``deadzone⁻¹(y) = play + y``, so
    ``width = 2·play + 2·τ*/k_mesh``: an over-estimate that tightens as the
    gate drops toward the torque noise floor. For this block's *smoothed* dead
    zone the inverse dips below ``play`` at very small gates, so a width read
    off simulated data with a tiny gate slightly under-reports instead.
    Either way the estimate is monotone in the gate and lands within a factor
    of about two of ``2·play`` at any sane setting — which is what a
    stiffness-free cross-check is for. Take the precise number from
    :func:`_fit_backlash`, and use this to catch the case where the fit is off
    by an order of magnitude or a sign.

    Args
    ----
    telemetry : BacklashTelemetry
        Dual-encoder trace covering at least one full reversal.
    engaged_fraction : float
        Torque threshold as a fraction of the trace's peak |torque|.

    Returns
    -------
    float
        Total lost motion, rad (≈ ``2·play``).

    Raises
    ------
    CalibrationError
        If no sample sits inside the dead zone (the trace never unloads, so
        the loop has no measurable width).

    Units
    -----
    rad.
    """
    delta = np.asarray(telemetry.theta_motor, dtype=np.float64) - np.asarray(
        telemetry.theta_joint, dtype=np.float64
    )
    tau = np.asarray(telemetry.tau_joint, dtype=np.float64)
    peak = float(np.max(np.abs(tau)))
    inside = np.abs(tau) <= engaged_fraction * peak
    if not np.any(inside):
        raise CalibrationError(
            "lost_motion found no unloaded samples: every sample carries more "
            f"than {engaged_fraction:.0%} of the peak torque "
            f"({peak:.3g} N·m), so the hysteresis loop has no measurable "
            "width. Fix: record through a full reversal including the "
            "torque-free crossing."
        )
    return float(np.ptp(delta[inside]))


def _fit_backlash(
    telemetry: BacklashTelemetry, *, engaged_fraction: float = _ENGAGED_FRACTION
) -> dict[str, float]:
    """Recover ``play`` and ``k_mesh`` from a dual-encoder reversal trace.

    Outside the dead zone the model is piecewise linear with a sign-dependent
    offset::

        τ = k_mesh·Δθ − k_mesh·play·sign(Δθ)

    so an OLS on ``[Δθ, −sign(Δθ)]`` over the *engaged* samples returns
    ``(k_mesh, k_mesh·play)`` and hence both parameters. Samples inside the
    dead zone are excluded by a torque gate: including them would drag the
    slope toward zero and shrink the apparent play. ``d_mesh`` is not fitted —
    it needs a fast reversal (velocity variation at constant deflection), which
    the slow protocol deliberately avoids.

    Private to ``backlash.py`` (to make the recovery test runnable); the formal
    ``sysid`` recipe home is later work.

    Args
    ----
    telemetry : BacklashTelemetry
        Dual-encoder reversal trace.
    engaged_fraction : float
        Torque gate as a fraction of the trace's peak |torque|.

    Returns
    -------
    dict[str, float]
        ``{"play": ..., "k_mesh": ...}`` in rad and N·m/rad.

    Raises
    ------
    CalibrationError
        If the trace has no engaged samples on both sides of the reversal
        (both branches are needed — one branch alone confounds ``play`` with a
        static encoder offset).

    Units
    -----
    rad, N·m.
    """
    delta = np.asarray(telemetry.theta_motor, dtype=np.float64) - np.asarray(
        telemetry.theta_joint, dtype=np.float64
    )
    tau = np.asarray(telemetry.tau_joint, dtype=np.float64)
    peak = float(np.max(np.abs(tau)))
    if peak <= 0.0:
        raise CalibrationError(
            "_fit_backlash: the trace transmits no torque at all, so neither "
            "the mesh stiffness nor the play is observable."
        )
    engaged = np.abs(tau) > engaged_fraction * peak
    if not (np.any(engaged & (delta > 0)) and np.any(engaged & (delta < 0))):
        raise CalibrationError(
            "_fit_backlash needs engaged samples on BOTH sides of the "
            "reversal; this trace loads only one direction, which cannot "
            "separate the play from a constant encoder offset. Fix: drive the "
            "motor through a full direction reversal under load."
        )
    x = delta[engaged]
    design = np.stack([x, -np.sign(x)], axis=1)
    coefficients, *_ = np.linalg.lstsq(design, tau[engaged], rcond=None)
    k_mesh = float(coefficients[0])
    return {"play": float(coefficients[1]) / k_mesh, "k_mesh": k_mesh}


__all__ = [
    "PositionalBacklash",
    "backlash_torque",
    "deadzone",
    "engagement",
    "lost_motion",
]
