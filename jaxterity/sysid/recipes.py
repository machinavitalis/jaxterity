# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Designed, identifiability-checked sysid recipes for actuator families.

The ``sysid.actuator`` recipe *fits* a device once you have telemetry;
this module adds the two things a field engineer needs around it for actuator
families beyond the SO-101 servo: a **designed excitation** (the input profile
that makes a fit well-posed) and an **identifiability check** that says — before
or after the run — whether the excitation actually constrains the parameters, so
a garbage fit on a flat excitation fails loudly instead of returning plausible
numbers.

Two families are covered here:

- **DC motor** (:class:`~jaxterity.actuators.dc_motor.DCMotorWithFriction`).
  The WPILib SysId relation ``V = Ks·sgn(v) + Kv·v + Ka·a`` is linear in the
  parameters, so identifiability is the rank / conditioning of the design matrix
  ``[sgn(v), v, a]``. The designed excitation is a zero-crossing,
  varying-acceleration velocity profile (a pure constant-acceleration ramp makes
  ``sgn(v)`` and ``a`` collinear — rank-deficient — which the check rejects).
- **BLDC** (:class:`~jaxterity.actuators.bldc.BLDC`). The current-loop dynamics
  are nonlinear, so identifiability is the conditioning of the **sensitivity
  Jacobian** of the predicted ``(v, i)`` trajectory w.r.t. the eight parameters,
  taken by **forward-mode** autodiff (the smoothed-friction term and the stiff
  current loop are traced, not linearized by hand). It surfaces the documented
  weak-identifiability of the electrical split ``R``/``L``/``Ke``/``Kp_i`` under
  a free-shaft chirp as named ``weak_params`` rather than silent over-confidence.

Both recipes reuse :func:`jaxterity.sysid.actuator` for the actual fit; what they
add is the excitation design and the up-front go/no-go on identifiability.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from ..actuators.bldc import BLDC
from ..actuators.dc_motor import DCMotorWithFriction
from ..data.schema import ActuatorChirpTelemetry, ActuatorRampTelemetry
from ..errors import CalibrationError
from . import actuator as _actuator_fit
from .result import SysidResult

#: Default conditioning ceiling: above this, ``JᵀJ`` is treated as effectively
#: rank-deficient (the excitation does not separate the parameters). 1e6 ≈ six
#: orders of magnitude between the strongest and weakest identified direction.
_DEFAULT_COND_MAX = 1e6


@dataclasses.dataclass(frozen=True)
class IdentifiabilityReport:
    """Whether an excitation constrains a fit's parameters, and how well.

    Built from the (sensitivity) Jacobian ``J`` of the fit residual w.r.t. the
    parameters: the singular values of ``J`` measure how strongly each
    orthogonal parameter direction is excited. A direction with a near-zero
    singular value is unconstrained — the fit there is arbitrary.

    Fields
    ------
    names : tuple[str, ...]
        Parameter names, in Jacobian-column order.
    singular_values : tuple[float, ...]
        Singular values of the (column-normalized) Jacobian, descending.
    rank : int
        Numerical rank (singular values above ``rank_tol``).
    condition_number : float
        ``σ_max / σ_min`` of the normalized Jacobian (``inf`` if rank-deficient).
    identifiable : bool
        True iff full column rank **and** ``condition_number <= cond_max``.
    weak_params : tuple[str, ...]
        Parameters the excitation constrains only weakly (largest marginal
        variance), even when the fit is nominally full-rank.
    cond_max : float
        The conditioning ceiling this report was judged against.
    """

    names: tuple[str, ...]
    singular_values: tuple[float, ...]
    rank: int
    condition_number: float
    identifiable: bool
    weak_params: tuple[str, ...]
    cond_max: float

    def raise_if_unidentifiable(self, context: str) -> None:
        """Raise :class:`CalibrationError` if the excitation is not identifiable.

        Args
        ----
        context : str
            Short label naming the excitation/recipe, woven into the message.

        Raises
        ------
        CalibrationError
            If ``not self.identifiable`` — names the rank deficit / condition
            number and the weak parameters, and suggests a richer excitation.
        """
        if self.identifiable:
            return
        deficit = len(self.names) - self.rank
        raise CalibrationError(
            f"{context}: the excitation is not identifiable — rank {self.rank} of "
            f"{len(self.names)} parameters (deficit {deficit}), condition number "
            f"{self.condition_number:.3g} > {self.cond_max:.3g}. Weakly/under-"
            f"constrained: {', '.join(self.weak_params) or '(rank-deficient)'}. "
            "Fix: use a richer excitation (wider frequency sweep / zero crossings "
            "/ longer horizon) that moves every parameter, or hold the "
            "unidentifiable parameters fixed."
        )


def identifiability_from_jacobian(
    jacobian,
    names: tuple[str, ...],
    *,
    cond_max: float = _DEFAULT_COND_MAX,
    rank_tol: float | None = None,
    weak_var_ratio: float = 1e3,
) -> IdentifiabilityReport:
    """Identifiability report from a residual/output sensitivity Jacobian.

    Column-normalizes ``J`` (so the conditioning reflects *relative* parameter
    sensitivity, not unit scale), takes its SVD for rank + condition number, and
    flags parameters whose marginal variance (the diagonal of
    ``(JᵀJ)⁻¹`` on the normalized Jacobian) exceeds ``weak_var_ratio`` × the
    smallest — the ones the excitation barely moves.

    Args
    ----
    jacobian : array-like, shape ``(n_obs, n_params)``
        Sensitivity of the residual (or predicted output) to each parameter.
    names : tuple[str, ...]
        Parameter names in column order.
    cond_max : float
        Conditioning ceiling for the ``identifiable`` verdict.
    rank_tol : float or None
        Singular-value floor for the rank count. Defaults to
        ``max(shape) · eps · σ_max`` (NumPy's ``matrix_rank`` convention).
    weak_var_ratio : float
        A parameter is "weak" if its normalized marginal variance exceeds this
        multiple of the smallest parameter's.

    Returns
    -------
    IdentifiabilityReport
    """
    jac = np.asarray(jacobian, dtype=np.float64)
    if jac.ndim != 2:
        jac = jac.reshape(jac.shape[0], -1)
    if jac.shape[1] != len(names):
        raise CalibrationError(
            f"identifiability: Jacobian has {jac.shape[1]} parameter columns but "
            f"{len(names)} names were given — they must match."
        )
    # Column-normalize so conditioning measures relative, not unit-scale, spread.
    col_norm = np.linalg.norm(jac, axis=0)
    col_norm = np.where(col_norm > 0, col_norm, 1.0)
    jn = jac / col_norm
    sv = np.linalg.svd(jn, compute_uv=False)
    sv_max = float(sv[0]) if sv.size else 0.0
    tol = (
        rank_tol
        if rank_tol is not None
        else max(jn.shape) * np.finfo(float).eps * sv_max
    )
    rank = int(np.sum(sv > tol))
    sv_min = float(sv[rank - 1]) if rank > 0 else 0.0
    condition_number = float(sv_max / sv_min) if sv_min > 0 else float("inf")
    # Marginal variance of each (normalized) parameter — large ⟹ weakly excited.
    weak: tuple[str, ...] = ()
    if rank == jn.shape[1] and jn.shape[1] > 0:
        var = np.diag(np.linalg.pinv(jn.T @ jn))
        ref = float(np.min(var))
        if ref > 0:
            weak = tuple(
                names[i] for i in range(len(names)) if var[i] > weak_var_ratio * ref
            )
    identifiable = rank == len(names) and condition_number <= cond_max
    return IdentifiabilityReport(
        names=tuple(names),
        singular_values=tuple(float(s) for s in sv),
        rank=rank,
        condition_number=condition_number,
        identifiable=identifiable,
        weak_params=weak,
        cond_max=cond_max,
    )


# --- DC motor --------------------------------------------------------------

_DC_PARAMS = ("Ks", "Kv", "Ka")


def design_dc_motor_excitation(
    *,
    fs: float = 200.0,
    duration: float = 2.0,
    v_peak: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Design an identifiable velocity profile for a DC-motor SysId run.

    Returns a velocity trajectory the actuator should be driven through (by the
    implied voltage). It is a growing-amplitude sinusoid so the velocity
    **crosses zero** (exciting both signs of ``sgn(v)``) and its **acceleration
    varies** (so ``a`` is not collinear with ``sgn(v)``) — the two conditions a
    pure linear ramp violates, leaving ``Ks`` and ``Ka`` unidentifiable.

    Args
    ----
    fs : float
        Sample rate, Hz.
    duration : float
        Profile length, s.
    v_peak : float
        Peak velocity, rad/s.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(t, v)`` — timestamps (s) and velocity (rad/s), each shape ``(n,)``.

    Units
    -----
    Time in seconds, velocity in rad/s.
    """
    n = int(fs * duration) + 1
    t = np.arange(n, dtype=np.float64) / fs
    # Two cycles over the window, amplitude ramping in → zero crossings + curvature.
    v = v_peak * (0.2 + 0.8 * t / duration) * np.sin(2.0 * np.pi * 2.0 * t / duration)
    return t, v


def _dc_design_matrix(
    v: np.ndarray, a: np.ndarray, *, v_eps: float = 1e-3
) -> np.ndarray:
    """The ``[sgn(v), v, a]`` regressor (smoothed sign, matching the model)."""
    return np.stack([np.tanh(v / v_eps), v, a], axis=1)


def dc_motor_identifiability(
    telemetry: ActuatorRampTelemetry,
    *,
    cond_max: float = _DEFAULT_COND_MAX,
) -> IdentifiabilityReport:
    """Identifiability of a DC-motor voltage-ramp excitation.

    The fit is linear in ``(Ks, Kv, Ka)``, so the residual Jacobian *is* the
    design matrix ``[sgn(v), v, a]`` (``a`` by central difference). A constant-
    acceleration ramp that never reverses gives ``sgn(v)`` and ``a`` constant —
    collinear — and the report is rank-deficient.

    Args
    ----
    telemetry : ActuatorRampTelemetry
        Voltage-ramp telemetry (``command`` = applied voltage, ``velocity``).
    cond_max : float
        Conditioning ceiling.

    Returns
    -------
    IdentifiabilityReport
    """
    v = np.asarray(telemetry.velocity, dtype=np.float64)
    t = np.asarray(telemetry.time, dtype=np.float64)
    a = np.gradient(v, t)
    jac = _dc_design_matrix(v, a)
    return identifiability_from_jacobian(jac, _DC_PARAMS, cond_max=cond_max)


def dc_motor_recipe(
    telemetry: ActuatorRampTelemetry,
    device: DCMotorWithFriction | None = None,
    *,
    cond_max: float = _DEFAULT_COND_MAX,
    require_identifiable: bool = True,
) -> SysidResult:
    """Fit a DC motor, gating on excitation identifiability first.

    Checks the voltage-ramp excitation with :func:`dc_motor_identifiability` and
    — when ``require_identifiable`` — **raises** on a non-identifiable profile
    before fitting; otherwise it folds a weak-identifiability note into the
    result's warnings. The fit itself is the shared
    :func:`jaxterity.sysid.actuator` closed-form recovery.

    Args
    ----
    telemetry : ActuatorRampTelemetry
        Voltage-ramp telemetry.
    device : DCMotorWithFriction or None
        The device to identify; defaults to ``DCMotorWithFriction.uncalibrated()``.
    cond_max : float
        Conditioning ceiling for the identifiability gate.
    require_identifiable : bool
        Raise on a non-identifiable excitation (default) vs. warn and proceed.

    Returns
    -------
    SysidResult
        The fitted ``(Ks, Kv, Ka)`` plus any weak-identifiability warning.

    Raises
    ------
    CalibrationError
        If ``require_identifiable`` and the excitation is not identifiable.
    """
    device = device or DCMotorWithFriction.uncalibrated()
    report = dc_motor_identifiability(telemetry, cond_max=cond_max)
    if require_identifiable:
        report.raise_if_unidentifiable("dc_motor_recipe")
    result = _actuator_fit(telemetry, device)
    if report.weak_params:
        note = (
            f"dc_motor_recipe: weakly identified under this excitation: "
            f"{', '.join(report.weak_params)} (condition number "
            f"{report.condition_number:.3g}). Treat them as loosely constrained."
        )
        result = SysidResult(
            fitted=result.fitted,
            covariance=result.covariance,
            residuals=result.residuals,
            telemetry_hash=result.telemetry_hash,
            recipe_name=result.recipe_name,
            recipe_version=result.recipe_version,
            warnings=(*result.warnings, note),
        )
    return result


# --- BLDC ------------------------------------------------------------------

_BLDC_PARAMS = ("R", "L", "Kt", "Ke", "Kp_i", "J", "b", "Fc")


def design_bldc_excitation(
    *,
    fs: float = 2000.0,
    duration: float = 1.0,
    f0: float = 0.5,
    f1: float = 50.0,
) -> np.ndarray:
    """Design a current-setpoint chirp for a BLDC SysId run.

    A linear chirp sweeping ``f0 → f1`` Hz over ``duration``, the excitation the
    BLDC recipe's identifiability check and fit both expect. The sweep must span
    the electrical bandwidth ``1/(2π·L/R)`` to constrain the electrical split.

    Args
    ----
    fs : float
        Sample rate, Hz (≥ 1 kHz to resolve the electrical time constant).
    duration : float
        Chirp length, s.
    f0, f1 : float
        Start / end chirp frequency, Hz.

    Returns
    -------
    numpy.ndarray, shape ``(n,)``
        The current setpoint ``i_ref(t)`` (A), to drive over the bus.
    """
    n = int(fs * duration) + 1
    t = np.arange(n, dtype=np.float64) / fs
    k = (f1 - f0) / duration
    return np.sin(2.0 * np.pi * (f0 * t + 0.5 * k * t * t))


def _bldc_predict(params, iref, t, *, v_eps: float = 1e-3):
    """Forward-mode-friendly BLDC rollout → stacked ``[v; i]`` (jnp).

    The same substepped RK4 the fitter / ``BLDC.validate`` use, written so
    :func:`jax.jacfwd` traces it (smoothed Coulomb friction, stiff current loop).
    """
    R, L, Kt, Ke, Kp_i, J, b, Fc = (params[i] for i in range(8))
    iref = jnp.asarray(iref, dtype=float)
    t = jnp.asarray(t, dtype=float)
    n_sub = 8

    def rhs(y, ir):
        v, i = y[1], y[2]
        didt = (Kp_i * (ir - i) - R * i - Ke * v) / L
        dvdt = (Kt * i - b * v - Fc * jnp.tanh(v / v_eps)) / J
        return jnp.array([v, dvdt, didt])

    def step(y, k):
        dt = t[k + 1] - t[k]
        sub = dt / n_sub

        def one(yy, _):
            k1 = rhs(yy, iref[k])
            k2 = rhs(yy + 0.5 * sub * k1, iref[k])
            k3 = rhs(yy + 0.5 * sub * k2, iref[k])
            k4 = rhs(yy + sub * k3, iref[k])
            return yy + (sub / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), None

        y, _ = jax.lax.scan(one, y, None, length=n_sub)
        return y, jnp.array([y[1], y[2]])

    y0 = jnp.zeros(3)
    _, out = jax.lax.scan(step, y0, jnp.arange(t.shape[0] - 1))
    # out: (n-1, 2) of (v, i); flatten to a single residual-shaped vector.
    return out.reshape(-1)


def bldc_identifiability(
    device: BLDC,
    telemetry: ActuatorChirpTelemetry,
    *,
    cond_max: float = _DEFAULT_COND_MAX,
) -> IdentifiabilityReport:
    """Identifiability of a BLDC chirp, by forward-mode sensitivity.

    Differentiates the predicted ``(v, i)`` trajectory w.r.t. the eight
    parameters with :func:`jax.jacfwd` at the device's current values, and reads
    rank + conditioning off the result. A free-shaft chirp typically leaves the
    electrical split weakly identified — surfaced as ``weak_params`` — while the
    mechanical ratio ``Kt/J`` is well constrained.

    Args
    ----
    device : BLDC
        The device whose parameters define the linearization point.
    telemetry : ActuatorChirpTelemetry
        Current-chirp telemetry (``setpoint``, ``velocity``, ``current``).
    cond_max : float
        Conditioning ceiling.

    Returns
    -------
    IdentifiabilityReport
    """
    params = jnp.array(
        [
            float(device.R),
            float(device.L),
            float(device.Kt),
            float(device.Ke),
            float(device.Kp_i),
            float(device.J),
            float(device.b),
            float(device.Fc),
        ]
    )
    iref = np.asarray(telemetry.setpoint, dtype=np.float64)
    t = np.asarray(telemetry.time, dtype=np.float64)
    jac = jax.jacfwd(lambda p: _bldc_predict(p, iref, t))(params)
    return identifiability_from_jacobian(
        np.asarray(jac), _BLDC_PARAMS, cond_max=cond_max
    )


def bldc_recipe(
    telemetry: ActuatorChirpTelemetry,
    device: BLDC | None = None,
    *,
    cond_max: float = _DEFAULT_COND_MAX,
    require_identifiable: bool = False,
    hints: dict | None = None,
) -> SysidResult:
    """Fit a BLDC, reporting excitation identifiability.

    Runs :func:`bldc_identifiability` and folds the weak-parameter set into the
    result's warnings (the electrical split is usually weak under a free-shaft
    chirp — that is expected, so ``require_identifiable`` defaults to ``False``;
    set it to gate on a fully-identifiable excitation). The fit is the shared
    :func:`jaxterity.sysid.actuator` Levenberg-Marquardt recovery.

    Args
    ----
    telemetry : ActuatorChirpTelemetry
        Current-chirp telemetry.
    device : BLDC or None
        The device to identify; defaults to ``BLDC.uncalibrated()``.
    cond_max : float
        Conditioning ceiling for the identifiability report.
    require_identifiable : bool
        Raise on a non-identifiable excitation. Default ``False`` (the electrical
        split is legitimately weak under a free-shaft chirp).
    hints : dict or None
        Optimizer hints forwarded to :func:`jaxterity.sysid.actuator`.

    Returns
    -------
    SysidResult
        The fitted eight parameters plus a weak-identifiability warning.

    Raises
    ------
    CalibrationError
        If ``require_identifiable`` and the excitation is not identifiable.
    """
    device = device or BLDC.uncalibrated()
    report = bldc_identifiability(device, telemetry, cond_max=cond_max)
    if require_identifiable:
        report.raise_if_unidentifiable("bldc_recipe")
    result = _actuator_fit(telemetry, device, hints=hints)
    if report.weak_params:
        note = (
            f"bldc_recipe: weakly identified under this chirp: "
            f"{', '.join(report.weak_params)} (condition number "
            f"{report.condition_number:.3g}). The mechanical ratio Kt/J is "
            "constrained; treat the electrical split as loosely identified."
        )
        result = SysidResult(
            fitted=result.fitted,
            covariance=result.covariance,
            residuals=result.residuals,
            telemetry_hash=result.telemetry_hash,
            recipe_name=result.recipe_name,
            recipe_version=result.recipe_version,
            warnings=(*result.warnings, note),
        )
    return result


__all__ = [
    "IdentifiabilityReport",
    "identifiability_from_jacobian",
    "design_dc_motor_excitation",
    "dc_motor_identifiability",
    "dc_motor_recipe",
    "design_bldc_excitation",
    "bldc_identifiability",
    "bldc_recipe",
]
