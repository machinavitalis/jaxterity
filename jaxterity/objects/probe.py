# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Differentiable probe design: gradient-optimize the heft itself.

Not every poke is equally informative. :func:`design_probe` differentiates the
probe's **predicted Fisher information** about the payload parameters with
respect to the probe itself (the coast excitation's initial state) and ascends
it — differentiable optimal experiment design at manipulation time. The
information matrix is the Gram matrix ``JᵀJ`` of the same sensitivity Jacobian
:func:`~jaxterity.objects.payload_identifiability` scores and the LM fit
iterates on, so "informative" here means *for this fit, exactly*.

With a ``prior_covariance`` from a previous fit, the objective becomes the
Bayesian D-optimal ``logdet(JᵀJ + P_prior)``: the ascent then pushes the probe
toward the parameter directions the *last* fit left widest — the
probe → fit → next-probe loop.

Everything is deterministic (fixed step count, no RNG) and ends at the same
hard gate as any excitation: the returned probe passes
:func:`~jaxterity.calibrate.active.validate_excitation`, with the candidate
scaled back toward rest if the ascent walked it out of limits.

Second-order autodiff note: the objective's gradient is
forward-over-forward (``jacfwd`` of a scalar built from an inner ``jacfwd``),
so it traverses MJX's solver the same way the forward-mode fit does — no
reverse-mode pass, no adjoint requirements.

Frame
-----
The probe lives in joint space: ``q0`` (rad) / ``qd0`` (rad/s) in the robot's
DOF order.

Units
-----
rad, rad/s; the information matrix is scale-free (columns normalized by the
parameter scales), so the objective is unchanged by unit choices.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from jaxonomy import SimulatorOptions

from ..calibrate.active import (
    _DEFAULT_OPTIONS,
    Excitation,
    design_coast_excitation,
    dof_joint_names,
    validate_excitation,
)
from ..errors import CalibrationError
from ..robot import Robot
from .payload import RigidPayload, _fit_layout, attach_payload

#: Fraction of a joint's limit the optimized probe may use (same spirit as the
#: lane's ``_LIMIT_MARGIN``, slightly tighter because the ascent explores).
_PROBE_MARGIN = 0.8
#: Scale floors so a zero prior (e.g. a CoM at the origin) still gets a
#: sensible column scale: mass kg, CoM m, principal inertia kg·m².
_SCALE_FLOORS = {"mass": 1e-2, "com": 0.05, "inertia_diag": 1e-5}


def _param_scales(payload: RigidPayload, fit: tuple[str, ...]) -> np.ndarray:
    """Per-scalar column scales: |prior| floored per group (units-normalizing)."""
    _, prior, _ = _fit_layout(payload, fit)
    scales: list[float] = []
    k = 0
    if "mass" in fit:
        scales.append(max(abs(float(prior[k])), _SCALE_FLOORS["mass"]))
        k += 1
    if "com" in fit:
        for i in range(3):
            scales.append(max(abs(float(prior[k + i])), _SCALE_FLOORS["com"]))
        k += 3
    if "inertia_diag" in fit:
        for i in range(3):
            scales.append(max(abs(float(prior[k + i])), _SCALE_FLOORS["inertia_diag"]))
        k += 3
    return np.asarray(scales, dtype=float)


def _state_observer(
    attached: Robot,
    payload: RigidPayload,
    fit: tuple[str, ...],
    sample_times: tuple[float, ...],
    options: SimulatorOptions,
):
    """``observe(flat_params, x0) -> stacked [q, qd]`` — the payload observer
    with the probe's initial state threaded as an input (so the objective can
    differentiate w.r.t. the probe)."""
    import jaxonomy

    diagram, system = attached.to_diagram(return_system=True)
    sid = system.system_id
    name = payload.name

    @jax.jit
    def observe(flat, x0):
        ctx = diagram.create_context()
        sub = ctx[sid]
        k = 0
        if "mass" in fit:
            sub = sub.with_parameter(f"{name}.mass", flat[k])
            k += 1
        if "com" in fit:
            sub = sub.with_parameter(f"{name}.com", flat[k : k + 3])
            k += 3
        if "inertia_diag" in fit:
            sub = sub.with_parameter(f"{name}.inertia_diag", flat[k : k + 3])
            k += 3
        sub = sub.with_continuous_state(x0)
        ctx = ctx.with_subcontext(sid, sub)
        states = [
            jaxonomy.simulate(diagram, ctx, (0.0, t), options=options)
            .context[sid]
            .continuous_state
            for t in sample_times
        ]
        return jnp.concatenate(states)

    return observe


def probe_information(
    robot: Robot,
    payload: RigidPayload,
    excitation: Excitation,
    *,
    fit: tuple[str, ...] = ("mass", "com"),
    prior_covariance: dict[str, float] | None = None,
    options: SimulatorOptions | None = None,
) -> float:
    """The (Bayesian) D-optimality of a probe: ``logdet(JᵀJ + P_prior)``.

    The scalar :func:`design_probe` ascends, exposed for comparing candidate
    probes and for tests. Columns of ``J`` are normalized by the payload's
    prior scales, so the value is unit-free; larger is more informative.

    Args
    ----
    robot : Robot
        The bare robot (the payload is attached internally).
    payload : RigidPayload
        The object and its priors (the linearization point).
    excitation : Excitation
        The candidate probe.
    fit : tuple[str, ...]
        Parameter groups the fit will target.
    prior_covariance : dict[str, float] or None
        Per-scalar marginal variance from a previous fit
        (``SysidResult.covariance`` keys); folds in as a diagonal prior
        precision. ``None`` uses a small ridge for definiteness.
    options : SimulatorOptions | None
        Integration options (defaults to the lane's fixed-step RK4).

    Returns
    -------
    float
        ``logdet`` of the scaled information matrix.

    Units
    -----
    Dimensionless (scale-normalized).
    """
    options = options or _DEFAULT_OPTIONS
    fit = tuple(fit)
    attached = attach_payload(robot, payload)
    names, prior, _ = _fit_layout(payload, fit)
    scales = _param_scales(payload, fit)
    observe = _state_observer(
        attached, payload, fit, tuple(excitation.sample_times), options
    )
    x0 = jnp.concatenate(
        [
            jnp.asarray(excitation.q0, dtype=float),
            jnp.asarray(excitation.qd0, dtype=float),
        ]
    )
    jac = jax.jit(jax.jacfwd(observe, argnums=0))(jnp.asarray(prior), x0)
    scaled = np.asarray(jac) * scales[None, :]
    info = scaled.T @ scaled + _prior_precision(names, scales, prior_covariance)
    sign, logdet = np.linalg.slogdet(info)
    return float(logdet) if sign > 0 else float("-inf")


def _prior_precision(
    names: tuple[str, ...],
    scales: np.ndarray,
    prior_covariance: dict[str, float] | None,
) -> np.ndarray:
    """Diagonal prior precision in scaled-parameter units (ridge if no prior)."""
    if prior_covariance is None:
        return 1e-9 * np.eye(len(names))
    precision = np.zeros(len(names))
    for i, name in enumerate(names):
        var = float(prior_covariance.get(name, np.inf))
        # Scaled parameter = theta / scale, so var_scaled = var / scale².
        precision[i] = (
            0.0 if not np.isfinite(var) or var <= 0.0 else scales[i] ** 2 / var
        )
    return np.diag(precision) + 1e-9 * np.eye(len(names))


def design_probe(
    robot: Robot,
    payload: RigidPayload,
    *,
    fit: tuple[str, ...] = ("mass", "com"),
    excitation0: Excitation | None = None,
    objective: str = "D-optimal",
    prior_covariance: dict[str, float] | None = None,
    steps: int = 12,
    rate: float = 0.25,
    margin: float = _PROBE_MARGIN,
    optimize_q0: bool = True,
    options: SimulatorOptions | None = None,
) -> Excitation:
    """Gradient-optimize a coast probe for payload identifiability.

    Ascends the probe's predicted Fisher information (``logdet`` of the scaled
    ``JᵀJ``, plus the prior precision when ``prior_covariance`` is given) with
    respect to the probe's initial joint state, by forward-over-forward
    autodiff through the simulation. Deterministic: a fixed number of
    projected-gradient steps, no RNG — the same inputs return the same probe.

    Each iterate is projected into the joint limit box (velocities to
    ``±margin × velocity_limit``, positions to ``margin ×`` the position
    range; unlimited joints are left free), and the final candidate must pass
    :func:`~jaxterity.calibrate.active.validate_excitation` — if the ascent
    walked it out of the simulated-trajectory limits, it is scaled back
    toward the starting probe until it validates.

    Args
    ----
    robot : Robot
        The bare robot (the payload is attached internally).
    payload : RigidPayload
        The object and its priors.
    fit : tuple[str, ...]
        Parameter groups the eventual fit targets (the information is about
        exactly these).
    excitation0 : Excitation | None
        Starting probe. Defaults to
        :func:`~jaxterity.calibrate.active.design_coast_excitation`.
    objective : str
        ``"D-optimal"`` (the only recognized objective — named so the loop
        reads honestly at call sites).
    prior_covariance : dict[str, float] or None
        A previous fit's per-scalar variance (``SysidResult.covariance``):
        turns the objective Bayesian, targeting the directions the last fit
        left widest.
    steps : int
        Projected-gradient ascent steps.
    rate : float
        Ascent step size (on the unit-scaled state).
    margin : float
        Fraction of each joint limit the probe may use.
    optimize_q0 : bool
        Also optimize the starting pose (default) — gravity/inertia coupling
        varies with configuration, and that variation is informative.
    options : SimulatorOptions | None
        Integration options (defaults to the lane's fixed-step RK4).

    Returns
    -------
    Excitation
        The optimized, validated probe (same sample times as the start).

    Raises
    ------
    CalibrationError
        If ``objective`` is not recognized, or no valid probe can be produced
        (even the starting probe fails validation).

    Units
    -----
    rad, rad/s.
    """
    if objective != "D-optimal":
        raise CalibrationError(
            f"design_probe: unknown objective {objective!r}. The recognized "
            "objective is 'D-optimal' (logdet of the scaled information matrix)."
        )
    options = options or _DEFAULT_OPTIONS
    fit = tuple(fit)
    excitation0 = excitation0 or design_coast_excitation(robot)
    attached = attach_payload(robot, payload)
    names, prior, _ = _fit_layout(payload, fit)
    scales = _param_scales(payload, fit)
    precision = jnp.asarray(_prior_precision(names, scales, prior_covariance))
    scales_j = jnp.asarray(scales)
    observe = _state_observer(
        attached, payload, fit, tuple(excitation0.sample_times), options
    )
    prior_j = jnp.asarray(prior)

    def logdet(x0):
        jac = jax.jacfwd(observe, argnums=0)(prior_j, x0)
        scaled = jac * scales_j[None, :]
        info = scaled.T @ scaled + precision
        sign, value = jnp.linalg.slogdet(info)
        return value

    grad_fn = jax.jit(jax.jacfwd(logdet))

    # Limit boxes in DOF order (0-limit ⇒ unlimited, mirroring the validator).
    jmap = robot.joint_map
    dof_names = dof_joint_names(robot)
    v_hi = np.asarray(
        [
            margin * jmap[n].limits.velocity if jmap[n].limits.velocity > 0 else np.inf
            for n in dof_names
        ]
    )
    q_lo = np.asarray(
        [
            margin * jmap[n].limits.lower
            if (jmap[n].limits.lower, jmap[n].limits.upper) != (0.0, 0.0)
            else -np.inf
            for n in dof_names
        ]
    )
    q_hi = np.asarray(
        [
            margin * jmap[n].limits.upper
            if (jmap[n].limits.lower, jmap[n].limits.upper) != (0.0, 0.0)
            else np.inf
            for n in dof_names
        ]
    )

    nv = len(dof_names)
    q0 = np.asarray(excitation0.q0, dtype=float)
    qd0 = np.asarray(excitation0.qd0, dtype=float)
    for _ in range(int(steps)):
        x0 = jnp.concatenate([jnp.asarray(q0), jnp.asarray(qd0)])
        g = np.asarray(grad_fn(x0))
        if not np.all(np.isfinite(g)):
            break  # ascent left the well-behaved region; keep the last iterate
        if optimize_q0:
            q0 = np.clip(q0 + rate * g[:nv], q_lo, q_hi)
        qd0 = np.clip(qd0 + rate * g[nv:], -v_hi, v_hi)

    # Hard gate: the optimized probe must validate; otherwise walk it back
    # toward the starting probe (deterministically) until it does.
    q0_start = np.asarray(excitation0.q0, dtype=float)
    qd0_start = np.asarray(excitation0.qd0, dtype=float)
    for k in range(9):
        blend = 0.5**k if k else 1.0
        candidate = dataclasses.replace(
            excitation0,
            q0=tuple(float(v) for v in (q0_start + blend * (q0 - q0_start))),
            qd0=tuple(float(v) for v in (qd0_start + blend * (qd0 - qd0_start))),
        )
        try:
            validate_excitation(attached, candidate, options=options)
            return candidate
        except CalibrationError:
            continue
    # blend → 0 reduces to the starting probe; if even that fails, say so.
    validate_excitation(attached, excitation0, options=options)
    return excitation0


__all__ = ["design_probe", "probe_information"]
