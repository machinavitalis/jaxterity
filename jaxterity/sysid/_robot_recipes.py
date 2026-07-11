# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Robot-level sysid recipes — inertial, full_body, contact.

These recipes fit Robot-level identifiable parameters (link.mass, joint.damping,
joint.friction) from :class:`~jaxterity.data.schema.JointTelemetry`. They share
a forward-mode Levenberg-Marquardt loop (the constraint solver inside MJX uses
``lax.while_loop``, which reverse-mode autodiff cannot traverse — see ADR-017;
forward-mode does, via :func:`jax.jacfwd`).

The private module name avoids the public-attribute / submodule collision the
same way :mod:`jaxterity.sysid._actuator` does; the public names are exposed
lazily through ``jaxterity/sysid/__init__.py``'s ``__getattr__``.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import jaxonomy
import numpy as np
from jaxonomy import SimulatorOptions

from ..data.schema import JointTelemetry
from ..errors import JaxterityError, JaxterityWarning
from ..robot import Robot
from .result import SysidResult

#: Default integrator options for forward-mode robot identification — fixed-step
#: RK4 with the custom adjoint off (forward-mode must trace the real ops).
_OPTIONS = SimulatorOptions(
    math_backend="jax",
    ode_solver_method="rk4",
    enable_autodiff=False,
    max_minor_step_size=0.005,
    max_major_steps=400,
)

INERTIAL_RECIPE = "sysid.inertial"
FULL_BODY_RECIPE = "sysid.full_body"
CONTACT_RECIPE = "sysid.contact"
RECIPE_VERSION = "1.0"


def _make_observer(robot: Robot, target_params: list[str], q0, qd0, sample_times):
    """Return a ``observe(theta_vec) -> stacked_states`` JAX-jitted closure."""
    diagram, system = robot.to_diagram(q0=q0, qd0=qd0, return_system=True)
    sid = system.system_id

    @jax.jit
    def observe(theta):
        ctx = diagram.create_context()
        sub = ctx[sid]
        for name, value in zip(target_params, theta):
            sub = sub.with_parameter(name, value)
        ctx = ctx.with_subcontext(sid, sub)
        states = [
            jaxonomy.simulate(diagram, ctx, (0.0, t), options=_OPTIONS)
            .context[sid]
            .continuous_state
            for t in sample_times
        ]
        return jnp.concatenate(states)

    return observe


def _fm_lm(telemetry_states, observe, theta0, *, iterations=10, lam=1e-2, max_step=0.5):
    """Forward-mode Levenberg-Marquardt — same algorithm as examples/sysid_so101.py.

    Damped Gauss-Newton step:
        Δ = -(JᵀJ + λ diag(JᵀJ))⁻¹ Jᵀr
    clipped by ``max_step`` per coordinate.
    """
    observe_jac = jax.jit(jax.jacfwd(observe))
    est = jnp.asarray(theta0, dtype=float)
    history = []
    for _ in range(iterations):
        residual = np.asarray(observe(est)) - telemetry_states
        jac = np.asarray(observe_jac(est))
        history.append(float(residual @ residual))
        hessian = jac.T @ jac
        damped = hessian + lam * np.diag(np.diag(hessian) + 1e-12)
        step = np.clip(-np.linalg.solve(damped, jac.T @ residual), -max_step, max_step)
        est = est + step
    return np.asarray(est), history


def _stack_states_from_telemetry(telemetry: JointTelemetry) -> np.ndarray:
    """Telemetry → ``concat([q, qd])`` flattened to match the observer output."""
    q = np.asarray(telemetry.q, dtype=np.float64)
    qd = np.asarray(telemetry.qd, dtype=np.float64)
    return np.concatenate([q, qd], axis=1).flatten()


def _recipe(
    recipe_name: str,
    target_params: list[str],
    telemetry: JointTelemetry,
    robot: Robot,
    *,
    theta0: list[float] | None,
    hints: dict | None,
) -> SysidResult:
    """Shared body — target_params is the list of '<comp>.<field>' to fit."""
    if not isinstance(telemetry, JointTelemetry):
        raise JaxterityError(
            f"{recipe_name} requires JointTelemetry; got {type(telemetry).__name__}."
        )
    hints = hints or {}
    n = len(target_params)
    # Sample times: every telemetry timestamp the observer should match.
    sample_times = np.asarray(telemetry.time, dtype=np.float64)
    # Initial state: first telemetry sample.
    q0 = np.asarray(telemetry.q[0], dtype=np.float64)
    qd0 = np.asarray(telemetry.qd[0], dtype=np.float64)
    observe = _make_observer(
        robot, target_params, q0=q0, qd0=qd0, sample_times=sample_times[1:]
    )
    truth_states = _stack_states_from_telemetry(telemetry)
    # Observer outputs samples 1..N (initial state matches q0/qd0 by construction).
    # Drop the first sample from the truth so the shapes match.
    n_state = len(q0) + len(qd0)
    truth = truth_states.reshape(-1, n_state)[1:].flatten()
    theta0_vec = jnp.zeros(n) if theta0 is None else jnp.asarray(theta0, dtype=float)
    iterations = int(hints.get("iterations", 8))
    est, history = _fm_lm(truth, observe, theta0_vec, iterations=iterations)
    fitted = {name: jnp.asarray(float(v)) for name, v in zip(target_params, est)}
    residuals = {"state_cost": float(history[-1]) if history else 0.0}
    stall = _recipe_stall_warnings(recipe_name, target_params, theta0_vec, est)
    return SysidResult(
        fitted=fitted,
        residuals=residuals,
        telemetry_hash=telemetry.hash().hex(),
        recipe_name=recipe_name,
        recipe_version=RECIPE_VERSION,
        warnings=tuple(stall),
    )


def _recipe_stall_warnings(
    recipe_name: str, target_params: list[str], theta0, est
) -> list[str]:
    """Flag forward-mode-LM parameters that never moved off the initial guess.

    The silent-no-op failure mode: ``frictionloss`` is non-smooth at
    velocity zero crossings, so its forward-mode Jacobian column is ~zero and
    the LM step doesn't move it — the recipe would otherwise return the warm
    start as if it were a fit. Uses an absolute + relative threshold so a
    parameter whose initial guess is 0.0 (a common friction prior) is handled.
    """
    theta0_arr = np.asarray(theta0, dtype=float)
    est_arr = np.asarray(est, dtype=float)
    moved = np.abs(est_arr - theta0_arr) > (1e-6 + 1e-4 * np.abs(theta0_arr))
    stalled = [p for p, m in zip(target_params, moved) if not m]
    if not stalled:
        return []
    return [
        f"{recipe_name}: parameters {stalled} did not move off the initial guess "
        "(forward-mode gradient ~zero — e.g. frictionloss is non-smooth at "
        "velocity zero crossings). Treat them as unidentified; for joint friction "
        "prefer fit_damping=True, which has a clean gradient."
    ]


# --- public recipe entry points -------------------------------------------


def inertial(
    telemetry: JointTelemetry,
    robot: Robot,
    *,
    link_names: list[str] | None = None,
    theta0: list[float] | None = None,
    hints: dict | None = None,
) -> SysidResult:
    """Fit per-link masses from joint telemetry.

    Args
    ----
    telemetry : JointTelemetry
        Joint positions and velocities over time. The arm should follow a
        chirp / sweep that excites the mass-sensitive modes (see CONTEXT.md
        §sysid: a fast trajectory at the joint of interest with the link
        below it loaded).
    robot : Robot
        The robot whose link masses we identify. Uncalibrated values are the
        initial guess unless ``theta0`` is supplied.
    link_names : list[str] or None
        Which links to identify. Defaults to every link except ``"world"`` /
        ``"base"`` if present.
    theta0 : list[float] or None
        Initial guesses, in ``link_names`` order. Defaults to the robot's
        current values.
    hints : dict or None
        Optimizer hints; supported: ``iterations`` (default 8).

    Scope: per ADR-017, ``com`` and ``inertia`` are
    not runtime-threaded for autodiff — they are refit by structural rebuild
    via :meth:`Robot.with_parameters` only. This recipe identifies mass; com
    + inertia stay at the prior values.
    """
    if link_names is None:
        link_names = [
            link.name for link in robot.links if link.name not in ("world", "base")
        ]
    target_params = [f"{name}.mass" for name in link_names]
    if theta0 is None:
        masses = {link.name: float(link.mass) for link in robot.links}
        theta0 = [masses[name] for name in link_names]
    return _recipe(
        INERTIAL_RECIPE,
        target_params,
        telemetry,
        robot,
        theta0=theta0,
        hints=hints,
    )


def full_body(
    telemetry: JointTelemetry,
    robot: Robot,
    *,
    fit_mass: bool = True,
    fit_damping: bool = True,
    theta0: list[float] | None = None,
    hints: dict | None = None,
) -> SysidResult:
    """Fit link masses AND joint dampings jointly from joint telemetry.

    Use when per-link mocap is not available — the only signal is the joint
    state, and both inertial and friction-like parameters must be identified
    together. Defaults to fitting both families; either can be turned off.
    """
    target_params: list[str] = []
    init: list[float] = []
    if fit_mass:
        for link in robot.links:
            if link.name in ("world", "base"):
                continue
            target_params.append(f"{link.name}.mass")
            init.append(float(link.mass))
    if fit_damping:
        for joint in robot.joints:
            try:
                target_params.append(f"{joint.name}.damping")
                init.append(float(joint.damping))
            except AttributeError:
                continue
    if not target_params:
        raise JaxterityError(
            "sysid.full_body: nothing to fit — pass fit_mass=True or fit_damping=True."
        )
    return _recipe(
        FULL_BODY_RECIPE,
        target_params,
        telemetry,
        robot,
        theta0=theta0 or init,
        hints=hints,
    )


def contact(
    telemetry: JointTelemetry,
    robot: Robot,
    *,
    joint_names: list[str] | None = None,
    fit_friction: bool = False,
    fit_damping: bool = True,
    force: bool = False,
    theta0: list[float] | None = None,
    hints: dict | None = None,
) -> SysidResult:
    """Fit per-joint friction / damping from joint telemetry.

    The recipe's "contact" name reflects its long-term scope: identifying the
    friction/contact parameters of a robot from impact-rich telemetry.

    **Default fits viscous damping** (MJX's ``damping``), which has a clean
    forward-mode gradient and recovers truth (verified in the recipe test).

    **Coulomb friction (``fit_friction=True``) is a known-weak path.**
    Forward-mode autodiff sees near-zero gradient through ``frictionloss``
    (``sign(v)`` is non-smooth at velocity zero crossings), so the LM step does
    not move the friction parameter — the fit returns the warm start looking
    legitimate. Full differentiable contact needs implicit
    differentiation of the contact solver — the long-term scope per ADR-017
    §consequences. The default was therefore flipped to ``fit_friction=False,
    fit_damping=True``. Requesting ``fit_friction=True`` emits a
    :class:`~jaxterity.errors.JaxterityWarning` unless ``force=True``; either
    way, any parameter that stalls is flagged in :attr:`SysidResult.warnings`.

    Args
    ----
    telemetry : JointTelemetry
    robot : Robot
    joint_names : list[str] or None
        Joints to identify. Defaults to every joint exposing the chosen
        parameter(s).
    fit_friction : bool
        Include ``joint.friction`` (default False — weak gradient, see above).
    fit_damping : bool
        Include ``joint.damping`` (default True, clean gradient — recommended).
    force : bool
        Suppress the weak-friction :class:`JaxterityWarning` when
        ``fit_friction=True`` (for callers who understand the limitation).
    theta0, hints
        Standard recipe knobs.

    Raises
    ------
    JaxterityError
        If both ``fit_friction`` and ``fit_damping`` are False.
    """
    if not (fit_friction or fit_damping):
        raise JaxterityError(
            "sysid.contact: nothing to fit — pass fit_friction=True or "
            "fit_damping=True (or both)."
        )
    if fit_friction and not force:
        warnings.warn(
            "sysid.contact(fit_friction=True): Coulomb friction has a near-zero "
            "forward-mode gradient (frictionloss is non-smooth at velocity zero "
            "crossings), so the fit will likely return the warm start unchanged. "
            "Prefer fit_damping=True for a recoverable viscous-friction fit, or "
            "pass force=True to suppress this warning. Stalled parameters are "
            "also reported in SysidResult.warnings.",
            JaxterityWarning,
            stacklevel=2,
        )
    if joint_names is None:
        joint_names = [j.name for j in robot.joints if hasattr(j, "friction")]
    joints_by_name = {j.name: j for j in robot.joints}
    target_params: list[str] = []
    init: list[float] = []
    for name in joint_names:
        j = joints_by_name[name]
        if fit_friction and hasattr(j, "friction"):
            target_params.append(f"{name}.friction")
            init.append(float(j.friction))
        if fit_damping and hasattr(j, "damping"):
            target_params.append(f"{name}.damping")
            init.append(float(j.damping))
    return _recipe(
        CONTACT_RECIPE,
        target_params,
        telemetry,
        robot,
        theta0=theta0 or init,
        hints=hints,
    )


__all__ = ["inertial", "full_body", "contact"]
