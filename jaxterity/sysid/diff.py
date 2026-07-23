# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Forward-mode differentiation of a robot simulation w.r.t. its parameters.

Differentiable system identification needs the gradient of a simulation w.r.t. a
robot's identifiable parameters (link ``mass``, joint ``damping`` …). For
constraint-free motion either autodiff mode works, but the moment a real robot's
joint limits or contacts engage, MJX solves them with an *iterative* solver that
runs under a ``lax.while_loop``. JAX **cannot reverse-differentiate** a
``while_loop`` (the backward pass needs a statically known trip count), so plain
``jax.grad`` through :func:`jaxonomy.simulate` raises on any limited-joint robot
— which is to say, on essentially every real arm.

**Forward-mode autodiff has no such restriction**: it carries tangents *forward*
through each loop iteration, so it traverses the constraint solver cleanly. This
module computes parameter Jacobians with :func:`jax.jacfwd`, giving correct
gradients on real robots (the full SO-101 included) where reverse-mode cannot go
— and without building a backward tape, so memory stays modest for many-DOF arms.

The cost of forward-mode scales with the number of *parameters* differentiated
(one tangent sweep each), so it suits the modest parameter counts of inertial /
friction identification. Reverse-mode remains preferable for constraint-free
problems with many parameters (use ``jax.grad`` with
``SimulatorOptions(enable_autodiff=True, ode_solver_method="bdf")`` there) —
and is now *also* available **through** the joint-limit constraint solver via
``implicit_vjp=True`` on :meth:`Robot.functional_dynamics` /
:meth:`Robot.to_diagram` (an implicit-function-theorem wrap of the solve; see
:mod:`jaxterity.robot.implicit`), which yields all-parameter gradients in one
backward pass. It pairs with explicit integrators only (``rk4``, not ``bdf``)
and forecloses forward-mode through the wrapped block.

Caveat — **active impacts are non-smooth.** A joint *slamming into* a limit (or a
hard contact making/breaking) is a genuine kink in the dynamics: the gradient
through the impact is ill-defined for *any* method (forward-mode, reverse-mode,
or finite differences disagree there). This is physics, not a bug. Identification
excitation should stay within joint limits and free of hard contact — standard
practice anyway. :func:`parameter_jacobian` does not police this; it computes the
derivative of the trajectory it is given.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

import jax
import jaxonomy
from jaxonomy import SimulatorOptions

from ..errors import JaxterityError
from ..robot import Robot

#: Default integration for forward-mode differentiation: a deterministic
#: fixed-step solver with the custom adjoint **off** (the adjoint is a
#: reverse-mode-only ``custom_vjp``; forward-mode must trace the real ops).
_DEFAULT_OPTIONS = SimulatorOptions(
    math_backend="jax",
    ode_solver_method="rk4",
    enable_autodiff=False,
    max_minor_step_size=0.005,
    max_major_steps=400,
)


def parameter_jacobian(
    robot: Robot,
    objective: Callable[[Any], Any],
    params: dict[str, Any],
    *,
    t_span: tuple[float, float],
    q0=None,
    qd0=None,
    options: SimulatorOptions | None = None,
) -> dict[str, Any]:
    """Forward-mode Jacobian of ``objective(final_state)`` w.r.t. ``params``.

    Builds the robot's diagram once, overrides the named identifiable parameters
    in the context, simulates over ``t_span``, and differentiates the objective
    w.r.t. the parameters with :func:`jax.jacfwd` (forward-mode, so it traverses
    MJX's constraint solver — see the module docstring).

    Args
    ----
    robot : Robot
        The robot to simulate (unchanged; parameters are overridden per call).
    objective : Callable
        Maps the final continuous state ``[q, qd]`` to a scalar (a loss) or a
        vector (residuals, for Gauss-Newton). The Jacobian shape follows it.
    params : dict[str, jax.Array]
        Identifiable parameters to differentiate w.r.t., keyed
        ``"<link_or_joint>.<field>"`` (the :meth:`Robot.parameters` convention),
        mapped to the values at which the Jacobian is evaluated.
    t_span : tuple[float, float]
        ``(t0, t1)`` integration window.
    q0, qd0 : array-like | None
        Initial joint positions / velocities. Default zeros.
    options : SimulatorOptions | None
        Integration options. ``enable_autodiff`` is forced to ``False`` (forward
        mode requires tracing the real ops, not the reverse-mode adjoint).
        Defaults to a fixed-step RK4 configuration.

    Returns
    -------
    dict[str, jax.Array]
        Each parameter name mapped to ``d(objective)/d(param)`` — a gradient when
        ``objective`` is scalar, a Jacobian (rows = residuals) when it is a
        vector.

    Notes
    -----
    Gradients through *active* joint-limit impacts or hard contacts are not
    meaningful for any differentiation method (the dynamics are non-smooth
    there); keep excitation within limits. See the module docstring.
    """
    opts = dataclasses.replace(options or _DEFAULT_OPTIONS, enable_autodiff=False)
    diagram, system = robot.to_diagram(q0=q0, qd0=qd0, return_system=True)
    sid = system.system_id

    # Only the runtime-threaded leaves (link mass, joint damping/friction/
    # armature) can be
    # differentiated here; com/inertia are refit by structural rebuild via
    # Robot.with_parameters, so guard with a typed error rather than
    # surfacing an opaque KeyError from the context.
    differentiable = set(system.dynamic_parameters)
    unknown = sorted(set(params) - differentiable)
    if unknown:
        raise JaxterityError(
            f"parameter_jacobian: {unknown} are not runtime-differentiable. "
            f"Differentiable now: {sorted(differentiable)}. "
            "(``<link>.com`` and ``<link>.inertia_diag`` are runtime-differentiable; "
            "the off-diagonal "
            "inertia tensor and the inertia-frame rotation are still refit "
            "by structural rebuild via Robot.with_parameters.)"
        )

    def rollout(p: dict[str, Any]):
        ctx = diagram.create_context()
        sub = ctx[sid]
        for name, value in p.items():
            sub = sub.with_parameter(name, value)
        ctx = ctx.with_subcontext(sid, sub)
        result = jaxonomy.simulate(diagram, ctx, t_span, options=opts)
        return objective(result.context[sid].continuous_state)

    return jax.jacfwd(rollout)(params)


__all__ = ["parameter_jacobian"]
