# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Gradient-correctness property tests.

Reverse-mode autodiff through articulated dynamics is load-bearing: if gradients
silently break, every sysid recipe becomes wrong without a visible failure. This
suite verifies finite-difference agreement of the parameter gradient through
``Robot.to_diagram()`` simulation, parametrized over (solver, dtype, robot,
parameter):

* **Reverse-mode** (``jax.grad`` + the in-engine adjoint) on **constraint-free**
  robots (unlimited ``CONTINUOUS`` joints), across every solver and both dtypes —
  the systematic sweep. Reverse-mode cannot traverse MJX's constraint solver, so
  it is confined to the constraint-free regime here (ADR-016/017). ``bdf`` and
  ``rk4`` are the trustworthy reverse-mode solvers; ``dopri5``'s adjoint is
  unreliable on sensitive trajectories (erratically tolerance-dependent) and is
  marked ``xfail(strict=False)``.
* **Forward-mode** (:func:`jaxterity.sysid.parameter_jacobian`) on the **real zoo
  robots** through their joint limits — the regime reverse-mode cannot reach. The
  enabler's targeted unit checks (incl. the full SO-101 and forward-vs-reverse
  agreement) live in ``test/sysid/test_diff.py``; here they anchor the
  property sweep.

Tiers (CONTEXT.md / ci.yml): an unmarked **PR-time subset** (~1 min) plus the
full grid behind ``@pytest.mark.nightly`` (~15 min). Tolerances come from
``jaxterity.testing.tolerances.gradient_rtol/atol`` (FD-truncation-bounded).

Note — **float32 + global x64.** jaxonomy runs under JAX x64, so MJX model arrays
are float64 and a float32 *input* parameter promotes internally; the float32
cases verify the pipeline handles float32 inputs to its (looser) tolerance, not a
fully float32 integration (a genuine float32 run needs x64 disabled process-wide,
out of scope here).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jaxonomy
import pytest
from jaxonomy import SimulatorOptions

from jaxterity import zoo
from jaxterity.robot import (
    Frame,
    FramedQuantity,
    Joint,
    JointLimits,
    JointType,
    Link,
    Position,
    Robot,
    Transform,
)
from jaxterity.sysid import parameter_jacobian
from jaxterity.testing import tolerances as tol

pytest.importorskip("mujoco")

_DTYPE = {"float64": jnp.float64, "float32": jnp.float32}


# --- constraint-free robot builders (reverse-mode differentiable) ----------


def _link(name: str, mass: float, com) -> Link:
    return Link(
        name=name,
        body_frame=Frame(name),
        mass=jnp.asarray(mass),
        com=jnp.asarray(com, dtype=float),
        inertia=jnp.eye(3) * 1e-6,
    )


def _hinge(name, parent, child, origin=(0, 0, 0), damping=0.5) -> Joint:
    return Joint(
        name=name,
        joint_type=JointType.CONTINUOUS,
        parent_link=parent,
        child_link=child,
        axis=FramedQuantity(Position(jnp.array([0.0, 1.0, 0.0])), Frame(parent)),
        limits=JointLimits(lower=-3.14, upper=3.14, effort=100.0, velocity=100.0),
        origin=Transform(
            jnp.eye(3),
            jnp.asarray(origin, dtype=float),
            source=Frame(parent),
            target=Frame(child),
        ),
        damping=jnp.asarray(damping),
        friction=jnp.asarray(0.0),
    )


def _pendulum() -> Robot:
    return Robot(
        links=(_link("base", 0.1, (0, 0, 0)), _link("bob", 1.0, (0, 0, -1.0))),
        joints=(_hinge("pivot", "base", "bob"),),
        name="pendulum",
    )


def _double_pendulum() -> Robot:
    return Robot(
        links=(
            _link("base", 0.1, (0, 0, 0)),
            _link("link1", 1.0, (0, 0, -1.0)),
            _link("link2", 1.0, (0, 0, -1.0)),
        ),
        joints=(
            _hinge("p1", "base", "link1"),
            _hinge("p2", "link1", "link2", origin=(0, 0, -1.0)),
        ),
        name="double_pendulum",
    )


# Constraint-free robots, the parameter to fit, and the initial state.
_ROBOTS = {
    "pendulum": (_pendulum, "pivot.damping", "bob.mass", [0.4], [0.0]),
    "double_pendulum": (
        _double_pendulum,
        "p1.damping",
        "link1.mass",
        [0.4, 0.2],
        [0.0, 0.0],
    ),
}
_PARAM_VALUE = {"damping": 0.5, "mass": 1.0}


def _solver_options(solver: str) -> SimulatorOptions:
    return SimulatorOptions(
        math_backend="jax",
        ode_solver_method=solver,
        enable_autodiff=True,  # reverse-mode adjoint
        rtol=1e-8,
        atol=1e-10,
        max_minor_step_size=0.01,
        max_major_steps=300,
    )


def _reverse_grad_vs_fd(robot, param, value, q0, qd0, solver, dtype):
    """Reverse-mode ``d(sum state²)/d(param)`` and its central finite difference."""
    dt = _DTYPE[dtype]
    diagram, system = robot.to_diagram(
        q0=jnp.asarray(q0, dt), qd0=jnp.asarray(qd0, dt), return_system=True
    )
    sid = system.system_id
    opts = _solver_options(solver)

    def loss(p):
        ctx = diagram.create_context()
        ctx = ctx.with_subcontext(sid, ctx[sid].with_parameter(param, p))
        res = jaxonomy.simulate(diagram, ctx, (0.0, 0.6), options=opts)
        return jnp.sum(res.context[sid].continuous_state ** 2)

    p0 = jnp.asarray(value, dt)
    grad = float(jax.grad(loss)(p0))
    h = 0.01 if dtype == "float64" else 0.02
    fd = float((loss(p0 + h) - loss(p0 - h)) / (2 * h))
    return grad, fd


# dopri5's reverse-mode adjoint is unreliable: its gradient error is erratically
# tolerance- and problem-dependent (the pendulum agrees with FD at rtol=1e-8 but
# the chaotic double pendulum is ~2% off and only converges near 1e-12, while at
# 1e-12 the pendulum itself drifts) — there is no robust dopri5 reverse-mode
# setting. bdf and rk4 are the trustworthy reverse-mode solvers. dopri5 cases are
# therefore xfail(strict=False): the suite still covers them (and flags any future
# upstream fix as XPASS) without asserting a value it cannot guarantee.
#
# Re-measured: 7 of the 8 dopri5 cases now XPASS, but
# ``double_pendulum-damping-dopri5-float64`` still XFAILs — i.e. the behaviour is
# still erratic, exactly as described above. So ``strict=False``
# stays: ``strict=True`` would fail on the 7 xpasses, and dropping the marker
# would surface the one real failure. Do not "tighten" this without first
# confirming all eight cases pass reproducibly across dtypes.
_DOPRI5_XFAIL = pytest.mark.xfail(
    reason="jaxonomy dopri5 reverse-mode adjoint is unreliable on sensitive "
    "trajectories (erratic — 7/8 xpass, double_pendulum-damping-float64 fails); "
    "bdf/rk4 are the trusted solvers",
    strict=False,
)

# Full (robot, param_kind, solver, dtype) grid; a small subset runs at PR time,
# the remainder is nightly.
_PR_SUBSET = {
    ("pendulum", "damping", "bdf", "float64"),
    ("pendulum", "mass", "rk4", "float64"),
}


def _case_marks(rname, kind, solver, dtype):
    marks = [] if (rname, kind, solver, dtype) in _PR_SUBSET else [pytest.mark.nightly]
    if solver == "dopri5":
        marks.append(_DOPRI5_XFAIL)
    return tuple(marks)


_REVERSE_CASES = [
    pytest.param(
        rname,
        kind,
        solver,
        dtype,
        marks=_case_marks(rname, kind, solver, dtype),
        id=f"{rname}-{kind}-{solver}-{dtype}",
    )
    for rname in _ROBOTS
    for kind in ("damping", "mass")
    for solver in tol.SOLVERS
    for dtype in tol.DTYPES
]


@pytest.mark.parametrize("robot_name, kind, solver, dtype", _REVERSE_CASES)
def test_reverse_grad_matches_fd(robot_name, kind, solver, dtype):
    """Reverse-mode parameter gradient agrees with finite differences."""
    builder, damping_param, mass_param, q0, qd0 = _ROBOTS[robot_name]
    param = damping_param if kind == "damping" else mass_param
    grad, fd = _reverse_grad_vs_fd(
        builder(), param, _PARAM_VALUE[kind], q0, qd0, solver, dtype
    )
    rtol, atol = tol.gradient_rtol(dtype), tol.gradient_atol(dtype)
    assert grad == pytest.approx(fd, rel=rtol, abs=atol), (
        f"reverse-mode gradient mismatch for {robot_name}.{param} "
        f"under solver={solver} dtype={dtype}: grad={grad:.6e} fd={fd:.6e}"
    )


# --- forward-mode through real zoo-robot joint limits -----------------------


def _forward_grad_vs_fd(robot, param, value, q0, qd0, t_span):
    grad = float(
        parameter_jacobian(
            robot,
            lambda s: jnp.sum(s**2),
            {param: jnp.asarray(value)},
            q0=q0,
            qd0=qd0,
            t_span=t_span,
        )[param]
    )
    diagram, system = robot.to_diagram(q0=q0, qd0=qd0, return_system=True)
    sid = system.system_id
    opts = SimulatorOptions(
        math_backend="jax", ode_solver_method="rk4", max_minor_step_size=0.005
    )

    def loss(p):
        ctx = diagram.create_context()
        ctx = ctx.with_subcontext(sid, ctx[sid].with_parameter(param, p))
        return float(
            jnp.sum(
                jaxonomy.simulate(diagram, ctx, t_span, options=opts)
                .context[sid]
                .continuous_state
                ** 2
            )
        )

    p0 = jnp.asarray(value)
    fd = float((loss(p0 + 0.01) - loss(p0 - 0.01)) / 0.02)
    return grad, fd


def test_forward_grad_cartpole_through_limit_matches_fd():
    """PR-time zoo anchor: forward-mode grad through cartpole's limited slider."""
    grad, fd = _forward_grad_vs_fd(
        zoo.load("cartpole"),
        "cart_to_pole.damping",
        0.1,
        [0.0, 0.2],
        [0.0, 0.0],
        (0.0, 0.3),
    )
    rtol, atol = tol.gradient_rtol("float64"), tol.gradient_atol("float64")
    assert grad == pytest.approx(fd, rel=rtol, abs=atol), f"grad={grad} fd={fd}"


@pytest.mark.nightly
@pytest.mark.parametrize(
    "param,value", [("shoulder_lift.damping", 0.3), ("shoulder_link.mass", 0.5)]
)
def test_forward_grad_so101_matches_fd(param, value):
    """Nightly: forward-mode grad on the full real SO-101 (gentle, within-limit)."""
    grad, fd = _forward_grad_vs_fd(
        zoo.load("so100"), param, value, [0.0] * 6, [0.05] * 6, (0.0, 0.15)
    )
    rtol, atol = tol.gradient_rtol("float64"), tol.gradient_atol("float64")
    assert grad == pytest.approx(fd, rel=rtol, abs=atol), (
        f"{param}: grad={grad} fd={fd}"
    )
