# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Forward-mode parameter-differentiation tests (forward-mode sysid enabler).

The headline guarantee: ``parameter_jacobian`` gives correct gradients w.r.t.
robot parameters *through* MJX's constraint solver (joint limits / contacts),
where reverse-mode autodiff raises. Verified against finite differences on a
limited-joint robot (cartpole) and the full real SO-101, and cross-checked
against reverse-mode on a constraint-free pendulum.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jaxonomy
import numpy as np
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

_GR = tol.gradient_rtol("float64")
_GA = tol.gradient_atol("float64")


def _pendulum(damping: float = 0.5) -> Robot:
    """A constraint-free (unlimited hinge) pendulum — reverse-mode differentiable."""
    base = Link(
        name="base",
        body_frame=Frame("base"),
        mass=jnp.asarray(0.1),
        com=jnp.zeros(3),
        inertia=jnp.eye(3) * 1e-3,
    )
    bob = Link(
        name="bob",
        body_frame=Frame("bob"),
        mass=jnp.asarray(1.0),
        com=jnp.array([0.0, 0.0, -1.0]),
        inertia=jnp.eye(3) * 1e-6,
    )
    pivot = Joint(
        name="pivot",
        joint_type=JointType.CONTINUOUS,
        parent_link="base",
        child_link="bob",
        axis=FramedQuantity(Position(jnp.array([0.0, 1.0, 0.0])), Frame("base")),
        limits=JointLimits(lower=-3.14, upper=3.14, effort=100.0, velocity=100.0),
        origin=Transform.identity(Frame("base")),
        damping=jnp.asarray(damping),
        friction=jnp.asarray(0.0),
    )
    return Robot(links=(base, bob), joints=(pivot,), name="pendulum")


def _fd(loss, p0, h=0.01):
    """Central finite difference of a scalar ``loss`` at ``p0``."""
    return float((loss(p0 + h) - loss(p0 - h)) / (2 * h))


@pytest.mark.slow  # compiles the BDF adjoint only to confirm it raises
def test_reverse_mode_raises_on_limited_joint():
    """Motivation: reverse-mode autodiff cannot traverse MJX's constraint solver.

    A limited joint makes ``mjx.forward`` run its iterative solver under a
    ``lax.while_loop``; ``jax.grad`` through ``simulate`` then raises — which is
    exactly why ``parameter_jacobian`` uses forward mode.
    """
    cartpole = zoo.load("cartpole")  # prismatic slider is a limited joint
    diagram, system = cartpole.to_diagram(
        q0=[0.0, 0.1], qd0=[0.0, 0.0], return_system=True
    )
    sid = system.system_id
    opts = SimulatorOptions(
        math_backend="jax",
        ode_solver_method="bdf",
        enable_autodiff=True,
        max_major_steps=100,
    )

    def loss(p):
        ctx = diagram.create_context()
        ctx = ctx.with_subcontext(
            sid, ctx[sid].with_parameter("cart_to_pole.damping", p)
        )
        res = jaxonomy.simulate(diagram, ctx, (0.0, 0.2), options=opts)
        return jnp.sum(res.context[sid].continuous_state ** 2)

    with pytest.raises(Exception, match="while_loop|reverse-mode|custom_vjp"):
        jax.grad(loss)(jnp.asarray(0.1))


def test_forward_mode_through_limited_joint_matches_fd():
    """Forward mode differentiates through cartpole's limited slider (vs FD)."""
    cartpole = zoo.load("cartpole")

    def objective(state):
        return jnp.sum(state**2)

    jac = parameter_jacobian(
        cartpole,
        objective,
        {"cart_to_pole.damping": jnp.asarray(0.1)},
        q0=[0.0, 0.2],
        qd0=[0.0, 0.0],
        t_span=(0.0, 0.3),
    )
    grad = float(jac["cart_to_pole.damping"])

    # finite-difference reference (same configuration, no active impact)
    diagram, system = cartpole.to_diagram(
        q0=[0.0, 0.2], qd0=[0.0, 0.0], return_system=True
    )
    sid = system.system_id
    opts = SimulatorOptions(
        math_backend="jax", ode_solver_method="rk4", max_minor_step_size=0.005
    )

    def loss(p):
        ctx = diagram.create_context()
        ctx = ctx.with_subcontext(
            sid, ctx[sid].with_parameter("cart_to_pole.damping", p)
        )
        return float(
            jnp.sum(
                jaxonomy.simulate(diagram, ctx, (0.0, 0.3), options=opts)
                .context[sid]
                .continuous_state
                ** 2
            )
        )

    fd = _fd(loss, jnp.asarray(0.1))
    assert grad == pytest.approx(fd, rel=_GR, abs=_GA), f"forward={grad} fd={fd}"


@pytest.mark.slow  # runs both AD modes (the BDF adjoint compile is heavy)
def test_forward_mode_agrees_with_reverse_mode():
    """On a constraint-free pendulum (where both modes work) they must agree."""
    robot = _pendulum(damping=0.5)

    jac = parameter_jacobian(
        robot,
        lambda s: s[0] ** 2,
        {"pivot.damping": jnp.asarray(0.5)},
        q0=[0.4],
        qd0=[0.0],
        t_span=(0.0, 1.0),
    )
    forward = float(jac["pivot.damping"])

    diagram, system = robot.to_diagram(q0=[0.4], qd0=[0.0], return_system=True)
    sid = system.system_id
    adj = SimulatorOptions(
        math_backend="jax",
        ode_solver_method="bdf",
        enable_autodiff=True,
        rtol=1e-7,
        atol=1e-9,
        max_major_steps=300,
    )

    def loss(p):
        ctx = diagram.create_context()
        ctx = ctx.with_subcontext(sid, ctx[sid].with_parameter("pivot.damping", p))
        return (
            jaxonomy.simulate(diagram, ctx, (0.0, 1.0), options=adj)
            .context[sid]
            .continuous_state[0]
            ** 2
        )

    reverse = float(jax.grad(loss)(jnp.asarray(0.5)))
    assert forward == pytest.approx(reverse, rel=_GR, abs=_GA), (
        f"forward={forward} reverse={reverse}"
    )


def test_non_differentiable_parameter_raises_typed_error():
    """Full 3×3 inertia tensor isn't runtime-threaded → typed error.

    Runtime threading exists for ``<link>.com`` (3-vec → body_ipos) and
    ``<link>.inertia_diag`` (3-vec → body_inertia). The off-diagonal inertia
    tensor (``<link>.inertia``) and the inertia-frame rotation (body_iquat)
    are still refit by structural rebuild via ``Robot.with_parameters``; this
    test pins that boundary.
    """
    from jaxterity.errors import JaxterityError

    with pytest.raises(JaxterityError, match="not runtime-differentiable"):
        parameter_jacobian(
            _pendulum(),
            lambda s: jnp.sum(s**2),
            {"bob.inertia": jnp.eye(3)},
            q0=[0.4],
            qd0=[0.0],
            t_span=(0.0, 0.1),
        )


def test_com_and_inertia_diag_are_runtime_differentiable():
    """<link>.com (3-vec) and <link>.inertia_diag (3-vec) flow gradients."""
    com_jac = parameter_jacobian(
        _pendulum(),
        lambda s: jnp.sum(s**2),
        {"bob.com": jnp.array([0.0, 0.0, -1.0])},
        q0=[0.4],
        qd0=[0.0],
        t_span=(0.0, 0.1),
    )
    assert "bob.com" in com_jac
    g = np.asarray(com_jac["bob.com"])
    assert g.shape == (3,) and np.all(np.isfinite(g))
    inertia_jac = parameter_jacobian(
        _pendulum(),
        lambda s: jnp.sum(s**2),
        {"bob.inertia_diag": jnp.array([1e-3, 1e-3, 1e-3])},
        q0=[0.4],
        qd0=[0.0],
        t_span=(0.0, 0.1),
    )
    assert "bob.inertia_diag" in inertia_jac
    g2 = np.asarray(inertia_jac["bob.inertia_diag"])
    assert g2.shape == (3,) and np.all(np.isfinite(g2))


def test_jacobian_multiple_params_and_vector_residual():
    """A vector objective over several params yields a dict of Jacobian rows."""
    robot = _pendulum()
    params = {"bob.mass": jnp.asarray(1.0), "pivot.damping": jnp.asarray(0.5)}
    jac = parameter_jacobian(
        robot,
        lambda s: s,  # residual = full final state [q, qd]
        params,
        q0=[0.4],
        qd0=[0.0],
        t_span=(0.0, 0.5),
    )
    assert set(jac) == set(params)
    for name in params:
        assert jac[name].shape == (2,)  # d[q, qd] / d(param)
        assert np.all(np.isfinite(np.asarray(jac[name])))
    # mass and damping both affect the trajectory non-trivially
    assert np.linalg.norm(np.asarray(jac["bob.mass"])) > 1e-6
    assert np.linalg.norm(np.asarray(jac["pivot.damping"])) > 1e-6


@pytest.mark.slow
def test_forward_mode_on_full_so101_matches_fd():
    """The headline: correct gradients on the full real 6-DOF SO-101.

    Gentle, within-limit excitation (no active impact) for both an inertial
    (``mass``) and a dissipative (``damping``) parameter — the regime real
    identification uses. Reverse-mode cannot do this (limited joints + memory).
    """
    so101 = zoo.load("so100")
    ndof = 6
    q0 = [0.0] * ndof
    qd0 = [0.05] * ndof
    t_span = (0.0, 0.15)

    for name, p0 in (("shoulder_lift.damping", 0.3), ("shoulder_link.mass", 0.5)):
        jac = parameter_jacobian(
            so101,
            lambda s: jnp.sum(s**2),
            {name: jnp.asarray(p0)},
            q0=q0,
            qd0=qd0,
            t_span=t_span,
        )
        grad = float(jac[name])

        diagram, system = so101.to_diagram(q0=q0, qd0=qd0, return_system=True)
        sid = system.system_id
        opts = SimulatorOptions(
            math_backend="jax", ode_solver_method="rk4", max_minor_step_size=0.005
        )

        def loss(p, _name=name):
            ctx = diagram.create_context()
            ctx = ctx.with_subcontext(sid, ctx[sid].with_parameter(_name, p))
            return float(
                jnp.sum(
                    jaxonomy.simulate(diagram, ctx, t_span, options=opts)
                    .context[sid]
                    .continuous_state
                    ** 2
                )
            )

        fd = _fd(loss, jnp.asarray(p0))
        assert grad == pytest.approx(fd, rel=_GR, abs=_GA), (
            f"{name}: fwd={grad} fd={fd}"
        )
