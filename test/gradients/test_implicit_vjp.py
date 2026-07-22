# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Reverse-mode AD through the MJX constraint solve (T-218).

The headline guarantee: with ``implicit_vjp=True``, ``jax.grad`` works
*through* MJX's joint-limit constraint solver — where the raw path raises —
and one backward pass yields gradients for every parameter at once, matching
finite differences and the forward-mode ``parameter_jacobian`` path.

The wrap is exact, not smoothed: the forward pass is byte-equivalent to the
raw solve, and the backward pass solves the adjoint of the solver's own KKT
condition (``jaxonomy.optimization.implicit_solver``, jaxonomy T-131).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxterity import zoo
from jaxterity.errors import JaxterityError
from jaxterity.robot import (
    CollisionGeom,
    Frame,
    FramedQuantity,
    Joint,
    JointLimits,
    JointType,
    Link,
    Position,
    Robot,
    Transform,
    constraint_residual,
    implicit_qacc,
)
from jaxterity.testing import tolerances as tol

pytest.importorskip("mujoco")

_GR = tol.gradient_rtol("float64")
_GA = tol.gradient_atol("float64")


def _limited_pendulum(
    damping: float = 0.05, friction: float = 0.0, ground_plane: bool = False
) -> Robot:
    """A hinge pendulum with tight joint limits (constraint-active near 0.1 rad).

    With ``ground_plane=True`` the bob carries a collision sphere so the model
    can generate contact rows (used by the scope-guard test).
    """
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
        mass=jnp.asarray(0.4),
        com=jnp.array([0.3, 0.0, 0.0]),
        inertia=jnp.eye(3) * 1e-3,
        collision=(
            (CollisionGeom(kind="sphere", size=(0.05,), pos=(0.3, 0.0, 0.0)),)
            if ground_plane
            else None
        ),
    )
    pivot = Joint(
        name="pivot",
        joint_type=JointType.REVOLUTE,
        parent_link="base",
        child_link="bob",
        axis=FramedQuantity(Position(jnp.array([0.0, 1.0, 0.0])), Frame("base")),
        limits=JointLimits(lower=-0.1, upper=0.1, effort=100.0, velocity=100.0),
        origin=Transform.identity(Frame("base")),
        damping=jnp.asarray(damping),
        friction=jnp.asarray(friction),
    )
    return Robot(
        links=(base, bob), joints=(pivot,), name="limited", ground_plane=ground_plane
    )


def _mjx_pieces(robot: Robot):
    system = robot.build_system()
    return system._mjx, system._mjx_model


def _fd(loss, p0, h=1e-4):
    return float((loss(p0 + h) - loss(p0 - h)) / (2.0 * h))


# --- the KKT residual reconstruction is exact ---------------------------------


@pytest.mark.parametrize("robot_name", ["cartpole", "so100"])
def test_residual_is_zero_at_the_solver_solution(robot_name):
    """The reconstructed KKT residual vanishes at MJX's own solution.

    Property test of the equation the IFT differentiates: if this holds, the
    backward pass is anchored to the same optimum the forward pass returns.
    Poses are pushed past the joint limits so the limit rows are *active*.
    """
    robot = zoo.load(robot_name)
    mjx, model = _mjx_pieces(robot)
    nv = int(model.nv)
    q = jnp.asarray(np.asarray(model.jnt_range)[:, 1] + 0.05)[: int(model.nq)]
    qd = jnp.full(nv, 0.3)
    tau = jnp.zeros(nv)

    data = mjx.make_data(model).replace(qpos=q, qvel=qd, qfrc_applied=tau)
    qacc = mjx.forward(model, data).qacc
    r = constraint_residual(mjx, qacc, model, q, qd, tau)
    # generalized-force scale of the problem, for a relative bound
    scale = float(jnp.abs(qacc).max()) + 1.0
    assert float(jnp.abs(r).max()) <= 1e-6 * scale


def test_forward_pass_is_byte_equivalent_to_raw_mjx():
    """implicit_vjp changes nothing about the primal — only the tape."""
    robot = _limited_pendulum()
    mjx, model = _mjx_pieces(robot)
    qacc_fn = implicit_qacc(mjx, model)
    q = jnp.asarray([0.12])  # beyond the +0.1 limit → limit row active
    qd = jnp.asarray([0.5])
    tau = jnp.asarray([0.2])

    data = mjx.make_data(model).replace(qpos=q, qvel=qd, qfrc_applied=tau)
    raw = mjx.forward(model, data).qacc
    wrapped = qacc_fn(model, q, qd, tau)
    assert jnp.array_equal(raw, wrapped)


# --- gradients on a single limit-active solve ---------------------------------


def test_grad_matches_fd_with_active_limit():
    """d(qacc)/d(damping) and d(qacc)/d(q) through an *active* limit vs FD."""
    robot = _limited_pendulum()
    mjx, model = _mjx_pieces(robot)
    qacc_fn = implicit_qacc(mjx, model)
    q0, qd = jnp.asarray([0.12]), jnp.asarray([0.5])
    tau = jnp.zeros(1)

    def loss_damping(d):
        return qacc_fn(model.replace(dof_damping=jnp.atleast_1d(d)), q0, qd, tau).sum()

    def loss_q(q):
        return qacc_fn(model, jnp.atleast_1d(q), qd, tau).sum()

    g_d = float(jax.grad(loss_damping)(jnp.asarray(0.05)))
    fd_d = _fd(loss_damping, 0.05)
    assert g_d == pytest.approx(fd_d, rel=_GR, abs=_GA), f"grad={g_d} fd={fd_d}"

    g_q = float(jax.grad(loss_q)(jnp.asarray(0.12)))
    fd_q = _fd(loss_q, 0.12)
    assert g_q == pytest.approx(fd_q, rel=_GR, abs=_GA), f"grad={g_q} fd={fd_q}"


def test_grad_matches_fd_with_active_friction_row():
    """A dof-friction (frictionloss) row in its sliding regime, vs FD."""
    robot = _limited_pendulum(friction=0.05)
    mjx, model = _mjx_pieces(robot)
    qacc_fn = implicit_qacc(mjx, model)
    q0, qd = jnp.asarray([0.0]), jnp.asarray([0.8])  # inside limits, sliding
    tau = jnp.zeros(1)

    def loss(fl):
        return qacc_fn(
            model.replace(dof_frictionloss=jnp.atleast_1d(fl)), q0, qd, tau
        ).sum()

    g = float(jax.grad(loss)(jnp.asarray(0.05)))
    fd = _fd(loss, 0.05)
    assert g == pytest.approx(fd, rel=_GR, abs=_GA), f"grad={g} fd={fd}"


# --- gradients through a rollout ----------------------------------------------


def test_rollout_grad_matches_fd_and_forward_mode():
    """jax.grad through a 60-step RK4 rollout that rides the joint limit.

    Cross-checked two ways: central finite differences, and the raw
    (forward-differentiable) dynamics path under ``jax.jacfwd``.
    """
    dt, n_steps = 0.005, 60
    tau = jnp.asarray([0.5])
    state0 = jnp.array([0.09, 0.0])

    def make_loss(implicit_vjp: bool):
        robot = _limited_pendulum()
        system = robot.build_system(actuation=True, implicit_vjp=implicit_vjp)
        mjx, base_model = system._mjx, system._mjx_model
        qacc_fn = system._qacc_fn

        def dynamics(model, state):
            q, qd = state[:1], state[1:]
            if qacc_fn is not None:
                qacc = qacc_fn(model, q, qd, tau)
            else:
                data = mjx.make_data(model).replace(qpos=q, qvel=qd, qfrc_applied=tau)
                qacc = mjx.forward(model, data).qacc
            return jnp.concatenate([qd, qacc])

        def loss(damping):
            model = base_model.replace(dof_damping=jnp.atleast_1d(damping))

            def rk4(state, _):
                k1 = dynamics(model, state)
                k2 = dynamics(model, state + 0.5 * dt * k1)
                k3 = dynamics(model, state + 0.5 * dt * k2)
                k4 = dynamics(model, state + dt * k3)
                return state + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), None

            state, _ = jax.lax.scan(rk4, state0, None, length=n_steps)
            return jnp.sum(state**2)

        return loss

    loss_implicit = make_loss(implicit_vjp=True)
    loss_raw = make_loss(implicit_vjp=False)

    p0 = jnp.asarray(0.05)
    g_rev = float(jax.grad(loss_implicit)(p0))
    g_fwd = float(jax.jacfwd(loss_raw)(p0))
    fd = _fd(loss_raw, 0.05, h=1e-3)

    assert g_rev == pytest.approx(g_fwd, rel=_GR, abs=_GA), (
        f"reverse={g_rev} forward={g_fwd}"
    )
    assert g_rev == pytest.approx(fd, rel=1e-3, abs=_GA), f"reverse={g_rev} fd={fd}"


@pytest.mark.slow
def test_one_backward_pass_covers_every_damping_on_so101():
    """The scaling claim: all SO-101 dampings from ONE backward pass.

    Forward mode costs one pass per parameter; the implicit wrap yields the
    full 6-vector gradient in a single reverse sweep. Verified entry-wise
    against forward-mode ``jacfwd`` of the raw path on the same rollout.
    """
    robot = zoo.load("so100")
    dt, n_steps = 0.005, 30
    system = robot.build_system(actuation=True, implicit_vjp=True)
    raw_system = robot.build_system(actuation=True)
    mjx = system._mjx
    nv = system._nv
    nq = system._nq
    tau = jnp.full(nv, 0.3)
    q0 = jnp.asarray(np.asarray(system._mjx_model.jnt_range)[:, 1] - 0.02)[:nq]
    state0 = jnp.concatenate([q0, jnp.zeros(nv)])

    def make_loss(qacc_fn, base_model):
        def dynamics(model, state):
            q, qd = state[:nq], state[nq:]
            return jnp.concatenate([qd, qacc_fn(model, q, qd, tau)])

        def loss(damping):
            model = base_model.replace(dof_damping=damping)

            def rk4(state, _):
                k1 = dynamics(model, state)
                k2 = dynamics(model, state + 0.5 * dt * k1)
                k3 = dynamics(model, state + 0.5 * dt * k2)
                k4 = dynamics(model, state + dt * k3)
                return state + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), None

            state, _ = jax.lax.scan(rk4, state0, None, length=n_steps)
            return jnp.sum(state**2)

        return loss

    def raw_qacc(model, q, qd, tau_):
        data = mjx.make_data(model).replace(qpos=q, qvel=qd, qfrc_applied=tau_)
        return mjx.forward(model, data).qacc

    damping0 = jnp.asarray(system._mjx_model.dof_damping)
    loss_implicit = make_loss(
        lambda m, q, qd, t: system._qacc_fn(m, q, qd, t), system._mjx_model
    )
    loss_raw = make_loss(raw_qacc, raw_system._mjx_model)

    g_rev = np.asarray(jax.grad(loss_implicit)(damping0))  # ONE backward pass
    g_fwd = np.asarray(jax.jacfwd(loss_raw)(damping0))  # nv forward passes

    assert g_rev.shape == (nv,)
    np.testing.assert_allclose(g_rev, g_fwd, rtol=_GR, atol=_GA)
    assert np.linalg.norm(g_rev) > 1e-8  # non-vacuous: limits + dynamics active


@pytest.mark.slow
def test_grad_through_simulate_matches_forward_mode():
    """jax.grad through ``jaxonomy.simulate`` on a limit-active cartpole.

    The exact scenario ``test_reverse_mode_raises_on_limited_joint`` pins as
    impossible on the raw path, now differentiable with
    ``to_diagram(implicit_vjp=True)`` — checked against forward-mode
    ``parameter_jacobian`` on the same trajectory functional.

    Solver note: ``rk4``, not ``bdf`` — BDF's Newton iteration takes a jvp of
    the RHS, which the ``custom_vjp`` wrap forecloses (documented limitation;
    see :mod:`jaxterity.robot.implicit`).
    """
    import jaxonomy
    from jaxonomy import SimulatorOptions

    from jaxterity.sysid import parameter_jacobian

    cartpole = zoo.load("cartpole")
    q0, qd0 = [0.0, 0.2], [0.0, 0.0]
    t_span = (0.0, 0.3)

    jac = parameter_jacobian(
        cartpole,
        lambda s: jnp.sum(s**2),
        {"cart_to_pole.damping": jnp.asarray(0.1)},
        q0=q0,
        qd0=qd0,
        t_span=t_span,
    )
    forward = float(jac["cart_to_pole.damping"])

    diagram, system = cartpole.to_diagram(
        q0=q0, qd0=qd0, return_system=True, implicit_vjp=True
    )
    sid = system.system_id
    opts = SimulatorOptions(
        math_backend="jax",
        ode_solver_method="rk4",
        enable_autodiff=True,
        max_minor_step_size=0.005,
        max_major_steps=200,
    )

    def loss(p):
        ctx = diagram.create_context()
        ctx = ctx.with_subcontext(
            sid, ctx[sid].with_parameter("cart_to_pole.damping", p)
        )
        res = jaxonomy.simulate(diagram, ctx, t_span, options=opts)
        return jnp.sum(res.context[sid].continuous_state ** 2)

    reverse = float(jax.grad(loss)(jnp.asarray(0.1)))
    assert reverse == pytest.approx(forward, rel=_GR, abs=_GA), (
        f"reverse={reverse} forward={forward}"
    )


# --- scope guard --------------------------------------------------------------


def test_contact_capable_model_raises_typed_error():
    """First-cut scope: contact-capable models are rejected at construction."""
    robot = _limited_pendulum(ground_plane=True)
    with pytest.raises(JaxterityError, match="contact"):
        robot.build_system(implicit_vjp=True)


def test_vmap_consistency():
    """Wrapped qacc under vmap equals the loop over the batch (invariant 3)."""
    robot = _limited_pendulum()
    mjx, model = _mjx_pieces(robot)
    qacc_fn = implicit_qacc(mjx, model)
    qs = jnp.linspace(-0.15, 0.15, 7)[:, None]  # straddles both limits
    qds = jnp.linspace(-1.0, 1.0, 7)[:, None]
    tau = jnp.zeros(1)

    batched = jax.vmap(lambda q, qd: qacc_fn(model, q, qd, tau))(qs, qds)
    looped = jnp.stack([qacc_fn(model, qs[i], qds[i], tau) for i in range(7)])
    np.testing.assert_allclose(np.asarray(batched), np.asarray(looped), rtol=1e-12)
