# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Robot-level recipes — `inertial`, `full_body`, `contact`.

Each recipe is tested by: generate truth telemetry from a robot with known
perturbed parameters, fit through the recipe starting from the original
parameters, assert the truth values are recovered within tolerance.
"""

from __future__ import annotations

import jax.numpy as jnp
import jaxonomy
import numpy as np
import pytest
from jaxonomy import SimulatorOptions

import jaxterity.sysid as sysid
from jaxterity.data import JointTelemetry
from jaxterity.errors import JaxterityError, JaxterityWarning
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

pytest.importorskip("mujoco")


_OPTS = SimulatorOptions(
    math_backend="jax",
    ode_solver_method="rk4",
    enable_autodiff=False,
    max_minor_step_size=0.005,
    max_major_steps=200,
)


def _pendulum(
    *, mass: float = 1.0, damping: float = 0.0, friction: float = 0.0
) -> Robot:
    """Single-joint pendulum with parameterizable mass / damping / friction."""
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
        mass=jnp.asarray(mass),
        com=jnp.array([0.0, 0.0, -1.0]),
        inertia=jnp.eye(3) * 1e-3,
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
        friction=jnp.asarray(friction),
    )
    return Robot(links=(base, bob), joints=(pivot,), name="pendulum")


def _telemetry_from_robot(
    robot: Robot, *, q0=(0.5,), qd0=(0.0,), t_span=(0.0, 0.6), n_samples=20
) -> JointTelemetry:
    """Forward-sim ``robot`` and bundle the trajectory as a JointTelemetry."""
    q0_arr, qd0_arr = jnp.asarray(q0, dtype=float), jnp.asarray(qd0, dtype=float)
    diagram, system = robot.to_diagram(q0=q0_arr, qd0=qd0_arr, return_system=True)
    sid = system.system_id
    sample_times = np.linspace(t_span[0], t_span[1], n_samples)
    qs, qds = [], []
    for t in sample_times:
        ctx = diagram.create_context()
        res = jaxonomy.simulate(diagram, ctx, (0.0, float(t) + 1e-9), options=_OPTS)
        state = np.asarray(res.context[sid].continuous_state)
        nq = state.size // 2
        qs.append(state[:nq])
        qds.append(state[nq:])
    return JointTelemetry(
        time=sample_times.astype(np.float64),
        sample_rate_hz=1.0 / (sample_times[1] - sample_times[0]),
        joint_names=("pivot",),
        q=np.asarray(qs, dtype=np.float64),
        qd=np.asarray(qds, dtype=np.float64),
        tau=np.zeros_like(np.asarray(qs)),
    )


# --- sysid.inertial -------------------------------------------------


def test_inertial_recipe_recovers_perturbed_mass():
    """Truth mass 1.5 → telemetry; recipe starting from 1.0 recovers 1.5."""
    truth_mass = 1.5
    truth = _pendulum(mass=truth_mass)
    tel = _telemetry_from_robot(truth)
    guess = _pendulum(mass=1.0)
    res = sysid.inertial(tel, guess, link_names=["bob"], hints={"iterations": 6})
    assert res.recipe_name == "sysid.inertial"
    fitted = float(res.fitted["bob.mass"])
    assert fitted == pytest.approx(truth_mass, rel=1e-2)


def test_inertial_recipe_rejects_wrong_telemetry():
    from jaxterity.data import ActuatorRampTelemetry

    tel = ActuatorRampTelemetry(
        time=np.arange(5, dtype=np.float64),
        sample_rate_hz=1.0,
        actuator_name="m",
        command=np.zeros(5),
        velocity=np.zeros(5),
    )
    with pytest.raises(JaxterityError, match="JointTelemetry"):
        sysid.inertial(tel, _pendulum())  # type: ignore[arg-type]


# --- sysid.full_body ------------------------------------------------


def test_full_body_recipe_recovers_damping_alone():
    """fit_mass=False → identifies damping from telemetry."""
    truth_damping = 0.4
    truth = _pendulum(damping=truth_damping)
    tel = _telemetry_from_robot(truth)
    guess = _pendulum(damping=0.0)
    res = sysid.full_body(
        tel,
        guess,
        fit_mass=False,
        fit_damping=True,
        hints={"iterations": 8},
    )
    assert res.recipe_name == "sysid.full_body"
    fitted = float(res.fitted["pivot.damping"])
    assert fitted == pytest.approx(truth_damping, abs=5e-2)


def test_full_body_rejects_empty_target():
    with pytest.raises(JaxterityError, match="nothing to fit"):
        sysid.full_body(
            _telemetry_from_robot(_pendulum()),
            _pendulum(),
            fit_mass=False,
            fit_damping=False,
        )


# --- sysid.contact --------------------------------------------------


def test_contact_recipe_damping_mode_recovers_truth():
    """fit_damping mode has clean gradient and recovers truth.

    The original sysid.contact (Coulomb-friction-only) has a near-zero
    forward-mode gradient. With fit_damping=True / fit_friction=False the
    recipe identifies viscous friction — the same signal sysid.full_body
    recovers, but exposed under the contact recipe's name for users whose
    mental model is "joint friction" rather than "joint damping".
    """
    truth_damping = 0.4
    truth = _pendulum(damping=truth_damping)
    tel = _telemetry_from_robot(truth)
    guess = _pendulum(damping=0.0)
    res = sysid.contact(
        tel,
        guess,
        joint_names=["pivot"],
        fit_friction=False,
        fit_damping=True,
        hints={"iterations": 8},
    )
    assert res.recipe_name == "sysid.contact"
    fitted = float(res.fitted["pivot.damping"])
    assert fitted == pytest.approx(truth_damping, abs=5e-2)


def test_contact_recipe_rejects_no_fit_targets():
    with pytest.raises(JaxterityError, match="nothing to fit"):
        sysid.contact(
            _telemetry_from_robot(_pendulum()),
            _pendulum(),
            fit_friction=False,
            fit_damping=False,
        )


def test_contact_recipe_dispatches_and_returns_correct_shape():
    """The friction path runs end-to-end and returns the friction leaf.

    Numerical recovery is **known to be weak** for the Coulomb-friction
    parameter under forward-mode autodiff: MJX maps ``frictionloss`` to a
    sign-of-velocity term which is non-smooth at the many zero crossings of a
    swinging pendulum, so the forward-mode Jacobian column for friction is
    ~zero and the LM step doesn't move. The default was flipped to fit_damping,
    and the friction path must be asked for explicitly;
    requesting it emits a JaxterityWarning, and the stalled parameter is flagged
    in SysidResult.warnings. This test asserts the recipe is correctly wired and
    is honest about the stall.
    """
    truth = _pendulum(friction=0.2)
    tel = _telemetry_from_robot(truth)
    guess = _pendulum(friction=0.0)
    with pytest.warns(JaxterityWarning, match="near-zero"):
        res = sysid.contact(
            tel,
            guess,
            joint_names=["pivot"],
            fit_friction=True,
            fit_damping=False,
            hints={"iterations": 4},
        )
    assert res.recipe_name == "sysid.contact"
    assert "pivot.friction" in res.fitted
    assert res.telemetry_hash and res.recipe_version == "1.0"
    # The friction column stalls (weak gradient) — the result says so.
    assert res.warnings
    assert "did not move" in res.warnings[0]


def test_contact_recipe_force_suppresses_friction_warning():
    """force=True lets a caller who understands the limitation skip the warning."""
    import warnings as _w

    truth = _pendulum(friction=0.2)
    tel = _telemetry_from_robot(truth)
    guess = _pendulum(friction=0.0)
    with _w.catch_warnings():
        _w.simplefilter("error", JaxterityWarning)  # any JaxterityWarning → failure
        sysid.contact(
            tel,
            guess,
            joint_names=["pivot"],
            fit_friction=True,
            fit_damping=False,
            force=True,
            hints={"iterations": 4},
        )
