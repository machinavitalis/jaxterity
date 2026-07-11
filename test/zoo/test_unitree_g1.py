# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Unitree G1 humanoid — topology, floating-base free-fall, contact.

The shared zoo harness (``test_zoo.py``) covers the *fixed-base* G1 against its
golden. This file covers what that harness structurally cannot: the
**floating-base** topology (``nq != nv``), which is the form a real biped needs
and which exercises the ADR-018 quaternion configuration map on a 29-DoF model,
plus the foot–ground contact form (``contact=True``) the locomotion stack
builds on.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jaxonomy
import numpy as np
import pytest
from jaxonomy import SimulatorOptions

from jaxterity import zoo
from jaxterity.errors import JaxterityError
from jaxterity.robot import JointType

pytest.importorskip("mujoco")

_RK4 = SimulatorOptions(
    math_backend="jax", ode_solver_method="rk4", max_minor_step_size=0.005
)


def test_fixed_base_topology():
    """The default load is fixed-base: 29 revolute joints, no free joint."""
    robot = zoo.load("unitree_g1")
    assert len(robot.joints) == 29
    assert all(j.joint_type is JointType.REVOLUTE for j in robot.joints)
    # Standard pelvis-welded root → square configuration space (builds cleanly).
    robot.to_diagram(q0=[0.0] * 29, qd0=[0.0] * 29)


def test_floating_base_adds_a_free_joint():
    """``floating_base=True`` prepends one FLOATING joint (world → pelvis)."""
    robot = zoo.unitree_g1.load(floating_base=True)
    free = [j for j in robot.joints if j.joint_type is JointType.FLOATING]
    assert len(free) == 1
    assert free[0].parent_link == "world" and free[0].child_link == "pelvis"
    # Free base ⇒ quaternion configuration: nq = nv + 1 over the whole model.
    mjcf = robot.to_mjcf()
    import mujoco

    m = mujoco.MjModel.from_xml_string(mjcf)
    assert (m.nq, m.nv) == (36, 35)


def test_floating_base_free_fall_is_ballistic():
    """Released from its default pose, the free-base pelvis follows ½·g·t².

    A clean ballistic drop (with the quaternion staying unit-norm) is the
    end-to-end check that the floating-base dynamics + config-derivative map
    are wired correctly for the humanoid — not just the unit free-body fixture.
    qpos layout for a free base is ``[x y z, quat(4), 29 joints]``.
    """
    t = 0.4
    robot = zoo.unitree_g1.load(floating_base=True)
    diagram, sysm = robot.to_diagram(return_system=True)  # default qpos0
    ctx = diagram.create_context()
    z0 = float(np.asarray(ctx[sysm.system_id].continuous_state)[2])

    res = jaxonomy.simulate(diagram, ctx, (0.0, t), options=_RK4)
    out = np.asarray(res.context[sysm.system_id].continuous_state)
    assert np.all(np.isfinite(out))

    q = out[:36]
    drop = z0 - float(q[2])
    expected = 0.5 * 9.81 * t**2  # rest → ballistic free fall
    # Whole-body inertia doesn't change a free base's CoM fall, and the pelvis
    # is rigid w.r.t. the base, so a few-% band is plenty.
    assert drop == pytest.approx(expected, rel=0.05), (drop, expected)
    # Quaternion stays on the unit sphere with no explicit renormalization
    # (ADR-018: the RK4 derivative is tangent to the sphere).
    assert abs(float(np.linalg.norm(q[3:7])) - 1.0) < 1e-6


# --- foot–ground contact -------------------------------------------------


def test_contact_requires_floating_base():
    """``contact=True`` on the welded form raises with guidance."""
    with pytest.raises(JaxterityError, match="floating_base=True"):
        zoo.unitree_g1.load(contact=True)


def test_contact_lowers_feet_and_floor():
    """The contact form carries exactly the floor + two foot boxes.

    The foot soles sit 0.794 m below the pelvis at zero configuration —
    the upstream standing height (pelvis z ≈ 0.793 in the source model) —
    so the Playground-matched box pose is wired in correctly.
    """
    import mujoco

    robot = zoo.unitree_g1.load(floating_base=True, contact=True)
    m = mujoco.MjModel.from_xml_string(robot.to_mjcf())
    names = {m.geom(i).name for i in range(m.ngeom)}
    assert names == {"floor", "left_foot", "right_foot"}

    d = mujoco.MjData(m)
    d.qpos[:] = m.qpos0
    mujoco.mj_forward(m, d)
    for foot in ("left_foot", "right_foot"):
        sole_z = float(d.geom_xpos[m.geom(foot).id][2]) - 0.008
        assert sole_z == pytest.approx(-0.794, abs=0.005), (foot, sole_z)

    # Geometry is part of the model description: the handle must move.
    bare = zoo.unitree_g1.load(floating_base=True)
    assert robot.attestation_handle != bare.attestation_handle


def test_contact_arrests_free_fall():
    """Dropped from just above standing height, the floor stops the fall.

    Zero torque, 0.5 s, fixed-step RK4 over ``functional_dynamics``. The
    unactuated humanoid buckles, but contact must arrest the fall: without
    the floor the pelvis would be at z ≈ -0.41 m (ballistic); with it the
    pelvis must stay well above the floor and the state finite throughout.
    """
    robot = zoo.unitree_g1.load(floating_base=True, contact=True)
    f = robot.functional_dynamics()
    dt, n_steps = 0.002, 250
    tau = jnp.zeros(35)
    q0 = jnp.zeros(36).at[2].set(0.82).at[3].set(1.0)
    state0 = jnp.concatenate([q0, jnp.zeros(35)])

    def rk4(state, _):
        k1 = f(state, tau)
        k2 = f(state + 0.5 * dt * k1, tau)
        k3 = f(state + 0.5 * dt * k2, tau)
        k4 = f(state + dt * k3, tau)
        s = state + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return s, s

    _, states = jax.jit(lambda s: jax.lax.scan(rk4, s, None, length=n_steps))(state0)
    states = np.asarray(states)
    assert np.all(np.isfinite(states))
    pelvis_z = states[:, 2]
    assert pelvis_z.min() > 0.5, pelvis_z.min()  # ballistic would be -0.41
    # Quaternion integrity holds through contact.
    assert abs(float(np.linalg.norm(states[-1, 3:7])) - 1.0) < 1e-5
