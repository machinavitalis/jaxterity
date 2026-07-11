# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Twin — identity/handle, PyTree, vmap, grad, composed rollout, diagram.

The composed-rollout tests cross-check the `Twin` against the standalone
building blocks: the body trajectory against a direct
`Robot.functional_dynamics()` RK4 (same integrator → bit-exact), the per-servo
temperature against the analytic `thermal_response`, and the bus voltage
against the standalone `bus_voltage`. The handle tests exercise the
ADR-008 contract extended off-body: sensitive to off-body params + serial
+ provenance, stable under the cosmetic set.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxonomy
import numpy as np
import pytest

from jaxterity.physics import bus_voltage
from jaxterity.physics.thermal import thermal_response
from jaxterity.testing import tolerances as tol
from jaxterity.twin import Twin, so101_twin

pytest.importorskip("mujoco")


# --- shared twin (building MJX once is the expensive part) ------------------


@pytest.fixture(scope="module")
def twin() -> Twin:
    return so101_twin("SO101-0001")


@pytest.fixture(scope="module")
def dims(twin):
    nq = twin.robot.build_system()._nq
    nv = twin.n_servos
    return nq, nv


def _battery_twin(twin, nv) -> Twin:
    return Twin(
        twin.robot,
        serial="LK-0001",
        C_th=twin.C_th,
        R_th=twin.R_th,
        k_pwr=twin.k_pwr,
        V_oc=12.0,
        R_int=0.4,
        k_i=tuple([1.5] * nv),
    )


# --- construction / validation ---------------------------------------------


def test_basic_shape(twin):
    assert twin.n_servos == 6
    assert not twin.has_battery
    assert twin.C_th.shape == (6,)


def test_thermal_shape_mismatch_raises(twin):
    with pytest.raises(ValueError, match="one per servo|shape"):
        Twin(
            twin.robot,
            serial="x",
            C_th=jnp.ones(6),
            R_th=jnp.ones(5),
            k_pwr=jnp.ones(6),
        )


def test_half_specified_battery_raises(twin):
    with pytest.raises(ValueError, match="half-specified"):
        Twin(
            twin.robot,
            serial="x",
            C_th=twin.C_th,
            R_th=twin.R_th,
            k_pwr=twin.k_pwr,
            V_oc=12.0,  # R_int omitted
        )


def test_battery_requires_matching_k_i(twin):
    with pytest.raises(ValueError, match="k_i"):
        Twin(
            twin.robot,
            serial="x",
            C_th=twin.C_th,
            R_th=twin.R_th,
            k_pwr=twin.k_pwr,
            V_oc=12.0,
            R_int=0.4,
            k_i=(1.0, 1.0),  # wrong length
        )


def test_battery_twin_constructs(twin, dims):
    _, nv = dims
    twb = _battery_twin(twin, nv)
    assert twb.has_battery
    assert twb.n_servos == nv


# --- attestation handle (ADR-008 extended off-body) ------------------


def test_handle_is_deterministic_hex(twin):
    h = twin.attestation_handle
    assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)
    assert h == so101_twin("SO101-0001").attestation_handle


def test_handle_distinct_for_distinct_serial(twin):
    # Two units, identical parameters, different serials → distinct handles.
    assert twin.attestation_handle != twin.with_serial("SO101-9999").attestation_handle


def test_handle_sensitive_to_offbody_params(twin):
    bumped = eqx.tree_at(lambda t: t.C_th, twin, twin.C_th * 1.01)
    assert bumped.attestation_handle != twin.attestation_handle
    bumped_k = eqx.tree_at(lambda t: t.k_pwr, twin, twin.k_pwr + 0.1)
    assert bumped_k.attestation_handle != twin.attestation_handle


def test_handle_sensitive_to_robot_params(twin):
    damping = float(twin.robot.parameters()["shoulder_pan.damping"])
    new_robot = twin.robot.with_parameters({"shoulder_pan.damping": damping + 0.1})
    moved = eqx.tree_at(lambda t: t.robot, twin, new_robot)
    assert moved.attestation_handle != twin.attestation_handle


def test_handle_sensitive_to_provenance(twin):
    with_prov = Twin(
        twin.robot,
        serial=twin.serial,
        C_th=twin.C_th,
        R_th=twin.R_th,
        k_pwr=twin.k_pwr,
        servo_names=twin.servo_names,
        provenance=(("sysid.thermal", "1.0", "tel-hash"),),
    )
    assert with_prov.attestation_handle != twin.attestation_handle


def test_handle_insensitive_to_offbody_representation(twin, dims):
    """The cosmetic set: how the off-body leaves are *spelled* must not move it.

    Passing the thermal params as Python lists vs jax arrays of the same values
    yields the same handle (the Twin coerces to canonical float64; byte-order /
    endianness invariance itself is exercised at the `compute_handle` level in
    `test/diagnostics/test_attestation.py`).
    """
    _, nv = dims
    as_arrays = twin
    as_lists = Twin(
        twin.robot,
        serial=twin.serial,
        C_th=[float(x) for x in twin.C_th],
        R_th=[float(x) for x in twin.R_th],
        k_pwr=[float(x) for x in twin.k_pwr],
        T_amb=twin.T_amb,
        servo_names=twin.servo_names,
    )
    assert as_lists.attestation_handle == as_arrays.attestation_handle


def test_battery_handle_differs_from_bench(twin, dims):
    _, nv = dims
    twb = _battery_twin(twin, nv)
    # Same serial-free comparison: bench vs battery with otherwise-equal params.
    bench_same_serial = twin.with_serial("LK-0001")
    assert twb.attestation_handle != bench_same_serial.attestation_handle


def test_building_twin_does_not_change_robot_handle(twin):
    """Composing a twin must not mutate the underlying calibrated robot (#7)."""
    from jaxterity import zoo

    assert twin.robot.attestation_handle == zoo.calibrated("so100").attestation_handle


# --- PyTree ----------------------------------------------------------------


def test_pytree_round_trip(twin, dims):
    leaves, treedef = jax.tree_util.tree_flatten(twin)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt.attestation_handle == twin.attestation_handle
    nq, nv = dims
    q0, qd0, T0 = jnp.zeros(nq), jnp.zeros(nv), jnp.full(nv, twin.T_amb)
    tau = jnp.tile(jnp.full(nv, 0.3), (4, 1))
    a = twin.rollout(q0, qd0, T0, tau, 0.02).temperature
    b = rebuilt.rollout(q0, qd0, T0, tau, 0.02).temperature
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


# --- composed rollout cross-checks the standalone blocks -------------------


def test_body_trajectory_matches_standalone_functional_dynamics(twin, dims):
    """Twin's body rollout == direct functional_dynamics RK4 (same integrator)."""
    nq, nv = dims
    q0, qd0, T0 = jnp.zeros(nq), jnp.zeros(nv), jnp.full(nv, twin.T_amb)
    tau_vec = jnp.full(nv, 0.4)
    n_steps, dt = 12, 0.02
    tau = jnp.tile(tau_vec, (n_steps, 1))
    out = twin.rollout(q0, qd0, T0, tau, dt)

    f = twin.robot.functional_dynamics()

    def rk4(state, u):
        k1 = f(state, u)
        k2 = f(state + 0.5 * dt * k1, u)
        k3 = f(state + 0.5 * dt * k2, u)
        k4 = f(state + dt * k3, u)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    state = jnp.concatenate([q0, qd0])
    for _ in range(n_steps):
        state = rk4(state, tau_vec)
    np.testing.assert_allclose(
        np.asarray(out.q[-1]),
        np.asarray(state[:nq]),
        atol=tol.atol("joint_position", "rk4", "float64"),
        rtol=tol.rtol("joint_position", "rk4", "float64"),
    )


def test_temperature_matches_analytic_first_order(twin, dims):
    """Twin per-servo temperature == analytic thermal_response (constant torque)."""
    nq, nv = dims
    q0, qd0 = jnp.zeros(nq), jnp.zeros(nv)
    T_amb = twin.T_amb
    T0 = jnp.full(nv, T_amb)
    tau_value = 0.5
    tau_vec = jnp.full(nv, tau_value)
    n_steps, dt = 40, 0.05
    tau = jnp.tile(tau_vec, (n_steps, 1))
    out = twin.rollout(q0, qd0, T0, tau, dt)
    t_end = n_steps * dt
    # per-servo params are uniform; each node follows the scalar analytic rise.
    T_exact = float(
        thermal_response(
            t_end,
            tau_value,
            T0=T_amb,
            T_amb=T_amb,
            C_th=float(twin.C_th[0]),
            R_th=float(twin.R_th[0]),
            k_pwr=float(twin.k_pwr[0]),
        )
    )
    for i in range(nv):
        assert float(out.temperature[-1, i]) == pytest.approx(
            T_exact,
            abs=tol.atol("motor_temperature", "rk4", "float64"),
            rel=tol.rtol("motor_temperature", "rk4", "float64"),
        )


def test_bus_voltage_matches_standalone(twin, dims):
    """Battery twin's bus voltage == standalone bus_voltage at each torque."""
    nq, nv = dims
    twb = _battery_twin(twin, nv)
    q0, qd0, T0 = jnp.zeros(nq), jnp.zeros(nv), jnp.full(nv, twb.T_amb)
    rng = np.random.default_rng(0)
    tau = jnp.asarray(rng.uniform(-1.0, 1.0, size=(8, nv)))
    out = twb.rollout(q0, qd0, T0, tau, 0.02)
    k_i = jnp.asarray(twb.k_i)
    for k in range(tau.shape[0]):
        v_ref = float(
            bus_voltage(
                tau[k], V_oc=twb.V_oc, R_int=twb.R_int, k_i=k_i, i_eps=twb.i_eps
            )
        )
        assert float(out.bus_voltage[k]) == pytest.approx(
            v_ref,
            abs=tol.atol("bus_voltage", "rk4", "float64"),
            rel=tol.rtol("bus_voltage", "rk4", "float64"),
        )


# --- vmap over a fleet (off-body leaves) -----------------------------------


def test_vmap_over_fleet_offbody(twin, dims):
    """A fleet (units sharing the body, differing off-body) vmaps in one call."""
    nq, nv = dims
    robot = twin.robot
    q0, qd0, T0 = jnp.zeros(nq), jnp.zeros(nv), jnp.full(nv, twin.T_amb)
    tau = jnp.tile(jnp.full(nv, 0.5), (10, 1))

    def final_temp(C_th, R_th, k_pwr):
        unit = Twin(
            robot,
            serial="fleet",
            C_th=C_th,
            R_th=R_th,
            k_pwr=k_pwr,
            T_amb=twin.T_amb,
            servo_names=twin.servo_names,
        )
        return unit.rollout(q0, qd0, T0, tau, 0.05).temperature[-1]

    C_batch = jnp.stack([twin.C_th, twin.C_th * 2.0])  # unit B has 2× heat capacity
    R_batch = jnp.stack([twin.R_th, twin.R_th])
    K_batch = jnp.stack([twin.k_pwr, twin.k_pwr])
    out = jax.vmap(final_temp)(C_batch, R_batch, K_batch)
    assert out.shape == (2, nv)
    # higher C_th → slower rise → lower temperature at the same time.
    assert float(out[1, 0]) < float(out[0, 0])


# --- gradients -------------------------------------------------------------


def test_grad_to_offbody_leaves_matches_finite_difference(twin, dims):
    """Reverse-mode grad of final temperature wrt off-body leaves vs central FD."""
    nq, nv = dims
    robot = twin.robot
    q0, qd0, T0 = jnp.zeros(nq), jnp.zeros(nv), jnp.full(nv, twin.T_amb)
    tau = jnp.tile(jnp.full(nv, 0.5), (10, 1))

    def loss(k_pwr0):
        k_pwr = twin.k_pwr.at[0].set(k_pwr0)
        unit = Twin(
            robot,
            serial="g",
            C_th=twin.C_th,
            R_th=twin.R_th,
            k_pwr=k_pwr,
            T_amb=twin.T_amb,
            servo_names=twin.servo_names,
        )
        return unit.rollout(q0, qd0, T0, tau, 0.05).temperature[-1, 0]

    x0 = float(twin.k_pwr[0])
    g = float(jax.grad(loss)(x0))
    h = 1e-4
    fd = (float(loss(x0 + h)) - float(loss(x0 - h))) / (2 * h)
    assert g == pytest.approx(
        fd, rel=tol.gradient_rtol("float64"), abs=tol.gradient_atol("float64")
    )


def test_robot_leaves_remain_differentiable_via_parameter_jacobian(twin, dims):
    """Robot leaves stay differentiable in the twin via the forward-mode path."""
    from jaxterity.sysid import parameter_jacobian

    nq, nv = dims
    q0, qd0 = jnp.zeros(nq), jnp.zeros(nv)
    key = "shoulder_pan.damping"
    jac = parameter_jacobian(
        twin.robot,
        lambda s: jnp.sum(s[nq:] ** 2),
        {key: twin.robot.parameters()[key]},
        t_span=(0.0, 0.05),
        q0=q0,
        qd0=qd0,
    )
    assert bool(jnp.all(jnp.isfinite(jac[key])))


# --- to_diagram ------------------------------------------------------------


def test_to_diagram_bench_simulates(twin):
    diagram = twin.to_diagram()
    res = jaxonomy.simulate(diagram, diagram.create_context(), (0.0, 0.05))
    assert res is not None


def test_to_diagram_battery_simulates(twin, dims):
    _, nv = dims
    twb = _battery_twin(twin, nv)
    diagram = twb.to_diagram()
    res = jaxonomy.simulate(diagram, diagram.create_context(), (0.0, 0.05))
    assert res is not None
