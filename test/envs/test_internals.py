# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Training-time internals layer.

The contract a paired (A/B) training study leans on, in order of import:
neutral mode is an *exact* no-op; the thermal state matches the standalone
thermal model; the actuation feedback follows the declared K_t(T)/throttle
forms; the brownout respects its dwell; and the condition sampler's
``correlated`` flag changes the joint distribution while preserving every
marginal.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxterity.envs import (
    InternalsParams,
    actuation_scale,
    init_state,
    sample_conditions,
    step_internals,
)
from jaxterity.errors import JaxterityError


def _hot_params(n=3) -> InternalsParams:
    return InternalsParams(
        C_th=jnp.full(n, 50.0),
        R_th=jnp.full(n, 2.0),
        k_pwr=jnp.full(n, 0.5),
        alpha_kt=jnp.full(n, 0.002),
        T_ref=298.15,
        T_throttle=jnp.full(n, 350.0),
        T_max=jnp.full(n, 380.0),
        throttle_floor=0.3,
        V_oc=48.0,
        R_int=0.05,
        k_i=jnp.full(n, 2.0),
        V_min=40.0,
        brownout_dwell_s=0.02,
        T_amb=298.15,
    )


def test_neutral_is_an_exact_noop():
    """The A arm: scale ≡ 1.0 bit-exactly, brownout unreachable."""
    params = InternalsParams.neutral(4)
    state = init_state(params)
    scale = np.asarray(actuation_scale(params, state))
    assert np.array_equal(scale, np.ones(4))  # exact, not approx

    # Even under absurd sustained torque: state evolves, nothing feeds back.
    tau = jnp.full(4, 500.0)
    for _ in range(100):
        state = step_internals(params, state, tau, dt=0.01)
    assert np.array_equal(np.asarray(actuation_scale(params, state)), np.ones(4))
    assert not bool(state.browned_out)
    assert float(state.V_bus) == 48.0  # R_int = 0 ⇒ no sag at all


def test_thermal_matches_the_standalone_model():
    """The Euler chain reproduces the closed-form first-order response."""
    from jaxterity.physics import thermal_response

    params = _hot_params(1)
    tau = jnp.asarray([4.0])
    dt, n_steps = 0.005, 4000  # 20 s
    state = init_state(params)
    for _ in range(n_steps):
        state = step_internals(params, state, tau, dt)
    expected = float(
        thermal_response(
            jnp.asarray(n_steps * dt),
            tau=4.0,
            C_th=50.0,
            R_th=2.0,
            k_pwr=0.5,
            T_amb=298.15,
            T0=298.15,
        )
    )
    assert float(state.T[0]) == pytest.approx(expected, rel=1e-3)


def test_actuation_feedback_forms():
    """K_t falls linearly with T; the throttle derates inside its band."""
    params = _hot_params(3)
    # Motor 0 at ref (scale 1), motor 1 hot (K_t down, below throttle band),
    # motor 2 inside the throttle band.
    state = init_state(params)._replace(T=jnp.asarray([298.15, 348.15, 365.0]))
    scale = np.asarray(actuation_scale(params, state))
    assert scale[0] == pytest.approx(1.0)
    assert scale[1] == pytest.approx(1.0 - 0.002 * 50.0)  # K_t only
    kt2 = 1.0 - 0.002 * (365.0 - 298.15)
    throttle2 = (380.0 - 365.0) / 30.0  # inside the band
    assert scale[2] == pytest.approx(kt2 * throttle2, rel=1e-6)

    # The floor holds at/above T_max.
    state = init_state(params)._replace(T=jnp.full(3, 500.0))
    floor_scale = np.asarray(actuation_scale(params, state))
    assert np.all(floor_scale <= 0.3 + 1e-9)
    assert np.all(floor_scale >= 0.0)


def test_brownout_respects_the_dwell():
    params = _hot_params(3)
    state = init_state(params)
    # Aggregate draw: V = 48 − 0.05 · Σ 2·|τ| ⇒ τ = 30 each ⇒ V = 39 < 40.
    heavy = jnp.full(3, 30.0)
    light = jnp.zeros(3)
    # A brief 10 ms dip (< 20 ms dwell) does not latch …
    state = step_internals(params, state, heavy, dt=0.010)
    state = step_internals(params, state, light, dt=0.010)
    assert not bool(state.browned_out)
    assert float(state.brownout_timer) == 0.0  # reset on recovery
    # … a sustained sag does.
    for _ in range(3):
        state = step_internals(params, state, heavy, dt=0.010)
    assert bool(state.browned_out)
    # Latched: recovery does not clear it (the controller already reset).
    state = step_internals(params, state, light, dt=0.010)
    assert bool(state.browned_out)


def test_apply_actuation_scale_matches_drift_semantics():
    pytest.importorskip("mujoco")
    from jaxterity.envs import apply_actuation_scale
    from jaxterity.locomotion import G1JoystickController

    controller = G1JoystickController()
    model = controller.mjx_model
    ids = jnp.asarray([3, 7])
    scaled = apply_actuation_scale(model, ids, jnp.asarray([0.5, 0.8]))
    assert float(scaled.actuator_gainprm[3, 0]) == pytest.approx(
        0.5 * float(model.actuator_gainprm[3, 0])
    )
    assert float(scaled.actuator_biasprm[7, 1]) == pytest.approx(
        0.8 * float(model.actuator_biasprm[7, 1])
    )
    untouched = 5
    assert float(scaled.actuator_gainprm[untouched, 0]) == pytest.approx(
        float(model.actuator_gainprm[untouched, 0])
    )


def test_conditions_marginals_match_correlation_differs():
    """The A/B contrast: identical marginals, different joint distribution."""
    n = 4000
    keys = jax.random.split(jax.random.PRNGKey(0), n)
    correlated = jax.vmap(lambda k: sample_conditions(k, correlated=True))(keys)
    independent = jax.vmap(lambda k: sample_conditions(k, correlated=False))(keys)

    for channel in ("friction_scale", "R_int_scale", "damping_scale"):
        a = np.asarray(getattr(correlated, channel))
        b = np.asarray(getattr(independent, channel))
        assert a.mean() == pytest.approx(b.mean(), rel=0.02)
        assert a.std() == pytest.approx(b.std(), rel=0.05)

    def corrcoef(draws):
        return float(
            np.corrcoef(
                np.asarray(draws.friction_scale), np.asarray(draws.R_int_scale)
            )[0, 1]
        )

    assert corrcoef(correlated) > 0.99  # one latent temperature
    assert abs(corrcoef(independent)) < 0.05  # independent latents

    # Cold draws push every channel up together in the correlated world.
    cold = np.asarray(correlated.T_amb) < 273.15
    assert np.all(np.asarray(correlated.friction_scale)[cold] > 1.0)
    assert np.all(np.asarray(correlated.R_int_scale)[cold] > 1.0)


def test_params_validation():
    with pytest.raises(JaxterityError, match="shape"):
        InternalsParams.neutral(3).__class__(
            **{
                **dataclasses_asdict(InternalsParams.neutral(3)),
                "R_th": jnp.ones(2),
            }
        )
    with pytest.raises(JaxterityError, match="throttle_floor"):
        InternalsParams.neutral(3).__class__(
            **{
                **dataclasses_asdict(InternalsParams.neutral(3)),
                "throttle_floor": 0.0,
            }
        )


def dataclasses_asdict(params):
    import dataclasses

    return {f.name: getattr(params, f.name) for f in dataclasses.fields(params)}


def test_with_welded_joints_carves_an_arm():
    pytest.importorskip("mujoco")
    from jaxterity import zoo
    from jaxterity.robot import JointType

    robot = zoo.load("unitree_g1")
    leg_parts = ("ankle", "hip", "knee")
    arm = tuple(
        j.name
        for j in robot.joints
        if j.name.startswith("right_") and not any(part in j.name for part in leg_parts)
    )
    weld = tuple(j.name for j in robot.joints if j.name not in arm)
    arm_robot = robot.with_welded_joints(weld)
    assert sum(j.joint_type is JointType.REVOLUTE for j in arm_robot.joints) == len(arm)
    system = arm_robot.build_system(actuation=True)
    assert system._nv == len(arm) == 7
    # Dynamics runs on the subset.
    f = arm_robot.functional_dynamics()
    out = f(jnp.zeros(2 * len(arm)), jnp.zeros(len(arm)))
    assert out.shape == (2 * len(arm),)
    # Structural change ⇒ new identity.
    assert arm_robot.attestation_handle != robot.attestation_handle

    with pytest.raises(JaxterityError, match="unknown joints"):
        robot.with_welded_joints(("bogus",))
    floating = zoo.unitree_g1.load(floating_base=True)
    with pytest.raises(JaxterityError, match="floating base"):
        floating.with_welded_joints(("floating_base",))
