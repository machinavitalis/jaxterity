# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Verifies the nanojaxterity cartpole demo actually balances.

Two tiers (mirrors ``test_sysid_so101``):

* **Fast** — load the *committed* trained policy + golden rollout, re-run the
  greedy rollout, and assert (a) it keeps the pole within the behavioural bound
  and (b) it reproduces the committed golden. No training, so it is cheap.
* **Nightly** — train PPO from scratch and assert the trained agent balances,
  proving the self-contained trainer actually converges.

Regenerate the goldens after a deliberate change with
``JAXTERITY_SAVE_GOLDEN=1 python -m examples.nanojaxterity``.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import equinox as eqx
import jax
import numpy as np
import pytest

pytest.importorskip("mujoco")

_EXAMPLES = pathlib.Path(__file__).resolve().parents[2] / "examples"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "nanojaxterity_demo", _EXAMPLES / "nanojaxterity.py"
    )
    module = importlib.util.module_from_spec(spec)
    # Register before exec: the @dataclass decorator looks the module up in
    # sys.modules while processing ``Config``.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


nj = _load_module()


def _calibrated_cartpole():
    return nj.calibrated_cartpole()


def test_env_step_is_finite_and_deterministic():
    """The functional env steps to finite states and is deterministic."""
    cfg = nj.Config()
    reset, step = nj.make_env(_calibrated_cartpole(), cfg)
    import jax.numpy as jnp

    state = jnp.array([0.0, 0.05, 0.0, 0.0])
    s1, r1, d1 = step(state, 2)  # +force
    s2, r2, d2 = step(state, 2)
    assert np.all(np.isfinite(np.asarray(s1)))
    np.testing.assert_array_equal(np.asarray(s1), np.asarray(s2))  # deterministic
    assert not bool(d1) and float(r1) <= 1.0


def test_committed_policy_balances_and_reproduces_golden():
    """The committed trained policy balances the pole and reproduces its golden.

    This is the reference check: a *behavioural* bound (the pole stays
    within ``theta_bound`` for the whole episode) plus *reproducibility* (the
    greedy rollout matches the committed golden). A confident policy's argmax is
    stable under ~1e-6 float noise, so the trajectory matches tightly across
    platforms; ``atol`` is loose enough to absorb that noise, tight enough to
    catch a real regression.
    """
    if not (nj.POLICY_PATH.exists() and nj.GOLDEN_PATH.exists()):
        pytest.skip(
            "nanojaxterity goldens not committed; regenerate with "
            "JAXTERITY_SAVE_GOLDEN=1 python -m examples.nanojaxterity"
        )
    cfg = nj.Config()
    robot = _calibrated_cartpole()
    like = nj.ActorCritic(jax.random.PRNGKey(0))
    model = eqx.tree_deserialise_leaves(nj.POLICY_PATH, like)

    traj = nj.greedy_rollout(model, robot, cfg)
    assert nj.balances(traj, cfg), (
        f"committed policy did not balance: max|θ|={np.max(np.abs(traj[:, 1])):.4f} "
        f">= bound {cfg.theta_bound}"
    )
    golden = np.load(nj.GOLDEN_PATH)["rollout"]
    np.testing.assert_allclose(traj, golden, atol=1e-3, rtol=0)


@pytest.mark.nightly
def test_ppo_trains_to_balance_from_scratch():
    """Full from-scratch PPO training converges to a balancing policy.

    Uses the shipped defaults (the same config that produced the committed
    goldens), so this also pins that the trainer is reproducible end-to-end.
    """
    cfg = nj.Config()
    robot = _calibrated_cartpole()
    model = nj.train(robot, cfg, verbose=False)
    traj = nj.greedy_rollout(model, robot, cfg)
    assert nj.balances(traj, cfg), (
        f"trained-from-scratch agent did not balance: "
        f"max|θ|={np.max(np.abs(traj[:, 1])):.4f}"
    )
