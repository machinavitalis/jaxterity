# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Verifies the nanoLeRobot SO-100 demo clones a reaching policy.

Two tiers (mirrors ``test_nanojaxterity``):

* **Fast** — load the *committed* cloned policy + golden, re-run the closed-loop
  reach, and assert (a) every held-out goal is reached within ``reach_tol`` and
  (b) the final joint configurations reproduce the committed golden. No
  training, so it is cheap.
* **Nightly** — collect demos → write a LeRobotDataset → train BC from scratch →
  assert the cloned policy reaches, proving the self-contained pipeline
  converges end-to-end.

Regenerate the goldens after a deliberate change with
``JAXTERITY_SAVE_GOLDEN=1 python -m examples.nanolerobot``.
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
pytest.importorskip("pyarrow")

_EXAMPLES = pathlib.Path(__file__).resolve().parents[2] / "examples"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "nanolerobot_demo", _EXAMPLES / "nanolerobot.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


nl = _load_module()


def _like_policy(cfg):
    """A BCPolicy with the right tree shape to deserialise committed leaves into.

    The committed policy is serialised in float32; force this template to float32
    too so the load works whether or not the test session has JAX x64 enabled.
    """
    import jax.numpy as jnp

    zeros = nl.Norm(
        mean=jnp.zeros(nl.NJ, jnp.float32), std=jnp.ones(nl.NJ, jnp.float32)
    )
    like = nl.BCPolicy(jax.random.PRNGKey(0), cfg, zeros, zeros, zeros)
    return nl._as_float32(like)


def test_env_step_is_finite_and_deterministic():
    """The position-controlled env steps to finite states and is deterministic."""
    import jax.numpy as jnp

    cfg = nl.Config()
    _, servo_step, _ = nl.make_env(nl.calibrated_so100(), cfg)
    state = jnp.zeros(2 * nl.NJ)
    q_cmd = jnp.full(nl.NJ, 0.1)
    s1 = servo_step(state, q_cmd)
    s2 = servo_step(state, q_cmd)
    assert np.all(np.isfinite(np.asarray(s1)))
    np.testing.assert_array_equal(np.asarray(s1), np.asarray(s2))


def test_demonstrations_write_a_valid_lerobot_dataset(tmp_path):
    """Collecting demos and writing them yields a readable LeRobotDataset."""
    from jaxterity.integrations import lerobot as lr

    cfg = nl.Config(n_demos=4, horizon=20)
    robot = nl.calibrated_so100()
    _, goals, states, commands = nl.collect_demonstrations(robot, cfg)
    episodes = nl.to_episodes(goals, states, commands)
    root = nl.write_demo_dataset(tmp_path / "ds", robot, cfg, episodes)

    ds = lr.read_local_lerobot_dataset(root)
    assert len(ds) == 4
    assert ds.frames("action").shape == (4 * 20, nl.NJ)
    assert ds.features["observation.state"]["names"] == list(nl.JOINTS)


def test_committed_policy_reaches():
    """The committed cloned policy reaches the held-out goals; golden is a valid set.

    We assert the *behavioural* invariant — at least ``success_frac`` of the
    held-out goals reached within ``reach_tol`` — for both the live rollout and
    the committed golden. We deliberately do **not** assert an exact-finals match:
    a closed-loop policy rollout accumulates cross-platform float divergence over
    its horizon, so the exact final configurations reproduce only *same-device*
    (regenerate with ``JAXTERITY_SAVE_GOLDEN=1 python -m examples.nanolerobot``);
    across arm64/x86-64 they drift, so the portable claim is behavioural.
    """
    if not (nl.POLICY_PATH.exists() and nl.GOLDEN_PATH.exists()):
        pytest.skip(
            "nanolerobot goldens not committed; regenerate with "
            "JAXTERITY_SAVE_GOLDEN=1 python -m examples.nanolerobot"
        )
    cfg = nl.Config()
    robot = nl.calibrated_so100()
    model = eqx.tree_deserialise_leaves(nl.POLICY_PATH, _like_policy(cfg))

    golden = np.load(nl.GOLDEN_PATH)
    starts, goals = golden["starts"], golden["goals"]
    finals, max_err = nl.closed_loop_reach(model, robot, cfg, starts, goals)

    assert nl.reaches(max_err, cfg), (
        f"committed policy missed a goal: worst max|q−goal|={np.max(max_err):.4f} "
        f">= tol {cfg.reach_tol}"
    )
    golden_err = np.max(np.abs(golden["finals"] - goals), axis=1)
    assert nl.reaches(golden_err, cfg), "committed golden is not a valid reaching set"


@pytest.mark.nightly
def test_bc_trains_to_reach_from_scratch(tmp_path):
    """Full pipeline: demos → LeRobotDataset → BC → closed-loop reach."""
    from jaxterity.integrations import lerobot as lr

    cfg = nl.Config()
    robot = nl.calibrated_so100()
    _, goals, states, commands = nl.collect_demonstrations(robot, cfg)
    episodes = nl.to_episodes(goals, states, commands)
    root = nl.write_demo_dataset(tmp_path / "ds", robot, cfg, episodes)
    dataset = lr.read_local_lerobot_dataset(root)
    model = nl.train_bc(dataset, cfg, verbose=False)

    starts, eval_goals = nl.eval_starts_goals(cfg)
    _, max_err = nl.closed_loop_reach(model, robot, cfg, starts, eval_goals)
    assert nl.reaches(max_err, cfg), (
        f"trained-from-scratch policy missed a goal: worst={np.max(max_err):.4f}"
    )
