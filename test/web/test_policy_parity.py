# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""The in-browser JS policy + physics match the JAX policy + a NumPy reference.

Runs the *generated bundle's* ``policy.js`` / ``physics.js`` under Node and
asserts:

* **policy parity** — the JS greedy action equals the JAX actor's ``argmax`` for
  the same random states (so "the browser runs the trained policy" is verified);
* **dynamics parity** — the JS closed-form derivative matches a NumPy reference
  of the same equations;
* **behaviour** — a closed-loop closed-form episode keeps ``|theta|`` within the
  failure bound (the browser demo actually balances).

Self-skips when Node or the committed policy is unavailable.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("mujoco")
jax = pytest.importorskip("jax")
import equinox as eqx  # noqa: E402

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_NODE = shutil.which("node")
_PARITY_JS = pathlib.Path(__file__).with_name("parity_node.mjs")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _reference_deriv(s, F, p):
    """NumPy port of ``physics.js`` ``cartpoleDeriv`` — the parity reference."""
    _, th, xd, thd = s
    sin, cos = np.sin(th), np.cos(th)
    tot = p["mc"] + p["mp"]
    den = tot - p["mp"] * cos * cos
    thdd = (
        p["g"] * sin * tot
        - cos * (F - p["cart_damping"] * xd + p["mp"] * p["L"] * thd * thd * sin)
        - p["pole_damping"] * thd * tot / (p["mp"] * p["L"])
    ) / (p["L"] * den)
    xdd = (
        F - p["cart_damping"] * xd + p["mp"] * p["L"] * (thd * thd * sin - thdd * cos)
    ) / tot
    return np.array([xd, thd, xdd, thdd])


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    web = _load("njweb", _ROOT / "examples" / "nanojaxterity_web.py")
    nj = _load("nanojaxterity_demo", _ROOT / "examples" / "nanojaxterity.py")
    if not nj.POLICY_PATH.exists():
        pytest.skip("nanojaxterity policy not committed")
    out = tmp_path_factory.mktemp("cpbundle") / "dist"
    web.build(out)
    return out


def _run_node(bundle_dir, states):
    states_path = bundle_dir / "_states.json"
    states_path.write_text(json.dumps([list(map(float, s)) for s in states]))
    proc = subprocess.run(
        [_NODE, str(_PARITY_JS), str(bundle_dir), str(states_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"node failed: {proc.stderr}"
    return json.loads(proc.stdout)


@pytest.mark.skipif(_NODE is None, reason="node not installed")
def test_js_policy_matches_jax_argmax(bundle):
    nj = sys.modules["nanojaxterity_demo"]
    model = eqx.tree_deserialise_leaves(
        nj.POLICY_PATH, nj.ActorCritic(jax.random.PRNGKey(0))
    )
    rng = np.random.default_rng(0)
    # states spanning the working envelope: [cart_x, theta, x_dot, theta_dot]
    states = rng.uniform([-1.0, -0.15, -1.0, -1.0], [1.0, 0.15, 1.0, 1.0], size=(64, 4))

    out = _run_node(bundle, states)
    import jax.numpy as jnp

    py_actions = [int(jnp.argmax(model(jnp.asarray(s))[0])) for s in states]
    assert out["actions"] == py_actions


@pytest.mark.skipif(_NODE is None, reason="node not installed")
def test_js_dynamics_matches_reference(bundle):
    policy = json.loads((bundle / "policy.json").read_text())
    p = policy["physics"]
    rng = np.random.default_rng(1)
    states = rng.uniform([-1.0, -0.3, -1.0, -1.0], [1.0, 0.3, 1.0, 1.0], size=(32, 4))

    out = _run_node(bundle, states)
    for s, js in zip(states, out["derivs"]):
        np.testing.assert_allclose(js, _reference_deriv(s, 5.0, p), rtol=0, atol=1e-9)


@pytest.mark.skipif(_NODE is None, reason="node not installed")
def test_browser_demo_balances(bundle):
    policy = json.loads((bundle / "policy.json").read_text())
    out = _run_node(bundle, [[0.0, 0.05, 0.0, 0.0]])
    assert out["rollout"]["maxTheta"] < policy["physics"]["theta_fail"], out["rollout"]
