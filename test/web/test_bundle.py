# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""The web bundle is well-formed: files present, MJCF valid, policy exported."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

pytest.importorskip("mujoco")
jax = pytest.importorskip("jax")
import equinox as eqx  # noqa: E402

from jaxterity.web import bundle as web_bundle  # noqa: E402

_ROOT = pathlib.Path(__file__).resolve().parents[2]

_EXPECTED_FILES = {
    "index.html",
    "style.css",
    "app.js",
    "policy.js",
    "physics.js",
    "MUJOCO_WASM.md",
    "policy.json",
    "cartpole.xml",
}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    web = _load("njweb", _ROOT / "examples" / "nanojaxterity_web.py")
    nj = _load("nanojaxterity_demo", _ROOT / "examples" / "nanojaxterity.py")
    if not nj.POLICY_PATH.exists():
        pytest.skip("nanojaxterity policy not committed")
    out = tmp_path_factory.mktemp("bundle") / "dist"
    web.build(out)
    return out, nj


def test_bundle_has_all_files(built):
    out, _ = built
    present = {p.name for p in out.iterdir()}
    assert _EXPECTED_FILES <= present, _EXPECTED_FILES - present


def test_exported_mjcf_loads_in_mujoco(built):
    import mujoco

    out, _ = built
    model = mujoco.MjModel.from_xml_string((out / "cartpole.xml").read_text())
    assert model.nq == 2  # cart slide + pole hinge
    assert model.ngeom >= 2  # at least the cart + pole visual geoms


def test_policy_json_schema_and_weights(built):
    out, nj = built
    payload = json.loads((out / "policy.json").read_text())
    assert payload["schema"] == "jaxterity.web/cartpole-actorcritic/1"
    assert payload["actions"] == list(nj.ACTIONS)
    assert payload["obs_scale"] == list(nj.OBS_SCALE)
    for key in ("g", "mc", "mp", "L", "cart_damping", "dt", "theta_fail"):
        assert key in payload["physics"]

    # the exported weights are exactly the committed actor's
    model = eqx.tree_deserialise_leaves(
        nj.POLICY_PATH, nj.ActorCritic(jax.random.PRNGKey(0))
    )
    layers = payload["mlp"]["layers"]
    assert len(layers) == len(model.actor.layers)
    for got, layer in zip(layers, model.actor.layers):
        np.testing.assert_allclose(
            np.asarray(got["W"]), np.asarray(layer.weight), rtol=0, atol=0
        )
        np.testing.assert_allclose(
            np.asarray(got["b"]), np.asarray(layer.bias), rtol=0, atol=0
        )


def test_index_html_references_assets(built):
    out, _ = built
    html = (out / "index.html").read_text()
    assert "./app.js" in html and "./style.css" in html


def test_write_bundle_overwrites_cleanly(built, tmp_path):
    out, nj = built
    dst = tmp_path / "dst"
    payload = json.loads((out / "policy.json").read_text())
    mjcf = (out / "cartpole.xml").read_text()
    web_bundle.write_bundle(dst, mjcf=mjcf, payload=payload)
    web_bundle.write_bundle(dst, mjcf=mjcf, payload=payload)  # second write replaces
    assert (dst / "policy.json").exists()
