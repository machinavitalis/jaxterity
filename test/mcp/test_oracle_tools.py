# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""The oracle verbs as MCP tools: JSON in, physics out, provenance attached.

These cover the tool layer's own contract — handle lifecycle, JSON shapes, the
adapters that turn agent-supplied dicts into envelopes and semantic priors, and
the labels that keep a simulated answer from reading as a measurement. The
physics itself is tested in ``test/test_oracle.py``.

The module imports without ``fastmcp`` (the tools are plain functions); only
the server-registration test needs the ``[mcp]`` extra.
"""

from __future__ import annotations

import json

import pytest

from jaxterity.errors import JaxterityError
from jaxterity.mcp import oracle_tools

pytest.importorskip("mujoco")


@pytest.fixture
def handle():
    oracle_tools.clear_twin_registry()
    yield oracle_tools.create_twin_handle("so101", "SO101-MCP")
    oracle_tools.clear_twin_registry()


@pytest.fixture
def envelope():
    return {"temp_ceiling_k": 320.0, "torque_margin_frac": 0.9}


# --- handle lifecycle -------------------------------------------------------


def test_handle_lifecycle_and_info(handle):
    assert handle.startswith("@twin-")
    assert oracle_tools.list_twin_handles() == [handle]
    info = oracle_tools.twin_info(handle)
    assert info["serial"] == "SO101-MCP"
    assert info["n_servos"] == 6
    assert info["has_battery"] is False
    assert len(info["attestation_handle"]) > 16
    # A zoo twin carries representative defaults and no fit covariance — and
    # the tool says so rather than letting an agent assume otherwise.
    assert info["uncertain_leaves"] == []
    assert "not a per-unit calibration" in info["provenance_note"]
    oracle_tools.release_twin_handle(handle)
    assert oracle_tools.list_twin_handles() == []


def test_handle_is_stable_for_the_same_unit():
    oracle_tools.clear_twin_registry()
    first = oracle_tools.create_twin_handle("so101", "SO101-A")
    second = oracle_tools.create_twin_handle("so101", "SO101-A")
    other = oracle_tools.create_twin_handle("so101", "SO101-B")
    assert first == second and first != other
    oracle_tools.clear_twin_registry()


def test_unknown_unit_and_handle_are_refused():
    with pytest.raises(JaxterityError, match="unknown twin unit"):
        oracle_tools.create_twin_handle("humanoid-9000", "X")
    with pytest.raises(JaxterityError, match="unknown twin handle"):
        oracle_tools.twin_info("@twin-deadbeefdeadbeef")


# --- simulate_plan ----------------------------------------------------------


def test_simulate_plan_returns_json_serializable_answer(handle, envelope):
    result = oracle_tools.oracle_simulate_plan(
        handle,
        tau=[0.05] * 6,
        n_steps=10,
        dt=0.005,
        envelope=envelope,
        name="gentle hold",
    )
    json.dumps(result)  # the whole answer must survive the wire
    assert result["outcome"] == "pass"
    assert result["question"] == "gentle hold"
    assert result["twin_handle"] == oracle_tools.twin_info(handle)["attestation_handle"]
    names = {m["name"] for m in result["margins"]}
    assert {"servo_temperature", "torque"} <= names
    assert all("units" in m for m in result["margins"])


def test_simulate_plan_reports_a_thermal_veto_with_time_to_breach(handle, envelope):
    result = oracle_tools.oracle_simulate_plan(
        handle, tau=[1.5] * 6, n_steps=10, dt=0.005, envelope=envelope, name="cook"
    )
    assert result["outcome"] == "veto"
    assert result["binding_constraint"] == "servo_temperature"
    assert result["time_to_breach_s"] > 0.0


def test_mistyped_limit_is_refused_not_ignored(handle):
    with pytest.raises(JaxterityError, match="unknown limit"):
        oracle_tools.oracle_simulate_plan(
            handle,
            tau=[0.05] * 6,
            n_steps=4,
            dt=0.005,
            envelope={"temp_ceiling": 320.0},
        )


def test_unknown_base_profile_is_refused(handle, envelope):
    with pytest.raises(JaxterityError, match="unknown base profile"):
        oracle_tools.oracle_simulate_plan(
            handle,
            tau=[0.05] * 6,
            n_steps=4,
            dt=0.005,
            envelope=envelope,
            base="forklift",
        )


# --- what_if ----------------------------------------------------------------


def test_what_if_ranks_candidates(handle, envelope):
    result = oracle_tools.oracle_what_if(
        handle,
        candidates=[
            {"name": "fast", "tau": [1.5] * 6, "n_steps": 10, "dt": 0.005},
            {"name": "slow", "tau": [0.05] * 6, "n_steps": 10, "dt": 0.005},
        ],
        envelope=envelope,
    )
    json.dumps(result)
    assert result["best"] == "slow"
    assert result["ranking"] == ["slow", "fast"]
    assert len(result["answers"]) == 2


def test_what_if_names_a_malformed_candidate(handle, envelope):
    with pytest.raises(JaxterityError, match="missing"):
        oracle_tools.oracle_what_if(
            handle,
            candidates=[{"name": "no torque", "n_steps": 4, "dt": 0.005}],
            envelope=envelope,
        )


# --- identify_payload -------------------------------------------------------


def test_identify_payload_labels_the_simulated_transport():
    result = oracle_tools.oracle_identify_payload(
        "cartpole",
        "pole",
        offset=[0.0, 0.0, 0.6],
        planted_mass=0.2,
        fit=["mass"],
        probe_speed=0.3,
        iterations=4,
        uncertainty=None,
    )
    json.dumps(result)
    assert result["transport"] == "loopback-simulated"
    assert result["params"]["payload.mass"] == pytest.approx(0.2, abs=1e-2)
    assert result["recipe"] == "objects.payload"


def test_identify_payload_accepts_a_material_prior_and_declares_it():
    result = oracle_tools.oracle_identify_payload(
        "cartpole",
        "pole",
        offset=[0.0, 0.0, 0.6],
        planted_mass=0.2,
        prior={
            "label": "steel tip weight",
            "material": "steel",
            "extent_m": [0.03, 0.03, 0.035],
            "source": "vlm/test",
        },
        fit=["mass"],
        probe_speed=0.3,
        iterations=4,
        uncertainty=None,
    )
    assert result["recipe"] == "objects.payload+semantic_prior"
    assert any("semantic prior" in w for w in result["warnings"])


def test_malformed_priors_are_refused():
    with pytest.raises(JaxterityError, match="needs `extent_m`"):
        oracle_tools._prior_from_mapping({"label": "x", "material": "steel"})
    with pytest.raises(JaxterityError, match="cannot bound the search"):
        oracle_tools._prior_from_mapping({"label": "x", "source": "s"})
    with pytest.raises(JaxterityError, match="unexpected key"):
        oracle_tools._prior_from_mapping(
            {"label": "x", "mass_kg": [0.1, 0.2], "source": "s", "colour": "red"}
        )


# --- server registration ----------------------------------------------------


def test_oracle_tools_are_registered_on_the_server():
    pytest.importorskip("fastmcp")
    import asyncio

    from jaxterity.mcp import server

    tools = asyncio.run(server.build_server().list_tools())
    names = {t.name for t in tools}
    assert {
        "create_twin_handle",
        "oracle_simulate_plan",
        "oracle_what_if",
        "oracle_identify_payload",
    } <= names
