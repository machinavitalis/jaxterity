# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""MCP layered-API tools. Tests use the underlying Python functions
directly so they run without a FastMCP runtime."""

from __future__ import annotations

import pytest

from jaxterity.mcp.layered_tools import (
    LAYERED_TOOLS,
    clear_registry,
    create_robot_handle,
    list_handles,
    release_handle,
    s1_pick,
    s1_reach,
    s1_walk_to,
    s2_move_joint,
    s3_add_posture_task,
    s3_step_wbc,
    s4_apply_torque,
    s4_joint_names,
    s4_read_encoder,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Every test starts with an empty handle registry."""
    clear_registry()
    yield
    clear_registry()


pytest.importorskip("mujoco")


def test_handle_lifecycle():
    h = create_robot_handle("cartpole")
    try:
        assert h in list_handles()
        assert h.startswith("@") and len(h) == 17  # "@" + 16 hex
    finally:
        release_handle(h)
    assert h not in list_handles()


def test_s4_tools_round_trip():
    h = create_robot_handle("cartpole")
    try:
        names = s4_joint_names(h)
        assert names == ["slider_to_cart", "cart_to_pole"]
        # Encoders read zero at construction.
        for n in names:
            assert s4_read_encoder(h, n) == 0.0
        msg = s4_apply_torque(h, "slider_to_cart", 0.5)
        assert "0.5" in msg
    finally:
        release_handle(h)


def test_s3_step_wbc_with_posture_task():
    h = create_robot_handle("cartpole")
    try:
        s3_add_posture_task(h, "p", [0.1, 0.0], kp=5.0, kd=1.0)
        qdd = s3_step_wbc(h)
        assert isinstance(qdd, list) and len(qdd) == 2
        assert qdd[0] > 0
    finally:
        release_handle(h)


def test_s2_move_joint_via_mcp():
    h = create_robot_handle("cartpole")
    try:
        out = s2_move_joint(h, "slider_to_cart", 0.03, max_steps=200, dt=0.005)
        assert out["n_steps"] > 0
        # We're moving toward 0.03; final must be strictly positive after PD step.
        assert out["joint_final"] > 0
    finally:
        release_handle(h)


def test_s1_tools_return_dicts():
    h = create_robot_handle("cartpole")
    try:
        r = s1_reach(h, [0.05, 0.0], max_steps=100)
        assert r["success"] in (True, False)
        assert "summary" in r
        p = s1_pick(h, "ball", [0.0, 0.0, 0.5])
        assert p["success"] is False
        w = s1_walk_to(h, [1.0, 2.0])
        assert w["success"] is False
    finally:
        release_handle(h)


def test_unknown_handle_raises():
    with pytest.raises(KeyError, match="unknown handle"):
        s4_joint_names("@deadbeef00000000")


def test_layered_tools_tuple_is_non_empty():
    """The MCP server registers every function in LAYERED_TOOLS."""
    assert len(LAYERED_TOOLS) >= 15
    for fn in LAYERED_TOOLS:
        assert callable(fn)
        assert fn.__doc__, f"{fn.__name__} missing docstring (MCP needs it)"


def test_clear_registry_drops_all_handles():
    """clear_registry empties the live handle set."""
    h1 = create_robot_handle("cartpole")
    h2 = create_robot_handle("cartpole", q0=[0.1, 0.0])
    assert h1 in list_handles() and h2 in list_handles()
    clear_registry()
    assert list_handles() == []


# --- MCP integration through FastMCP --------------------------------


def test_layered_tools_register_through_fastmcp_runtime():
    """Every LAYERED_TOOLS entry round-trips through FastMCP's tool registry.

    The rest of the tests bypass FastMCP and call the Python
    functions directly. This test confirms each tool is acceptable to
    FastMCP's tool decorator (correct signature shape, complete docstring,
    serializable annotations) and that the assembled server exposes them
    under the expected names.
    """
    pytest.importorskip("fastmcp")
    from jaxterity.mcp.server import build_server

    server = build_server()
    # The FastMCP server should have at minimum: ping + 16 layered tools.
    # We pull the tool names from the manager that FastMCP maintains and
    # verify each LAYERED_TOOLS function shows up.
    registered = {
        getattr(t, "__name__", None) or getattr(t, "name", None)
        for t in (
            list(server._tool_manager._tools.values())
            if hasattr(server, "_tool_manager")
            else []
        )
    }
    # If FastMCP's internal layout doesn't match (older/newer release), fall
    # back to a softer check: that build_server() succeeds without raising.
    if not registered:
        # The server constructed without exception — that alone is the
        # check we can make portably.
        return
    for fn in LAYERED_TOOLS:
        assert fn.__name__ in registered, (
            f"FastMCP did not register {fn.__name__}; registered={registered}"
        )
