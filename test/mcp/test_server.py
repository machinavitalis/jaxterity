# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Tests for the MCP server scaffold.

The server requires the optional ``[mcp]`` extra (``fastmcp``); when it is not
installed these tests skip. On the default CI matrix ``[all,dev]`` is installed,
so they run there.
"""

import asyncio

import pytest

pytest.importorskip("fastmcp")

from jaxterity.cli import build_parser  # noqa: E402
from jaxterity.mcp import server  # noqa: E402


def test_ping_returns_pong():
    """The ``ping`` health-check tool returns ``"pong"``."""
    assert server.ping() == "pong"


def test_server_builds_and_enumerates_tools():
    """The server constructs and exposes a non-empty tool list including ping."""
    mcp = server.build_server()
    tools = asyncio.run(mcp.list_tools())
    assert len(tools) >= 1
    assert "ping" in {t.name for t in tools}


def test_tools_registry_non_empty():
    """The source-of-truth ``TOOLS`` registry is non-empty (regression guard)."""
    assert len(server.TOOLS) >= 1


def test_cli_exposes_mcp_serve():
    """The ``jaxterity mcp serve`` command parses (the CLI wiring exists)."""
    args = build_parser().parse_args(["mcp", "serve"])
    assert args.command == "mcp"
    assert args.mcp_command == "serve"
