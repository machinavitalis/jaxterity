# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""FastMCP server exposing Jaxterity as agent-callable tools.

Starts from a single ``ping`` health-check tool, enough to prove an
MCP-aware client (Claude Code, Cursor) can launch the server and enumerate a
tool. The layered S1–S4 API and the sysid recipes are registered here as
well, and so are the **oracle verbs** (``oracle_simulate_plan`` /
``oracle_what_if`` / ``oracle_identify_payload``) — the physics answers an
LLM planner cannot get from language. Launch with ``jaxterity mcp serve``
(stdio transport).

``fastmcp`` is an optional dependency (the ``[mcp]`` extra); importing this
module requires it. The ``jaxterity.mcp`` package ``__init__`` does not import
this module, so the rest of the package imports without ``fastmcp`` installed.
"""

from __future__ import annotations

from fastmcp import FastMCP

SERVER_NAME = "jaxterity"


def ping() -> str:
    """Health check: confirm the Jaxterity MCP server is reachable.

    Returns
    -------
    str
        Always ``"pong"``.

    Examples
    --------
    >>> ping()
    'pong'
    """
    return "pong"


# Source-of-truth registry of the tools exposed on the server. The layered
# S1-S4 + sysid recipes live in ``layered_tools`` and the oracle verbs
# (simulate_plan / what_if / identify_payload) in ``oracle_tools``; they're
# appended here so a single `mcp.tool(fn)` loop registers them all.
from .layered_tools import LAYERED_TOOLS  # noqa: E402
from .oracle_tools import ORACLE_TOOLS  # noqa: E402

TOOLS = (ping, *LAYERED_TOOLS, *ORACLE_TOOLS)


def build_server(name: str = SERVER_NAME) -> FastMCP:
    """Construct the Jaxterity FastMCP server with every tool in ``TOOLS``.

    Args
    ----
    name : str
        The server name advertised to MCP clients. Defaults to ``"jaxterity"``.

    Returns
    -------
    FastMCP
        A configured server, ready to ``.run()`` or to introspect.
    """
    mcp = FastMCP(name)
    for fn in TOOLS:
        mcp.tool(fn)  # type: ignore[call-overload]
    return mcp


def serve() -> None:
    """Run the Jaxterity MCP server on stdio (blocking).

    stdio is FastMCP's default transport — the one MCP clients spawn as a
    subprocess. Invoked by ``jaxterity mcp serve``.
    """
    build_server().run()
