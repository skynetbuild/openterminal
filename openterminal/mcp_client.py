"""MCP (Model Context Protocol) client.

Connects to servers declared in config at startup and wraps each tool they
expose as an ordinary `Tool` (tools/base.py) — the agent loop never has to
know a given call is going out over stdio or HTTP instead of running
in-process. Same adapter idea as the provider layer: one small class
translates a foreign shape into ours, and everything above it stays generic.

MCP servers are arbitrary code (often a `npx ...` one-liner someone pasted
from a README) — every MCP tool is `dangerous=True` regardless of what it
claims to do, so it always goes through the permission gate.
"""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from openterminal.agent.context import AgentContext
from openterminal.tools.base import Tool, ToolRunResult

logger = logging.getLogger("openterminal.mcp")


@dataclass
class MCPServerConfig:
    name: str
    command: str | None = None  # stdio server: the executable
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    url: str | None = None  # streamable-HTTP server instead of stdio; mutually exclusive with `command`


class MCPTool(Tool):
    """One tool from one MCP server. Namespaced as mcp__{server}__{tool} so
    two servers exposing a same-named tool (e.g. both have "search") don't
    collide in the model's tool list."""

    dangerous = True

    def __init__(self, server_name: str, mcp_tool: Any, manager: "MCPManager") -> None:
        self.server_name = server_name
        self.mcp_tool_name = mcp_tool.name
        self.name = f"mcp__{server_name}__{mcp_tool.name}"
        self.description = f"[MCP:{server_name}] {mcp_tool.description or mcp_tool.name}"
        self.parameters = mcp_tool.input_schema or {"type": "object", "properties": {}}
        self._manager = manager

    def summary(self, args: dict[str, Any]) -> str:
        arg_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
        return f"{self.server_name}.{self.mcp_tool_name}({arg_str})"

    async def run(self, args: dict[str, Any], ctx: AgentContext) -> ToolRunResult:
        ok = await ctx.permissions.check(self.name, self.summary(args))
        if not ok:
            return ToolRunResult(content="User declined this MCP tool call.", is_error=True)

        session = self._manager.session(self.server_name)
        if session is None:
            return ToolRunResult(content=f"MCP server '{self.server_name}' is not connected.", is_error=True)

        try:
            result = await session.call_tool(self.mcp_tool_name, arguments=args)
        except Exception as e:  # noqa: BLE001 — a broken MCP server shouldn't crash the loop
            return ToolRunResult(content=f"MCP call failed: {e}", is_error=True)

        text = "\n".join(getattr(block, "text", "") for block in result.content if getattr(block, "text", None))
        return ToolRunResult(content=text or "(no output)", is_error=bool(result.is_error))


class MCPManager:
    """Owns the connections for the life of one CLI session. `connect_all`
    opens every configured server (skipping — with a warning, not a crash —
    any that fail) and returns the flattened tool list; `close` tears every
    connection down together via one AsyncExitStack."""

    def __init__(self) -> None:
        self._stack = AsyncExitStack()
        self._sessions: dict[str, Any] = {}

    def session(self, server_name: str) -> Any | None:
        return self._sessions.get(server_name)

    @property
    def connected_servers(self) -> list[str]:
        return list(self._sessions.keys())

    async def connect_all(self, servers: list[MCPServerConfig]) -> list[Tool]:
        tools: list[Tool] = []
        for s in servers:
            try:
                session = await self._connect_one(s)
            except Exception as e:  # noqa: BLE001 — one bad server config shouldn't block the others
                logger.warning("MCP server '%s' failed to connect: %s", s.name, e)
                continue
            self._sessions[s.name] = session
            try:
                listed = await session.list_tools()
            except Exception as e:  # noqa: BLE001
                logger.warning("MCP server '%s' connected but list_tools failed: %s", s.name, e)
                continue
            for t in listed.tools:
                tools.append(MCPTool(s.name, t, self))
        return tools

    async def _connect_one(self, s: MCPServerConfig) -> Any:
        from mcp import ClientSession

        if s.url:
            from mcp.client.streamable_http import streamable_http_client

            read, write = await self._stack.enter_async_context(streamable_http_client(s.url))
        elif s.command:
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client

            params = StdioServerParameters(command=s.command, args=s.args, env=s.env)
            read, write = await self._stack.enter_async_context(stdio_client(params))
        else:
            raise ValueError(f"MCP server '{s.name}' has neither `command` nor `url` set.")

        session = await self._stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return session

    async def close(self) -> None:
        await self._stack.aclose()
