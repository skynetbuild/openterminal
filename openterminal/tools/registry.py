from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from openterminal.tools.bash import BashTool
from openterminal.tools.base import Tool
from openterminal.tools.dispatch_agent import DispatchAgentTool
from openterminal.tools.fs_read import GlobTool, ListDirTool, ReadFileTool
from openterminal.tools.fs_write import EditFileTool, WriteFileTool
from openterminal.tools.grep import GrepTool

if TYPE_CHECKING:
    from openterminal.config import Config


def default_tools(config: "Config | None" = None, model_id: str | None = None, cwd: Path | None = None) -> list[Tool]:
    """The built-in kit. `dispatch_agent` is only included when the caller
    passes what it needs to spin up a sub-agent (config/model/cwd) — the
    read-only tool smoke tests and anything that doesn't care about
    sub-agents can still call `default_tools()` bare."""
    tools: list[Tool] = [
        ReadFileTool(),
        ListDirTool(),
        GlobTool(),
        GrepTool(),
        WriteFileTool(),
        EditFileTool(),
        BashTool(),
    ]
    if config is not None and model_id is not None and cwd is not None:
        tools.append(DispatchAgentTool(config=config, model_id=model_id, cwd=cwd))
    return tools
