from __future__ import annotations

from openterminal.tools.bash import BashTool
from openterminal.tools.base import Tool
from openterminal.tools.fs_read import GlobTool, ListDirTool, ReadFileTool
from openterminal.tools.fs_write import EditFileTool, WriteFileTool
from openterminal.tools.grep import GrepTool


def default_tools() -> list[Tool]:
    return [
        ReadFileTool(),
        ListDirTool(),
        GlobTool(),
        GrepTool(),
        WriteFileTool(),
        EditFileTool(),
        BashTool(),
    ]
