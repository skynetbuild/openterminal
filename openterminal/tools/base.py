"""The Tool contract.

A tool declares its own JSON-Schema `parameters` (so it doubles as the spec
sent to the model — no separate description to keep in sync) and returns a
`ToolRunResult`: `content` is what goes back to the model as text, `display`
is an optional Rich renderable the UI shows the human instead of (or as well
as) that raw text — a diff instead of a wall of unified-diff text, a
truncated file listing instead of 400 raw paths.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from openterminal.agent.context import AgentContext
from openterminal.types import ToolSpec


@dataclass
class ToolRunResult:
    content: str
    is_error: bool = False
    display: Any = None


class Tool(ABC):
    name: str
    description: str
    parameters: dict[str, Any]
    # Gates the permission prompt (agent/permissions.py). Reads are cheap and
    # reversible by nature — nothing about looking at a file needs a human in
    # the loop. Anything that writes to disk or spawns a process does.
    dangerous: bool = False

    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, parameters=self.parameters)

    def summary(self, args: dict[str, Any]) -> str:
        """One line shown in the permission prompt and the transcript before
        the tool has actually run — override for anything where the raw args
        dict isn't already a good enough summary."""
        return f"{self.name}({', '.join(f'{k}={v!r}' for k, v in args.items())})"

    @abstractmethod
    async def run(self, args: dict[str, Any], ctx: AgentContext) -> ToolRunResult:
        raise NotImplementedError
