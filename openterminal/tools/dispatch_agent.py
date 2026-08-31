"""dispatch_agent — the main agent's way to spawn a scoped sub-agent.

For a broad, self-contained sub-task ("find every place X is parsed and
summarize the formats handled"), running it inline burns the main
conversation's context on dozens of intermediate read/grep calls the user
never needed to see. This tool runs that sub-task as its own fresh
`AgentLoop` — new message list, no memory of the parent conversation — with
a read-only tool set (no write_file/edit_file/bash, and deliberately no
dispatch_agent itself: no recursive sub-agents in v1) and hands back only
the sub-agent's final answer.

Read-only means nothing it does needs a permission prompt, so the sub-agent
runs to completion unattended.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from openterminal.agent.context import AgentContext
from openterminal.agent.permissions import Decision, PermissionManager
from openterminal.tools.base import Tool, ToolRunResult
from openterminal.tools.fs_read import GlobTool, ListDirTool, ReadFileTool
from openterminal.tools.grep import GrepTool
from openterminal.types import Role

if TYPE_CHECKING:
    from openterminal.config import Config

MAX_SUBAGENT_ROUNDS_NOTE = (
    "You are a read-only research sub-agent, spawned by a parent agent to investigate one "
    "self-contained task. You have no memory of the parent conversation — the task below is "
    "everything you know. You can read and search the project, but you cannot write files or "
    "run commands. Answer with a concise, complete summary; the parent only sees your final "
    "message, not your intermediate steps."
)


async def _deny_always(_tool: str, _summary: str, _detail: str) -> Decision:
    # Every tool in the sub-agent's kit is read-only (dangerous=False), so this
    # never actually fires — it exists so PermissionManager has a well-defined
    # callback rather than a silent no-op if that ever changes.
    return Decision.DENY


class DispatchAgentTool(Tool):
    name = "dispatch_agent"
    description = (
        "Delegate a self-contained, read-only investigation to a fresh sub-agent — e.g. "
        "\"find every place the config file is parsed and list the fields each one reads\". "
        "The sub-agent can read and search the project but can't write files or run commands, "
        "and it reports back one summary instead of its full step-by-step work. Use this for "
        "broad exploration before you make a targeted edit, not for the edit itself."
    )
    parameters = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": (
                    "A fully self-contained description of what to investigate — the sub-agent "
                    "cannot see this conversation, so include everything it needs to know."
                ),
            },
        },
        "required": ["task"],
    }
    dangerous = False  # nothing this tool does *directly* touches disk — the sub-agent's own tools are all read-only

    def __init__(self, config: Config, model_id: str, cwd: Path) -> None:
        self.config = config
        self.model_id = model_id
        self.cwd = cwd

    def summary(self, args: dict[str, Any]) -> str:
        task = args.get("task", "")
        return f"Sub-agent: {task[:100]}{'…' if len(task) > 100 else ''}"

    async def run(self, args: dict[str, Any], ctx: AgentContext) -> ToolRunResult:
        # Deferred imports: AgentLoop importing tools/registry (which imports
        # this file) would be a circular import at module load time otherwise.
        from openterminal.agent.context import build_system_prompt
        from openterminal.agent.loop import AgentLoop, FatalError
        from openterminal.types import Message

        sub_ctx = AgentContext(cwd=self.cwd, permissions=PermissionManager(ask_fn=_deny_always))
        sub_tools = [ReadFileTool(), ListDirTool(), GlobTool(), GrepTool()]
        sub_system = f"{build_system_prompt(self.cwd, self.model_id)}\n\n{MAX_SUBAGENT_ROUNDS_NOTE}"
        sub_loop = AgentLoop(config=self.config, tools=sub_tools, system_prompt=sub_system, ctx=sub_ctx)

        messages = [Message.user(args["task"])]
        error: str | None = None
        async for event in sub_loop.run_turn(messages, model_id=self.model_id):
            if isinstance(event, FatalError):
                error = event.message

        if error:
            return ToolRunResult(content=f"Sub-agent failed: {error}", is_error=True)

        final = messages[-1] if messages else None
        text = final.text() if final and final.role == Role.ASSISTANT else ""
        return ToolRunResult(content=text or "Sub-agent finished without a final answer.", display=text)
