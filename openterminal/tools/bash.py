from __future__ import annotations

import asyncio
from typing import Any

from openterminal.agent.context import AgentContext
from openterminal.tools.base import Tool, ToolRunResult

DEFAULT_TIMEOUT_SEC = 120
MAX_OUTPUT_CHARS = 30_000


class BashTool(Tool):
    name = "bash"
    description = (
        "Run a shell command in the project directory and return its combined stdout/stderr. "
        "Prefer the narrowest command that answers the question — `git status` over poking "
        "around by hand, a project's own lint/test script over ad hoc equivalents."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "timeout_sec": {
                "type": "integer",
                "description": f"Kill the command after this many seconds (default {DEFAULT_TIMEOUT_SEC}).",
            },
        },
        "required": ["command"],
    }
    dangerous = True

    def summary(self, args: dict[str, Any]) -> str:
        return args.get("command", "")

    async def run(self, args: dict[str, Any], ctx: AgentContext) -> ToolRunResult:
        command = args["command"]
        timeout = args.get("timeout_sec") or DEFAULT_TIMEOUT_SEC

        ok = await ctx.permissions.check("bash", command)
        if not ok:
            return ToolRunResult(content="User declined to run this command.", is_error=True)

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=ctx.cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except OSError as e:
            return ToolRunResult(content=f"Failed to start: {e}", is_error=True)

        chunks: list[str] = []
        truncated = False

        async def pump() -> None:
            nonlocal truncated
            assert proc.stdout is not None
            total = 0
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                text = line.decode(errors="replace")
                if ctx.on_output:
                    ctx.on_output.write(text)
                total += len(text)
                if total <= MAX_OUTPUT_CHARS:
                    chunks.append(text)
                elif not truncated:
                    truncated = True
                    chunks.append(f"\n... output truncated at {MAX_OUTPUT_CHARS:,} chars ...\n")

        try:
            await asyncio.wait_for(pump(), timeout=timeout)
            code = await asyncio.wait_for(proc.wait(), timeout=5)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            output = "".join(chunks)
            return ToolRunResult(
                content=f"Command timed out after {timeout}s and was killed.\n\nOutput so far:\n{output}",
                is_error=True,
            )

        output = "".join(chunks).rstrip("\n")
        content = f"$ {command}\n(exit {code})\n{output}" if output else f"$ {command}\n(exit {code}, no output)"
        return ToolRunResult(content=content, is_error=code != 0, display=output)
