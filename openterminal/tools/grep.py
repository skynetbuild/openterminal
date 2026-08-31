from __future__ import annotations

import re
from typing import Any

from openterminal.agent.context import AgentContext
from openterminal.tools.base import Tool, ToolRunResult
from openterminal.tools.fs_read import IGNORED_DIRS
from openterminal.tools.paths import PathEscapeError, resolve_in_project

MAX_MATCHES = 200
MAX_FILE_BYTES = 2_000_000  # skip huge/binary-ish files rather than choke on them


class GrepTool(Tool):
    name = "grep"
    description = (
        "Search file contents for a regex pattern across the project (or a subdirectory). "
        "Returns matching lines with file:line prefixes, like `grep -rn`."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Python-flavored regex."},
            "path": {"type": "string", "description": "Directory to search from. Defaults to the project root."},
            "glob": {"type": "string", "description": "Restrict to files matching this glob, e.g. \"*.py\"."},
            "case_sensitive": {"type": "boolean"},
        },
        "required": ["pattern"],
    }

    def summary(self, args: dict[str, Any]) -> str:
        return f"Search for /{args.get('pattern')}/"

    async def run(self, args: dict[str, Any], ctx: AgentContext) -> ToolRunResult:
        base = ctx.cwd
        if args.get("path"):
            try:
                base = resolve_in_project(ctx.cwd, args["path"])
            except PathEscapeError as e:
                return ToolRunResult(content=str(e), is_error=True)

        try:
            flags = 0 if args.get("case_sensitive", True) else re.IGNORECASE
            regex = re.compile(args["pattern"], flags)
        except re.error as e:
            return ToolRunResult(content=f"Bad regex: {e}", is_error=True)

        file_glob = args.get("glob") or "**/*"
        results: list[str] = []
        for p in sorted(base.glob(file_glob)):
            if len(results) >= MAX_MATCHES:
                break
            if not p.is_file():
                continue
            rel = p.relative_to(ctx.cwd)
            if any(part in IGNORED_DIRS for part in rel.parts):
                continue
            try:
                if p.stat().st_size > MAX_FILE_BYTES:
                    continue
                text = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    results.append(f"{rel}:{i}: {line.strip()[:300]}")
                    if len(results) >= MAX_MATCHES:
                        break

        if not results:
            return ToolRunResult(content="No matches.")
        suffix = f"\n... capped at {MAX_MATCHES} matches" if len(results) >= MAX_MATCHES else ""
        return ToolRunResult(content="\n".join(results) + suffix)
