from __future__ import annotations

from typing import Any

from openterminal.agent.context import AgentContext
from openterminal.tools.base import Tool, ToolRunResult
from openterminal.tools.paths import PathEscapeError, resolve_in_project

MAX_READ_BYTES = 300_000  # ~ a very large source file; past this, ask for a range instead
MAX_LIST_ENTRIES = 400


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "Read a UTF-8 text file from the project, optionally a line range. "
        "Returns the content with 1-indexed line numbers, like `cat -n`."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to the project root."},
            "start_line": {"type": "integer", "description": "1-indexed, inclusive. Omit for the top."},
            "end_line": {"type": "integer", "description": "1-indexed, inclusive. Omit for the end."},
        },
        "required": ["path"],
    }

    def summary(self, args: dict[str, Any]) -> str:
        return f"Read {args.get('path')}"

    async def run(self, args: dict[str, Any], ctx: AgentContext) -> ToolRunResult:
        try:
            p = resolve_in_project(ctx.cwd, args["path"])
        except PathEscapeError as e:
            return ToolRunResult(content=str(e), is_error=True)
        if not p.exists():
            return ToolRunResult(content=f"No such file: {args['path']}", is_error=True)
        if p.is_dir():
            return ToolRunResult(content=f"{args['path']} is a directory, not a file.", is_error=True)
        if p.stat().st_size > MAX_READ_BYTES:
            return ToolRunResult(
                content=(
                    f"{args['path']} is {p.stat().st_size:,} bytes — larger than the "
                    f"{MAX_READ_BYTES:,}-byte limit. Pass start_line/end_line to read a slice."
                ),
                is_error=True,
            )
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return ToolRunResult(content=f"Couldn't read {args['path']}: {e}", is_error=True)

        lines = text.splitlines()
        start = max(args.get("start_line") or 1, 1)
        end = min(args.get("end_line") or len(lines), len(lines))
        numbered = "\n".join(f"{i:>5}\t{lines[i - 1]}" for i in range(start, end + 1))
        return ToolRunResult(content=numbered or "(empty file)")


class ListDirTool(Tool):
    name = "list_dir"
    description = "List the immediate contents of a directory in the project (not recursive — use glob for that)."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": 'Directory, relative to the project root. Use "." for the root.'},
        },
        "required": ["path"],
    }

    def summary(self, args: dict[str, Any]) -> str:
        return f"List {args.get('path')}"

    async def run(self, args: dict[str, Any], ctx: AgentContext) -> ToolRunResult:
        try:
            p = resolve_in_project(ctx.cwd, args["path"])
        except PathEscapeError as e:
            return ToolRunResult(content=str(e), is_error=True)
        if not p.is_dir():
            return ToolRunResult(content=f"Not a directory: {args['path']}", is_error=True)
        entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
        lines = [f"{'📄' if e.is_file() else '📁'} {e.name}" for e in entries[:MAX_LIST_ENTRIES]]
        if len(entries) > MAX_LIST_ENTRIES:
            lines.append(f"... and {len(entries) - MAX_LIST_ENTRIES} more")
        return ToolRunResult(content="\n".join(lines) or "(empty directory)")


class GlobTool(Tool):
    name = "glob"
    description = "Find files by a glob pattern (e.g. \"**/*.py\", \"src/**/*.tsx\"), relative to the project root."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string", "description": "Directory to search from. Defaults to the project root."},
        },
        "required": ["pattern"],
    }

    def summary(self, args: dict[str, Any]) -> str:
        return f"Glob {args.get('pattern')}"

    async def run(self, args: dict[str, Any], ctx: AgentContext) -> ToolRunResult:
        base = ctx.cwd
        if args.get("path"):
            try:
                base = resolve_in_project(ctx.cwd, args["path"])
            except PathEscapeError as e:
                return ToolRunResult(content=str(e), is_error=True)
        matches = sorted(
            p for p in base.glob(args["pattern"])
            if not any(part in IGNORED_DIRS for part in p.relative_to(ctx.cwd).parts)
        )
        rel = [str(p.relative_to(ctx.cwd)) for p in matches[:MAX_LIST_ENTRIES]]
        if len(matches) > MAX_LIST_ENTRIES:
            rel.append(f"... and {len(matches) - MAX_LIST_ENTRIES} more")
        return ToolRunResult(content="\n".join(rel) or "No matches.")


IGNORED_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".next"}
