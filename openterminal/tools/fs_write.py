from __future__ import annotations

import difflib
from typing import Any

from openterminal.agent.context import AgentContext
from openterminal.tools.base import Tool, ToolRunResult
from openterminal.tools.paths import PathEscapeError, resolve_in_project


def _diff(before: str, after: str, path: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )


class WriteFileTool(Tool):
    name = "write_file"
    description = (
        "Create a new file or overwrite an existing one with the given full content. "
        "For changing part of an existing file, prefer edit_file — it's reviewable as a "
        "small diff instead of replacing the whole file."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    }
    dangerous = True

    def summary(self, args: dict[str, Any]) -> str:
        return f"Write {args.get('path')}"

    async def run(self, args: dict[str, Any], ctx: AgentContext) -> ToolRunResult:
        try:
            p = resolve_in_project(ctx.cwd, args["path"])
        except PathEscapeError as e:
            return ToolRunResult(content=str(e), is_error=True)

        before = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
        after = args["content"]
        diff = _diff(before, after, args["path"])

        ok = await ctx.permissions.check(
            self.name, f"Write {args['path']} ({len(after.splitlines())} lines)", diff
        )
        if not ok:
            return ToolRunResult(content="User declined this write.", is_error=True)

        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(after, encoding="utf-8")
        return ToolRunResult(content=f"Wrote {args['path']} ({len(after)} bytes).", display=diff)


class EditFileTool(Tool):
    name = "edit_file"
    description = (
        "Replace an exact, unique substring within an existing file — the surgical way to "
        "change part of a file. `old_string` must match exactly once in the file (include "
        "enough surrounding context to make it unique); if it doesn't, nothing is changed "
        "and you're told why, so you can retry with a more specific match."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
            "replace_all": {
                "type": "boolean",
                "description": "Replace every occurrence instead of requiring exactly one match.",
            },
        },
        "required": ["path", "old_string", "new_string"],
    }
    dangerous = True

    def summary(self, args: dict[str, Any]) -> str:
        return f"Edit {args.get('path')}"

    async def run(self, args: dict[str, Any], ctx: AgentContext) -> ToolRunResult:
        try:
            p = resolve_in_project(ctx.cwd, args["path"])
        except PathEscapeError as e:
            return ToolRunResult(content=str(e), is_error=True)
        if not p.is_file():
            return ToolRunResult(content=f"No such file: {args['path']}", is_error=True)

        before = p.read_text(encoding="utf-8", errors="replace")
        old, new = args["old_string"], args["new_string"]
        count = before.count(old)

        if count == 0:
            return ToolRunResult(
                content=(
                    f"old_string not found in {args['path']}. Re-read the file to check it "
                    "hasn't changed, and make sure old_string matches exactly (including "
                    "whitespace)."
                ),
                is_error=True,
            )
        if count > 1 and not args.get("replace_all"):
            return ToolRunResult(
                content=(
                    f"old_string matches {count} places in {args['path']} — it must be unique. "
                    "Add more surrounding context, or pass replace_all=true if that's intended."
                ),
                is_error=True,
            )

        after = before.replace(old, new) if args.get("replace_all") else before.replace(old, new, 1)
        diff = _diff(before, after, args["path"])

        ok = await ctx.permissions.check(self.name, f"Edit {args['path']}", diff)
        if not ok:
            return ToolRunResult(content="User declined this edit.", is_error=True)

        p.write_text(after, encoding="utf-8")
        return ToolRunResult(content=f"Edited {args['path']}.", display=diff)
