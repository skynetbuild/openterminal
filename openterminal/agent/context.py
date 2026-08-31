"""Everything a tool needs to act, and the project context folded into the
system prompt at the start of a session.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from openterminal.agent.permissions import PermissionManager

# Checked in this order — first one found wins. OPENTERMINAL.md lets a repo
# speak to us specifically; falling back to AGENTS.md / CLAUDE.md means a
# project that already wrote instructions for another agent doesn't have to
# duplicate them just to work well with this one.
PROJECT_INSTRUCTION_FILES = ("OPENTERMINAL.md", "AGENTS.md", "CLAUDE.md")


@dataclass
class AgentContext:
    cwd: Path
    permissions: PermissionManager
    # Populated by the UI layer so tools that stream output (bash) can push
    # it live instead of buffering silently until they return.
    on_output: OutputSink | None = None


class OutputSink:
    """Minimal interface the UI implements to receive live tool output."""

    def write(self, text: str) -> None:  # pragma: no cover — UI concern
        raise NotImplementedError


def find_project_instructions(cwd: Path) -> tuple[str, str] | None:
    for name in PROJECT_INSTRUCTION_FILES:
        p = cwd / name
        if p.is_file():
            try:
                return name, p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
    return None


def git_summary(cwd: Path) -> str | None:
    try:
        branch = _run(cwd, ["git", "rev-parse", "--abbrev-ref", "HEAD"])
        status = _run(cwd, ["git", "status", "--short"])
    except (OSError, subprocess.CalledProcessError):
        return None
    if branch is None:
        return None
    lines = [f"Branch: {branch}"]
    if status:
        n = len(status.splitlines())
        lines.append(f"Working tree: {n} uncommitted change(s)")
    else:
        lines.append("Working tree: clean")
    return "\n".join(lines)


def _run(cwd: Path, args: list[str]) -> str | None:
    r = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=5)
    if r.returncode != 0:
        return None
    return r.stdout.strip()


def build_system_prompt(cwd: Path, model_id: str) -> str:
    parts = [BASE_SYSTEM_PROMPT.format(cwd=cwd)]

    git = git_summary(cwd)
    if git:
        parts.append(f"## Git\n{git}")

    instructions = find_project_instructions(cwd)
    if instructions:
        name, text = instructions
        parts.append(f"## Project instructions ({name})\n{text.strip()}")

    return "\n\n".join(parts)


BASE_SYSTEM_PROMPT = """\
You are OpenTerminal, an agentic coding assistant running in a real terminal \
against a real project at {cwd}. You have tools to read and search the \
project, propose file edits, and run shell commands.

Ground rules:
- Prefer reading before writing. Don't guess at a file's contents when a \
tool can show you.
- Edits go through the edit_file/write_file tools, never inline in your \
reply — those tools are what actually touch disk and what the user reviews \
before approving.
- Keep shell commands narrow and inspectable; explain briefly why one is \
needed before running it if it's not obvious.
- Match the surrounding code's style instead of imposing your own.
- When you're done, stop — don't keep narrating or re-summarizing work \
already shown in your tool calls.\
"""
