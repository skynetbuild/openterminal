"""The approval gate between "the model wants to" and "it happened".

Reading is never gated — a model that can't look at the project is useless.
Anything that changes state on disk or runs a process (write_file, edit_file,
bash) goes through here first. Three ways out: allow once, allow this tool
for the rest of the session, or deny (which feeds a rejection back to the
model as a tool result, not a crash — it can propose something else).

The actual prompt UI lives outside this module (`ui/permission_prompt.py`)
and is injected as `ask_fn`, so the agent loop and the tools never import
Rich/Textual directly — this stays testable without a terminal.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from enum import Enum


class Decision(str, Enum):
    ALLOW_ONCE = "allow_once"
    ALLOW_SESSION = "allow_session"
    DENY = "deny"


AskFn = Callable[[str, str, str], Awaitable[Decision]]
"""(tool_name, one_line_summary, detail_for_display) -> Decision"""


class PermissionManager:
    def __init__(self, ask_fn: AskFn, auto_approve: set[str] | None = None) -> None:
        self._ask = ask_fn
        self._session_allowed: set[str] = set(auto_approve or ())
        self._always_deny: set[str] = set()

    def is_pre_approved(self, tool_name: str) -> bool:
        return tool_name in self._session_allowed

    async def check(self, tool_name: str, summary: str, detail: str = "") -> bool:
        if tool_name in self._session_allowed:
            return True
        if tool_name in self._always_deny:
            return False
        decision = await self._ask(tool_name, summary, detail)
        if decision == Decision.ALLOW_SESSION:
            self._session_allowed.add(tool_name)
        return decision != Decision.DENY
