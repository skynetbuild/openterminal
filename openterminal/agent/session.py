"""Durable, resumable conversations.

Every session is one JSON file under `~/.local/share/openterminal/sessions/`
(platform-appropriate via `platformdirs`), named by its id. `openterminal
--continue` picks up the most recently touched one for the current project;
`openterminal --resume <id>` picks a specific one. This is what makes
closing the terminal not lose the conversation — same idea as Claude Code's
session resume, just file-backed instead of anything fancier.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from platformdirs import user_data_dir

from openterminal.types import (
    ContentBlock,
    Message,
    Role,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
    new_id,
)

APP_NAME = "openterminal"


def sessions_dir() -> Path:
    d = Path(user_data_dir(APP_NAME)) / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class SessionMeta:
    id: str
    project_path: str
    model: str
    created_at: float
    updated_at: float
    title: str = ""  # first ~60 chars of the first user message, for listing


@dataclass
class Session:
    meta: SessionMeta
    messages: list[Message] = field(default_factory=list)

    @classmethod
    def create(cls, project_path: Path, model: str) -> Session:
        now = time.time()
        return cls(
            meta=SessionMeta(
                id=new_id("sess"), project_path=str(project_path), model=model, created_at=now, updated_at=now
            )
        )

    def path(self) -> Path:
        return sessions_dir() / f"{self.meta.id}.json"

    def touch_title(self) -> None:
        if self.meta.title:
            return
        for m in self.messages:
            if m.role == Role.USER:
                text = m.text().strip()
                if text:
                    self.meta.title = (text[:60] + "…") if len(text) > 60 else text
                return

    def save(self) -> None:
        self.meta.updated_at = time.time()
        self.touch_title()
        data = {"meta": asdict(self.meta), "messages": [_message_to_dict(m) for m in self.messages]}
        self.path().write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, session_id: str) -> Session:
        path = sessions_dir() / f"{session_id}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            meta=SessionMeta(**data["meta"]),
            messages=[_message_from_dict(m) for m in data["messages"]],
        )

    @staticmethod
    def latest_for_project(project_path: Path) -> Session | None:
        candidates = []
        for p in sessions_dir().glob("*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if data.get("meta", {}).get("project_path") == str(project_path):
                candidates.append((data["meta"]["updated_at"], data["meta"]["id"]))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return Session.load(candidates[0][1])

    @staticmethod
    def list_all() -> list[SessionMeta]:
        out = []
        for p in sessions_dir().glob("*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                out.append(SessionMeta(**data["meta"]))
            except (OSError, json.JSONDecodeError, TypeError):
                continue
        out.sort(key=lambda m: m.updated_at, reverse=True)
        return out


def _message_to_dict(m: Message) -> dict[str, Any]:
    blocks = []
    for b in m.content:
        if isinstance(b, TextBlock):
            blocks.append({"type": "text", "text": b.text})
        elif isinstance(b, ToolCallBlock):
            blocks.append({"type": "tool_call", "id": b.id, "name": b.name, "arguments": b.arguments})
        elif isinstance(b, ToolResultBlock):
            blocks.append(
                {"type": "tool_result", "tool_call_id": b.tool_call_id, "content": b.content, "is_error": b.is_error}
            )
    return {"id": m.id, "role": m.role.value, "created_at": m.created_at, "content": blocks}


def _message_from_dict(d: dict[str, Any]) -> Message:
    blocks: list[ContentBlock] = []
    for b in d["content"]:
        if b["type"] == "text":
            blocks.append(TextBlock(b["text"]))
        elif b["type"] == "tool_call":
            blocks.append(ToolCallBlock(id=b["id"], name=b["name"], arguments=b["arguments"]))
        elif b["type"] == "tool_result":
            blocks.append(
                ToolResultBlock(tool_call_id=b["tool_call_id"], content=b["content"], is_error=b.get("is_error", False))
            )
    return Message(role=Role(d["role"]), content=blocks, id=d["id"], created_at=d["created_at"])
