"""Provider-agnostic conversation and streaming types.

Every provider adapter (Anthropic, OpenAI-compatible, Gemini) translates its
own wire format into these dataclasses on the way in, and these back into its
own format on the way out. The agent loop, the tools, the session store, and
the UI never see a provider-specific shape — that's what makes swapping the
model (or adding a new provider) a one-file change instead of a rewrite.

The content-block model is deliberately closest to Anthropic's, since it's
the most expressive of the three native shapes (it keeps tool calls and text
interleaved in order); OpenAI's and Gemini's flatter shapes fold into it
without losing anything, whereas going the other direction would lose
ordering.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


def new_id(prefix: str = "id") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


# ── Content blocks ──────────────────────────────────────────────────────────


@dataclass
class TextBlock:
    text: str
    type: Literal["text"] = "text"


@dataclass
class ToolCallBlock:
    """The model asking to invoke a tool. `arguments` is already-parsed JSON
    by the time this is constructed — each provider adapter is responsible
    for buffering and parsing its own streaming argument deltas."""

    id: str
    name: str
    arguments: dict[str, Any]
    type: Literal["tool_call"] = "tool_call"


@dataclass
class ToolResultBlock:
    """The result of running a tool, fed back to the model as part of the
    next user-role turn (that's how Anthropic/OpenAI/Gemini all expect it,
    despite each wiring it up slightly differently under the hood)."""

    tool_call_id: str
    content: str
    is_error: bool = False
    type: Literal["tool_result"] = "tool_result"


ContentBlock = TextBlock | ToolCallBlock | ToolResultBlock


@dataclass
class Message:
    role: Role
    content: list[ContentBlock]
    id: str = field(default_factory=lambda: new_id("msg"))
    created_at: float = field(default_factory=time.time)

    @staticmethod
    def user(text: str) -> "Message":
        return Message(role=Role.USER, content=[TextBlock(text)])

    @staticmethod
    def assistant(blocks: list[ContentBlock]) -> "Message":
        return Message(role=Role.ASSISTANT, content=blocks)

    @staticmethod
    def tool_results(results: list[ToolResultBlock]) -> "Message":
        # Tool results travel back as a user-role turn — every provider
        # expects them addressed to it that way, not as their own role.
        return Message(role=Role.USER, content=list(results))

    def text(self) -> str:
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))

    def tool_calls(self) -> list[ToolCallBlock]:
        return [b for b in self.content if isinstance(b, ToolCallBlock)]


# ── Tool specs (what the model is told exists) ──────────────────────────────


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema, `type: object`


# ── Streaming events yielded by a Provider while a turn is generating ──────


@dataclass
class TextDelta:
    text: str
    type: Literal["text_delta"] = "text_delta"


@dataclass
class ToolCallStart:
    id: str
    name: str
    type: Literal["tool_call_start"] = "tool_call_start"


@dataclass
class ToolCallDelta:
    id: str
    arguments_delta: str  # raw JSON text fragment, buffered by the caller
    type: Literal["tool_call_delta"] = "tool_call_delta"


@dataclass
class ToolCallEnd:
    id: str
    type: Literal["tool_call_end"] = "tool_call_end"


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


StopReason = Literal["end_turn", "tool_use", "max_tokens", "error"]


@dataclass
class TurnEnd:
    stop_reason: StopReason
    usage: Usage
    type: Literal["turn_end"] = "turn_end"


@dataclass
class StreamError:
    message: str
    retryable: bool = False
    type: Literal["error"] = "error"


StreamEvent = TextDelta | ToolCallStart | ToolCallDelta | ToolCallEnd | TurnEnd | StreamError
