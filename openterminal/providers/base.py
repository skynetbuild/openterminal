"""The contract every provider adapter implements.

Adding a provider means writing one class that turns `(messages, system,
tools)` into a stream of `StreamEvent`s — nothing else in the codebase (the
agent loop, the tools, the UI, the session store) needs to know it exists
beyond registering it. See providers/registry.py for how a *user* adds one
without writing Python at all (any OpenAI-compatible endpoint).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

from openterminal.types import Message, StreamEvent, ToolSpec


class ProviderError(Exception):
    """Raised for config/auth problems (missing key, bad model id) — distinct
    from a StreamError event, which is a mid-stream failure the agent loop
    can react to (e.g. retry) without crashing the whole session."""


@dataclass
class ProviderInfo:
    id: str
    display_name: str
    default_model: str
    models: list[str]
    requires_api_key: bool = True
    env_var: str | None = None


class Provider(ABC):
    info: ProviderInfo

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self.api_key = api_key
        self.base_url = base_url

    @abstractmethod
    def stream(
        self,
        *,
        messages: list[Message],
        system: str,
        tools: list[ToolSpec],
        model: str,
        temperature: float | None = None,
        max_tokens: int = 8192,
    ) -> AsyncIterator[StreamEvent]:
        """Stream one assistant turn. Must yield TextDelta/ToolCall* events as
        they arrive and end with exactly one TurnEnd (or StreamError)."""
        raise NotImplementedError
