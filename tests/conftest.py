"""Shared fixtures.

The point of these tests is to exercise the real agent loop, tools, config,
and permission logic without ever making a network call — a `FakeProvider`
stands in for a real model, scripted to yield exactly the StreamEvents a
real one would for a given scenario (plain text, a tool call, a fallback
after an error). Everything downstream of that (tool execution, permission
gating, message-history bookkeeping) is the real, unmocked code.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from openterminal.agent.context import AgentContext
from openterminal.agent.permissions import Decision, PermissionManager
from openterminal.config import Config
from openterminal.providers.base import Provider, ProviderInfo
from openterminal.types import (
    StreamError,
    StreamEvent,
    TextDelta,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    TurnEnd,
    Usage,
)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """An empty project directory — every filesystem tool test gets its own,
    so tests can't step on each other or leak into the real repo."""
    return tmp_path


@pytest.fixture
def allow_all_ctx(project: Path) -> AgentContext:
    async def always_allow(_tool: str, _summary: str, _detail: str) -> Decision:
        return Decision.ALLOW_ONCE

    return AgentContext(cwd=project, permissions=PermissionManager(ask_fn=always_allow))


@pytest.fixture
def deny_all_ctx(project: Path) -> AgentContext:
    async def always_deny(_tool: str, _summary: str, _detail: str) -> Decision:
        return Decision.DENY

    return AgentContext(cwd=project, permissions=PermissionManager(ask_fn=always_deny))


class FakeProvider(Provider):
    """A Provider whose `stream()` replays a pre-scripted list of "turns" —
    each turn itself a list of StreamEvents — one turn per call. Lets a test
    say exactly what "the model" does at each round-trip: answer with text,
    ask for a tool, or fail, without any of that being real."""

    info = ProviderInfo(id="fake", display_name="Fake", default_model="fake-model", models=["fake-model"])

    def __init__(self, turns: list[list[StreamEvent]]) -> None:
        super().__init__()
        self._turns = list(turns)
        self.calls = 0

    def stream(
        self, *, messages, system, tools, model, temperature=None, max_tokens=8192
    ) -> AsyncIterator[StreamEvent]:
        self.calls += 1
        turn = self._turns.pop(0) if self._turns else [TurnEnd(stop_reason="end_turn", usage=Usage())]

        async def gen() -> AsyncIterator[StreamEvent]:
            for event in turn:
                yield event

        return gen()


def text_turn(text: str) -> list[StreamEvent]:
    return [TextDelta(text=text), TurnEnd(stop_reason="end_turn", usage=Usage(output_tokens=len(text)))]


def tool_call_turn(call_id: str, name: str, arguments: dict) -> list[StreamEvent]:
    return [
        ToolCallStart(id=call_id, name=name),
        ToolCallDelta(id=call_id, arguments_delta=json.dumps(arguments)),
        ToolCallEnd(id=call_id),
        TurnEnd(stop_reason="tool_use", usage=Usage()),
    ]


def error_turn(message: str, retryable: bool) -> list[StreamEvent]:
    return [StreamError(message=message, retryable=retryable)]


@pytest.fixture
def base_config() -> Config:
    return Config(model="fake/fake-model")
