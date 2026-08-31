"""The most important tests in this suite: the actual agent loop, driven by
a FakeProvider so no test here makes a network call or needs an API key —
but every tool call it produces runs for real (a real file gets written and
its diff computed; the loop's tool-result bookkeeping is the real code).
"""

from __future__ import annotations

from pathlib import Path

from conftest import FakeProvider, error_turn, text_turn, tool_call_turn

from openterminal.agent.context import AgentContext
from openterminal.agent.loop import (
    AgentLoop,
    FatalError,
    ModelSwitched,
    TextChunk,
    ToolCallFinished,
    ToolCallStarted,
    TurnComplete,
)
from openterminal.config import Config
from openterminal.tools.fs_read import ReadFileTool
from openterminal.tools.fs_write import WriteFileTool
from openterminal.types import Message


def _loop(config: Config, tools: list, ctx: AgentContext) -> AgentLoop:
    return AgentLoop(config=config, tools=tools, system_prompt="test system prompt", ctx=ctx)


def _patch_provider(monkeypatch, provider_or_factory) -> None:
    factory = provider_or_factory if callable(provider_or_factory) else (lambda pid, cfg: provider_or_factory)
    monkeypatch.setattr("openterminal.agent.loop.get_provider", factory)


async def test_plain_text_turn_ends_without_tool_calls(base_config, allow_all_ctx, monkeypatch):
    provider = FakeProvider([text_turn("hello, world")])
    _patch_provider(monkeypatch, provider)

    loop = _loop(base_config, [], allow_all_ctx)
    messages = [Message.user("hi")]
    events = [e async for e in loop.run_turn(messages)]

    texts = [e.text for e in events if isinstance(e, TextChunk)]
    assert "".join(texts) == "hello, world"
    assert any(isinstance(e, TurnComplete) and e.stop_reason == "end_turn" for e in events)
    # The assistant's reply is appended to the real message history.
    assert messages[-1].text() == "hello, world"


async def test_tool_call_round_trip_actually_runs_the_tool(base_config, allow_all_ctx, project: Path, monkeypatch):
    (project / "a.txt").write_text("file content here\n", encoding="utf-8")
    provider = FakeProvider(
        [
            tool_call_turn("call_1", "read_file", {"path": "a.txt"}),
            text_turn("the file says: file content here"),
        ]
    )
    _patch_provider(monkeypatch, provider)

    loop = _loop(base_config, [ReadFileTool()], allow_all_ctx)
    messages = [Message.user("what's in a.txt?")]
    events = [e async for e in loop.run_turn(messages)]

    started = [e for e in events if isinstance(e, ToolCallStarted)]
    finished = [e for e in events if isinstance(e, ToolCallFinished)]
    assert started[0].name == "read_file"
    assert not finished[0].result.is_error
    assert "file content here" in finished[0].result.content
    # Two model round-trips: the tool-call turn, then the follow-up answer.
    assert provider.calls == 2
    assert messages[-1].text() == "the file says: file content here"


async def test_dangerous_tool_denied_feeds_rejection_back_to_model(
    base_config, deny_all_ctx, project: Path, monkeypatch
):
    provider = FakeProvider(
        [
            tool_call_turn("call_1", "write_file", {"path": "new.txt", "content": "hi"}),
            text_turn("okay, I won't write it"),
        ]
    )
    _patch_provider(monkeypatch, provider)

    loop = _loop(base_config, [WriteFileTool()], deny_all_ctx)
    messages = [Message.user("write a file")]
    events = [e async for e in loop.run_turn(messages)]

    finished = [e for e in events if isinstance(e, ToolCallFinished)][0]
    assert finished.result.is_error
    assert "declined" in finished.result.content
    assert not (project / "new.txt").exists()


async def test_fallback_kicks_in_when_primary_errors_before_output(base_config, allow_all_ctx, monkeypatch):
    base_config.fallback_models = ["fake/fallback-model"]
    primary = FakeProvider([error_turn("connection refused", retryable=True)])
    fallback = FakeProvider([text_turn("answered by the fallback")])

    calls = {"n": 0}

    def get_provider(pid, cfg):
        calls["n"] += 1
        return primary if calls["n"] == 1 else fallback

    _patch_provider(monkeypatch, get_provider)

    loop = _loop(base_config, [], allow_all_ctx)
    messages = [Message.user("hi")]
    events = [e async for e in loop.run_turn(messages)]

    switched = [e for e in events if isinstance(e, ModelSwitched)]
    assert len(switched) == 1
    assert switched[0].to_model == "fake/fallback-model"
    assert messages[-1].text() == "answered by the fallback"


async def test_non_retryable_error_does_not_fall_back(base_config, allow_all_ctx, monkeypatch):
    base_config.fallback_models = ["fake/fallback-model"]
    provider = FakeProvider([error_turn("invalid api key", retryable=False)])
    _patch_provider(monkeypatch, provider)

    loop = _loop(base_config, [], allow_all_ctx)
    messages = [Message.user("hi")]
    events = [e async for e in loop.run_turn(messages)]

    assert any(isinstance(e, FatalError) for e in events)
    assert not any(isinstance(e, ModelSwitched) for e in events)
    assert provider.calls == 1  # never even tried the fallback
