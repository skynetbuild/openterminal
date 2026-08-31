"""The agent loop: model turn -> tool calls -> tool results -> model turn ...
until the model stops asking for tools.

This is deliberately provider-agnostic — it only ever talks to `Provider` and
`Message`/`StreamEvent` from `openterminal.types`, so everything above this
file (the CLI, the UI, sessions) is also provider-agnostic by construction.

Fallback: if the primary model errors out before producing any content (bad
key, the provider is down, a rate limit on the very first chunk), the loop
tries the next model in `config.fallback_models` — each a "provider/model"
string — before giving up. Once a model has started producing output for a
turn we don't switch mid-stream; a half-finished answer from model A handed
to model B as "continue this" would be confusing at best.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal

from openterminal.agent.context import AgentContext
from openterminal.config import Config, split_model_id
from openterminal.providers.registry import get_provider
from openterminal.tools.base import Tool, ToolRunResult
from openterminal.types import (
    Message,
    StreamError,
    TextBlock,
    TextDelta,
    ToolCallBlock,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    ToolResultBlock,
    ToolSpec,
    TurnEnd,
)

MAX_TOOL_ROUNDS = 50  # a runaway tool-call loop stops here instead of spinning forever


# ── Events the UI actually consumes ─────────────────────────────────────────


@dataclass
class TextChunk:
    text: str


@dataclass
class ToolCallStarted:
    id: str
    name: str
    args: dict
    summary: str


@dataclass
class ToolCallFinished:
    id: str
    name: str
    result: ToolRunResult


@dataclass
class ModelSwitched:
    """A fallback kicked in — the UI should say so, not fail silently."""

    from_model: str
    to_model: str
    reason: str


@dataclass
class TurnComplete:
    stop_reason: Literal["end_turn", "max_tokens", "max_tool_rounds"]


@dataclass
class FatalError:
    message: str


AgentEvent = TextChunk | ToolCallStarted | ToolCallFinished | ModelSwitched | TurnComplete | FatalError


class AgentLoop:
    def __init__(
        self,
        config: Config,
        tools: list[Tool],
        system_prompt: str,
        ctx: AgentContext,
    ) -> None:
        self.config = config
        self.tools: dict[str, Tool] = {t.name: t for t in tools}
        self.tool_specs: list[ToolSpec] = [t.spec() for t in tools]
        self.system_prompt = system_prompt
        self.ctx = ctx

    async def run_turn(
        self, messages: list[Message], model_id: str | None = None
    ) -> AsyncIterator[AgentEvent]:
        """Mutates `messages` in place (appends the assistant/tool-result
        turns as they happen) and yields UI events as the turn progresses.
        A "turn" here means everything up to the model producing a final
        answer with no more tool calls — which may be several round-trips.
        """
        candidates = [model_id or self.config.model, *self.config.fallback_models]

        for _round_num in range(1, MAX_TOOL_ROUNDS + 1):
            assistant_blocks: list = []
            stop_reason = "end_turn"
            error: str | None = None
            async for event in self._stream_one_turn(messages, candidates):
                if isinstance(event, _TurnResult):
                    assistant_blocks, stop_reason, error = event.blocks, event.stop_reason, event.error
                else:
                    yield event

            if error is not None:
                yield FatalError(message=error)
                return

            messages.append(Message.assistant(assistant_blocks))
            tool_calls = [b for b in assistant_blocks if isinstance(b, ToolCallBlock)]

            if stop_reason != "tool_use" or not tool_calls:
                yield TurnComplete(stop_reason="max_tokens" if stop_reason == "max_tokens" else "end_turn")
                return

            results: list[ToolResultBlock] = []
            for call in tool_calls:
                tool = self.tools.get(call.name)
                if tool is None:
                    results.append(
                        ToolResultBlock(
                            tool_call_id=call.id,
                            content=f"Unknown tool '{call.name}'.",
                            is_error=True,
                        )
                    )
                    continue
                yield ToolCallStarted(
                    id=call.id, name=call.name, args=call.arguments, summary=tool.summary(call.arguments)
                )
                run_result = await tool.run(call.arguments, self.ctx)
                yield ToolCallFinished(id=call.id, name=call.name, result=run_result)
                results.append(
                    ToolResultBlock(
                        tool_call_id=call.id, content=run_result.content, is_error=run_result.is_error
                    )
                )

            messages.append(Message.tool_results(results))
            # loop again — the model gets the tool results and decides whether
            # it needs another round or is ready to answer.

        yield TurnComplete(stop_reason="max_tool_rounds")

    async def _stream_one_turn(
        self, messages: list[Message], candidates: list[str]
    ) -> AsyncIterator[AgentEvent | _TurnResult]:
        """Runs the provider stream for the first candidate model that
        doesn't fail immediately. Yields TextChunk as text arrives (live),
        a ModelSwitched notice if a fallback kicks in, and ends with exactly
        one `_TurnResult` carrying the finished blocks/stop_reason/error for
        the caller to pick up — Python async generators can't cleanly return
        a value alongside yields, so this is the sentinel-event workaround."""
        last_error: str | None = None

        for i, model_ref in enumerate(candidates):
            provider_id, model_name = split_model_id(model_ref)
            try:
                provider = get_provider(provider_id, self.config)
            except Exception as e:  # noqa: BLE001 — bad config shouldn't crash the loop
                last_error = str(e)
                continue

            if i > 0:
                yield ModelSwitched(from_model=candidates[i - 1], to_model=model_ref, reason=last_error or "error")

            blocks: list = []
            text_buf = ""
            tool_buf: dict[str, dict] = {}  # id -> {"name": str, "args": str}
            tool_order: list[str] = []
            produced_any_output = False
            stream_error: str | None = None
            stream_error_retryable = False
            stop_reason = "end_turn"

            async for event in provider.stream(
                messages=messages,
                system=self.system_prompt,
                tools=self.tool_specs,
                model=model_name,
            ):
                if isinstance(event, TextDelta):
                    produced_any_output = True
                    text_buf += event.text
                    yield TextChunk(text=event.text)
                elif isinstance(event, ToolCallStart):
                    produced_any_output = True
                    tool_buf[event.id] = {"name": event.name, "args": ""}
                    tool_order.append(event.id)
                elif isinstance(event, ToolCallDelta):
                    if event.id in tool_buf:
                        tool_buf[event.id]["args"] += event.arguments_delta
                elif isinstance(event, ToolCallEnd):
                    pass  # finalized below, once we also have TurnEnd's stop_reason
                elif isinstance(event, TurnEnd):
                    stop_reason = event.stop_reason
                elif isinstance(event, StreamError):
                    stream_error = event.message
                    stream_error_retryable = event.retryable
                    if not produced_any_output and event.retryable and i + 1 < len(candidates):
                        break  # try the next candidate below
                    last_error = event.message

            # Fall back only for the same reason the inner loop's early `break`
            # does: retryable, nothing shown yet, and another candidate left to
            # try. Anything else (a non-retryable error like a bad key, or a
            # retryable one that still produced partial output) is final —
            # switching models mid-answer, or retrying an error that isn't
            # transient, isn't the fallback's job.
            can_fall_back = stream_error_retryable and not produced_any_output and i + 1 < len(candidates)
            if stream_error and not can_fall_back:
                yield _TurnResult(blocks=blocks, stop_reason="error", error=stream_error)
                return
            if stream_error:
                continue

            if text_buf:
                blocks.append(TextBlock(text_buf))
            for tid in tool_order:
                buf = tool_buf[tid]
                try:
                    args = json.loads(buf["args"]) if buf["args"] else {}
                except json.JSONDecodeError:
                    args = {"_raw": buf["args"]}
                blocks.append(ToolCallBlock(id=tid, name=buf["name"], arguments=args))

            yield _TurnResult(blocks=blocks, stop_reason=stop_reason, error=None)
            return

        yield _TurnResult(blocks=[], stop_reason="error", error=last_error or "No model available.")


@dataclass
class _TurnResult:
    """Internal-only sentinel — never part of the public AgentEvent union
    consumers switch on; see `_stream_one_turn`'s docstring."""

    blocks: list
    stop_reason: str
    error: str | None
