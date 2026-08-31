from __future__ import annotations

import os
from collections.abc import AsyncIterator

from openterminal.providers.base import Provider, ProviderError, ProviderInfo
from openterminal.types import (
    Message,
    Role,
    StreamError,
    StreamEvent,
    TextBlock,
    TextDelta,
    ToolCallBlock,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    ToolResultBlock,
    ToolSpec,
    TurnEnd,
    Usage,
)


class AnthropicProvider(Provider):
    info = ProviderInfo(
        id="anthropic",
        display_name="Anthropic",
        default_model="claude-sonnet-4-5",
        models=[
            "claude-opus-4-5",
            "claude-sonnet-4-5",
            "claude-haiku-4-5",
        ],
        env_var="ANTHROPIC_API_KEY",
    )

    def _client(self):
        try:
            from anthropic import AsyncAnthropic
        except ImportError as e:  # pragma: no cover
            raise ProviderError(
                "The `anthropic` package isn't installed. Run: pip install anthropic"
            ) from e
        key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ProviderError(
                "No Anthropic API key. Set ANTHROPIC_API_KEY or run `openterminal auth anthropic`."
            )
        return AsyncAnthropic(api_key=key, base_url=self.base_url)

    @staticmethod
    def _to_wire(messages: list[Message]) -> list[dict]:
        wire: list[dict] = []
        for m in messages:
            content: list[dict] = []
            for b in m.content:
                if isinstance(b, TextBlock):
                    if b.text:
                        content.append({"type": "text", "text": b.text})
                elif isinstance(b, ToolCallBlock):
                    content.append(
                        {"type": "tool_use", "id": b.id, "name": b.name, "input": b.arguments}
                    )
                elif isinstance(b, ToolResultBlock):
                    content.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": b.tool_call_id,
                            "content": b.content,
                            "is_error": b.is_error,
                        }
                    )
            wire.append({"role": "user" if m.role == Role.USER else "assistant", "content": content})
        return wire

    @staticmethod
    def _to_wire_tools(tools: list[ToolSpec]) -> list[dict]:
        return [
            {"name": t.name, "description": t.description, "input_schema": t.parameters}
            for t in tools
        ]

    async def stream(
        self,
        *,
        messages: list[Message],
        system: str,
        tools: list[ToolSpec],
        model: str,
        temperature: float | None = None,
        max_tokens: int = 8192,
    ) -> AsyncIterator[StreamEvent]:
        # Everything here — including client construction, which is where a
        # missing API key raises — stays inside the try. A ProviderError (or
        # anything else) that escapes uncaught would crash the whole process
        # instead of becoming a StreamError the agent loop can react to (and,
        # with a fallback model configured, recover from).
        try:
            client = self._client()
            kwargs: dict = dict(
                model=model,
                system=system,
                messages=self._to_wire(messages),
                max_tokens=max_tokens,
            )
            if tools:
                kwargs["tools"] = self._to_wire_tools(tools)
            if temperature is not None:
                kwargs["temperature"] = temperature

            # Buffers keyed by the block's index within the current message —
            # Anthropic streams tool_use input as raw JSON text fragments
            # (input_json_delta) that only parse cleanly once fully assembled.
            async with client.messages.stream(**kwargs) as stream:
                current_tool_id: dict[int, str] = {}
                async for event in stream:
                    et = event.type
                    if et == "content_block_start":
                        block = event.content_block
                        if block.type == "tool_use":
                            current_tool_id[event.index] = block.id
                            yield ToolCallStart(id=block.id, name=block.name)
                    elif et == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            yield TextDelta(text=delta.text)
                        elif delta.type == "input_json_delta":
                            tid = current_tool_id.get(event.index)
                            if tid:
                                yield ToolCallDelta(id=tid, arguments_delta=delta.partial_json)
                    elif et == "content_block_stop":
                        tid = current_tool_id.get(event.index)
                        if tid:
                            yield ToolCallEnd(id=tid)
                    elif et == "message_delta":
                        pass  # stop_reason/usage arrive on message_stop's final snapshot below

                final = await stream.get_final_message()
                stop = final.stop_reason
                reason = "tool_use" if stop == "tool_use" else "max_tokens" if stop == "max_tokens" else "end_turn"
                yield TurnEnd(
                    stop_reason=reason,  # type: ignore[arg-type]
                    usage=Usage(
                        input_tokens=final.usage.input_tokens,
                        output_tokens=final.usage.output_tokens,
                    ),
                )
        except Exception as e:  # noqa: BLE001 — surfaced to the agent loop as a stream event
            yield StreamError(message=str(e), retryable=_looks_retryable(e))


def _looks_retryable(e: Exception) -> bool:
    name = type(e).__name__.lower()
    return "ratelimit" in name or "overloaded" in name or "timeout" in name or "apiconnection" in name
