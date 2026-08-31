"""One adapter, several providers.

OpenAI, xAI (Grok), Ollama, and any custom endpoint a user points us at all
speak the same Chat Completions wire format — that's the whole reason
"OpenAI-compatible" became the de facto standard for new model providers.
So instead of one class per provider, this is one class *parameterized* by
provider metadata (id, base_url, env var, model list); `providers/registry.py`
instantiates it once per known provider and once more per user-defined
custom provider from config.
"""

from __future__ import annotations

import asyncio
import json
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


class OpenAICompatProvider(Provider):
    def __init__(
        self,
        info: ProviderInfo,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__(api_key=api_key, base_url=base_url)
        self.info = info

    def _client(self):
        try:
            from openai import AsyncOpenAI
        except ImportError as e:  # pragma: no cover
            raise ProviderError("The `openai` package isn't installed. Run: pip install openai") from e
        key = self.api_key or (os.environ.get(self.info.env_var) if self.info.env_var else None)
        if self.info.requires_api_key and not key:
            raise ProviderError(
                f"No API key for {self.info.display_name}. Set {self.info.env_var} "
                f"or run `openterminal auth {self.info.id}`."
            )
        # Ollama and some custom local endpoints don't check the key at all —
        # the SDK still requires a non-empty string to construct the client.
        return AsyncOpenAI(api_key=key or "not-needed", base_url=self.base_url)

    @staticmethod
    def _to_wire(messages: list[Message], system: str) -> list[dict]:
        wire: list[dict] = [{"role": "system", "content": system}] if system else []
        for m in messages:
            texts = [b.text for b in m.content if isinstance(b, TextBlock)]
            calls = [b for b in m.content if isinstance(b, ToolCallBlock)]
            results = [b for b in m.content if isinstance(b, ToolResultBlock)]

            if m.role == Role.ASSISTANT:
                entry: dict = {"role": "assistant", "content": "".join(texts) or None}
                if calls:
                    entry["tool_calls"] = [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                        }
                        for c in calls
                    ]
                wire.append(entry)
            else:
                if texts:
                    wire.append({"role": "user", "content": "".join(texts)})
                # Tool results are their own role in this wire format, one
                # message per result, each tagged with the call it answers.
                for r in results:
                    wire.append(
                        {"role": "tool", "tool_call_id": r.tool_call_id, "content": r.content}
                    )
        return wire

    @staticmethod
    def _to_wire_tools(tools: list[ToolSpec]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
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
        # Client construction (where a missing API key raises) stays inside
        # the try along with everything else — see the note in
        # anthropic_provider.py's stream() for why that matters. `client`
        # starts as None so the `finally` below can tell whether there's
        # actually a connection to close (AsyncOpenAI opens an httpx client
        # eagerly; leaving it open past this call is what surfaces as a
        # "generator didn't stop" warning from httpcore during shutdown).
        client = None
        try:
            client = self._client()
            kwargs: dict = dict(
                model=model,
                messages=self._to_wire(messages, system),
                stream=True,
                stream_options={"include_usage": True},
            )
            if tools:
                kwargs["tools"] = self._to_wire_tools(tools)
            if temperature is not None:
                kwargs["temperature"] = temperature
            # Not every OpenAI-compatible backend accepts max_tokens the same
            # way (some reject it entirely on reasoning models) — best-effort.
            kwargs["max_completion_tokens"] = max_tokens

            # index -> id, so later chunks that only carry the index (some
            # backends omit the id after the first fragment) still resolve.
            id_by_index: dict[int, str] = {}
            name_by_index: dict[int, str] = {}
            started: set[int] = set()
            finish_reason: str | None = None
            usage = Usage()

            stream = await client.chat.completions.create(**kwargs)
            try:
                async for chunk in stream:
                    if chunk.usage:
                        usage = Usage(
                            input_tokens=chunk.usage.prompt_tokens or 0,
                            output_tokens=chunk.usage.completion_tokens or 0,
                        )
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason
                    delta = choice.delta
                    if delta.content:
                        yield TextDelta(text=delta.content)
                    for tc in delta.tool_calls or []:
                        idx = tc.index
                        if tc.id:
                            id_by_index[idx] = tc.id
                        if tc.function and tc.function.name:
                            name_by_index[idx] = tc.function.name
                        tid = id_by_index.get(idx, f"call_{idx}")
                        if idx not in started and idx in name_by_index:
                            started.add(idx)
                            yield ToolCallStart(id=tid, name=name_by_index[idx])
                        if tc.function and tc.function.arguments:
                            yield ToolCallDelta(id=tid, arguments_delta=tc.function.arguments)
            finally:
                # Closing the client alone isn't enough — the stream response
                # itself holds the open httpcore2 byte-stream generator that
                # otherwise gets torn down (noisily) at interpreter shutdown
                # instead of cleanly here, inside a still-running loop.
                await stream.close()

            for idx in started:
                yield ToolCallEnd(id=id_by_index.get(idx, f"call_{idx}"))

            reason = (
                "tool_use" if finish_reason == "tool_calls"
                else "max_tokens" if finish_reason == "length"
                else "end_turn"
            )
            yield TurnEnd(stop_reason=reason, usage=usage)  # type: ignore[arg-type]
        except Exception as e:  # noqa: BLE001
            yield StreamError(message=str(e), retryable=_looks_retryable(e))
        finally:
            if client is not None:
                await client.close()
                # httpx's connection-pool cleanup is itself async and doesn't
                # always finish within close() — giving the loop one more
                # tick lets it drain before asyncio.run() tears the loop down,
                # instead of it failing later during interpreter shutdown
                # with no loop left to run in.
                await asyncio.sleep(0)


def _looks_retryable(e: Exception) -> bool:
    name = type(e).__name__.lower()
    return "ratelimit" in name or "timeout" in name or "connection" in name or "internalserver" in name
