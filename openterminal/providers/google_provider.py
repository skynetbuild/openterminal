from __future__ import annotations

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


class GoogleProvider(Provider):
    info = ProviderInfo(
        id="google",
        display_name="Google Gemini",
        default_model="gemini-3-pro",
        models=["gemini-3-pro", "gemini-3-flash"],
        env_var="GEMINI_API_KEY",
    )

    def _client(self):
        try:
            from google import genai
        except ImportError as e:  # pragma: no cover
            raise ProviderError(
                "The `google-genai` package isn't installed. Run: pip install google-genai"
            ) from e
        key = self.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ProviderError(
                "No Gemini API key. Set GEMINI_API_KEY or run `openterminal auth google`."
            )
        return genai.Client(api_key=key)

    @staticmethod
    def _to_wire(messages: list[Message]) -> list[dict]:
        wire: list[dict] = []
        for m in messages:
            parts: list[dict] = []
            for b in m.content:
                if isinstance(b, TextBlock):
                    if b.text:
                        parts.append({"text": b.text})
                elif isinstance(b, ToolCallBlock):
                    parts.append({"function_call": {"name": b.name, "args": b.arguments}})
                elif isinstance(b, ToolResultBlock):
                    # Gemini keys a function_response by name, not call id — it
                    # has no separate id concept, so we carry ours inside the
                    # response payload purely for our own bookkeeping/logs.
                    parts.append(
                        {
                            "function_response": {
                                "name": b.tool_call_id.rsplit(":", 1)[0],
                                "response": {"result": b.content, "is_error": b.is_error},
                            }
                        }
                    )
            wire.append({"role": "model" if m.role == Role.ASSISTANT else "user", "parts": parts})
        return wire

    @staticmethod
    def _to_wire_tools(tools: list[ToolSpec]) -> list:
        from google.genai import types

        return [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name=t.name, description=t.description, parameters=t.parameters
                    )
                    for t in tools
                ]
            )
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
        from google.genai import types

        client = None
        try:
            client = self._client()
            config = types.GenerateContentConfig(
                system_instruction=system or None,
                tools=self._to_wire_tools(tools) if tools else None,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            call_seq = 0
            saw_tool_call = False
            usage = Usage()
            stream = await client.aio.models.generate_content_stream(
                model=model, contents=self._to_wire(messages), config=config
            )
            async for chunk in stream:
                if chunk.usage_metadata:
                    usage = Usage(
                        input_tokens=chunk.usage_metadata.prompt_token_count or 0,
                        output_tokens=chunk.usage_metadata.candidates_token_count or 0,
                    )
                if not chunk.candidates:
                    continue
                content = chunk.candidates[0].content
                if not content or not content.parts:
                    continue
                for part in content.parts:
                    if part.text:
                        yield TextDelta(text=part.text)
                    elif part.function_call:
                        saw_tool_call = True
                        call_seq += 1
                        # Our own synthetic id: Gemini doesn't hand out one, so
                        # {name}:{n} keeps repeated calls to the same tool
                        # distinguishable within a single turn.
                        cid = f"{part.function_call.name}:{call_seq}"
                        yield ToolCallStart(id=cid, name=part.function_call.name)
                        yield ToolCallDelta(
                            id=cid, arguments_delta=json.dumps(dict(part.function_call.args or {}))
                        )
                        yield ToolCallEnd(id=cid)

            yield TurnEnd(stop_reason="tool_use" if saw_tool_call else "end_turn", usage=usage)
        except Exception as e:  # noqa: BLE001
            yield StreamError(message=str(e), retryable=_looks_retryable(e))
        finally:
            # The google-genai SDK doesn't always expose a close() — guard
            # rather than assume, so this never masks the real error above.
            aclose = getattr(client, "aclose", None) or getattr(getattr(client, "_api_client", None), "aclose", None)
            if callable(aclose):
                await aclose()


def _looks_retryable(e: Exception) -> bool:
    name = type(e).__name__.lower()
    return "resourceexhausted" in name or "timeout" in name or "serviceunavailable" in name
