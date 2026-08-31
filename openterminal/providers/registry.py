"""Provider lookup by id.

Five providers ship built in: `anthropic` and `google` get native adapters
(their tool-calling formats are distinct enough to be worth it); `openai`,
`xai`, and `ollama` all ride the shared OpenAI-compatible adapter with
different base URLs. Anything in the user's `[custom_providers.*]` config
also rides that same adapter — that's the whole point of standardizing on
it: a user can point OpenTerminal at literally any OpenAI-compatible
endpoint (a self-hosted vLLM server, a new provider we've never heard of,
their employer's internal gateway) without anyone writing a line of code.
"""

from __future__ import annotations

from openterminal.config import Config
from openterminal.providers.base import Provider, ProviderError, ProviderInfo
from openterminal.providers.openai_compat import OpenAICompatProvider

_BUILTIN_OPENAI_COMPAT = {
    "openai": ProviderInfo(
        id="openai",
        display_name="OpenAI",
        default_model="gpt-5.2",
        models=["gpt-5.2", "gpt-5.2-mini", "o4-mini"],
        env_var="OPENAI_API_KEY",
    ),
    "xai": ProviderInfo(
        id="xai",
        display_name="xAI",
        default_model="grok-4",
        models=["grok-4", "grok-4-fast"],
        env_var="XAI_API_KEY",
    ),
    "ollama": ProviderInfo(
        id="ollama",
        display_name="Ollama (local)",
        default_model="qwen2.5-coder:32b",
        models=[],  # discovered locally — the model picker shells to `ollama list`
        requires_api_key=False,
        env_var=None,
    ),
    # LM Studio and vLLM both serve an OpenAI-compatible /v1 endpoint like
    # Ollama does, just on different default ports — same adapter, zero extra
    # code. (llama.cpp's `server` binary and SGLang do too, and slot in the
    # same way as a [custom_providers.*] entry if someone's running one.)
    "lmstudio": ProviderInfo(
        id="lmstudio",
        display_name="LM Studio (local)",
        default_model="",
        models=[],
        requires_api_key=False,
        env_var=None,
    ),
    "vllm": ProviderInfo(
        id="vllm",
        display_name="vLLM (local/self-hosted)",
        default_model="",
        models=[],
        requires_api_key=False,
        env_var=None,
    ),
}

_BUILTIN_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "xai": "https://api.x.ai/v1",
    "ollama": "http://localhost:11434/v1",
    "lmstudio": "http://localhost:1234/v1",
    "vllm": "http://localhost:8000/v1",
}


def list_providers(config: Config | None = None) -> list[ProviderInfo]:
    infos = [_native_info("anthropic"), _native_info("google"), *_BUILTIN_OPENAI_COMPAT.values()]
    if config:
        for c in config.custom_providers:
            infos.append(
                ProviderInfo(
                    id=c.id,
                    display_name=c.display_name,
                    default_model=c.default_model,
                    models=c.models,
                    requires_api_key=c.api_key is None and c.env_var is None,
                    env_var=c.env_var,
                )
            )
    return infos


def get_provider(provider_id: str, config: Config) -> Provider:
    if provider_id == "anthropic":
        from openterminal.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(api_key=config.provider_api_keys.get("anthropic"))

    if provider_id == "google":
        from openterminal.providers.google_provider import GoogleProvider

        return GoogleProvider(api_key=config.provider_api_keys.get("google"))

    if provider_id in _BUILTIN_OPENAI_COMPAT:
        info = _BUILTIN_OPENAI_COMPAT[provider_id]
        base_url = config.provider_base_urls.get(provider_id, _BUILTIN_BASE_URLS[provider_id])
        return OpenAICompatProvider(
            info=info, api_key=config.provider_api_keys.get(provider_id), base_url=base_url
        )

    for c in config.custom_providers:
        if c.id == provider_id:
            info = ProviderInfo(
                id=c.id,
                display_name=c.display_name,
                default_model=c.default_model,
                models=c.models,
                requires_api_key=bool(c.api_key or c.env_var),
                env_var=c.env_var,
            )
            return OpenAICompatProvider(info=info, api_key=c.api_key, base_url=c.base_url)

    raise ProviderError(
        f"Unknown provider '{provider_id}'. Run `openterminal providers` to list the available ones, "
        f"or add it under [custom_providers.{provider_id}] in your config."
    )


def _native_info(provider_id: str) -> ProviderInfo:
    if provider_id == "anthropic":
        from openterminal.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider.info
    from openterminal.providers.google_provider import GoogleProvider

    return GoogleProvider.info
