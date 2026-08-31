"""User + project configuration.

Two layers, later wins:
  1. `~/.config/openterminal/config.toml` (or the platform equivalent) — the
     user's defaults: which provider/model to use, API keys, and any custom
     OpenAI-compatible providers they've registered.
  2. `.openterminal/config.toml` in the current project — overrides for this
     repo only (e.g. "always use gpt-5.2 here", or a project-scoped key).

API keys resolve in this order for a given provider: config file -> env var
-> (for Ollama/custom local endpoints) no key needed at all. That mirrors
every other CLI in this space, so switching from one to OpenTerminal doesn't
mean re-learning where secrets live.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomli_w
from platformdirs import user_config_dir

APP_NAME = "openterminal"


def split_model_id(model_id: str) -> tuple[str, str]:
    """"anthropic/claude-sonnet-4-5" -> ("anthropic", "claude-sonnet-4-5").

    A bare model name with no slash (rare, but someone will type it) is
    treated as belonging to the default provider rather than raising —
    friendlier than failing on a typo'd flag.
    """
    if "/" in model_id:
        provider, _, model = model_id.partition("/")
        return provider, model
    return "anthropic", model_id


def user_config_path() -> Path:
    return Path(user_config_dir(APP_NAME)) / "config.toml"


def project_config_path(cwd: Path | None = None) -> Path:
    return (cwd or Path.cwd()) / ".openterminal" / "config.toml"


@dataclass
class CustomProviderConfig:
    id: str
    display_name: str
    base_url: str
    default_model: str
    models: list[str] = field(default_factory=list)
    api_key: str | None = None
    env_var: str | None = None


@dataclass
class Config:
    # The canonical way to name a model everywhere in OpenTerminal (CLI flags,
    # config, session files) is a single "provider/model" string — e.g.
    # "anthropic/claude-sonnet-4-5" — rather than two separate fields. One
    # token to copy-paste, one thing to log, one thing a fallback list is a
    # list *of*. `default_provider`/`default_model` stay as a legacy escape
    # hatch for hand-edited configs that predate this.
    model: str = "anthropic/claude-sonnet-4-5"
    fallback_models: list[str] = field(default_factory=list)
    provider_api_keys: dict[str, str] = field(default_factory=dict)
    provider_base_urls: dict[str, str] = field(default_factory=dict)
    custom_providers: list[CustomProviderConfig] = field(default_factory=list)
    auto_approve_tools: list[str] = field(default_factory=list)  # e.g. ["read_file", "glob"]
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def default_provider(self) -> str:
        return split_model_id(self.model)[0]

    @property
    def default_model(self) -> str:
        return split_model_id(self.model)[1]

    @classmethod
    def load(cls, cwd: Path | None = None) -> "Config":
        merged: dict[str, Any] = {}
        for path in (user_config_path(), project_config_path(cwd)):
            if path.exists():
                merged.update(_deep_merge(merged, tomllib.loads(path.read_text(encoding="utf-8"))))

        providers = merged.get("providers", {})
        keys = {pid: v["api_key"] for pid, v in providers.items() if isinstance(v, dict) and v.get("api_key")}
        base_urls = {
            pid: v["base_url"] for pid, v in providers.items() if isinstance(v, dict) and v.get("base_url")
        }
        custom = [
            CustomProviderConfig(
                id=pid,
                display_name=v.get("display_name", pid),
                base_url=v["base_url"],
                default_model=v.get("default_model", ""),
                models=v.get("models", []),
                api_key=v.get("api_key"),
                env_var=v.get("env_var"),
            )
            for pid, v in merged.get("custom_providers", {}).items()
        ]

        # Legacy `default_provider`/`default_model` pair still wins over the
        # built-in default if `model` itself wasn't set explicitly.
        model = merged.get("model")
        if not model and (merged.get("default_provider") or merged.get("default_model")):
            model = f"{merged.get('default_provider', 'anthropic')}/{merged.get('default_model', '')}"

        return cls(
            model=model or cls.model,
            fallback_models=merged.get("fallback_models", []),
            provider_api_keys=keys,
            provider_base_urls=base_urls,
            custom_providers=custom,
            auto_approve_tools=merged.get("auto_approve_tools", []),
            raw=merged,
        )

    def set_api_key(self, provider_id: str, api_key: str, *, project: bool = False, cwd: Path | None = None) -> None:
        path = project_config_path(cwd) if project else user_config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        data.setdefault("providers", {}).setdefault(provider_id, {})["api_key"] = api_key
        path.write_text(tomli_w.dumps(data), encoding="utf-8")


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
