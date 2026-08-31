from __future__ import annotations

from pathlib import Path

from openterminal.config import Config, split_model_id


def test_split_model_id_with_slash():
    assert split_model_id("anthropic/claude-sonnet-4-5") == ("anthropic", "claude-sonnet-4-5")


def test_split_model_id_ollama_model_name_with_colon():
    # Ollama model names contain a colon (the tag) — must not be confused
    # with the provider/model slash split.
    assert split_model_id("ollama/qwen2.5-coder:32b") == ("ollama", "qwen2.5-coder:32b")


def test_split_model_id_no_slash_falls_back_to_anthropic():
    assert split_model_id("gpt-5.2") == ("anthropic", "gpt-5.2")


def test_config_default_provider_and_model_properties():
    c = Config(model="openai/gpt-5.2")
    assert c.default_provider == "openai"
    assert c.default_model == "gpt-5.2"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_load_reads_user_config(tmp_path: Path, monkeypatch):
    user_cfg = tmp_path / "user" / "config.toml"
    _write(
        user_cfg,
        """
        model = "openai/gpt-5.2"
        fallback_models = ["ollama/qwen2.5-coder:32b"]

        [providers.anthropic]
        api_key = "sk-ant-test"

        [custom_providers.myprov]
        display_name = "My Provider"
        base_url = "https://api.example.com/v1"
        default_model = "my-model"
        """,
    )
    monkeypatch.setattr("openterminal.config.user_config_path", lambda: user_cfg)
    project = tmp_path / "project"
    project.mkdir()

    cfg = Config.load(cwd=project)
    assert cfg.model == "openai/gpt-5.2"
    assert cfg.fallback_models == ["ollama/qwen2.5-coder:32b"]
    assert cfg.provider_api_keys["anthropic"] == "sk-ant-test"
    assert cfg.custom_providers[0].id == "myprov"
    assert cfg.custom_providers[0].base_url == "https://api.example.com/v1"


def test_project_config_overrides_user_config(tmp_path: Path, monkeypatch):
    user_cfg = tmp_path / "user" / "config.toml"
    _write(user_cfg, 'model = "anthropic/claude-sonnet-4-5"\n')
    monkeypatch.setattr("openterminal.config.user_config_path", lambda: user_cfg)

    project = tmp_path / "project"
    _write(project / ".openterminal" / "config.toml", 'model = "openai/gpt-5.2"\n')

    cfg = Config.load(cwd=project)
    assert cfg.model == "openai/gpt-5.2"  # project layer wins


def test_load_parses_mcp_servers(tmp_path: Path, monkeypatch):
    user_cfg = tmp_path / "user" / "config.toml"
    _write(
        user_cfg,
        """
        [mcp_servers.filesystem]
        command = "npx"
        args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]

        [mcp_servers.remote]
        url = "https://example.com/mcp"
        """,
    )
    monkeypatch.setattr("openterminal.config.user_config_path", lambda: user_cfg)
    project = tmp_path / "project"
    project.mkdir()

    cfg = Config.load(cwd=project)
    by_name = {s.name: s for s in cfg.mcp_servers}
    assert by_name["filesystem"].command == "npx"
    assert by_name["filesystem"].args == ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    assert by_name["remote"].url == "https://example.com/mcp"


def test_set_api_key_writes_and_is_readable_back(tmp_path: Path, monkeypatch):
    user_cfg = tmp_path / "user" / "config.toml"
    monkeypatch.setattr("openterminal.config.user_config_path", lambda: user_cfg)

    Config().set_api_key("anthropic", "sk-ant-newkey")
    cfg = Config.load(cwd=tmp_path)
    assert cfg.provider_api_keys["anthropic"] == "sk-ant-newkey"
