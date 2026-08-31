# OpenTerminal

[![CI](https://github.com/skynetbuild/openterminal/actions/workflows/ci.yml/badge.svg)](https://github.com/skynetbuild/openterminal/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)

A multi-provider agentic coding CLI. By [SkynetBuild](https://skynet.build).

```bash
❯ openterminal
OpenTerminal — anthropic/claude-sonnet-4-5 · C:\code\my-project
Type your request, or /help for commands. Ctrl+C to interrupt, /exit to quit.

❯ fix the off-by-one in the pagination helper and add a test for it
⏺ Read lib/pagination.py
  ✓ Read lib/pagination.py
⏺ Edit lib/pagination.py
  Permission needed: Edit lib/pagination.py
  --- a/lib/pagination.py
  +++ b/lib/pagination.py
  @@ -12,7 +12,7 @@
  -    return items[offset : offset + limit]
  +    return items[offset : offset + limit + 1]
  Allow this? [y/a/n]: y
  ✓ Edit lib/pagination.py
...
```

## Install

```bash
pip install openterminalai
```

```bash
openterminal auth anthropic   # or: export ANTHROPIC_API_KEY=...
openterminal                  # in any project directory
```

## Usage

```bash
openterminal                     # interactive session
openterminal --tui               # same, full-screen Textual UI (beta)
openterminal --continue          # resume the last session for this project
openterminal --resume <id>       # resume a specific one
openterminal run "prompt"        # one-shot, prints and exits — scripting/CI
openterminal --model openai/gpt-5.2
openterminal --model ollama/qwen2.5-coder:32b   # local, no API key
```

Model IDs are `provider/model`. Providers: `anthropic`, `openai`, `google`,
`xai`, `ollama`, `ollama-cloud`, `lmstudio`, `vllm`, or anything OpenAI-compatible
you add yourself (see below). `openterminal providers` lists what's configured.

## Config

`~/.config/openterminal/config.toml` (platform equivalent elsewhere), overridable
per-project in `.openterminal/config.toml`:

```toml
model = "anthropic/claude-sonnet-4-5"
fallback_models = ["openai/gpt-5.2", "ollama/qwen2.5-coder:32b"]

[providers.anthropic]
api_key = "sk-ant-..."

[custom_providers.myprovider]
display_name = "My Provider"
base_url = "https://api.myprovider.com/v1"
default_model = "my-model"

[mcp_servers.filesystem]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/path"]

[mcp_servers.remote]
url = "https://example.com/mcp"
```

- **`fallback_models`** — tried in order if the primary errors out before answering.
- **`custom_providers`** — any OpenAI-compatible `/v1/chat/completions` endpoint.
  `openterminal --model myprovider/my-model`.
- **`mcp_servers`** — stdio or streamable-HTTP. Tools show up namespaced
  `mcp__server__tool`, gated by the same permission prompt as `write_file`/`bash`.
  `openterminal mcp` checks a config without starting a session.
- **`OPENTERMINAL.md`** (or `AGENTS.md`/`CLAUDE.md`) in a repo root — folded
  into the system prompt if present.

`dispatch_agent` is a built-in tool the model can call to delegate a
self-contained, read-only investigation to a fresh sub-agent (own message
history, no writes/bash, no nesting) instead of spending the main
conversation on intermediate searches.

## Architecture

```
openterminal/
  types.py            provider-agnostic Message/ToolCall/StreamEvent shapes
  providers/           anthropic and google are native adapters; openai/xai/
                        ollama/lmstudio/vllm/custom ride one OpenAI-compatible adapter
  tools/                read_file, list_dir, glob, grep, write_file, edit_file,
                        bash, dispatch_agent
  mcp_client.py          MCP servers -> Tool instances
  agent/
    loop.py             stream -> tool calls -> results -> repeat
    permissions.py       approval gate for writes/bash/MCP tools
    session.py            JSON-backed, resumable
    context.py             system-prompt assembly (git state, OPENTERMINAL.md)
  ui/
    console.py          Rich REPL (default)
    tui.py               Textual UI (--tui) — same AgentLoop, different consumer
  cli.py               entry point (Typer)
```

Adding a provider that isn't OpenAI-compatible means implementing
`Provider.stream()`; nothing else needs to know it exists beyond a registry
entry. Same for a UI — `console.py` and `tui.py` both just consume
`AgentLoop.run_turn()`'s event stream.

## Status

Pre-1.0. Core loop, five providers, permissions, sessions, the TUI, MCP, and
sub-agents are implemented. Not yet: predefined sub-agent types, nested
sub-agents, Windows/Linux binary distribution.

## License

MIT
