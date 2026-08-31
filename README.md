# OpenTerminal

A real, multi-provider agentic coding CLI. By **SkynetBuild** — [skynet.build](https://skynet.build) · [openterminal.org](https://openterminal.org)

Run it in a project, talk to it in plain language, and it reads/searches/edits
files and runs shell commands to get things done — with your choice of model
behind it, not just one vendor.

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

## Why this exists

Every agentic coding CLI today locks you into one model vendor. OpenTerminal
doesn't: Anthropic, OpenAI, Google Gemini, xAI Grok, local models (Ollama,
LM Studio, vLLM), or **any OpenAI-compatible endpoint you point it at** — same
tools, same permission model, same sessions, whichever one is answering.

## Install

```bash
pip install openterminalai          # once published
# or, from source:
git clone https://github.com/skynetbuild/openterminal
cd openterminal && pip install -e .
```

## Quick start

```bash
openterminal auth anthropic        # paste your API key once, it's saved locally
openterminal                       # interactive session in the current directory
openterminal --tui                 # same session, full-screen Textual UI (beta)
openterminal --continue            # resume the most recent session for this project
openterminal run "explain this repo's structure"   # one-shot, prints and exits — good for scripts/CI
```

Switch models per-run or set a default:

```bash
openterminal --model openai/gpt-5.2
openterminal --model ollama/qwen2.5-coder:32b      # fully local, no API key
```

## Add your own provider

Anything that speaks the OpenAI-compatible `/v1/chat/completions` API works —
add it to `~/.config/openterminal/config.toml` (or the platform equivalent):

```toml
[custom_providers.myprovider]
display_name = "My Provider"
base_url = "https://api.myprovider.com/v1"
api_key = "..."
default_model = "my-model"
models = ["my-model", "my-model-mini"]
```

Then: `openterminal --model myprovider/my-model`.

## Fallback between models

```toml
model = "anthropic/claude-sonnet-4-5"
fallback_models = ["openai/gpt-5.2", "ollama/qwen2.5-coder:32b"]
```

If the primary model errors out before producing any output (bad key, an
outage, a rate limit), OpenTerminal tries the next one automatically and
tells you it did.

## Project instructions

Drop an `OPENTERMINAL.md` in your repo root (or it'll fall back to an
existing `AGENTS.md`/`CLAUDE.md`) and it's folded into every session's system
prompt — conventions, commands, things not to touch.

## MCP servers

Any MCP server — stdio or streamable-HTTP — adds its tools to the session:

```toml
[mcp_servers.filesystem]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allow"]

[mcp_servers.remote]
url = "https://example.com/mcp"
```

`openterminal mcp` connects to everything configured and lists what each
server exposes, without starting a session — useful for checking a server's
actually reachable before relying on it. Every MCP tool goes through the
same permission gate as `write_file`/`bash`, since it's arbitrary code we
didn't write.

## Sub-agents

The model has a `dispatch_agent` tool for delegating a self-contained,
read-only investigation ("find every place X is parsed and summarize the
formats") to a fresh sub-agent instead of burning the main conversation on
dozens of intermediate searches. The sub-agent gets its own message history
(no memory of your conversation), a read-only tool set (no writes, no bash,
no nested sub-agents), and reports back one summary.

## Architecture

```
openterminal/
  types.py            provider-agnostic Message/ToolCall/StreamEvent shapes
  providers/           one adapter per wire format, not per vendor —
                        anthropic, google are native; openai/xai/ollama/
                        lmstudio/vllm/custom all ride one OpenAI-compatible adapter
  tools/                read_file, list_dir, glob, grep, write_file, edit_file, bash,
                          dispatch_agent (spawns a read-only sub-agent)
  mcp_client.py          MCP servers -> Tool instances, same adapter idea as providers/
  agent/
    loop.py             the actual agent loop: stream -> tool calls -> results -> repeat
    permissions.py       the approval gate for anything that writes or executes
    session.py            durable, resumable conversations (JSON on disk)
    context.py             project system-prompt assembly (git state, OPENTERMINAL.md)
  ui/
    console.py          Rich-based plain REPL (the default)
    tui.py               Textual full-screen UI (`--tui`, beta) — same
                          agent loop and events, just a different consumer
  cli.py               the `openterminal` / `ot` entry point (Typer)
```

Adding a provider that isn't already OpenAI-compatible means writing one
class implementing `Provider.stream()` — nothing else in the codebase needs
to know it exists beyond a registry entry. Same idea for a UI: both
`ui/console.py` and `ui/tui.py` are independent consumers of the same
`AgentLoop.run_turn()` event stream — a third frontend (a web UI, say) would
be a new file, not a fork of the agent logic.

## Status

Early — the core loop, all five day-one providers, the permission system,
session persistence, a Textual TUI, MCP servers, and sub-agents are real and
working (tested live against a real model: tool calls, the permission
modal, an MCP round-trip, and a dispatch_agent delegation that came back
with an accurate answer). Not yet: predefined sub-agent types (today there's
one general-purpose read-only kind), nested sub-agents.

## License

MIT
