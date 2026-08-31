"""The `openterminal` / `ot` entry point.

    openterminal                    interactive session in the current directory
    openterminal --continue         resume the most recent session for this project
    openterminal --resume <id>      resume a specific session
    openterminal run "prompt"       one-shot: print the answer and exit (scripting/CI)
    openterminal auth <provider>    save an API key
    openterminal providers          list providers and their models
    openterminal sessions           list saved sessions
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import typer
from rich.console import Console

# Windows only, and only when stdout/stderr aren't already UTF-8: without
# this, piping/redirecting output (`openterminal run ... > out.txt`, running
# under an older cmd.exe, CI) makes Rich fall back to a legacy console writer
# that encodes as cp1252 — which can't represent the box-drawing/status glyphs
# used throughout the UI (❯, ⏺, ✓, ✗, ⚠) and crashes with UnicodeEncodeError.
# A real modern terminal (Windows Terminal, PowerShell 7) doesn't need this,
# but forcing it is harmless there too.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

from openterminal.agent.context import AgentContext, build_system_prompt
from openterminal.agent.loop import (
    AgentLoop,
    FatalError,
    ModelSwitched,
    TextChunk,
    ToolCallFinished,
    ToolCallStarted,
    TurnComplete,
)
from openterminal.agent.permissions import PermissionManager
from openterminal.agent.session import Session
from openterminal.config import Config, split_model_id
from openterminal.mcp_client import MCPManager
from openterminal.providers.registry import list_providers
from openterminal.tools.registry import default_tools
from openterminal.types import Message
from openterminal.ui.console import ConsoleOutputSink, TerminalUI

app = typer.Typer(add_completion=False, no_args_is_help=False, invoke_without_command=True)


@app.callback()
def main(
    ctx: typer.Context,
    model: str = typer.Option(None, "--model", "-m", help="provider/model, e.g. anthropic/claude-sonnet-4-5"),
    cont: bool = typer.Option(False, "--continue", "-c", help="Resume the most recent session for this project."),
    resume: str = typer.Option(None, "--resume", help="Resume a specific session id."),
    tui: bool = typer.Option(False, "--tui", help="Full-screen Textual interface instead of the plain REPL (beta)."),
) -> None:
    if ctx.invoked_subcommand is not None:
        return
    if tui:
        _launch_tui(model=model, cont=cont, resume=resume)
    else:
        asyncio.run(_interactive(model=model, cont=cont, resume=resume))


def _launch_tui(model: str | None, cont: bool, resume: str | None) -> None:
    from openterminal.ui.tui import run_tui

    config = Config.load()
    cwd = Path.cwd()

    session: Session | None = None
    if resume:
        try:
            session = Session.load(resume)
        except FileNotFoundError:
            typer.echo(f"No session '{resume}'.")
            raise typer.Exit(1) from None
    elif cont:
        session = Session.latest_for_project(cwd)

    model_id = model or (session.meta.model if session else config.model)
    if session is None:
        session = Session.create(cwd, model_id)

    run_tui(config=config, model_id=model_id, session=session, cwd=cwd)


@app.command()
def run(
    prompt: str = typer.Argument(..., help="The request to send. Prints the final answer and exits."),
    model: str = typer.Option(None, "--model", "-m"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Auto-approve every tool call (writes, bash) — for CI."),
) -> None:
    """One-shot mode: no REPL, no persisted session. Exit code is non-zero on error."""
    asyncio.run(_one_shot(prompt, model=model, auto_approve_all=yes))


@app.command()
def auth(provider: str = typer.Argument(..., help="e.g. anthropic, openai, google, xai")) -> None:
    """Save an API key for a provider to your user config."""
    config = Config.load()
    infos = {i.id: i for i in list_providers(config)}
    if provider not in infos:
        typer.echo(f"Unknown provider '{provider}'. Run `openterminal providers` to see the list.")
        raise typer.Exit(1)
    key = typer.prompt(f"API key for {infos[provider].display_name}", hide_input=True)
    config.set_api_key(provider, key)
    typer.echo(f"Saved to {config.__class__.__name__} — {provider} is ready to use.")


@app.command()
def providers() -> None:
    """List available providers and their known models."""
    config = Config.load()
    console = Console()
    for info in list_providers(config):
        key_state = "no key needed" if not info.requires_api_key else (
            "key set" if info.id in config.provider_api_keys or _env_has_key(info.env_var) else "no key"
        )
        console.print(f"[bold]{info.id}[/]  [dim]({info.display_name} · {key_state})[/]")
        for m in info.models[:6]:
            console.print(f"    {info.id}/{m}")
        if not info.models:
            console.print(f"    {info.id}/<model-id> — discovered locally, pass any model name")


@app.command()
def sessions() -> None:
    """List saved sessions, most recent first."""
    console = Console()
    metas = Session.list_all()
    if not metas:
        console.print("[dim]No sessions yet.[/]")
        return
    for m in metas:
        console.print(f"[bold]{m.id}[/]  [dim]{m.model} · {m.project_path}[/]\n  {m.title or '(untitled)'}")


@app.command(name="mcp")
def mcp_command() -> None:
    """Connect to every configured MCP server and list the tools they expose.

    Doesn't start a session — this is for checking a server config is right
    (command spelled correctly, URL reachable) before relying on it.
    """
    asyncio.run(_mcp_list())


async def _mcp_list() -> None:
    config = Config.load()
    console = Console()
    if not config.mcp_servers:
        console.print(
            r"[dim]No MCP servers configured. Add one under \[mcp_servers.<name>] in your config.[/]"
        )
        return
    manager = MCPManager()
    tools = await manager.connect_all(config.mcp_servers)
    by_server: dict[str, list[str]] = {}
    for t in tools:
        by_server.setdefault(t.server_name, []).append(t.mcp_tool_name)  # type: ignore[attr-defined]
    for s in config.mcp_servers:
        if s.name in manager.connected_servers:
            names = ", ".join(by_server.get(s.name, [])) or "(no tools)"
            console.print(f"[bold]{s.name}[/]  [green]connected[/]  {names}")
        else:
            console.print(f"[bold]{s.name}[/]  [red]failed to connect[/]")
    await manager.close()


def _env_has_key(env_var: str | None) -> bool:
    import os

    return bool(env_var and os.environ.get(env_var))


# ── interactive REPL ─────────────────────────────────────────────────────


async def _interactive(model: str | None, cont: bool, resume: str | None) -> None:
    config = Config.load()
    cwd = Path.cwd()
    ui = TerminalUI()

    session: Session | None = None
    if resume:
        try:
            session = Session.load(resume)
        except FileNotFoundError:
            ui.error(f"No session '{resume}'.")
            raise typer.Exit(1) from None
    elif cont:
        session = Session.latest_for_project(cwd)

    model_id = model or (session.meta.model if session else config.model)
    if session is None:
        session = Session.create(cwd, model_id)

    permissions = PermissionManager(ask_fn=ui.ask_permission, auto_approve=set(config.auto_approve_tools))
    agent_ctx = AgentContext(cwd=cwd, permissions=permissions, on_output=ConsoleOutputSink(ui.console))
    system_prompt = build_system_prompt(cwd, model_id)

    tools = default_tools(config, model_id, cwd)
    mcp_manager = MCPManager()
    if config.mcp_servers:
        try:
            tools += await mcp_manager.connect_all(config.mcp_servers)
        except Exception as e:  # noqa: BLE001 — a broken MCP config shouldn't block the session
            ui.error(f"MCP setup failed: {e}")

    loop = AgentLoop(config=config, tools=tools, system_prompt=system_prompt, ctx=agent_ctx)

    ui.banner(model_id, str(cwd))

    try:
        while True:
            try:
                user_text = ui.console.input("[bold]❯[/] ")
            except (EOFError, KeyboardInterrupt):
                ui.console.print()
                break

            if not user_text.strip():
                continue
            if user_text.startswith("/"):
                if _handle_slash(user_text.strip(), ui, session):
                    break
                continue

            session.messages.append(Message.user(user_text))
            await _drive_turn(loop, session, ui, model_id)
            session.save()

        session.save()
        ui.info(f"Session saved: {session.meta.id}  (resume with `openterminal --resume {session.meta.id}`)")
    finally:
        await mcp_manager.close()
        # See the matching comment in _one_shot — lets provider HTTP clients
        # finish draining their connection pool before the loop shuts down.
        await asyncio.sleep(0.1)


def _handle_slash(cmd: str, ui: TerminalUI, session: Session) -> bool:
    """Returns True if the REPL should exit."""
    if cmd in ("/exit", "/quit"):
        return True
    if cmd == "/help":
        ui.info("/exit  /clear  /help — that's v1. More slash commands land as the tool grows.")
    elif cmd == "/clear":
        session.messages.clear()
        ui.info("Conversation cleared (session id kept).")
    else:
        ui.info(f"Unknown command: {cmd}")
    return False


async def _drive_turn(loop: AgentLoop, session: Session, ui: TerminalUI, model_id: str) -> None:
    try:
        async for event in loop.run_turn(session.messages, model_id=model_id):
            _render_event(event, ui)
    except KeyboardInterrupt:
        ui.info("Interrupted.")
    ui.end_text()


def _render_event(event, ui: TerminalUI) -> None:  # noqa: ANN001 — AgentEvent union, kept loose here
    if isinstance(event, TextChunk):
        ui.text_delta(event.text)
    elif isinstance(event, ToolCallStarted):
        ui.tool_started(event.summary)
    elif isinstance(event, ToolCallFinished):
        ui.tool_finished(event.result.content.splitlines()[0] if event.result.content else event.name, event.result.is_error, event.result.display)
    elif isinstance(event, ModelSwitched):
        ui.model_switched(event.from_model, event.to_model, event.reason)
    elif isinstance(event, FatalError):
        ui.error(event.message)
    elif isinstance(event, TurnComplete):
        pass


# ── one-shot mode ────────────────────────────────────────────────────────


async def _one_shot(prompt: str, model: str | None, auto_approve_all: bool) -> None:
    config = Config.load()
    cwd = Path.cwd()
    model_id = model or config.model

    async def auto_yes(_tool: str, _summary: str, _detail: str):
        from openterminal.agent.permissions import Decision

        return Decision.ALLOW_ONCE

    permissions = PermissionManager(ask_fn=auto_yes if auto_approve_all else _deny_in_ci)
    ctx = AgentContext(cwd=cwd, permissions=permissions)
    system_prompt = build_system_prompt(cwd, model_id)

    tools = default_tools(config, model_id, cwd)
    mcp_manager = MCPManager()
    console = Console()
    if config.mcp_servers:
        try:
            tools += await mcp_manager.connect_all(config.mcp_servers)
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]MCP setup failed: {e}[/]")

    loop = AgentLoop(config=config, tools=tools, system_prompt=system_prompt, ctx=ctx)

    messages = [Message.user(prompt)]
    had_error = False
    try:
        async for event in loop.run_turn(messages, model_id=model_id):
            if isinstance(event, TextChunk):
                console.print(event.text, end="", markup=False, highlight=False)
            elif isinstance(event, FatalError):
                console.print(f"\n[red]{event.message}[/]")
                had_error = True
        console.print()
    finally:
        await mcp_manager.close()
        # Some provider SDKs' HTTP layers (httpcore2, as of this writing) don't
        # fully drain their connection pool inside client.close() — without this,
        # asyncio.run() tears the loop down before that finishes and the cleanup
        # fails loudly (but harmlessly) during interpreter shutdown instead.
        await asyncio.sleep(0.1)
    if had_error:
        raise typer.Exit(1)


async def _deny_in_ci(_tool: str, summary: str, _detail: str):
    from openterminal.agent.permissions import Decision

    # `run` without --yes refuses anything dangerous rather than hang waiting
    # for a prompt no one's there to answer — the model gets told why, and
    # can either finish without it or the caller re-runs with --yes.
    return Decision.DENY
