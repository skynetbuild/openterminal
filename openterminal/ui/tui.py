"""The Textual TUI — opt-in via `openterminal --tui`.

Same agent loop, same events, same permission gate as the plain Rich REPL in
cli.py; this file is purely a different *consumer* of `AgentLoop.run_turn`'s
event stream. That split (loop emits structured events, a UI renders them) is
what makes swapping frontends here a new file instead of a fork of the agent
logic — see the note in ui/console.py.

Textual is already asyncio-native, so the permission prompt awaits a modal
screen directly (`push_screen_wait`) instead of needing the callback
indirection the plain REPL uses.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Input, Static

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
from openterminal.agent.permissions import Decision, PermissionManager
from openterminal.agent.session import Session
from openterminal.config import Config
from openterminal.tools.registry import default_tools
from openterminal.types import Message

CSS = """
Screen {
    background: $surface;
}
#chat {
    padding: 1 2;
    scrollbar-gutter: stable;
}
#chat > Static {
    margin-bottom: 1;
    width: 100%;
}
.user-line {
    color: $text;
    text-style: bold;
}
.assistant-line {
    color: $text;
}
.tool-line {
    color: $text-muted;
}
.tool-line.ok { color: $success; }
.tool-line.err { color: $error; }
.tool-detail {
    color: $text-muted;
    background: $panel;
    padding: 0 1;
    margin-left: 2;
    border-left: thick $primary;
}
.error-line {
    color: $error;
    text-style: bold;
}
.notice-line {
    color: $warning;
}
#prompt-row {
    dock: bottom;
    height: 3;
    padding: 0 1;
}
#prompt {
    width: 1fr;
}
PermissionModal {
    align: center middle;
}
#perm-box {
    width: 70%;
    max-width: 90;
    height: auto;
    max-height: 80%;
    background: $panel;
    border: thick $primary;
    padding: 1 2;
}
#perm-summary {
    text-style: bold;
    margin-bottom: 1;
}
#perm-detail {
    color: $text-muted;
    background: $surface;
    padding: 1;
    margin-bottom: 1;
    max-height: 20;
    overflow-y: auto;
}
#perm-buttons {
    height: 3;
    align: right middle;
}
#perm-buttons Button {
    margin-left: 1;
}
"""


class PermissionModal(ModalScreen[Decision]):
    """Blocks the loop (not the whole UI — Textual keeps rendering) until the
    human picks once / for the session / deny. Mirrors the Rich REPL's
    y/a/n prompt, just as buttons instead of a keystroke."""

    def __init__(self, tool_name: str, summary: str, detail: str) -> None:
        super().__init__()
        self.tool_name = tool_name
        self.summary = summary
        self.detail = detail

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="perm-box"):
            yield Static(f"Permission needed: {self.summary}", id="perm-summary", markup=False)
            if self.detail.strip():
                yield Static(self.detail, id="perm-detail", markup=False)
            with Horizontal(id="perm-buttons"):
                yield Button("Deny", id="deny", variant="error")
                yield Button("Allow session", id="session", variant="warning")
                yield Button("Allow once", id="once", variant="success")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        mapping = {"once": Decision.ALLOW_ONCE, "session": Decision.ALLOW_SESSION, "deny": Decision.DENY}
        self.dismiss(mapping[event.button.id or "deny"])


class OpenTerminalTUI(App[None]):
    CSS = CSS
    TITLE = "OpenTerminal"
    BINDINGS = [("ctrl+q", "quit", "Quit")]

    busy: reactive[bool] = reactive(False)

    def __init__(
        self,
        config: Config,
        model_id: str,
        session: Session,
        cwd: Path,
        extra_tools: list | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.model_id = model_id
        self.session = session
        self.cwd = cwd
        self.permissions = PermissionManager(ask_fn=self._ask_permission, auto_approve=set(config.auto_approve_tools))
        self.agent_ctx = AgentContext(cwd=cwd, permissions=self.permissions)
        self.agent_loop = AgentLoop(
            config=config,
            tools=default_tools(config, model_id, cwd) + (extra_tools or []),
            system_prompt=build_system_prompt(cwd, model_id),
            ctx=self.agent_ctx,
        )
        # Set for real in on_mount (see the comment there) — declared here so
        # every reference below has a known type instead of springing into
        # existence dynamically.
        self._chat: VerticalScroll = None  # type: ignore[assignment]
        self._prompt: Input = None  # type: ignore[assignment]

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(id="chat")
        with Horizontal(id="prompt-row"):
            yield Input(placeholder=f"Ask OpenTerminal ({self.model_id})…", id="prompt")
        yield Footer()

    def on_mount(self) -> None:
        # Cached once, used everywhere below, instead of re-querying by id:
        # `query_one` without an explicit screen resolves against whichever
        # screen is *currently on top* of the stack — once the permission
        # modal is pushed, that's the modal, which has no #chat/#prompt, and
        # a plain `self.query_one("#prompt")` from inside run_turn's `finally`
        # (which can fire while the modal is still up, or right as it closes)
        # raises NoMatches instead of finding the base screen's input.
        self._chat = self.query_one("#chat", VerticalScroll)
        self._prompt = self.query_one("#prompt", Input)

        self.sub_title = str(self.cwd)
        for m in self.session.messages:
            text = m.text()
            if text:
                cls = "user-line" if m.role.value == "user" else "assistant-line"
                self._log(text, cls)
        self._prompt.focus()

    def _log(self, text: str, css_class: str) -> Static:
        # markup=False: this renders model output, tool summaries, and shell
        # output — none of it is a UI string we wrote, and Textual's Static
        # parses `[...]` as markup by default. Unescaped, a Python `list[str]`
        # type hint or a markdown link in a real response silently vanishes
        # (`[str]` gets swallowed as a style tag) instead of displaying.
        widget = Static(text, classes=css_class, markup=False)
        self._chat.mount(widget)
        self._chat.scroll_end(animate=False)
        return widget

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text or self.busy:
            return
        if text in ("/exit", "/quit"):
            self.exit()
            return
        if text == "/clear":
            self.session.messages.clear()
            await self._chat.remove_children()
            self._log("Conversation cleared.", "notice-line")
            return

        self._log(text, "user-line")
        self.session.messages.append(Message.user(text))
        self.run_turn()

    @work(exclusive=True)
    async def run_turn(self) -> None:
        self.busy = True
        self._prompt.disabled = True
        assistant_widget: Static | None = None
        buf = ""
        try:
            async for event in self.agent_loop.run_turn(self.session.messages, model_id=self.model_id):
                if isinstance(event, TextChunk):
                    if assistant_widget is None:
                        assistant_widget = self._log("", "assistant-line")
                    buf += event.text
                    assistant_widget.update(buf)
                    self._chat.scroll_end(animate=False)
                elif isinstance(event, ToolCallStarted):
                    assistant_widget = None
                    buf = ""
                    self._log(f"⏺ {event.summary}", "tool-line")
                elif isinstance(event, ToolCallFinished):
                    icon = "✗" if event.result.is_error else "✓"
                    cls = "tool-line err" if event.result.is_error else "tool-line ok"
                    self._log(f"  {icon} {event.name}", cls)
                    if isinstance(event.result.display, str) and event.result.display.strip():
                        lines = event.result.display.splitlines()
                        shown = "\n".join(lines[:30])
                        self._log(shown, "tool-detail")
                elif isinstance(event, ModelSwitched):
                    self._log(f"⚠ {event.from_model} unavailable — falling back to {event.to_model}", "notice-line")
                elif isinstance(event, FatalError):
                    self._log(f"✗ {event.message}", "error-line")
                elif isinstance(event, TurnComplete):
                    pass
        finally:
            self.session.save()
            self.busy = False
            self._prompt.disabled = False
            self._prompt.focus()

    async def _ask_permission(self, tool_name: str, summary: str, detail: str) -> Decision:
        result = await self.push_screen_wait(PermissionModal(tool_name, summary, detail))
        return result if result is not None else Decision.DENY


def run_tui(config: Config, model_id: str, session: Session, cwd: Path) -> None:
    """Sync entry point (what cli.py calls) — wraps `_run_tui_async` in its
    own `asyncio.run`. MCP connections have to be opened in the *same* loop
    Textual's app runs in (asyncio pipes/streams are loop-bound — opening
    them in one loop via a throwaway `asyncio.run()` and then using them from
    a second, different loop breaks), so this drives `app.run_async()`
    itself instead of calling the sync `app.run()` convenience wrapper,
    which would open its own separate loop."""
    asyncio.run(_run_tui_async(config, model_id, session, cwd))


async def _run_tui_async(config: Config, model_id: str, session: Session, cwd: Path) -> None:
    from openterminal.mcp_client import MCPManager

    mcp_manager = MCPManager()
    extra_tools: list = []
    if config.mcp_servers:
        try:
            extra_tools = await mcp_manager.connect_all(config.mcp_servers)
        except Exception:  # noqa: BLE001 — a broken MCP config shouldn't block the TUI from starting
            pass

    app = OpenTerminalTUI(config=config, model_id=model_id, session=session, cwd=cwd, extra_tools=extra_tools)
    try:
        await app.run_async()
    finally:
        await mcp_manager.close()
