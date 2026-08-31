"""Terminal rendering — the human-facing half of the agent loop's events.

Kept separate from `agent/loop.py` on purpose: the loop yields structured
events (`TextChunk`, `ToolCallStarted`, ...) and knows nothing about Rich,
color, or how a diff should look. That's what makes the loop testable
headless and, longer term, makes a second frontend (a web UI, a Textual TUI)
a matter of writing a new consumer of the same event stream instead of
forking the agent logic.
"""

from __future__ import annotations

from openterminal.agent.context import OutputSink
from openterminal.agent.permissions import Decision
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax

ACCENT = "bright_cyan"
DIM = "grey58"
ERROR = "bright_red"
OK = "bright_green"


class TerminalUI:
    def __init__(self) -> None:
        self.console = Console(highlight=False)
        self._line_open = False  # tracks whether the last thing printed needs a trailing newline

    # ── chrome ───────────────────────────────────────────────────────────

    def banner(self, model: str, cwd: str) -> None:
        self.console.print(f"[bold {ACCENT}]OpenTerminal[/] [dim]— {model} · {cwd}[/]")
        self.console.print("[dim]Type your request, or /help for commands. Ctrl+C to interrupt, /exit to quit.[/]\n")

    def user_echo(self, text: str) -> None:
        self.console.print(f"[bold]❯[/] {escape(text)}")

    # ── streaming assistant text ────────────────────────────────────────

    def text_delta(self, text: str) -> None:
        # markup=False: this is the model's own text, not one of our UI
        # strings — code containing `[]` (a Python list type, a markdown
        # link, `array[i]`) is common enough that treating it as Rich markup
        # would silently mangle or crash on real output.
        self.console.print(text, end="", markup=False)
        self._line_open = True

    def end_text(self) -> None:
        if self._line_open:
            self.console.print()
            self._line_open = False

    # ── tool calls ───────────────────────────────────────────────────────

    def tool_started(self, summary: str) -> None:
        self.end_text()
        self.console.print(f"[{DIM}]⏺[/] {escape(summary)}")

    def tool_finished(self, summary: str, is_error: bool, display: object) -> None:
        icon = f"[{ERROR}]✗[/]" if is_error else f"[{OK}]✓[/]"
        self.console.print(f"  {icon} [{DIM}]{escape(summary)}[/]")
        if isinstance(display, str) and display.strip():
            self._print_diff_or_output(display)

    def _print_diff_or_output(self, text: str) -> None:
        looks_like_diff = text.lstrip().startswith(("---", "+++", "@@"))
        lexer = "diff" if looks_like_diff else "text"
        # Long tool output gets a scroll-past panel, not a page-filling wall.
        lines = text.splitlines()
        shown = "\n".join(lines[:40])
        syntax = Syntax(shown, lexer, theme="ansi_dark", word_wrap=True, background_color="default")
        self.console.print(Panel(syntax, border_style=DIM, padding=(0, 1)))
        if len(lines) > 40:
            self.console.print(f"  [{DIM}]... {len(lines) - 40} more lines[/]")

    # ── status / errors ─────────────────────────────────────────────────

    def model_switched(self, from_model: str, to_model: str, reason: str) -> None:
        self.console.print(f"[{DIM}]⚠ {from_model} unavailable ({reason}) — falling back to {to_model}[/]")

    def error(self, message: str) -> None:
        self.end_text()
        self.console.print(f"[{ERROR}]✗ {escape(message)}[/]")

    def info(self, message: str) -> None:
        self.console.print(f"[{DIM}]{escape(message)}[/]")

    # ── permission prompt (wired into PermissionManager as `ask_fn`) ───

    async def ask_permission(self, tool_name: str, summary: str, detail: str) -> Decision:
        self.end_text()
        self.console.print(f"\n[{ACCENT}]Permission needed:[/] {escape(summary)}")
        if detail.strip():
            self._print_diff_or_output(detail)
        choice = Prompt.ask(
            "  Allow this?",
            choices=["y", "a", "n"],
            default="y",
            show_choices=False,
        )
        self.console.print("  [dim](y = once, a = always this session, n = no)[/]")
        return {"y": Decision.ALLOW_ONCE, "a": Decision.ALLOW_SESSION, "n": Decision.DENY}[choice]


class ConsoleOutputSink(OutputSink):
    """Feeds live bash output straight to the terminal as it streams, instead
    of only showing it once the command finishes."""

    def __init__(self, console: Console) -> None:
        self.console = console

    def write(self, text: str) -> None:
        # Same reasoning as text_delta: this is a shell command's raw stdout,
        # not a UI string — `ls` output, JSON, anything with `[`/`]` in it
        # would otherwise be parsed as Rich markup.
        self.console.print(text, end="", style=DIM, markup=False)
