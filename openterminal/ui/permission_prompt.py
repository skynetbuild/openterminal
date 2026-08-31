"""Re-exported for discoverability — the actual implementation lives on
TerminalUI.ask_permission in ui/console.py, since it needs the same Console
instance everything else prints through (a second Console would risk
interleaving badly with concurrent output, e.g. a streaming bash tool)."""

from openterminal.ui.console import TerminalUI

__all__ = ["TerminalUI"]
