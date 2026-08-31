from __future__ import annotations

import sys
from pathlib import Path

from openterminal.agent.context import AgentContext
from openterminal.tools.bash import BashTool

# `echo` behaves identically as a shell built-in on both cmd.exe and POSIX
# shells, so this doesn't need to special-case Windows vs. everything else.
ECHO_CMD = "echo hello-from-bash-tool"


async def test_bash_runs_and_captures_output(project: Path, allow_all_ctx: AgentContext):
    result = await BashTool().run({"command": ECHO_CMD}, allow_all_ctx)
    assert not result.is_error
    assert "hello-from-bash-tool" in result.content


async def test_bash_denied_never_runs(project: Path, deny_all_ctx: AgentContext):
    marker = project / "should-not-exist.txt"
    cmd = f'python -c "open(\'{marker.name}\', \'w\').close()"' if sys.platform == "win32" else f"touch {marker.name}"
    result = await BashTool().run({"command": cmd}, deny_all_ctx)
    assert result.is_error
    assert "declined" in result.content
    assert not marker.exists()


async def test_bash_nonzero_exit_is_reported_as_error(project: Path, allow_all_ctx: AgentContext):
    fail_cmd = "exit 1" if sys.platform != "win32" else "cmd /c exit 1"
    result = await BashTool().run({"command": fail_cmd}, allow_all_ctx)
    assert result.is_error


async def test_bash_timeout_kills_the_process(project: Path, allow_all_ctx: AgentContext):
    sleep_cmd = (
        'python -c "import time; time.sleep(5)"'
        if sys.platform == "win32"
        else "sleep 5"
    )
    result = await BashTool().run({"command": sleep_cmd, "timeout_sec": 1}, allow_all_ctx)
    assert result.is_error
    assert "timed out" in result.content
