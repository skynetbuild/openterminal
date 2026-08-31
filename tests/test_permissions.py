from __future__ import annotations

from openterminal.agent.permissions import Decision, PermissionManager


async def test_allow_once_does_not_persist():
    calls = []

    async def ask(tool, summary, detail):
        calls.append(tool)
        return Decision.ALLOW_ONCE

    mgr = PermissionManager(ask_fn=ask)
    assert await mgr.check("bash", "echo hi") is True
    assert await mgr.check("bash", "echo hi again") is True
    assert calls == ["bash", "bash"]  # asked both times — "once" never sticks


async def test_allow_session_persists_without_asking_again():
    calls = 0

    async def ask(tool, summary, detail):
        nonlocal calls
        calls += 1
        return Decision.ALLOW_SESSION

    mgr = PermissionManager(ask_fn=ask)
    assert await mgr.check("bash", "first") is True
    assert await mgr.check("bash", "second") is True
    assert calls == 1  # only asked once — session-approved after that


async def test_deny_blocks_the_call():
    async def ask(tool, summary, detail):
        return Decision.DENY

    mgr = PermissionManager(ask_fn=ask)
    assert await mgr.check("bash", "rm -rf /") is False


async def test_pre_approved_tools_skip_the_prompt_entirely():
    async def ask(tool, summary, detail):
        raise AssertionError("should never be asked — auto-approved")

    mgr = PermissionManager(ask_fn=ask, auto_approve={"read_file"})
    assert mgr.is_pre_approved("read_file") is True
    assert await mgr.check("read_file", "read a.txt") is True
