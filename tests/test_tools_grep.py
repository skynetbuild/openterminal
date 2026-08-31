from __future__ import annotations

from pathlib import Path

from openterminal.agent.context import AgentContext
from openterminal.tools.grep import GrepTool


async def test_grep_finds_matching_lines(project: Path, allow_all_ctx: AgentContext):
    (project / "a.py").write_text("def foo():\n    pass\n\ndef bar():\n    pass\n", encoding="utf-8")
    result = await GrepTool().run({"pattern": r"def \w+\("}, allow_all_ctx)
    assert "a.py:1" in result.content
    assert "a.py:4" in result.content


async def test_grep_no_matches(project: Path, allow_all_ctx: AgentContext):
    (project / "a.py").write_text("nothing here\n", encoding="utf-8")
    result = await GrepTool().run({"pattern": "zzz_not_present"}, allow_all_ctx)
    assert not result.is_error
    assert result.content == "No matches."


async def test_grep_respects_glob_filter(project: Path, allow_all_ctx: AgentContext):
    (project / "a.py").write_text("target\n", encoding="utf-8")
    (project / "b.txt").write_text("target\n", encoding="utf-8")
    result = await GrepTool().run({"pattern": "target", "glob": "*.py"}, allow_all_ctx)
    assert "a.py" in result.content
    assert "b.txt" not in result.content


async def test_grep_bad_regex(project: Path, allow_all_ctx: AgentContext):
    result = await GrepTool().run({"pattern": "("}, allow_all_ctx)
    assert result.is_error
    assert "Bad regex" in result.content
