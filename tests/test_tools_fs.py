from __future__ import annotations

from pathlib import Path

import pytest

from openterminal.agent.context import AgentContext
from openterminal.tools.fs_read import GlobTool, ListDirTool, ReadFileTool
from openterminal.tools.fs_write import EditFileTool, WriteFileTool


async def test_read_file_returns_numbered_lines(project: Path, allow_all_ctx: AgentContext):
    (project / "a.txt").write_text("first\nsecond\nthird\n", encoding="utf-8")
    result = await ReadFileTool().run({"path": "a.txt"}, allow_all_ctx)
    assert not result.is_error
    assert "1\tfirst" in result.content
    assert "3\tthird" in result.content


async def test_read_file_respects_line_range(project: Path, allow_all_ctx: AgentContext):
    (project / "a.txt").write_text("\n".join(f"line{i}" for i in range(1, 11)), encoding="utf-8")
    result = await ReadFileTool().run({"path": "a.txt", "start_line": 3, "end_line": 5}, allow_all_ctx)
    assert "line3" in result.content
    assert "line5" in result.content
    assert "line1" not in result.content
    assert "line6" not in result.content


async def test_read_file_missing(project: Path, allow_all_ctx: AgentContext):
    result = await ReadFileTool().run({"path": "nope.txt"}, allow_all_ctx)
    assert result.is_error


@pytest.mark.parametrize("path", ["../outside.txt", "../../etc/passwd", "/etc/passwd"])
async def test_read_file_refuses_path_escape(project: Path, allow_all_ctx: AgentContext, path: str):
    result = await ReadFileTool().run({"path": path}, allow_all_ctx)
    assert result.is_error
    assert "outside the project root" in result.content


async def test_glob_finds_matches_and_ignores_vendor_dirs(project: Path, allow_all_ctx: AgentContext):
    (project / "src").mkdir()
    (project / "src" / "a.py").write_text("", encoding="utf-8")
    (project / "src" / "b.py").write_text("", encoding="utf-8")
    (project / "node_modules").mkdir()
    (project / "node_modules" / "junk.py").write_text("", encoding="utf-8")

    result = await GlobTool().run({"pattern": "**/*.py"}, allow_all_ctx)
    assert "a.py" in result.content
    assert "b.py" in result.content
    assert "junk.py" not in result.content


async def test_list_dir(project: Path, allow_all_ctx: AgentContext):
    (project / "one.txt").write_text("", encoding="utf-8")
    (project / "sub").mkdir()
    result = await ListDirTool().run({"path": "."}, allow_all_ctx)
    assert "one.txt" in result.content
    assert "sub" in result.content


async def test_write_file_creates_and_reports_diff(project: Path, allow_all_ctx: AgentContext):
    result = await WriteFileTool().run({"path": "new.txt", "content": "hello\n"}, allow_all_ctx)
    assert not result.is_error
    assert (project / "new.txt").read_text(encoding="utf-8") == "hello\n"
    assert "+hello" in result.display


async def test_write_file_denied_leaves_disk_untouched(project: Path, deny_all_ctx: AgentContext):
    result = await WriteFileTool().run({"path": "new.txt", "content": "hello\n"}, deny_all_ctx)
    assert result.is_error
    assert not (project / "new.txt").exists()


async def test_edit_file_replaces_unique_match(project: Path, allow_all_ctx: AgentContext):
    (project / "a.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
    result = await EditFileTool().run({"path": "a.py", "old_string": "x = 1", "new_string": "x = 100"}, allow_all_ctx)
    assert not result.is_error
    assert (project / "a.py").read_text(encoding="utf-8") == "x = 100\ny = 2\n"


async def test_edit_file_rejects_ambiguous_match(project: Path, allow_all_ctx: AgentContext):
    (project / "a.py").write_text("x = 1\nx = 1\n", encoding="utf-8")
    result = await EditFileTool().run({"path": "a.py", "old_string": "x = 1", "new_string": "x = 2"}, allow_all_ctx)
    assert result.is_error
    assert "matches 2 places" in result.content
    # Refused, so the file must be untouched.
    assert (project / "a.py").read_text(encoding="utf-8") == "x = 1\nx = 1\n"


async def test_edit_file_replace_all(project: Path, allow_all_ctx: AgentContext):
    (project / "a.py").write_text("x = 1\nx = 1\n", encoding="utf-8")
    result = await EditFileTool().run(
        {"path": "a.py", "old_string": "x = 1", "new_string": "x = 2", "replace_all": True}, allow_all_ctx
    )
    assert not result.is_error
    assert (project / "a.py").read_text(encoding="utf-8") == "x = 2\nx = 2\n"


async def test_edit_file_no_match(project: Path, allow_all_ctx: AgentContext):
    (project / "a.py").write_text("x = 1\n", encoding="utf-8")
    result = await EditFileTool().run({"path": "a.py", "old_string": "z = 9", "new_string": "z = 10"}, allow_all_ctx)
    assert result.is_error
    assert "not found" in result.content
