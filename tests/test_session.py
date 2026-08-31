from __future__ import annotations

from pathlib import Path

from openterminal.agent.session import Session
from openterminal.types import Message, ToolCallBlock, ToolResultBlock


def test_session_round_trips_through_disk(project: Path, monkeypatch, tmp_path: Path):
    # Sessions live under platformdirs' user-data location, not the project —
    # redirect that to a scratch dir so the test doesn't touch the real one.
    data_dir = tmp_path / "data"
    monkeypatch.setattr("openterminal.agent.session.user_data_dir", lambda *_a, **_k: str(data_dir))

    s = Session.create(project, "anthropic/claude-sonnet-4-5")
    s.messages.append(Message.user("hello there"))
    s.messages.append(Message.assistant([ToolCallBlock(id="c1", name="read_file", arguments={"path": "a.py"})]))
    s.messages.append(Message.tool_results([ToolResultBlock(tool_call_id="c1", content="file contents")]))
    s.save()

    loaded = Session.load(s.meta.id)
    assert loaded.meta.model == "anthropic/claude-sonnet-4-5"
    assert loaded.meta.title == "hello there"
    assert len(loaded.messages) == 3
    assert loaded.messages[0].text() == "hello there"
    call = loaded.messages[1].tool_calls()[0]
    assert call.name == "read_file"
    assert call.arguments == {"path": "a.py"}


def test_latest_for_project_picks_the_most_recently_saved(project: Path, monkeypatch, tmp_path: Path):
    data_dir = tmp_path / "data"
    monkeypatch.setattr("openterminal.agent.session.user_data_dir", lambda *_a, **_k: str(data_dir))

    older = Session.create(project, "anthropic/claude-sonnet-4-5")
    older.messages.append(Message.user("older"))
    older.save()

    newer = Session.create(project, "openai/gpt-5.2")
    newer.messages.append(Message.user("newer"))
    newer.meta.updated_at = older.meta.updated_at + 10
    newer.save()

    latest = Session.latest_for_project(project)
    assert latest is not None
    assert latest.meta.id == newer.meta.id


def test_latest_for_project_none_when_no_sessions(tmp_path: Path, monkeypatch):
    data_dir = tmp_path / "data"
    monkeypatch.setattr("openterminal.agent.session.user_data_dir", lambda *_a, **_k: str(data_dir))
    assert Session.latest_for_project(tmp_path / "some-other-project") is None
