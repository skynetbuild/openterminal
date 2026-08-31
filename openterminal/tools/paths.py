"""Path safety shared by every filesystem tool.

Every tool that touches disk resolves through here first. The model only
ever sees paths relative to the project root, and a path that escapes it
(`../../etc/passwd`, an absolute path outside the tree, a symlink pointing
out) is refused before anything reads or writes — not caught after the fact.
"""

from __future__ import annotations

from pathlib import Path


class PathEscapeError(Exception):
    pass


def resolve_in_project(cwd: Path, relative: str) -> Path:
    candidate = (cwd / relative).resolve()
    root = cwd.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as e:
        raise PathEscapeError(
            f"'{relative}' resolves outside the project root ({root}) — refusing."
        ) from e
    return candidate
