"""Keep retracted-affiliation exemptions adjacent to their rationale."""

from __future__ import annotations

import ast
from pathlib import Path

_SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "check_affiliation_claims.py"


def _allowlist_entries(tree: ast.Module) -> list[tuple[str, int]]:
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "RETRACTED_INSTITUTIONS"
    )
    assert isinstance(assignment.value, ast.Dict)

    entries = []
    for paths in assignment.value.values:
        assert isinstance(paths, ast.Tuple)
        for path in paths.elts:
            assert isinstance(path, ast.Constant) and isinstance(path.value, str)
            entries.append((path.value, path.lineno))
    return entries


def _has_adjacent_note(lines: list[str], lineno: int, entry_lines: set[int]) -> bool:
    if "#" in lines[lineno - 1]:
        return True

    def is_comment(index: int) -> bool:
        return index >= 0 and lines[index].lstrip().startswith("#")

    if is_comment(lineno - 2):
        return True
    return lineno - 1 in entry_lines and is_comment(lineno - 3)


def test_retracted_allowlist_entries_have_an_adjacent_explanatory_note():
    source = _SOURCE.read_text(encoding="utf-8")
    lines = source.splitlines()
    entries = _allowlist_entries(ast.parse(source, filename=str(_SOURCE)))
    entry_lines = {lineno for _, lineno in entries}

    missing = [
        f"{path} (line {lineno})"
        for path, lineno in entries
        if not _has_adjacent_note(lines, lineno, entry_lines)
    ]
    assert not missing, "Allowlist entries without an adjacent note:\n" + "\n".join(missing)
