"""The repo-root .semgrepignore may hide only tests/ and the custom rules' fixtures.

A repo-root .semgrepignore REPLACES semgrep's default ignore list, and it sets the
scope of both semgrep jobs in security.yml: the blocking `semgrep-custom` job (its
pickle, joblib.load and shell=True rules among them) and the advisory registry scan.
Adding a source directory to it would leave both jobs green while they scanned less.
This test allows exactly two kinds of entry: `tests/`, and a file under
`semgrep-rules/` that is not a rule file and carries a `# ruleid:` annotation (a
fixture, which scripts/check_semgrep_rule_fixtures.py scans by name instead). It
also requires that file to be the only tracked .semgrepignore: semgrep honours a
nested one too, so `src/.semgrepignore` holding `*` would narrow both scans
without touching the root file.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RULE_SUFFIXES = (".yml", ".yaml")


def _entries() -> list[str]:
    text = (ROOT / ".semgrepignore").read_text(encoding="utf-8")
    stripped = (line.strip() for line in text.splitlines())
    return [line for line in stripped if line and not line.startswith("#")]


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_the_root_semgrepignore_is_the_only_one() -> None:
    if not (ROOT / ".git").exists():
        pytest.skip("not a git work tree, so the tracked file set cannot be read")
    listed = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--", ":(glob)**/.semgrepignore"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert listed == [".semgrepignore"], f"nested .semgrepignore files narrow the scans: {listed}"


def _is_fixture(entry: str) -> bool:
    path = ROOT / entry
    return (
        entry.startswith("semgrep-rules/")
        and path.is_file()
        and path.suffix not in RULE_SUFFIXES
        and re.search(r"#\s*ruleid:", path.read_text(encoding="utf-8")) is not None
    )


def test_semgrepignore_lists_tests_and_at_least_one_fixture() -> None:
    entries = _entries()
    assert "tests/" in entries
    assert any(_is_fixture(entry) for entry in entries)


def test_semgrepignore_hides_nothing_but_tests_and_rule_fixtures() -> None:
    unexpected = [entry for entry in _entries() if entry != "tests/" and not _is_fixture(entry)]
    assert not unexpected, (
        ".semgrepignore narrows both semgrep jobs; these entries are neither tests/ nor an "
        f"annotated rule fixture: {unexpected}"
    )
