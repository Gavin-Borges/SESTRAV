"""Every bandit suppression in scanned code names the test it suppresses.

A bare `# nosec` silences EVERY bandit test on its line, including tests added
after it was written, so it hides future findings rather than one reviewed
finding. The one bare `# nosec` that stood in scanned code, on a `torch.save`
call in `src/ann_benchmark.py`, suppressed nothing: measured 2026-10-08 with
bandit 1.9.4, that file reports 0 issues and 0 skipped lines both with and
without `--ignore-nosec`. It was removed rather than scoped.

The scope is the tree CI's bandit job scans: every tracked `.py` file outside
the directories its `-x` list excludes. Comments are read with `tokenize`, so
the word inside a string literal does not count, and a comment that merely
quotes the directive in prose is outside scope only because `tests/` is.
"""

from __future__ import annotations

import io
import re
import subprocess
import tokenize
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# The -x list of the bandit step in .github/workflows/security.yml.
EXCLUDED_PREFIXES = ("tests/", ".ci_test_venv/", ".venv/", "build/")

NOSEC = re.compile(r"#\s*nosec\b")
TARGETED = re.compile(r"#\s*nosec\s*:?\s*B\d{3}")


def _scanned_files() -> list[str]:
    if not (ROOT / ".git").exists():
        pytest.skip("not a git work tree, so the tracked file set cannot be read")
    out = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z", "--", "*.py"],
        capture_output=True,
        check=True,
    ).stdout
    return [p for p in out.decode("utf-8").split("\0") if p and not p.startswith(EXCLUDED_PREFIXES)]


def _bare_nosec_lines(rel_path: str) -> list[int]:
    text = (ROOT / rel_path).read_text(encoding="utf-8")
    return [
        tok.start[0]
        for tok in tokenize.generate_tokens(io.StringIO(text).readline)
        if tok.type == tokenize.COMMENT and NOSEC.search(tok.string) and not TARGETED.search(tok.string)
    ]


def test_every_nosec_in_scanned_code_names_its_bandit_test() -> None:
    files = _scanned_files()
    assert files, "git ls-files returned no Python files, so this test would pass vacuously"
    bare = [(path, line) for path in files for line in _bare_nosec_lines(path)]
    assert bare == [], (
        "these # nosec comments name no bandit test ID; scope each to the ID it suppresses "
        f"(for example `# nosec B603`) or delete it if it suppresses nothing: {bare}"
    )


def test_the_targeted_pattern_accepts_both_spellings_bandit_reads() -> None:
    """The guard must not reject a scoped suppression in either form bandit accepts."""
    for comment in ("# nosec B603", "# nosec: B310", "#nosec B404 - fixed argv"):
        assert NOSEC.search(comment) and TARGETED.search(comment), comment
    assert NOSEC.search("# nosec") and not TARGETED.search("# nosec"), "a bare directive must be caught"
