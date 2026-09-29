"""The pre-commit hook must scan staged files whose PATH git has to quote.

Why this test exists
--------------------
`scripts/hooks/pre-commit` builds one file list at the top and all four gates
iterate it. That list came from `git diff --cached --name-only`, and
`core.quotePath` defaults to TRUE, so any path holding a byte outside ASCII was
emitted C-quoted as `"caf\\303\\251.txt"`. Each gate then asked
`git show ":${file}"` for that quoted string, git exited 128 because no such
path is in the index, and a `|| continue` treated the failure as "this file is
clean".

Measured 2026-09-23 with the real hook in a throwaway repo: a staged file named
with a non-ASCII character and carrying an AWS access key ID committed at
**rc=0**, while byte-identical content in `plain.txt` was blocked at **rc=1**.
The gate was not weak on that file; it never looked at it.

Two independent fixes, and this module exercises both
-----------------------------------------------------
1. `-c core.quotePath=false` on the enumeration, so ordinary non-ASCII names
   resolve and are scanned normally.
2. A failed `git show` now BLOCKS instead of continuing. Unquoting does not
   cover every shape git quotes - a backslash in a name is still quoted - so
   the residual cases must fail CLOSED rather than silently pass. This is the
   same conflation `scripts/check_secrets.py` already fixed for unreadable
   files, where it reports `[UNREADABLE]` and returns 1 rather than folding
   "could not look" into "looked and found nothing".

Anti-vacuity
------------
`test_non_ascii_path_with_clean_content_is_allowed` is the load-bearing
partner. Fix 2 alone would block every non-ASCII path, which would also turn
the blocking test below green while making the hook unusable. That test fails
against such a degenerate fix, so the two cannot be confused.

`test_ascii_control_is_still_blocked` proves the fixture can trigger the gate at
all, so a blocking result is attributable to the path and not to a broken
fixture.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK_SOURCE = REPO_ROOT / "scripts" / "hooks" / "pre-commit"

# Built at runtime so no string literal in this module carries a non-ASCII
# character. tests/test_encoding_ascii_output.py walks ast.Constant nodes and
# inspects the PARSED value, so a backslash-u escape would not help either: the
# parsed string still holds the codepoint and still fails that gate.
NON_ASCII_NAME = "caf" + chr(0xE9) + ".txt"

# Split so this file never contains a literal in AWS access-key-id FORMAT.
# pre-commit Gate 2 reads staged content and scripts/check_secrets.py scans
# tests/ as tracked source, so a whole literal here would be flagged by the
# very gates under test.
AWS_KEY_LINE = 'aws_access_key_id = "' + "AKIA" + 'IOSFODNN7EXAMPLE"\n'
CLEAN_BODY = "ordinary ascii content with no credential\n"

# git reads the developer's GLOBAL and SYSTEM config even inside a throwaway
# repo, so without this the result depends on the machine rather than on the
# hook. Mirrors tests/test_pre_commit_hook_sees_renames.py, which records the
# specific settings measured to break a hook fixture.
GIT_ENV = {
    **os.environ,
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
}

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None,
    reason="pre-commit is a bash script; without bash it cannot run at all",
)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    # encoding="utf-8" is required, not tidiness. git writes path bytes as UTF-8
    # while text=True alone decodes with locale.getpreferredencoding(), which is
    # cp1252 on the Windows workstation. Without this the captured name is
    # double-encoded and compares unequal to the same name built in Python -
    # the assertion reads `assert 'cafe.txt' in 'cafe.txt'` and fails, which is
    # the identical defect this module's subject fixes in the hook itself.
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="surrogateescape",
        check=True,
        env=GIT_ENV,
    )


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """A throwaway repo with the hook under test copied in and one baseline commit."""
    repo = tmp_path / "sandbox"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    # newline="" preserves the hook's own line endings. This checkout is CRLF
    # and bash rejects a CRLF shebang outright, which was measured to produce a
    # confident rc=1 from a hook that had not run at all.
    (repo / "hook").write_text(
        HOOK_SOURCE.read_text(encoding="utf-8"), encoding="utf-8", newline=""
    )
    (repo / "baseline.md").write_text(CLEAN_BODY, encoding="utf-8")
    _git(repo, "add", "baseline.md")
    _git(repo, "commit", "--quiet", "-m", "baseline")
    return repo


def _run_hook(repo: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "hook"],
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="surrogateescape",
        check=False,
        env=GIT_ENV,
    )


def _stage(repo: Path, name: str, body: str) -> None:
    (repo / name).write_text(body, encoding="utf-8")
    _git(repo, "add", name)


def test_enumeration_returns_the_unquoted_name(repo: Path) -> None:
    """The premise, asserted directly.

    If git ever stopped quoting non-ASCII paths by default, the blocking test
    below would pass against the unfixed hook and stop testing anything.
    """
    _stage(repo, NON_ASCII_NAME, CLEAN_BODY)
    quoted = _git(repo, "diff", "--cached", "--name-only").stdout
    unquoted = _git(
        repo, "-c", "core.quotePath=false", "diff", "--cached", "--name-only"
    ).stdout
    assert NON_ASCII_NAME not in quoted, (
        "git no longer quotes non-ASCII paths by default; this suite's premise is gone"
    )
    assert NON_ASCII_NAME in unquoted


def test_non_ascii_path_with_a_credential_is_blocked(repo: Path) -> None:
    """The measured defect: this committed at rc=0 before the fix."""
    _stage(repo, NON_ASCII_NAME, AWS_KEY_LINE)
    result = _run_hook(repo)
    assert result.returncode == 1, (
        f"credential on a non-ASCII path was not blocked; "
        f"rc={result.returncode} stderr={result.stderr[:400]}"
    )


def test_ascii_control_is_still_blocked(repo: Path) -> None:
    """Byte-identical content on an ASCII path. Proves the fixture can trigger
    the gate, so a block above is attributable to the path handling."""
    _stage(repo, "plain.txt", AWS_KEY_LINE)
    assert _run_hook(repo).returncode == 1


def test_non_ascii_path_with_clean_content_is_allowed(repo: Path) -> None:
    """Anti-vacuity anchor.

    Failing the git show read CLOSED, on its own, blocks every path the
    enumeration cannot resolve - which would include every ordinary non-ASCII
    filename and make the hook unusable. This asserts the unquoting half is
    really there, so the blocking test cannot be satisfied by the degenerate
    "reject everything non-ASCII".
    """
    _stage(repo, NON_ASCII_NAME, CLEAN_BODY)
    result = _run_hook(repo)
    assert result.returncode == 0, (
        f"clean non-ASCII path was rejected; the hook is over-blocking. "
        f"stderr={result.stderr[:400]}"
    )


@pytest.mark.skipif(
    os.name == "nt",
    reason=(
        "a backslash cannot appear in a Windows filename, so the shape that git "
        "quotes even under core.quotePath=false is unconstructible here; measured "
        "on WSL2 Ubuntu instead"
    ),
)
def test_a_path_git_quotes_anyway_fails_closed(repo: Path) -> None:
    """Unquoting does not reach every shape. A backslash in a name is still
    emitted quoted, `git show` still fails, and the hook must BLOCK rather than
    skip. Measured on Linux: the pre-fix hook exits 0 on exactly this input.
    """
    name = "back" + chr(0x5C) + "slash.txt"
    _stage(repo, name, AWS_KEY_LINE)
    result = _run_hook(repo)
    assert result.returncode == 1, (
        f"unreadable staged path was treated as clean; rc={result.returncode}"
    )
    assert "NOT cleared" in result.stderr, (
        f"blocked, but not by the fail-closed branch; stderr={result.stderr[:400]}"
    )
