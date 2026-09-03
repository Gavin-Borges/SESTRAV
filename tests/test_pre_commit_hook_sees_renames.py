"""The pre-commit hook must scan RENAMED files, not only added and modified ones.

Why this test exists
--------------------
`scripts/hooks/pre-commit` builds exactly one file list at the top and all four gates
iterate it. That list used `--diff-filter=ACM`, which omits status **R**. Because git
performs rename detection by default, a moved file was reported as R and therefore
dropped from the list, so a `git mv` bypassed every gate at once: blocked paths,
credential content, em-dashes and workstation paths.

Measured on commit ce31e02 before the fix: 8 changed paths, 6 of them R100, and only
1 survived `--diff-filter=ACM`.

Renames are not only pure moves. This history contains an R056, i.e. 44% of the
content changed and git still classified it R, so content could be introduced through
the hole rather than merely relocated.

The em-dash gate (gate 3) is used as the probe because it is the sharpest
demonstration of the bug: its outer detector reads the whole staged diff and DOES see
the character, but the inner loop that actually sets the failure flag iterates the
filtered list. Before the fix the hook detected the banned character and exited 0.

Anti-vacuity
------------
`test_rename_is_actually_classified_as_a_rename` is the load-bearing anchor. If git
had staged the change as an add plus a delete instead of a rename, the destination
would carry status A, the unfixed `ACM` filter would have caught it, and the blocking
test below would pass against the broken hook. That test asserts the premise directly
so the suite cannot quietly stop testing the thing it is named for.

The two companions guard the other directions: a plain add must still be blocked
(proving the fixture can trigger the gate at all), and a clean rename must still be
allowed (proving the fix is not the degenerate "reject every rename").
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK_SOURCE = REPO_ROOT / "scripts" / "hooks" / "pre-commit"

# Built with chr() on purpose, and deliberately NOT as a string literal.
# A bare U+2014 here would be caught by the very gate under test. An escape keeps
# the SOURCE ascii but does not help either: the allowlist gate in
# tests/test_encoding_ascii_output.py walks ast.Constant nodes and inspects the
# PARSED VALUE, so an escaped literal still holds U+2014 and still fails.
# chr() builds the codepoint at runtime, so no string literal in this module
# carries it. Allowlisting this module instead would also have worked, but that
# gate skips the whole FILE, which would stop it noticing a genuinely stray
# literal added here later.
EM_DASH = chr(0x2014)

# Long enough that appending one line keeps similarity far above git's 50% rename
# threshold, so the staged change is classified R rather than A plus D.
BASELINE_BODY = "\n".join(f"line {i} of ordinary ASCII content" for i in range(40)) + "\n"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None,
    reason="pre-commit is a bash script; without bash it cannot run at all",
)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    )


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """A throwaway repo with the hook under test copied in and one baseline commit."""
    repo = tmp_path / "sandbox"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    # Do not install via core.hooksPath: the hook is invoked directly below so that
    # its exit code and stderr are observable rather than wrapped by git commit.
    (repo / "hook").write_text(
        HOOK_SOURCE.read_text(encoding="utf-8"), encoding="utf-8", newline=""
    )
    (repo / "baseline.md").write_text(BASELINE_BODY, encoding="utf-8")
    _git(repo, "add", "baseline.md")
    _git(repo, "commit", "--quiet", "-m", "baseline")
    return repo


def _run_hook(repo: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "hook"], cwd=repo, capture_output=True, text=True, check=False
    )


def _stage_renamed_file_with_em_dash(repo: Path) -> None:
    _git(repo, "mv", "baseline.md", "moved.md")
    moved = repo / "moved.md"
    moved.write_text(
        BASELINE_BODY + f"a line with an {EM_DASH} in it\n", encoding="utf-8"
    )
    _git(repo, "add", "moved.md")


def test_rename_is_actually_classified_as_a_rename(repo: Path) -> None:
    """Anti-vacuity anchor: the premise of the blocking test must actually hold.

    If this fails, the fixture is staging an add plus a delete rather than a rename,
    and the blocking test below would pass even against the unfixed hook.
    """
    _stage_renamed_file_with_em_dash(repo)
    status = _git(repo, "diff", "--cached", "--name-status").stdout
    assert any(
        line.startswith("R") for line in status.splitlines()
    ), f"expected a rename (R) in the staged status, got:\n{status}"
    # And the precise defect: the old filter cannot see it, the new one can.
    acm = _git(repo, "diff", "--cached", "--name-only", "--diff-filter=ACM").stdout
    acmr = _git(repo, "diff", "--cached", "--name-only", "--diff-filter=ACMR").stdout
    assert "moved.md" not in acm, "ACM unexpectedly sees the rename; probe is invalid"
    assert "moved.md" in acmr, "ACMR must yield the rename destination path"


def test_renamed_file_with_banned_content_is_blocked(repo: Path) -> None:
    """The regression itself: a renamed file carrying an em-dash must be rejected."""
    _stage_renamed_file_with_em_dash(repo)
    result = _run_hook(repo)
    assert result.returncode != 0, (
        "hook exited 0 on a renamed file containing U+2014; the staged-file "
        f"enumeration is not seeing renames.\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    # Blocked for the RIGHT reason, not merely blocked.
    assert "Em-dash" in result.stderr, (
        f"expected the em-dash gate to fire; stderr was:\n{result.stderr}"
    )


def test_plain_add_with_banned_content_is_still_blocked(repo: Path) -> None:
    """Companion: proves the fixture can trigger the gate through an ordinary path."""
    (repo / "added.md").write_text(f"an {EM_DASH} here\n", encoding="utf-8")
    _git(repo, "add", "added.md")
    result = _run_hook(repo)
    assert result.returncode != 0, "an added file with U+2014 must be blocked"
    assert "Em-dash" in result.stderr


def test_clean_rename_is_still_allowed(repo: Path) -> None:
    """Companion: the fix must widen the scan, not reject every rename.

    A hook changed to block renames outright would satisfy the two tests above, so
    this pins the other side of the contract.
    """
    _git(repo, "mv", "baseline.md", "renamed_cleanly.md")
    _git(repo, "add", "-A")
    status = _git(repo, "diff", "--cached", "--name-status").stdout
    assert any(line.startswith("R") for line in status.splitlines()), (
        f"fixture did not produce a rename; got:\n{status}"
    )
    result = _run_hook(repo)
    assert result.returncode == 0, (
        "a rename carrying no banned content must pass.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
