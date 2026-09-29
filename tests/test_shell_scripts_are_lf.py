"""Tracked shell scripts and git hooks must stay LF, or Linux bash cannot parse them.

The motivating defect is a cross-platform one that no existing gate could see.
`.gitattributes` carries an extensively audited `eol=lf` policy, but every block
of it exists so that a RECORDED sha256 stays portable across platforms. Shell
scripts were never pinned, and `core.autocrlf=true` on a Windows workstation
therefore checks them out with CRLF.

Measured 2026-09-23 with `bash -n`, which parses without executing, under WSL2
Ubuntu (GNU bash, grep 3.12) against a Windows checkout: **12 of the 18 tracked
shell files failed to parse**, each reporting a syntax error, while all 18
parsed cleanly once CR bytes were stripped. The control arm is what isolates the
cause to the line ending rather than the script.

The 12 included all five files under `scripts/hooks/` (the four hooks and their
installer) and three of the four `*_wsl.sh` scripts, which exist precisely to be
run from WSL against this checkout. Running `scripts/hooks/commit-msg` directly
under Linux bash exited 2 without reaching any gate; the same hook with CR
stripped exited 0 on a clean message and 1 on an em-dash, which is the
documented behaviour. So the CRLF form is not a stricter hook, it is an absent
one.

The remaining 6 parsed only because they carry no multi-line construct for a
stray CR to break. They are latently exposed rather than sound, which is why
this test covers the whole population instead of the 12 that happened to fail.

Both directions are pinned, because either alone can pass while the defect is
live: the committed blob must be free of CR bytes, AND `.gitattributes` must
resolve `eol=lf` for the path. A blob can be LF today while an unpinned path
lets the next Windows contributor commit CRLF, and a pin can exist while a blob
already carries CRLF, since a pin is not retroactive.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None, reason="needs git to read tracked paths and attributes"
)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _tracked_shell_files() -> list[str]:
    """Every tracked `*.sh` plus everything under `scripts/hooks/`.

    `scripts/hooks/` is matched as a directory rather than by extension because
    four of its five members carry no extension at all.
    """
    try:
        listed = _git("ls-files").splitlines()
    except (subprocess.CalledProcessError, OSError):
        # Deliberately swallowed HERE and nowhere else. This function runs at
        # COLLECTION time, inside the parametrize decorators below, so an
        # exception escaping it takes the whole module down as a collection
        # error - and a collection error means ZERO tests ran while reporting as
        # a single red line. Returning empty instead routes the failure through
        # test_discovery_is_not_vacuous, which fails loudly and names the cause.
        # Measured: `git ls-files` exits non-zero inside a worktree created by
        # Windows git and read from WSL, because the worktree's `.git` file
        # points at a `C:/...` gitdir that Linux git cannot resolve.
        return []
    return sorted(
        p for p in listed if p.endswith(".sh") or p.startswith("scripts/hooks/")
    )


# The load-bearing anchor. Without it every assertion below is vacuously true
# whenever discovery returns nothing - a wrong cwd, a non-repo, or a renamed
# directory would all read as a pass. These five are the gates whose absence
# would be most costly, so they are named rather than counted: a count would go
# stale the first time a script is added, and this repo has been burned by
# exactly that.
REQUIRED_MEMBERS = (
    "scripts/hooks/commit-msg",
    "scripts/hooks/pre-commit",
    "scripts/hooks/pre-push",
    "scripts/hooks/prepare-commit-msg",
    "scripts/hooks/install.sh",
)


def test_discovery_is_not_vacuous() -> None:
    found = _tracked_shell_files()
    missing = [p for p in REQUIRED_MEMBERS if p not in found]
    assert not missing, (
        "shell-file discovery did not return known tracked hooks "
        f"{missing}; every other test in this file would pass vacuously. "
        f"Discovery returned {len(found)} path(s). If it returned 0, `git "
        "ls-files` failed outright - check that this is a readable checkout "
        "and not a Windows-created worktree being read from WSL."
    )


@pytest.mark.parametrize("path", _tracked_shell_files())
def test_committed_blob_has_no_carriage_returns(path: str) -> None:
    blob = subprocess.run(
        ["git", "show", f"HEAD:{path}"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    ).stdout
    count = blob.count(b"\r")
    assert count == 0, (
        f"{path} carries {count} CR byte(s) in its committed blob, so a Linux "
        "checkout gets CRLF and bash may fail to parse it. Re-commit with LF."
    )


@pytest.mark.parametrize("path", _tracked_shell_files())
def test_gitattributes_pins_eol_lf(path: str) -> None:
    out = _git("check-attr", "eol", "--", path).strip()
    assert out.endswith(": lf"), (
        f"{path} has no eol=lf pin ({out!r}), so a Windows checkout with "
        "core.autocrlf=true materializes it with CRLF even though the blob is "
        "LF. Add it to the shell-script block in .gitattributes."
    )
