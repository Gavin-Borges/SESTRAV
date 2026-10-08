"""pre-push Check 6, pass (b): a pushed blob with a non-ASCII name is scanned.

Why this test exists
--------------------
Pass (b) scans every blob that the pushed commits add or modify, so content
crossing to the remote is scanned even when the checkout no longer holds it. It
listed those paths with `git diff-tree --name-only` and no `-z`. git then applies
core.quotePath, which defaults to true, and returns a name like `caf<e-acute>.py`
as the quoted, octal-escaped string `"caf\\303\\251.py"`. `_is_scannable`,
imported from scripts/check_secrets.py, rejects that string because it ends in a
quote rather than `.py`, so the blob was never scanned and a credential in it was
pushed with "Pushed commit ranges contain no credential assignments." printed.

How the hook is driven
----------------------
The whole tracked hook is copied into a throwaway repository and run with bash,
as git runs it, with the pushed ref on stdin. A `python` shim placed first on
PATH runs pass (b)'s heredoc, the hook's only `python -` call, with the
interpreter running this test, and answers every other python call with 0, so
no other check can block or pass the push on these cases. scripts/check_secrets.py
is copied beside the hook because pass (b) imports its scanner from there.

The fixture pins core.quotePath to true in its own config, and the global and
system config to the null device, so the result does not depend on the
developer's git settings. `test_the_fixture_name_is_quoted_without_z` asserts
that premise directly.

Anti-vacuity
------------
`test_an_ascii_named_secret_is_still_blocked` is the control: the same payload
under an ASCII name must block, so the case above cannot pass because the scan
never ran at all. `test_a_clean_non_ascii_named_file_still_passes` guards the
other direction, so the fix cannot be "block every non-ASCII name" or a name
that `git show` cannot resolve.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK_SOURCE = REPO_ROOT / "scripts" / "hooks" / "pre-push"
SCANNER_SOURCE = REPO_ROOT / "scripts" / "check_secrets.py"

# Built with chr() so this source file stays pure ASCII.
NON_ASCII_NAME = "caf" + chr(0xE9) + ".py"
ASCII_NAME = "plain.py"

FLAGGED = "[FLAGGED-RANGE]"
UNREADABLE = "[UNREADABLE-RANGE]"
RANGE_CLEAN = "Pushed commit ranges contain no credential assignments."
RANGE_BLOCKED = "Pushed commit range credential scan failed."
PUSH_ALLOWED = "All checks passed -- push allowed."

# Runs pass (b)'s heredoc with a real interpreter; every other python call exits 0.
SHIM = (
    "#!/bin/sh\n"
    'if [ "$1" = "-" ]; then\n'
    '    exec "$PREPUSH_REAL_PYTHON" "$@"\n'
    "fi\n"
    "exit 0\n"
)

# git reads the developer's GLOBAL and SYSTEM config even for a throwaway directory.
# Pinned so the result depends on the hook. The repository-discovery variables are
# dropped so every git call resolves to the fixture repository.
BASE_ENV = {
    key: value
    for key, value in os.environ.items()
    if key not in {"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_PREFIX", "REPO_ROOT"}
}
BASE_ENV.update(
    {
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_AUTHOR_NAME": "fixture",
        "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
        "GIT_COMMITTER_NAME": "fixture",
        "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
    }
)

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None,
    reason="pre-push is a bash script; without bash it cannot run at all",
)


def _payload() -> str:
    """One credential-keyword assignment that scan_file flags, assembled at runtime.

    The value is mixed-alphabet, longer than 8 and above the 3.0 entropy floor.
    It is not a real credential.
    """
    keyword = "api_" + "key"
    value = "a8f3k9d2" + "m1q7x4z0b5"
    return keyword + ' = "' + value + '"\n'


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "core.autocrlf=false", *args],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
        env=BASE_ENV,
        timeout=60,
    ).stdout


def _fixture(tmp_path: Path, name: str, content: str) -> Path:
    """A repository whose one commit adds `name`; the hook, scanner and shim are not committed."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "hook").write_text(
        HOOK_SOURCE.read_text(encoding="utf-8"), encoding="utf-8", newline=""
    )
    scanner = root / "scripts" / "check_secrets.py"
    scanner.parent.mkdir()
    shutil.copyfile(SCANNER_SOURCE, scanner)
    shim_dir = root / "shim"
    shim_dir.mkdir()
    shim = shim_dir / "python"
    shim.write_text(SHIM, encoding="utf-8", newline="\n")
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    (root / name).write_text(content, encoding="utf-8", newline="\n")
    _git(root, "init", "-q")
    _git(root, "config", "core.quotePath", "true")
    _git(root, "add", "--", name)
    _git(root, "commit", "-q", "-m", "fixture")
    return root


def _push(root: Path) -> subprocess.CompletedProcess:
    """Push a new branch whose tip is the fixture's commit, as git hands it to the hook."""
    local_sha = _git(root, "rev-parse", "HEAD").strip()
    env = dict(BASE_ENV)
    env.update(
        {
            # Skips the hook's conda activation, so the shim stays the interpreter.
            "CONDA_DEFAULT_ENV": "sestrav",
            "PREPUSH_REAL_PYTHON": Path(sys.executable).as_posix(),
        }
    )
    env["PATH"] = str(root / "shim") + os.pathsep + env.get("PATH", "")
    stdin = f"refs/heads/feature {local_sha} refs/heads/feature {'0' * 40}\n"
    return subprocess.run(
        ["bash", "hook", "origin", "https://example.invalid/fixture.git"],
        cwd=root,
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        env=env,
        timeout=120,
    )


def _load_scanner():
    spec = importlib.util.spec_from_file_location("check_secrets", SCANNER_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_fixture_name_is_quoted_without_z(tmp_path: Path) -> None:
    """Premise: without -z, git hands back a quoted name that is not scannable."""
    root = _fixture(tmp_path, NON_ASCII_NAME, _payload())
    listed = _git(root, "diff-tree", "--root", "--no-commit-id", "--name-only", "-r", "HEAD")
    assert listed.strip() == '"caf\\303\\251.py"', listed
    scanner = _load_scanner()
    assert not scanner._is_scannable(listed.strip())
    assert scanner._is_scannable(NON_ASCII_NAME)
    assert scanner.scan_file(str(root / NON_ASCII_NAME)) == [1], "payload is not flagged"


def test_a_non_ascii_named_secret_is_blocked(tmp_path: Path) -> None:
    result = _push(_fixture(tmp_path, NON_ASCII_NAME, _payload()))
    assert result.returncode != 0, result.stdout + result.stderr
    assert f"{FLAGGED} caf" in result.stderr, result.stdout + result.stderr
    assert UNREADABLE not in result.stderr, result.stderr
    assert RANGE_BLOCKED in result.stderr, result.stderr
    assert PUSH_ALLOWED not in result.stdout, result.stdout


def test_an_ascii_named_secret_is_still_blocked(tmp_path: Path) -> None:
    result = _push(_fixture(tmp_path, ASCII_NAME, _payload()))
    assert result.returncode != 0, result.stdout + result.stderr
    assert f"{FLAGGED} {ASCII_NAME} at" in result.stderr, result.stdout + result.stderr
    assert RANGE_BLOCKED in result.stderr, result.stderr


def test_a_clean_non_ascii_named_file_still_passes(tmp_path: Path) -> None:
    result = _push(_fixture(tmp_path, NON_ASCII_NAME, "print('hello')\n"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert RANGE_CLEAN in result.stdout, result.stdout + result.stderr
    assert PUSH_ALLOWED in result.stdout, result.stdout
    assert UNREADABLE not in result.stderr, result.stderr
