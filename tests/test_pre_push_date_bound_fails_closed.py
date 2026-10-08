"""pre-push Check 1b: the date-released lower bound fails closed, as release.yml does.

Why this test exists
--------------------
Check 1b blocks a version tag whose CITATION.cff `date-released` precedes the committer
date of the commit being tagged. It reads that date with `git show -s --format=%cs` and,
before this test, compared it to `date-released` as a string with nothing checking that
git had printed a date at all.

For a commit whose committer line holds no parseable date, git does not print an empty
string: it exits 0 and prints the placeholder itself, `%cs`. Measured 2026-10-07 on Git
for Windows 2.55.0 and on Linux git 2.53.0, for a committer line with no timestamp, a
non-numeric one, no timezone, no email, and no committer line at all. `%` sorts before
every digit, so `[[ "$cff_date" < "$commit_date" ]]` was false and the bound was skipped:
the hook printed its pass banner. The Release workflow records the same `git show` output
as COMMIT_DATE and refuses any value that is not YYYY-MM-DD, so the local copy passed a
tag its own workflow fails.

git's own commit commands refuse such a date (`git commit` and `git commit-tree` exit 128
with "invalid date format" for GIT_COMMITTER_DATE=garbage), so such a commit comes from
a hand-built object (as here, through `git hash-object --literally`) or from
another git implementation. The hook runs before anything is sent, so it still has to
give the workflow's verdict.

The oracle
----------
`test_the_release_workflow_refuses_that_commit_date` runs the workflow's own gate script,
extracted the way tests/test_release_citation_date_gate.py extracts it, with COMMIT_DATE
set to exactly what `git show` printed for the fixture commit. The assertion on the hook
is therefore an assertion that the two agree, not a restatement of either.

Anti-vacuity
------------
`test_a_readable_commit_date_still_passes` runs the same carriers on an ordinary commit,
so the fix cannot be the degenerate "block every tag". The premise test asserts that the
undated commit's carriers are readable and that only its date is not, so the blocking
case exercises the lower bound and no earlier check.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK_SOURCE = REPO_ROOT / "scripts" / "hooks" / "pre-push"
RELEASE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release.yml"
STEP_NAME = "Verify tag matches package version"

TAG = "v2.0.3"
VERSION = "2.0.3"
RELEASED = "2024-01-01"
SUCCESS_BANNER = "version carriers agree, date-released is not future"
UNREADABLE_DATE = "has no readable committer date"
BOUND_UNCHECKED = "lower bound cannot be checked"
FIXTURE_COMMIT_DATE = "1970-01-01T00:00:00+0000"

# The workflow's own test for COMMIT_DATE, written as it is in release.yml.
WORKFLOW_DATE_SHAPE = re.compile(r"\d{4}-\d{2}-\d{2}")

# git reads the developer's GLOBAL and SYSTEM config even for a throwaway directory, and
# both the fixture and the hook make git calls. Pinned so the result depends on the hook.
# The repository-discovery variables are dropped so every git call resolves to the
# fixture repository, never to the repository running this test.
BASE_ENV = {
    key: value
    for key, value in os.environ.items()
    if key not in {"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_PREFIX"}
}
BASE_ENV.update(
    {
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_AUTHOR_NAME": "fixture",
        "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
        "GIT_COMMITTER_NAME": "fixture",
        "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
        # Pinned to the epoch so RELEASED does not precede the ordinary commit.
        "GIT_AUTHOR_DATE": FIXTURE_COMMIT_DATE,
        "GIT_COMMITTER_DATE": FIXTURE_COMMIT_DATE,
    }
)

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None,
    reason="pre-push is a bash script; without bash it cannot run at all",
)


def _git(root: Path, *args: str) -> str:
    """Run git in the fixture repository. autocrlf off, so every file is stored byte for byte."""
    return subprocess.run(
        ["git", "-c", "core.autocrlf=false", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
        env=BASE_ENV,
        timeout=60,
    ).stdout.strip()


def _fixture(tmp_path: Path) -> Path:
    """A repository whose one ordinary commit holds two carriers that agree with TAG."""
    root = tmp_path / "carriers"
    root.mkdir()
    (root / "hook").write_text(
        HOOK_SOURCE.read_text(encoding="utf-8"), encoding="utf-8", newline=""
    )
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "fixture"\nversion = "{VERSION}"\n', encoding="utf-8", newline=""
    )
    (root / "CITATION.cff").write_text(
        f"cff-version: 1.2.0\ntitle: fixture\nversion: {VERSION}\ndate-released: {RELEASED}\n",
        encoding="utf-8",
        newline="",
    )
    # Everything after Check 1b needs a Python with pytest. A `python` that exits 1 makes
    # the hook stop at its pytest pre-flight instead of running a real suite.
    shim_dir = root / "shim"
    shim_dir.mkdir()
    shim = shim_dir / "python"
    shim.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8", newline="\n")
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    _git(root, "init", "-q")
    _git(root, "add", "pyproject.toml", "CITATION.cff")
    _git(root, "commit", "-q", "-m", "carriers")
    return root


def _undated_commit(root: Path) -> str:
    """The ordinary commit's tree, re-committed with a committer line that has no date."""
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    body = (
        f"tree {tree}\n"
        "author fixture <fixture@example.invalid> 0 +0000\n"
        "committer fixture <fixture@example.invalid>\n"
        "\n"
        "carriers, undated\n"
    )
    # Bytes, not text: text-mode stdin on Windows would turn every LF into CRLF.
    return subprocess.run(
        ["git", "hash-object", "-t", "commit", "-w", "--literally", "--stdin"],
        cwd=root,
        input=body.encode("ascii"),
        capture_output=True,
        check=True,
        env=BASE_ENV,
        timeout=60,
    ).stdout.decode("ascii").strip()


def _push_tag(root: Path, local_sha: str) -> subprocess.CompletedProcess:
    env = dict(BASE_ENV)
    env["PATH"] = str(root / "shim") + os.pathsep + env.get("PATH", "")
    stdin = f"refs/tags/{TAG} {local_sha} refs/tags/{TAG} {'0' * 40}\n"
    return subprocess.run(
        ["bash", "hook", "origin", "https://example.invalid/fixture.git"],
        cwd=root,
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=120,
    )


def _gate_script() -> str:
    """The release workflow's gate, extracted as tests/test_release_citation_date_gate.py does."""
    workflow = yaml.safe_load(RELEASE_WORKFLOW.read_text(encoding="utf-8"))
    matches = [
        step
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if step.get("name") == STEP_NAME
    ]
    assert len(matches) == 1, f"expected one {STEP_NAME!r} step, found {len(matches)}"
    lines = matches[0]["run"].splitlines()
    assert lines[0] == "python - <<'PY'", lines[0]
    assert lines[-1] == "PY", lines[-1]
    return "\n".join(lines[1:-1]) + "\n"


def test_the_premise_only_the_commit_date_is_unreadable(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    sha = _undated_commit(root)
    assert _git(root, "rev-parse", "--verify", "--quiet", f"{sha}^{{commit}}") == sha
    assert f'version = "{VERSION}"' in _git(root, "cat-file", "blob", f"{sha}:pyproject.toml")
    assert f"date-released: {RELEASED}" in _git(root, "cat-file", "blob", f"{sha}:CITATION.cff")
    printed = _git(root, "show", "-s", "--format=%cs", sha)
    assert not WORKFLOW_DATE_SHAPE.fullmatch(printed), printed
    # The ordinary commit's date does read, so the control below tests the same bound.
    assert _git(root, "show", "-s", "--format=%cs", "HEAD") == "1970-01-01"


def test_the_release_workflow_refuses_that_commit_date(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    printed = _git(root, "show", "-s", "--format=%cs", _undated_commit(root))
    env = dict(os.environ, TAG_NAME=TAG, COMMIT_DATE=printed)
    result = subprocess.run(
        [sys.executable, "-"],
        input=_gate_script(),
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "fails closed" in result.stdout, result.stdout


def test_an_unreadable_commit_date_blocks_the_tag(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    result = _push_tag(root, _undated_commit(root))
    assert result.returncode != 0
    assert UNREADABLE_DATE in result.stderr, result.stdout + result.stderr
    assert BOUND_UNCHECKED in result.stderr, result.stderr
    assert SUCCESS_BANNER not in result.stdout, result.stdout


def test_a_readable_commit_date_still_passes(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    result = _push_tag(root, _git(root, "rev-parse", "HEAD"))
    assert SUCCESS_BANNER in result.stdout, result.stdout + result.stderr
    assert UNREADABLE_DATE not in result.stderr, result.stderr
