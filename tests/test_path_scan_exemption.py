"""CHANGELOG.md is scanned for workstation paths like every other tracked file.

Why this test exists
--------------------
Both path scanners, the `pii_scan.yml` workflow and `scripts/hooks/pre-commit`, used to
exempt `CHANGELOG.md`. The exemption was granted for the file's one matching line, which
the hook's comment described as an example path quoted by a historical-cleanup entry. It
was not an example: it is a real workstation username, published in the commit titled
"Release: SESTRAV v2.0.0-rc1". That commit also created the workflow, whose diff scan then
dropped any added line containing "README.md", and this line does (a bypass since closed).
The line never appeared in a later diff, and both scans added afterwards, the hook's path
scan and the workflow's published-ref scan, exempted the file from the start. The line
itself is removed by a separate change; this one removes the exemption, so the next such
line is caught. The workflow's published-ref scan reads published `main`, so it fails
until that line is gone.

What each test proves
---------------------
- `test_workflow_does_not_exempt_the_changelog` drives the workflow's real `is_exempt`
  function, extracted from the YAML, and fails if the exemption returns through it. A
  `scan_blob` test alone cannot: the workflow applies `is_exempt` in its tracked-file
  loop, not inside `scan_blob`.
- `test_workflow_scan_ref_reports_a_changelog_path` runs the workflow's real `scan_ref`
  over a throwaway repository, so it also fails if the file is skipped any other way
  inside that loop, for example by an inline `continue`.
- `test_workflow_diff_scan_excludes_only_itself` pins the diff scan's `PATHSPEC`, which
  `is_exempt` does not govern, so an exclusion added there fails too.
- `test_workflow_scan_blob_catches_a_changelog_path` proves the detector itself fires
  on a real-shape path in a file named `CHANGELOG.md`.
- `test_pre_commit_hook_catches_a_staged_changelog` runs the real hook against a
  staged `CHANGELOG.md` in a throwaway repository.
- `test_the_empty_exemption_loop_is_safe_on_old_bash` pins the loop's expansion form.
  With the exemption list now EMPTY, a bare `"${arr[@]}"` aborts under `set -u` on
  bash before 4.4, and the hook is deliberately kept runnable under macOS's bash 3.2.
  No bash that old is available to CI, so the form is pinned instead of executed.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "pii_scan.yml"
HOOK = ROOT / "scripts" / "hooks" / "pre-commit"

# git reads the developer's GLOBAL and SYSTEM config even inside a throwaway repo.
# Pinned so the result depends on the hook, not on the machine running it.
GIT_ENV = {
    **os.environ,
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
}

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None,
    reason="both scanners are bash; without bash neither can run",
)


def _workflow_block(start_marker: str, end_marker: str) -> str:
    source = WORKFLOW.read_text(encoding="utf-8")
    assert source.count(start_marker) == 1, start_marker
    assert source.count(end_marker) == 1, end_marker
    body = source.split(start_marker, 1)[1].split(end_marker, 1)[0]
    return textwrap.dedent(start_marker + body)


def _workflow_function(name: str) -> str:
    """One shell function from the workflow, from its opening line to its own closing brace.

    Cutting at a later marker would also capture whatever the workflow runs after the
    function, which then executes here without the setup it depends on.
    """
    lines = WORKFLOW.read_text(encoding="utf-8").splitlines(keepends=True)
    opener = f"          {name}() {{\n"
    assert lines.count(opener) == 1, opener
    start = lines.index(opener)
    end = lines.index("          }\n", start)
    return textwrap.dedent("".join(lines[start : end + 1]))


def _workflow_scan_setup() -> str:
    """The workflow's real pattern setup and function definitions, up to its canary."""
    setup = _workflow_block('          WORK="$(mktemp -d)"', "          # Positive control.")
    return setup.replace('WORK="$(mktemp -d)"', 'WORK="$SCAN_WORK"', 1)


def _synthetic_path() -> str:
    # Assembled at runtime so no literal workstation path sits in this file, which
    # the very hook under test would otherwise refuse to commit.
    slash = chr(92)
    return f"X:{slash}Users{slash}fakeuser123{slash}project{slash}artifact.txt"


def _bash(script: str, **env: str) -> subprocess.CompletedProcess:
    full_env = dict(GIT_ENV)
    full_env.update(env)
    return subprocess.run(
        ["bash", "-c", script],
        cwd=ROOT,
        env=full_env,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("path", "exempt"),
    [
        ("CHANGELOG.md", False),
        ("docs/notes_CHANGELOG.md", False),
        (".github/workflows/pii_scan.yml", True),
    ],
)
def test_workflow_does_not_exempt_the_changelog(tmp_path: Path, path: str, exempt: bool) -> None:
    script = _workflow_scan_setup() + f'\nis_exempt "{path}"\n'
    result = _bash(script, SCAN_WORK=str(tmp_path))
    assert (result.returncode == 0) is exempt, result.stderr


def test_workflow_scan_ref_reports_a_changelog_path(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    work = tmp_path / "work"
    work.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True, env=GIT_ENV)
    (repo / "CHANGELOG.md").write_text(_synthetic_path() + "\n", encoding="utf-8")
    (repo / "notes.txt").write_text("no path here\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, env=GIT_ENV)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
        + ["commit", "-q", "-m", "fixture"],
        check=True,
        env=GIT_ENV,
    )
    scan_ref = _workflow_function("scan_ref")
    script = (
        _workflow_scan_setup()
        + "\nFAILED=0\n"
        + scan_ref
        + textwrap.dedent(
            """
            cd "$SCAN_REPO" || exit 99
            scan_ref HEAD enforce
            echo "FAILED=$FAILED"
            """
        )
    )
    # MSYS_NO_PATHCONV: Git Bash rewrites some `<rev>:<path>` arguments to `git show`.
    result = _bash(script, SCAN_WORK=str(work), SCAN_REPO=str(repo), MSYS_NO_PATHCONV="1")
    assert result.returncode == 0, result.stderr
    assert "scanned=2" in result.stdout, result.stdout
    assert "--- CHANGELOG.md" in result.stdout, result.stdout
    assert "--- notes.txt" not in result.stdout, result.stdout
    assert "FAILED=1" in result.stdout, result.stdout


def test_workflow_diff_scan_excludes_only_itself() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    assert source.count("PATHSPEC=(") == 1
    match = re.search(r"^[ ]*PATHSPEC=\(\n(.*?)^[ ]*\)$", source, re.M | re.S)
    assert match, "PATHSPEC block not found"
    entries = [line.strip() for line in match.group(1).splitlines() if line.strip()]
    assert entries == ["'.'", "':(exclude).github/workflows/pii_scan.yml'"], entries


def test_workflow_scan_blob_catches_a_changelog_path(tmp_path: Path) -> None:
    script = _workflow_scan_setup() + textwrap.dedent(
        """
        printf '%s\\n' "$SCAN_CANARY" > "$SCAN_WORK/CHANGELOG.md"
        scan_blob "$SCAN_WORK/CHANGELOG.md"
        """
    )
    result = _bash(script, SCAN_WORK=str(tmp_path), SCAN_CANARY=_synthetic_path())
    assert result.returncode == 0, result.stderr
    assert len(result.stdout.splitlines()) == 1, result.stdout


def test_pre_commit_hook_catches_a_staged_changelog(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, env=GIT_ENV)
    (tmp_path / "CHANGELOG.md").write_text(_synthetic_path() + "\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "CHANGELOG.md"], check=True, env=GIT_ENV)
    result = subprocess.run(
        ["bash", str(HOOK)],
        cwd=tmp_path,
        env=GIT_ENV,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1, result.stderr
    assert re.search(
        r"Hardcoded workstation path detected in staged 'CHANGELOG[.]md'", result.stderr
    ), result.stderr


def test_the_empty_exemption_loop_is_safe_on_old_bash() -> None:
    source = HOOK.read_text(encoding="utf-8")
    assert re.search(r"^PATH_SCAN_EXEMPT=\(\)$", source, re.M), "the list is expected empty"
    assert 'for allowed in ${PATH_SCAN_EXEMPT[@]+"${PATH_SCAN_EXEMPT[@]}"}; do' in source
    assert 'for allowed in "${PATH_SCAN_EXEMPT[@]}"; do' not in source
