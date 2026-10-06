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


# ---------------------------------------------------------------------------------------------
# A NUL byte, or a binary blob, used to blank these scans entirely. Both were reproduced on GNU
# grep 3.0 before the fix: one NUL anywhere in diff.txt took the hit count for an unrelated
# leaked path from 1 to 0 and the step printed PASSED, and a path hidden inside a binary blob
# scored 0 hits under -I against 1 without it. A scanner that reports clean because it stopped
# looking is the failure mode this file exists for.
# ---------------------------------------------------------------------------------------------


def _path_ere() -> str:
    """The workflow's own path ERE, read from its quoted heredoc rather than restated here."""
    source = WORKFLOW.read_text(encoding="utf-8")
    marker = 'pat_paths.txt" <<'
    assert source.count(marker) == 1, marker
    after = source.split(marker, 1)[1]
    for line in after.splitlines()[1:]:
        stripped = line.strip()
        if stripped and not stripped.startswith("PATEOF"):
            return stripped
    raise AssertionError("the pat_paths heredoc has no pattern line")


def _diff_scan_greps() -> list[str]:
    """Every line in the workflow that greps diff.txt."""
    return [
        line.strip()
        for line in WORKFLOW.read_text(encoding="utf-8").splitlines()
        if "diff.txt" in line and line.strip().startswith(("USER_LEAKS=", "HOME_LEAKS="))
    ]


def test_every_diff_scan_grep_reads_binary_as_text() -> None:
    """-a on each grep of diff.txt, or one NUL in the diff silently empties the whole scan."""
    greps = _diff_scan_greps()
    assert len(greps) == 2, f"expected the two leak scans over diff.txt, found {greps}"
    for line in greps:
        assert " -a " in line, f"this grep of diff.txt does not pass -a: {line}"


def test_a_nul_byte_would_blank_the_added_line_scan_without_dash_a(tmp_path: Path) -> None:
    """The mechanism, with the workflow's own pattern, so -a is shown to be load-bearing.

    Without -a the leak below is reported 0 times even though it sits in a different file of the
    same diff; with -a it is reported once. If a future grep stops needing the flag this test
    fails and the comment in the workflow can go with it.

    The canary is the Git Bash form on purpose, and that was measured rather than chosen. Whether
    the no--a scan collapses depends on the content: this layout with a Git Bash path reproduces
    the blanking on GNU grep 3.0 across three variants (plain or added-line filler, pattern given
    as an argument or in a file), while the same layout carrying a backslash drive-letter path does
    NOT. The fix does not rest on that detail - -a makes grep treat every input as text, so no
    content can suppress output - but a test asserting the negative half has to use a layout that
    actually exhibits it. The form also carries no backslash, so no quoting layer can mangle it.
    """
    leak = "+see /c/" + "U" + "sers/someone/private here"
    diff = tmp_path / "diff.txt"
    with diff.open("wb") as handle:
        handle.write(b"diff --git a/x b/x\n")
        handle.write(leak.encode("utf-8") + b"\n")
        handle.write(b"diff --git a/big b/big\n")
        handle.write(b"+" + b"A" * 8100 + b"\n")
        handle.write(b"+binary\x00payload\n")
    pattern = tmp_path / "pat.txt"
    pattern.write_text(_path_ere() + "\n", encoding="utf-8", newline="\n")

    def hits(extra: str) -> int:
        # The diff is fed on STDIN, not named as an argument, and that is measured rather than
        # stylistic. Without -a grep prints "Binary file <path> matches" in place of the lines, and
        # when the file is named, <path> is the temp directory - which under pytest on Windows lives
        # beneath the very profile directory these patterns hunt. The full ERE then matches grep's
        # OWN diagnostic and the scan reports 1, for a reason that has nothing to do with the diff.
        # On stdin the message carries no path, so the count reflects the diff alone.
        script = (
            f'grep {extra} -E "^\\+[^+]" < "{diff.as_posix()}" '
            f'| grep -c -i -E -f "{pattern.as_posix()}" || true'
        )
        return int((_bash(script).stdout or "0").strip() or 0)

    assert hits("") == 0, "premise: without -a the NUL hides the leak"
    assert hits("-a") == 1, "with -a the leak in the other file is still found"


def test_scan_blob_finds_a_path_hidden_inside_a_binary_blob(tmp_path: Path) -> None:
    """Drives the workflow's REAL scan_blob, so the -I to -a change is proved where it is used."""
    work = tmp_path / "work"
    work.mkdir()
    blob = tmp_path / "blob.bin"
    with blob.open("wb") as handle:
        handle.write(b"PK\x03\x04 junk ")
        handle.write(("see " + _synthetic_path() + " here").encode("utf-8"))
        handle.write(b" \x00 more junk\n")
    script = _workflow_scan_setup() + f'\nscan_blob "{blob.as_posix()}"\n'
    result = _bash(script, SCAN_WORK=str(work))
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip(), "scan_blob reported nothing for a path inside a binary blob"


def test_scan_blob_keeps_dash_capital_i_on_the_looser_home_pattern() -> None:
    """The asymmetry is deliberate and worth pinning.

    The PATH patterns are specific enough to read binary bytes; /home/<name> is not, so a chance
    byte run in a binary added later would block a push on noise. Measured when this landed: both
    patterns gave 0 hits with -a over every tracked binary, which bounds the tree of that day and
    not the next binary added to it.
    """
    setup = _workflow_function("scan_blob")
    path_line = next(line for line in setup.splitlines() if "pat_paths.txt" in line)
    home_line = next(line for line in setup.splitlines() if "pat_home.txt" in line)
    assert " -a " in path_line and " -I " not in path_line, path_line
    assert " -I " in home_line and " -a " not in home_line, home_line
