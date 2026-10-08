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
    """Every grep in the diff scan, from the first read of diff.txt to the LEAKS join.

    Comment lines are dropped first, so a grep that is only quoted in prose is not counted.
    """
    block = _workflow_block("          USER_LEAKS=$(grep", '          if [ -n "$LEAKS" ]; then')
    code = "\n".join(line for line in block.splitlines() if not line.lstrip().startswith("#"))
    return re.findall(r"\bgrep\b[^|]*", code)


def test_every_diff_scan_grep_reads_binary_as_text() -> None:
    """-a on EVERY grep of the scan, not only the two that read diff.txt itself.

    The first grep passes NUL-bearing and non-UTF-8 lines through unchanged, so each later grep
    reads the same bytes; without -a, GNU grep 3.12 prints nothing on stdout for them. An earlier
    form of this test checked only the lines starting USER_LEAKS= and HOME_LEAKS=, which are the
    first grep of each pipeline, and passed while the second grep of both dropped every leak in
    any pull request that also added a binary file."""
    greps = _diff_scan_greps()
    assert len(greps) == 6, f"expected 2 + 3 + 1 greps in the scan, found {len(greps)}: {greps}"
    for grep in greps:
        assert re.match(r"grep -a\b", grep), f"this grep in the diff scan does not pass -a: {grep}"


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


@pytest.mark.parametrize("carrier", ["a binary file beside the leak", "a Latin-1 byte on the leak line"])
def test_the_shipped_diff_scan_reports_every_leak_despite_binary_bytes(
    tmp_path: Path, carrier: str
) -> None:
    """The workflow's REAL diff scan, run on a pull request, must block AND name both leaks.

    The test above restates the pipeline with `grep -c` as its last stage, and a count survives
    binary input, so it could not see the second grep of each pipeline dropping its lines. This
    one runs the step's own text, from PATHSPEC to the verdict. Both carriers were measured on
    GNU grep 3.12 with -a on the first grep only: the first PASSED the pull request, the second
    reported 1 of the 2 leaks. It asserts the leaks by CONTENT, because GNU grep 3.0 (Git Bash)
    prints its 'Binary file (standard input) matches' notice on stdout, so a non-empty LEAKS
    alone would pass there on the notice rather than on a path. The canaries are assembled at
    runtime, so no literal workstation path sits in this file."""
    import subprocess as sp

    user_leak = "X:/" + "U" + "sers/fakeuser123/project"
    home_leak = "/ho" + "me/fakeuser123"
    repo = tmp_path / "r"
    repo.mkdir()

    def git(*argv: str) -> sp.CompletedProcess:
        return sp.run(
            ["git", "-C", str(repo), "-c", "user.email=t@example.com",
             "-c", "user.name=T", "-c", "commit.gpgsign=false", *argv],
            capture_output=True, text=True, check=False, env=GIT_ENV,
        )

    git("init", "-q")
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-q", "-m", "seed")
    base = git("rev-parse", "HEAD").stdout.strip()
    assert base, "the fixture repo has no base commit"
    git("update-ref", "refs/remotes/origin/main", base)

    tail = b'\xe9"\n' if carrier.startswith("a Latin-1") else b'"\n'
    (repo / "leak.py").write_bytes(
        b'p = "' + user_leak.encode() + tail + b'q = "' + home_leak.encode() + b'/project' + tail
    )
    if carrier.startswith("a binary"):
        (repo / "fig.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00")
    git("add", "-A")
    git("commit", "-q", "-m", "the pull request")

    step = _workflow_block("          PATHSPEC=(", '          echo "Path check PASSED."')
    result = sp.run(
        ["bash", "-c", step],
        cwd=repo,
        env={**GIT_ENV, "EVENT_NAME": "pull_request", "BASE_REF": "main", "LC_ALL": "C.UTF-8"},
        capture_output=True,
        check=False,
    )
    out = result.stdout.decode("utf-8", errors="replace")
    assert result.returncode == 1, f"the scan did not block ({carrier}): {out!r}"
    assert user_leak in out, f"the user-profile leak was not reported ({carrier}): {out!r}"
    assert home_leak in out, f"the /home leak was not reported ({carrier}): {out!r}"


def _shipped_diff_scan(tmp_path: Path, files: dict[str, bytes]) -> tuple[int, str]:
    """The workflow's REAL diff scan on a pull request that adds `files`: (exit code, stdout).

    The harness of the test above: a throwaway repo whose base is refs/remotes/origin/main, the
    step's own text from PATHSPEC to the verdict, and the environment CI gives it."""
    import subprocess as sp

    repo = tmp_path / "r"
    repo.mkdir()

    def git(*argv: str) -> sp.CompletedProcess:
        return sp.run(
            ["git", "-C", str(repo), "-c", "user.email=t@example.com",
             "-c", "user.name=T", "-c", "commit.gpgsign=false", *argv],
            capture_output=True, text=True, check=False, env=GIT_ENV,
        )

    git("init", "-q")
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-q", "-m", "seed")
    base = git("rev-parse", "HEAD").stdout.strip()
    assert base, "the fixture repo has no base commit"
    git("update-ref", "refs/remotes/origin/main", base)
    for name, data in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_bytes(data)
    git("add", "-A")
    git("commit", "-q", "-m", "the pull request")
    assert git("rev-parse", "HEAD").stdout.strip() != base, "the pull request commit failed"

    step = _workflow_block("          PATHSPEC=(", '          echo "Path check PASSED."')
    result = sp.run(
        ["bash", "-c", step],
        cwd=repo,
        env={**GIT_ENV, "EVENT_NAME": "pull_request", "BASE_REF": "main", "LC_ALL": "C.UTF-8"},
        capture_output=True,
        check=False,
    )
    return result.returncode, result.stdout.decode("utf-8", errors="replace")


_PLUS_USER_LEAK = "X:/" + "U" + "sers/fakeuser123/project"
_PLUS_HOME_LEAK = "/ho" + "me/fakeuser123"


@pytest.mark.parametrize(
    ("content", "leak"),
    [
        # Reaches the diff as '++X:/...'.
        ("+" + _PLUS_USER_LEAK, _PLUS_USER_LEAK),
        # Reaches the diff as '+++ b/...', byte for byte a file header.
        ("++ b" + _PLUS_HOME_LEAK + "/x", _PLUS_HOME_LEAK),
        # The control: an ordinary added leak, which the old filter did catch.
        (_PLUS_USER_LEAK, _PLUS_USER_LEAK),
    ],
    ids=["plus-led", "header-shaped", "ordinary"],
)
def test_the_shipped_diff_scan_reads_an_added_line_whatever_it_starts_with(
    tmp_path: Path, content: str, leak: str
) -> None:
    """An added line whose CONTENT starts with '+' is still an added line.

    Measured with this test before the fix, on both platforms: the step exited 0 and reported
    nothing for the first two cases, because the old filter '^\\+[^+]' required the character
    after git's '+' indicator to be something other than '+'. The control blocked."""
    rc, out = _shipped_diff_scan(tmp_path, {"leak.txt": (content + "\n").encode("utf-8")})
    assert rc == 1, f"the scan did not block: {out!r}"
    assert leak in out, f"the leak was not reported: {out!r}"


def test_the_shipped_diff_scan_does_not_read_the_file_header_as_content(tmp_path: Path) -> None:
    """The '+++ b/<path>' header is attribution, not added content, so it is not scanned.

    The file sits under a directory shaped like a home path, so its header carries one, while
    its content is clean. A filter that took every line starting with '+' would block this pull
    request on its file name."""
    rc, out = _shipped_diff_scan(tmp_path, {"ho" + "me/fakeuser123/notes.txt": b"nothing here\n"})
    assert rc == 0, f"the scan blocked a clean file on its header: {out!r}"
    assert "fakeuser123" not in out, out


def _wsl_unc_leaks() -> list[str]:
    """WSL's Windows-side UNC share, one per branch of the path ERE's last alternative.

    Assembled at runtime, like every canary in this file, and spelled with real backslashes, so
    the YAML, bash single-quote and heredoc layers each have to carry the ERE's bracket
    expressions intact for these to be found."""
    bs = chr(92)
    share = bs * 2 + "wsl"
    home = "ho" + "me"
    profile = "U" + "sers"
    return [
        f"{share}.localhost{bs}distro{bs}{home}{bs}fakeuser123{bs}project",
        f"{share}${bs}distro{bs}{home}{bs}fakeuser123{bs}project",
        f"{share}.localhost{bs}distro{bs}mnt{bs}c{bs}{profile}{bs}fakeuser123{bs}project",
    ]


def test_the_shipped_diff_scan_reports_a_backslash_wsl_unc_path(tmp_path: Path) -> None:
    """The INLINE copy of the ERE, run as the step runs it on a pull request.

    Measured before the fix: the step exited 0 and reported none of the three."""
    import subprocess as sp

    leaks = _wsl_unc_leaks()
    repo = tmp_path / "r"
    repo.mkdir()

    def git(*argv: str) -> sp.CompletedProcess:
        return sp.run(
            ["git", "-C", str(repo), "-c", "user.email=t@example.com",
             "-c", "user.name=T", "-c", "commit.gpgsign=false", *argv],
            capture_output=True, text=True, check=False, env=GIT_ENV,
        )

    git("init", "-q")
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-q", "-m", "seed")
    base = git("rev-parse", "HEAD").stdout.strip()
    assert base, "the fixture repo has no base commit"
    git("update-ref", "refs/remotes/origin/main", base)
    (repo / "leak.py").write_bytes(b"".join(b'p = r"' + leak.encode() + b'"\n' for leak in leaks))
    git("add", "-A")
    git("commit", "-q", "-m", "the pull request")

    step = _workflow_block("          PATHSPEC=(", '          echo "Path check PASSED."')
    result = sp.run(
        ["bash", "-c", step],
        cwd=repo,
        env={**GIT_ENV, "EVENT_NAME": "pull_request", "BASE_REF": "main", "LC_ALL": "C.UTF-8"},
        capture_output=True,
        check=False,
    )
    out = result.stdout.decode("utf-8", errors="replace")
    assert result.returncode == 1, f"the scan did not block: {out!r}"
    for leak in leaks:
        assert leak in out, f"this WSL UNC leak was not reported: {leak!r} in {out!r}"


def test_the_shipped_positive_control_detects_every_canary(tmp_path: Path) -> None:
    """The HEREDOC copy, through the real scan_blob and the control's own canaries and count.

    The control is the one place a bash quoting slip in a canary would surface, and before this
    test it surfaced only on CI, as a failed job."""
    block = _workflow_block('          WORK="$(mktemp -d)"', "          # A second control exercises")
    script = block.replace('WORK="$(mktemp -d)"', 'WORK="$SCAN_WORK"', 1)
    result = _bash(script, SCAN_WORK=str(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Positive control PASSED: 9/9 canary forms detected." in result.stdout, result.stdout


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
    # NOT a non-empty check. That is how this test passed for the WRONG REASON, and the
    # measurement is the whole point: with no -a on scan_blob's final stage, GNU grep 3.0
    # (Git Bash here) prints "Binary file (standard input) matches" to STDOUT, so a
    # non-empty assertion passes on grep's NOTICE while the path is silently dropped. GNU
    # grep 3.12 (Linux, which is what CI runs) puts that notice on STDERR instead, leaving
    # stdout EMPTY, so the same code fails there. rc is 0 in both cases, so the exit status
    # gives no signal either. Assert the PATH, and forbid the notice.
    assert "fakeuser123" in result.stdout, (
        "scan_blob did not report the matched PATH for a path hidden inside a binary blob. "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "Binary file" not in result.stdout, (
        "scan_blob emitted grep's binary-file NOTICE instead of the matched line, which is "
        "exactly what a missing -a on its final stage produces. "
        f"stdout={result.stdout!r}"
    )


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


def _diff_constructions() -> list[str]:
    """Every line in the workflow that BUILDS diff.txt with git diff."""
    return [
        line.strip()
        for line in WORKFLOW.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("git diff") and "> diff.txt" in line
    ]


def test_every_diff_construction_reads_binary_as_text() -> None:
    """--text on each git diff that builds diff.txt.

    This is the OTHER HALF of the `-a` requirement its sibling test above pins, and
    neither flag substitutes for the other. `-a` makes grep read a NUL-bearing
    diff.txt; --text is what puts the content INTO diff.txt in the first place. Without
    it git writes "Binary files ... differ" and no content at all, so there is nothing
    for any grep to read, flagged or not."""
    builds = _diff_constructions()
    assert len(builds) == 2, f"expected the PR and push diff builds, found {builds}"
    for line in builds:
        assert "--text" in line, f"this git diff does not pass --text: {line}"


def test_a_minus_diff_attribute_or_a_nul_hides_a_leak_without_text(tmp_path: Path) -> None:
    """The mechanism, measured per carrier so one cannot mask the other.

    Two carriers reach the same blind spot, and the FIRST is attacker-selectable: a pull
    request can add `nodiff.txt -diff` to .gitattributes in the same commit that adds
    the leak, so the file stays ordinary text a reviewer can read while this gate is
    told to treat it as binary. The second is an ordinary binary file holding a NUL.

    Without --text each carrier yields ZERO visible leak lines; with it, one each. The
    canary is assembled at runtime and uses the Git Bash home form, so no literal
    workstation path sits in this file."""
    import subprocess as sp

    users = "U" + "sers"
    leak = "see /c/" + users + "/someone/private/key here"
    repo = tmp_path / "r"
    repo.mkdir()

    def git(*argv: str) -> sp.CompletedProcess:
        return sp.run(
            ["git", "-C", str(repo), "-c", "user.email=t@example.com",
             "-c", "user.name=T", "-c", "commit.gpgsign=false", *argv],
            capture_output=True, text=True, check=False,
        )

    git("init", "-q")
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-q", "-m", "seed")
    base = git("rev-parse", "HEAD").stdout.strip()
    assert base, "the fixture repo has no base commit"

    (repo / ".gitattributes").write_text("nodiff.txt -diff\n", encoding="utf-8")
    (repo / "nodiff.txt").write_text(leak + "\n", encoding="utf-8")
    (repo / "blob.bin").write_bytes(b"\x00\x01" + leak.encode() + b"\n\x00")
    git("add", "-A")
    git("commit", "-q", "-m", "add both carriers")

    def visible(carrier: str, *extra: str) -> int:
        out = git("diff", *extra, base + "...HEAD", "--", carrier).stdout
        return sum(1 for line in out.splitlines() if "private/key" in line)

    for carrier in ("nodiff.txt", "blob.bin"):
        assert visible(carrier) == 0, (
            f"{carrier} leaked content without --text, so this fixture no longer "
            f"reproduces the blind spot and the test proves nothing"
        )
        assert visible(carrier, "--text") == 1, (
            f"{carrier} is still invisible WITH --text, so the fix does not work"
        )
def test_every_diff_construction_ignores_textconv_and_external_diff() -> None:
    """`--text` is not sufficient on its own, so all three flags are required together.

    textconv and GIT_EXTERNAL_DIFF each replace the diff BODY wholesale, so `--text`
    still produces a diff.txt with nothing in it for the scan to read. Asserted per
    construction rather than once, because the pull_request and push branches are
    separate commands and a fix applied to one is the shape this file keeps catching."""
    builds = _diff_constructions()
    assert len(builds) == 2, f"expected the PR and push diff builds, found {builds}"
    for line in builds:
        for flag in ("--text", "--no-textconv", "--no-ext-diff"):
            assert flag in line, f"this git diff does not pass {flag}: {line}"


def _leak_canary() -> str:
    # Assembled at runtime; the Git Bash home form, so no literal workstation path sits
    # in this file for the very gate under test to refuse.
    return "see /c/" + ("U" + "sers") + "/someone/private/key here"


def _repo_with_leak(where: Path) -> tuple[str, str]:
    """A two-commit repo whose second commit adds the canary. Returns (base, leak)."""
    import subprocess as sp

    where.mkdir(parents=True, exist_ok=True)

    def git(*argv: str) -> sp.CompletedProcess:
        return sp.run(
            ["git", "-C", str(where), "-c", "user.email=t@example.com",
             "-c", "user.name=T", "-c", "commit.gpgsign=false", *argv],
            capture_output=True, text=True, check=False,
        )

    git("init", "-q")
    (where / "seed.txt").write_text("seed\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-q", "-m", "seed")
    base = git("rev-parse", "HEAD").stdout.strip()
    assert base, "fixture repo has no base commit"
    leak = _leak_canary()
    (where / "leak.txt").write_text(leak + "\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-q", "-m", "add the canary")
    return base, leak


def test_textconv_defeats_text_and_no_textconv_restores_it(tmp_path: Path) -> None:
    """Measured, in a repo where ONLY textconv is configured.

    The isolation is deliberate and was learned the hard way: a first probe tested
    GIT_EXTERNAL_DIFF in a repo that still had textconv configured, so `--no-ext-diff`
    read as STILL BLIND and nearly shipped as "that flag does not help". Two mechanisms
    that blank the same output cannot share one fixture, because the second reading is
    the first mechanism's shadow."""
    import subprocess as sp

    repo = tmp_path / "textconv"
    base, _ = _repo_with_leak(repo)
    scrub = repo / "scrub.sh"
    scrub.write_text("#!/bin/sh\necho SCRUBBED\n", encoding="utf-8", newline="\n")

    def git(*argv: str) -> sp.CompletedProcess:
        return sp.run(["git", "-C", str(repo), *argv], capture_output=True, text=True,
                      encoding="utf-8", errors="replace", check=False)

    (repo / ".gitattributes").write_text("leak.txt diff=scrub\n", encoding="utf-8")
    git("add", "-A")
    git("-c", "user.email=t@e.com", "-c", "user.name=T", "-c", "commit.gpgsign=false",
        "commit", "-q", "-m", "name a driver")
    # The driver BODY lives in config, which a pull request cannot commit.
    git("config", "diff.scrub.textconv", "sh " + scrub.as_posix())

    def visible(*extra: str) -> int:
        out = git("diff", *extra, base + "...HEAD", "--", "leak.txt").stdout
        return sum(1 for line in out.splitlines() if "private/key" in line)

    assert visible("--text") == 0, (
        "textconv no longer blanks the diff, so this fixture no longer reproduces the "
        "mechanism and the assertion below would pass for the wrong reason"
    )
    assert visible("--text", "--no-textconv") == 1


def test_external_diff_defeats_text_and_no_ext_diff_restores_it(tmp_path: Path) -> None:
    """Measured in a repo with NO textconv configured, for the reason above."""
    import os
    import subprocess as sp

    repo = tmp_path / "extdiff"
    base, _ = _repo_with_leak(repo)
    ext = repo / "ext.sh"
    ext.write_text("#!/bin/sh\necho EXTERNAL\n", encoding="utf-8", newline="\n")
    env = dict(os.environ)
    env["GIT_EXTERNAL_DIFF"] = "sh " + ext.as_posix()

    def visible(*extra: str) -> int:
        out = sp.run(["git", "-C", str(repo), "diff", *extra, base + "...HEAD",
                      "--", "leak.txt"], capture_output=True, text=True,
                     encoding="utf-8", errors="replace", env=env, check=False).stdout
        return sum(1 for line in out.splitlines() if "private/key" in line)

    assert visible("--text") == 0, (
        "GIT_EXTERNAL_DIFF no longer blanks the diff, so this fixture no longer "
        "reproduces the mechanism"
    )
    assert visible("--text", "--no-ext-diff") == 1


def test_a_committed_attribute_naming_an_undefined_driver_is_inert(tmp_path: Path) -> None:
    """Why this unit is HARDENING and not a live PR-reachable gap.

    A pull request can commit `.gitattributes`, but the driver body lives in
    `diff.<driver>.textconv` in `.git/config`, which is never committed. With the
    attribute present and the driver undefined, the leak is still visible, so a
    contributor cannot reach this blind spot. These flags defend against a hostile
    runner config or a future workflow step that sets one."""
    import subprocess as sp

    repo = tmp_path / "inert"
    base, _ = _repo_with_leak(repo)
    (repo / ".gitattributes").write_text("leak.txt diff=nowhere\n", encoding="utf-8")
    sp.run(["git", "-C", str(repo), "add", "-A"], capture_output=True, check=False)
    sp.run(["git", "-C", str(repo), "-c", "user.email=t@e.com", "-c", "user.name=T",
            "-c", "commit.gpgsign=false", "commit", "-q", "-m", "undefined driver"],
           capture_output=True, check=False)
    out = sp.run(["git", "-C", str(repo), "diff", "--text", base + "...HEAD", "--",
                  "leak.txt"], capture_output=True, text=True, encoding="utf-8",
                 errors="replace", check=False).stdout
    assert sum(1 for line in out.splitlines() if "private/key" in line) == 1
