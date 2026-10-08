"""Regression tests for pre-commit path and staged-change coverage gaps."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
import subprocess

import pytest

_HOOK = Path(__file__).resolve().parents[1] / "scripts" / "hooks" / "pre-commit"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "waveI-scratch")
    _git(repo, "config", "user.email", "waveI-scratch@invalid")
    return repo


def _run(repo: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(_HOOK)], cwd=repo, capture_output=True, text=True, check=False
    )


def _blocked(repo: Path) -> None:
    result = _run(repo)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "[BLOCKED]" in result.stderr


def test_type_change_is_scanned_for_credentials(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    target = repo / "candidate.txt"
    try:
        os.symlink("missing-target", target)
    except OSError as exc:
        # Not a skipif: the constraint is a PRIVILEGE, not a platform, so it is
        # probed rather than asserted. .claude/rules/deletion-safety-battery.md
        # records that a skip reason is a claim and not a measurement, so this
        # one carries the error it actually got.
        #
        # Staging a type change (git's T filter, which this test exists to
        # cover) requires replacing a symlink, and creating one needs
        # SeCreateSymbolicLinkPrivilege. Measured on Windows 11 without
        # Developer Mode: OSError WinError 1314. The behaviour is UNREACHABLE
        # here, not broken, and it runs normally on the CI runners and on WSL.
        pytest.skip(f"cannot create a symlink to stage a type change: {exc!r}")
    _git(repo, "add", "candidate.txt")
    _git(repo, "commit", "-qm", "Add symlink")
    target.unlink()
    value = "AK" + "IA" + "A1B2C3D4E5F6G7H8"
    target.write_text(f"value={value}\n", encoding="utf-8")
    _git(repo, "add", "candidate.txt")
    _blocked(repo)


@pytest.mark.parametrize(
    "relative_path",
    [
        "clau" + "de.md",
        "docs/" + "CLAU" + "DE.md",
        ".Clau" + "de/settings.json",
        "pkg/.clau" + "de/config.json",
    ],
)
def test_ai_config_patterns_are_case_insensitive_and_nested(
    tmp_path: Path, relative_path: str
) -> None:
    repo = _repo(tmp_path)
    target = repo / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("configuration\n", encoding="utf-8")
    _git(repo, "add", "-f", relative_path)
    _blocked(repo)


def test_lowercase_windows_profile_path_is_blocked(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    target = repo / "candidate.txt"
    path = "c:" + "/" + "users" + "/developer/private"
    target.write_text(path + "\n", encoding="utf-8")
    _git(repo, "add", "candidate.txt")
    _blocked(repo)


def test_allowlisted_home_cannot_mask_another_home_on_same_line(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    target = repo / "candidate.txt"
    allowed = "/home/" + "runner/work"
    private = "/home/" + "developer/private"
    target.write_text(f"{allowed} {private}\n", encoding="utf-8")
    _git(repo, "add", "candidate.txt")
    _blocked(repo)


def test_false_positive_guidance_names_the_owning_hook() -> None:
    text = _HOOK.read_text(encoding="utf-8")
    guidance = text.split("Secrets must not be committed", 1)[1].split("# ---- Gate 3", 1)[0]
    assert "scripts/hooks/pre-commit" in guidance
    assert "scripts/check_secrets.py" not in guidance


def _allowed(repo: Path) -> None:
    result = _run(repo)
    assert result.returncode == 0, result.stdout + result.stderr


def _staged(tmp_path: Path, payload: str) -> Path:
    repo = _repo(tmp_path)
    (repo / "candidate.txt").write_text(payload + "\n", encoding="utf-8")
    _git(repo, "add", "candidate.txt")
    return repo


# Every path below is ASSEMBLED FROM FRAGMENTS, as the cases above are. That is load-bearing
# rather than stylistic: a literal workstation path in a tracked file is exactly what this hook
# blocks, so writing one here would make this file uncommittable.
_PROFILE = "U" + "sers"
_CYGWIN = "cyg" + "drive"
_WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "pii_scan.yml"


def test_git_bash_home_path_is_blocked(tmp_path: Path) -> None:
    """The form this project's own Git Bash prints, and the one the gate used to miss.

    The bare-POSIX alternative requires a non-path character before the profile directory, and in
    a Git Bash path that character is the drive letter, so it could never fire. Measured before the
    fix, on a throwaway repository: the hook exited 0 on this payload while the drive-letter
    control exited 1.
    """
    _blocked(_staged(tmp_path, "see /c/" + _PROFILE + "/developer/private here"))


def test_cygwin_home_path_is_blocked(tmp_path: Path) -> None:
    _blocked(_staged(tmp_path, "see /" + _CYGWIN + "/c/" + _PROFILE + "/developer/private here"))


_HOME = "ho" + "me"
# WSL's Windows-side UNC share, one per branch of the hook's alternative: the wsl.localhost and
# wsl$ host names, and a home directory or a drive mount's profile root. Measured before the
# fix: the hook exited 0 on all three, and both workflow copies of the path ERE matched none.
_WSL_UNC_LEAKS = [
    "x \\\\wsl.localhost\\distro\\" + _HOME + "\\someone\\y",
    "x \\\\wsl$\\distro\\" + _HOME + "\\someone\\y",
    "x \\\\wsl.localhost\\distro\\mnt\\c\\" + _PROFILE + "\\someone\\y",
]


@pytest.mark.parametrize("payload", _WSL_UNC_LEAKS, ids=["localhost-home", "dollar-home", "mnt-profile"])
def test_a_backslash_wsl_unc_path_is_blocked(tmp_path: Path, payload: str) -> None:
    _blocked(_staged(tmp_path, payload))


def test_a_url_path_carrying_the_profile_word_is_not_blocked(tmp_path: Path) -> None:
    """The new drive-letter-shaped alternative must not fire on an ordinary URL path."""
    _allowed(_staged(tmp_path, "see https://example.org/a/" + _PROFILE + "/b here"))


def test_a_ci_home_and_a_system_path_are_not_blocked(tmp_path: Path) -> None:
    _allowed(_staged(tmp_path, "see /home/runner/work and C:/Program Files/thing here"))


def _hook_patterns() -> list[str]:
    """The hook's WORKSTATION_PATTERNS array, read from the hook rather than restated here."""
    body = _HOOK.read_text(encoding="utf-8").split("WORKSTATION_PATTERNS=(", 1)[1]
    body = body.split("\n)", 1)[0]
    patterns = []
    for line in body.splitlines():
        line = line.strip()
        if line.startswith("'"):
            patterns.append(line[1 : line.index("'", 1)])
    return patterns


def _balanced(text: str, start: int) -> str:
    """The substring from text[start] == '(' through its matching ')'."""
    assert text[start] == "("
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise AssertionError("unbalanced parentheses from index %d" % start)


def _workflow_eres() -> list[str]:
    """Both copies in pii_scan.yml: the inline diff scan and the pat_paths heredoc.

    Found by their shared opening alternative and read with a balanced-paren scan, so neither a
    quoting layer nor a trailing `|| true` can truncate what this test compares.
    """
    needle = "([A-Za-z]:[/"
    found = []
    for line in _WORKFLOW.read_text(encoding="utf-8").splitlines():
        index = line.find(needle)
        if index != -1:
            found.append(_balanced(line, index))
    assert len(found) == 2, f"expected two copies of the path ERE, found {len(found)}"
    return found


def _greps(pattern: str, text: str) -> bool:
    """grep -i -E with the pattern passed in a FILE rather than in argv.

    Measured while writing this test, and the reason it is not the obvious one-liner: handing the
    pattern to grep as an ARGUMENT makes a bracket expression ending in a backslash unreliable,
    because Python quotes backslashes for the Windows command line and that escaping does not
    survive. The hook's own first pattern ends that way, so an argv-based comparison reported the
    hook missing a form it demonstrably blocks in a real commit. The hook has no such layer, and
    the workflow's heredoc copy already uses grep -f, so -f is both the faithful and the portable
    form.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".pat", delete=False, newline="\n") as handle:
        handle.write(pattern + "\n")
        pattern_file = handle.name
    try:
        completed = subprocess.run(
            ["grep", "-q", "-i", "-E", "-f", pattern_file],
            input=text + "\n",
            text=True,
            check=False,
        )
    finally:
        os.unlink(pattern_file)
    return completed.returncode == 0


_CORPUS_POSITIVE = [
    "x C:\\" + _PROFILE + "\\someone\\y",
    "x C:/" + _PROFILE + "/someone/y",
    "x /mnt/c/" + _PROFILE + "/someone/y",
    "x /" + _PROFILE + "/someone/y",
    "x /c/" + _PROFILE + "/someone/y",
    "x /" + _CYGWIN + "/c/" + _PROFILE + "/someone/y",
    *_WSL_UNC_LEAKS,
    # The same share as it appears inside an escaped string literal, every backslash doubled.
    _WSL_UNC_LEAKS[0].replace("\\", "\\\\"),
]
_CORPUS_NEGATIVE = [
    "/home/runner/work/repo",
    "C:/Program Files/thing",
    "https://example.org/a/" + _PROFILE + "/b",
    "models/v5/rf_oof_predictions.csv",
    "/usr/share/doc",
    "C:/Windows/Temp/x",
    # A WSL share that names no user: the alternative is scoped to a home or a profile root.
    "\\\\wsl.localhost\\distro\\etc\\hosts",
    "\\\\wsl$\\distro",
    "\\\\wsl.localhost\\distro\\mnt\\c\\Windows\\Temp",
]


@pytest.mark.skipif(shutil.which("grep") is None, reason="the corpus comparison shells out to grep")
def test_the_hook_and_both_workflow_copies_agree_on_one_corpus() -> None:
    """Three copies of one rule cannot be byte-identical, so agreement is the invariant.

    The hook keeps a bash array; the workflow keeps two EREs, one inline and one in a quoted
    heredoc. Comparing their text would fail by construction, so they are compared on behaviour
    over a shared corpus. A partial revert - fixing one copy and not the others - fails here.
    """
    inline, heredoc = _workflow_eres()
    readers = {
        "hook": _hook_patterns(),
        "workflow-inline": [inline],
        "workflow-heredoc": [heredoc],
    }
    assert len(readers["hook"]) >= 5, f"the hook's pattern array shrank: {readers['hook']}"
    for payload in _CORPUS_POSITIVE:
        for name, patterns in readers.items():
            assert any(_greps(p, payload) for p in patterns), f"{name} missed {payload!r}"
    for payload in _CORPUS_NEGATIVE:
        for name, patterns in readers.items():
            assert not any(_greps(p, payload) for p in patterns), (
                f"{name} false-positive on {payload!r}"
            )


def test_the_workflow_positive_control_counts_every_scanned_form() -> None:
    """The control asserts an exact hit count, so a new form without a new canary weakens it."""
    workflow = _WORKFLOW.read_text(encoding="utf-8")
    assert "-ne 9 ]" in workflow, "the canary count no longer matches the nine canary forms"
    assert "9/9 canary forms detected" in workflow
