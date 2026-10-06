"""check_lockfile_freshness.py must not fall back to a pruned walk inside a git checkout.

Check 4 discovers tracked ``requirements*.in`` files through ``git ls-files`` and
fails on any that LOCKFILE_PAIRS does not map. When git could not answer,
``_git_tracked_in_files`` returned None and discovery fell back to a filesystem
walk that skips ``results``, ``_local`` and ``.claude`` among others, so a tracked
``.in`` file under one of those names went unchecked and the gate still passed.
The fallback exists for a non-git context (an unpacked sdist, a vendored copy),
and there it is kept. Inside a git checkout, a git that cannot list the tracked
files now fails the gate, naming why.

git is replaced by patching the module's ``subprocess.run``: on Windows, Python's
process launch cannot execute an extension-less shim script.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tools import check_lockfile_freshness as freshness

_DUBIOUS = "fatal: detected dubious ownership in repository"


def _break_ls_files(monkeypatch, *, missing: bool = False) -> None:
    real_run = freshness.subprocess.run

    def fake_run(argv, *args, **kwargs):
        if "ls-files" in argv:
            if missing:
                raise FileNotFoundError(2, "No such file or directory", "git")
            return subprocess.CompletedProcess(argv, 128, stdout="", stderr=_DUBIOUS + "\n")
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(freshness.subprocess, "run", fake_run)


def _git_repo_with_hidden_in_file(tmp_path: Path) -> Path:
    """A git repo whose only requirements*.in is TRACKED under results/, a name the walk skips."""
    repo = tmp_path / "repo"
    (repo / "results").mkdir(parents=True)
    (repo / "results" / "requirements-extra.in").write_text("biopython==1.87\n", encoding="utf-8")
    for argv in (["init", "-q"], ["add", "-f", "results/requirements-extra.in"]):
        subprocess.run(["git", "-C", str(repo), *argv], check=True, capture_output=True)
    return repo


def test_the_hidden_file_is_found_when_git_works(tmp_path: Path, monkeypatch) -> None:
    """Anchor: with a working git the tracked results/ file IS discovered as unmapped."""
    monkeypatch.setattr(freshness, "REPO_ROOT", _git_repo_with_hidden_in_file(tmp_path))
    assert freshness.discover_unmapped_in_files(set()) == ["results/requirements-extra.in"]


@pytest.mark.parametrize("missing", [False, True])
def test_a_git_that_cannot_list_a_work_tree_raises(
    tmp_path: Path, monkeypatch, missing: bool
) -> None:
    monkeypatch.setattr(freshness, "REPO_ROOT", _git_repo_with_hidden_in_file(tmp_path))
    _break_ls_files(monkeypatch, missing=missing)
    with pytest.raises(freshness.GitListingError) as excinfo:
        freshness.discover_unmapped_in_files(set())
    assert (
        ("exited 128" in str(excinfo.value))
        if not missing
        else ("could not be run" in str(excinfo.value))
    )


def test_a_root_that_is_not_a_work_tree_still_walks(tmp_path: Path, monkeypatch) -> None:
    """The documented fallback is kept where it was meant: no .git entry, no listing owed."""
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "requirements.in").write_text("biopython==1.87\n", encoding="utf-8")
    monkeypatch.setattr(freshness, "REPO_ROOT", plain)
    _break_ls_files(monkeypatch)
    assert freshness._git_tracked_in_files() is None
    assert freshness.discover_unmapped_in_files(set()) == ["requirements.in"]


def test_main_fails_the_real_repository_when_git_cannot_list_it(monkeypatch, capsys) -> None:
    """End to end on this checkout: every pair is fresh, so only the listing can fail it."""
    _break_ls_files(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["check_lockfile_freshness.py", "--check"])
    assert freshness.main() == 1
    out = capsys.readouterr().out
    assert "could not list the tracked requirements*.in files" in out
    assert _DUBIOUS in out
