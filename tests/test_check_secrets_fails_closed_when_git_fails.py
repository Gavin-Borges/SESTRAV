"""check_secrets.py must not pass a git checkout whose tracked files it could not list.

EXCLUDE_DIRS prunes the walk by directory NAME, and ``results`` is one of those
names, so tracked files under it are reached ONLY through ``_tracked_paths``,
which reads ``git ls-files``. That function returned ``[]`` on any failure, by
design, so that a non-git checkout scans what the walk finds. Inside a git
checkout the same ``[]`` silently dropped every tracked file under a pruned
directory. Measured 2026-10-06 against the scanner as it stood: with git exiting
128 ("dubious ownership", which is what git says on a checkout owned by another
user, as in a container), a credential in a tracked ``results/`` file passed and
the scan printed [SUCCESS].

Now a scan root that IS a git work tree (it holds a ``.git`` entry) fails closed
when git cannot list it, and a root that is not one still scans what the walk
finds, exactly as before.

git is replaced by patching the module's ``subprocess.run``, not by a PATH shim:
on Windows, Python's process launch cannot execute an extension-less script.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_secrets.py"
_DUBIOUS = "fatal: detected dubious ownership in repository"


def _load():
    spec = importlib.util.spec_from_file_location("check_secrets_fail_closed", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _git(repo: Path, *argv: str) -> None:
    subprocess.run(["git", "-C", str(repo), *argv], check=True, capture_output=True)


def _repo(tmp_path: Path, *, secret: bool) -> Path:
    """A git repo with one TRACKED file under results/, a pruned directory name."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    target = repo / "results" / "note.md"
    target.parent.mkdir(parents=True)
    if secret:
        # Mixed alphabet, length above 8, entropy above 3.0. Not a real credential.
        target.write_text("api_" + "key" + ' = "' + "a8f3k9d2m1q7x4z0b5" + '"\n', encoding="utf-8")
    else:
        target.write_text("nothing to see\n", encoding="utf-8")
    _git(repo, "add", "-f", "results/note.md")
    return repo


def _break_ls_files(mod, monkeypatch, *, missing: bool = False) -> None:
    real_run = mod.subprocess.run

    def fake_run(argv, *args, **kwargs):
        if "ls-files" in argv:
            if missing:
                raise FileNotFoundError(2, "No such file or directory", "git")
            return subprocess.CompletedProcess(argv, 128, stdout="", stderr=_DUBIOUS + "\n")
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)


def test_the_tracked_file_is_found_when_git_works(tmp_path: Path) -> None:
    """Anchor: without a fault the tracked results/ file IS scanned and flagged."""
    mod = _load()
    repo = _repo(tmp_path, secret=True)
    assert mod.scan_tree(str(repo), min_files=0) == 1


@pytest.mark.parametrize("secret", [True, False])
def test_a_git_that_cannot_list_a_work_tree_fails_the_scan(
    tmp_path: Path, monkeypatch, capsys, secret: bool
) -> None:
    mod = _load()
    repo = _repo(tmp_path, secret=secret)
    _break_ls_files(mod, monkeypatch)
    assert mod.scan_tree(str(repo), min_files=0) == 1
    out = capsys.readouterr().out
    assert "could not list the tracked files" in out
    assert "dubious ownership" in out
    assert "[SUCCESS]" not in out


def test_a_missing_git_in_a_work_tree_fails_the_scan(tmp_path: Path, monkeypatch, capsys) -> None:
    mod = _load()
    repo = _repo(tmp_path, secret=False)
    _break_ls_files(mod, monkeypatch, missing=True)
    assert mod.scan_tree(str(repo), min_files=0) == 1
    assert "could not list the tracked files" in capsys.readouterr().out


def test_tracked_paths_reports_none_for_a_work_tree_git_cannot_list(
    tmp_path: Path, monkeypatch
) -> None:
    mod = _load()
    repo = _repo(tmp_path, secret=False)
    _break_ls_files(mod, monkeypatch)
    assert mod._tracked_paths(str(repo)) is None


def test_a_root_that_is_not_a_work_tree_still_scans_what_the_walk_finds(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """The ADDITIVE design is kept where it was meant: no .git, no tracked list owed."""
    mod = _load()
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "readme.md").write_text("nothing to see\n", encoding="utf-8")
    _break_ls_files(mod, monkeypatch)
    assert mod._tracked_paths(str(plain)) == []
    assert mod.scan_tree(str(plain), min_files=0) == 0
    assert "[SUCCESS]" in capsys.readouterr().out


def test_a_later_run_is_not_blocked_by_an_earlier_runs_git_failure(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """The record is per run, like UNREADABLE_PATHS: one module, two scans."""
    mod = _load()
    repo = _repo(tmp_path, secret=False)
    _break_ls_files(mod, monkeypatch)
    assert mod.scan_tree(str(repo), min_files=0) == 1
    monkeypatch.undo()
    assert mod.scan_tree(str(repo), min_files=0) == 0
    assert capsys.readouterr().out.rstrip().endswith("[SUCCESS] No secrets detected.")
