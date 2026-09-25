"""Regression tests for pre-commit path and staged-change coverage gaps."""

from __future__ import annotations

import os
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
    guidance = text.split("Secrets must not be committed", 1)[1].split(
        "# ---- Gate 3", 1
    )[0]
    assert "scripts/hooks/pre-commit" in guidance
    assert "scripts/check_secrets.py" not in guidance
