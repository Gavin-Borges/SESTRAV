"""Live-tree coverage for the secret-scanning gate.

.github/workflows/security.yml and scripts/hooks/pre-push both run
scripts/check_secrets.py, but tests/test_check_secrets.py scans only fixture trees; its
live-root tests enumerate files without scanning them. This runs the full gate
unpatched on the checkout itself, then stages a credential-shaped assignment in a
throwaway clone to prove the same invocation is able to fail.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _require_git_checkout() -> None:
    # The gate unions the filesystem walk with `git ls-files`, so a tree with no
    # repository (a `git archive` extraction) is not what CI or pre-push scans.
    try:
        probe = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        pytest.skip(f"git is not runnable here: {exc}")
    if probe.returncode != 0 or Path(probe.stdout.strip()).resolve() != REPO_ROOT:
        pytest.skip(
            f"not a git checkout of this repository (git rev-parse exited {probe.returncode})"
        )


def _run(repo: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/check_secrets.py"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )


def test_secret_gate_passes_the_live_tree_and_fails_a_planted_secret(tmp_path):
    _require_git_checkout()
    clean = _run(REPO_ROOT)
    assert clean.returncode == 0, clean.stdout + clean.stderr

    repo = tmp_path / "repo"
    subprocess.run(
        ["git", "clone", "--quiet", "--no-hardlinks", str(REPO_ROOT), str(repo)],
        check=True,
        capture_output=True,
    )
    probe = repo / "planted_secret_probe.py"
    # Assembled so this file does not itself carry the credential shape.
    assignment = "API" + "_KEY = " + repr("planted_super_secret_value_0123456789") + "\n"
    probe.write_text(assignment, encoding="utf-8")
    subprocess.run(["git", "add", probe.name], cwd=repo, check=True, capture_output=True)
    planted = _run(repo)
    assert planted.returncode == 1, planted.stdout + planted.stderr
    assert "[FLAGGED]" in planted.stdout
    assert probe.name in planted.stdout
    assert "planted_super_secret_value" not in planted.stdout
