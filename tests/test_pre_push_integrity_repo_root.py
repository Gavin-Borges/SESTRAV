"""Pre-push must pass --repo-root so the harness cannot certify another tree.

integrity_check.py resolves REPO_ROOT from __file__. Invoking the main
checkout's copy from a worktree without --repo-root is the lying gate
git-instruments.md rule 9 describes. This test pins the hook source; the
harness itself is gitignored and is exercised only when present.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PRE_PUSH = REPO_ROOT / "scripts" / "hooks" / "pre-push"
HARNESS = REPO_ROOT / "_local" / "integrity" / "integrity_check.py"


def test_pre_push_invokes_harness_only_with_repo_root() -> None:
    text = PRE_PUSH.read_text(encoding="utf-8")
    assert 'python "${HARNESS}" --repo-root "${REPO_ROOT}"' in text
    live_lines = [
        line
        for line in text.splitlines()
        if "integrity_check.py" in line and not line.lstrip().startswith("#")
    ]
    assert live_lines, "pre-push no longer names the harness"
    for line in live_lines:
        if "HARNESS=" in line:
            continue
        assert "--repo-root" in line, line


def test_harness_repo_root_flag_requires_a_path() -> None:
    if not HARNESS.is_file():
        pytest.skip("local integrity harness is not in this tree")
    proc = subprocess.run(
        [sys.executable, str(HARNESS), "--repo-root"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "--repo-root requires a path" in proc.stderr


def test_harness_repo_root_header_names_the_passed_tree(tmp_path: Path) -> None:
    if not HARNESS.is_file():
        pytest.skip("local integrity harness is not in this tree")
    other = tmp_path / "other_tree"
    other.mkdir()
    proc = subprocess.run(
        [sys.executable, str(HARNESS), "--repo-root", str(other)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    header = f"repo: {other.resolve()}"
    assert header in proc.stdout, proc.stdout[:500]
