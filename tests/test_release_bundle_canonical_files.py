"""PR-time guard for the canonical release-bundle inputs."""

from __future__ import annotations

import subprocess
from pathlib import Path

from src.release_bundle import CANONICAL_RESULT_FILES


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_canonical_release_bundle_files_exist_and_are_tracked() -> None:
    """Every canonical bundle input must exist in and be tracked by the clone."""
    for relative_path in CANONICAL_RESULT_FILES:
        artifact = REPO_ROOT / relative_path
        assert artifact.is_file(), f"canonical release-bundle input is missing: {relative_path}"

        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative_path],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert tracked.returncode == 0, (
            f"canonical release-bundle input is not tracked: {relative_path}\n{tracked.stderr}"
        )
