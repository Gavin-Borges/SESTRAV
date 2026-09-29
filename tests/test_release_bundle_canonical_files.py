"""PR-time guard for the canonical release-bundle inputs."""

from __future__ import annotations

import subprocess
from pathlib import Path

from src.release_bundle import CANONICAL_RESULT_FILES


REPO_ROOT = Path(__file__).resolve().parents[1]

# A floor for the anti-vacuity guard below, matching the idiom in
# tests/test_dev_extra_runs_the_test_suite.py. CANONICAL_RESULT_FILES held seven
# paths when this guard was written; the floor is the same seven rather than a
# margin, because this list is a fixed release contract that grows by deliberate
# edit, not by churn.
MINIMUM_CANONICAL_FILES = 7


def test_the_canonical_list_is_not_empty() -> None:
    """Anti-vacuity guard, and it is not hypothetical.

    The companion test below is a bare `for` over CANONICAL_RESULT_FILES with no
    count assertion, so emptying or truncating that list makes it iterate zero
    times and PASS GREEN while checking nothing. The failure mode this whole
    file exists to catch is a canonical artifact going missing; a guard that
    reports safety it never checked is worse than no guard.
    """
    assert len(CANONICAL_RESULT_FILES) >= MINIMUM_CANONICAL_FILES, (
        f"expected at least {MINIMUM_CANONICAL_FILES} canonical release-bundle "
        f"inputs, found {len(CANONICAL_RESULT_FILES)}; the list has shrunk and "
        "the companion test is therefore vacuous"
    )


def test_canonical_release_bundle_files_exist_and_are_tracked() -> None:
    """Every canonical bundle input must exist in and be tracked by the clone."""
    checked = 0
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
        checked += 1

    # The loop above is the only thing that can fail. Assert it actually ran, so
    # this test cannot pass by iterating nothing.
    assert checked >= MINIMUM_CANONICAL_FILES, (
        f"only {checked} canonical input(s) were checked; expected at least "
        f"{MINIMUM_CANONICAL_FILES}"
    )
