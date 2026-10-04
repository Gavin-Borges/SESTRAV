"""Live-tree coverage for the required line-citation gate.

scripts/check_doc_line_citations.py backs the REQUIRED context "Cited lines still hold
their content", yet 19 of the 20 tests in tests/test_check_doc_line_citations.py build
synthetic repositories under tmp_path, and the one that does not is an idempotence
check over normalize() parametrised on three inputs, which runs no gate at all, so a
local run could not see live citation rot. This runs
the gate unpatched on the checkout itself, then drifts one pinned line in a throwaway
clone to prove the same invocation is able to fail.

The planted line is chosen from docs/line_citations.json at run time rather than named
here: re-anchoring any one citation to its symbol, which the gate itself recommends,
must not break this test.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _require_git_checkout() -> None:
    # The gate enumerates files with `git ls-files`, so a tree with no repository
    # (a `git archive` extraction) cannot run it. Measured here rather than asserted.
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
        [sys.executable, "scripts/check_doc_line_citations.py"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _a_single_line_pin(repo: Path) -> tuple[str, int]:
    baseline = json.loads((repo / "docs" / "line_citations.json").read_text(encoding="utf-8"))
    pins = [
        (entry["target"], int(entry["lines"]))
        for entry in baseline["citations"]
        if str(entry["lines"]).isdigit() and (repo / entry["target"]).is_file()
    ]
    assert pins, "docs/line_citations.json pins no single-line citation to plant against"
    return pins[0]


def test_line_citation_gate_passes_the_live_tree_and_fails_planted_drift(tmp_path):
    _require_git_checkout()
    clean = _run(REPO_ROOT)
    assert clean.returncode == 0, clean.stdout + clean.stderr

    repo = tmp_path / "repo"
    _git(tmp_path, "clone", "--quiet", "--no-hardlinks", str(REPO_ROOT), str(repo))
    target, line = _a_single_line_pin(repo)
    raw = (repo / target).read_bytes().split(b"\n")
    eol = b"\r" if raw[line - 1].endswith(b"\r") else b""
    raw[line - 1] = b"PLANTED_DRIFT = True" + eol
    (repo / target).write_bytes(b"\n".join(raw))
    _git(repo, "add", target)
    _git(
        repo,
        "-c",
        "user.name=planted",
        "-c",
        "user.email=planted@invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "--no-verify",
        "-qm",
        "plant citation drift",
    )
    planted = _run(repo)
    assert planted.returncode == 1, planted.stdout + planted.stderr
    assert "DRIFTED" in planted.stdout
    assert f"{target}:{line}" in planted.stdout
