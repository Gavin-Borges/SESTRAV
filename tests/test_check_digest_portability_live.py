"""Live-tree coverage for the digest-portability gate.

.github/workflows/digest_portability.yml runs scripts/check_digest_portability.py,
but tests/test_check_digest_portability.py drives helpers or synthetic inputs, and the
only one of its tests that reads the live repository THROUGH scan_repository,
test_the_doi_manifest_is_in_scope_and_every_digest_resolves, asserts scope and pairing
only and never that a digest still reproduces, so the suite could not see a tracked
digest that stopped reproducing. Two of the twelve read live tracked content: the other
reads digest_portability.yml to check STRICT_FAILING against it, which is the workflow's
contract rather than the tree's digests. This runs the gate unpatched on the checkout
itself, following
tests/test_check_doc_commit_refs.py::test_gate_passes_on_the_live_tree, then plants a
mismatch in a throwaway clone to prove the same invocation is able to fail.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _require_git_checkout() -> None:
    # The gate reads git blobs, so a tree with no repository (a `git archive`
    # extraction) cannot run it. Measured here rather than asserted.
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
        [sys.executable, "scripts/check_digest_portability.py", "--strict"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def test_digest_gate_passes_the_live_tree_and_fails_a_planted_mismatch(tmp_path):
    _require_git_checkout()
    clean = _run(REPO_ROOT)
    assert clean.returncode == 0, clean.stdout + clean.stderr

    repo = tmp_path / "repo"
    _git(tmp_path, "clone", "--quiet", "--no-hardlinks", str(REPO_ROOT), str(repo))
    (repo / "planted_artifact.txt").write_text("actual\n", encoding="utf-8")
    (repo / "planted.provenance.json").write_text(
        json.dumps({"artifact": "planted_artifact.txt", "sha256": "0" * 64}), encoding="utf-8"
    )
    # The gate compares HEAD blobs, so the plant has to be committed. Signing and
    # hooks are switched off so a developer's global git config cannot interfere.
    _git(repo, "add", "planted_artifact.txt", "planted.provenance.json")
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
        "plant digest mismatch",
    )
    planted = _run(repo)
    assert planted.returncode == 1, planted.stdout + planted.stderr
    assert "MISMATCH" in planted.stdout
    assert "planted_artifact.txt" in planted.stdout
