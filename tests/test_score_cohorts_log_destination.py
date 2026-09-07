"""The cohort scoring run log must not land on a tracked docs path.

`scripts/score_validation_cohorts.py` writes a markdown table carrying AUC-PR,
AUC-ROC and ISSR to four decimals with bootstrap CIs, plus a YES/NO verdict
against an unpublished 0.75 threshold. None of it is bound through
`_local/integrity/claims_manifest.toml` or passed by claims-auditor.

It used to write that to `docs/stage3_results_log.md`. That path is NOT
gitignored and WAS tracked historically (added in 328202f, deleted in b6c9d50),
so any run followed by `git add -A` would publish unbound numbers into `docs/`.
No gate watches for an untracked file appearing there: the citation gate, the
retracted-token sweep and the reconcile check all read files that already exist
in the tree, so a freshly written one is invisible to every one of them until it
is committed - at which point it is already published.

The destination is now `results/`, which `.gitignore` covers.

These tests pin the PROPERTY (the target is ignored) rather than the string, so
they keep biting if the path is renamed within `results/`, and they do not
depend on the numbers themselves ever having been produced.
"""

from __future__ import annotations

import pathlib
import re
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "score_validation_cohorts.py"


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )


def _results_log_path() -> pathlib.Path:
    from scripts.score_validation_cohorts import RESULTS_LOG_PATH

    return pathlib.Path(RESULTS_LOG_PATH)


def test_results_log_target_is_gitignored():
    """The property that actually protects the repository."""
    target = _results_log_path()
    relative = target.relative_to(REPO_ROOT).as_posix()
    probe = _git("check-ignore", "-v", "--", relative)
    if probe.returncode == 128:
        pytest.skip(f"git unavailable: {probe.stderr.strip()}")
    assert probe.returncode == 0, (
        f"{relative} is NOT gitignored, so a run followed by `git add -A` would "
        "commit unbound experimental metrics"
    )


def test_results_log_is_not_under_docs():
    relative = _results_log_path().relative_to(REPO_ROOT).as_posix()
    assert not relative.startswith("docs/"), (
        f"the run log target is {relative}; docs/ is a reader-facing tracked tree "
        "and these numbers are unbound"
    )
    assert relative.startswith("results/"), relative


def test_script_source_never_writes_a_log_into_docs():
    """Anti-regression over the source, so a SECOND write cannot reintroduce this."""
    source = SCRIPT.read_text(encoding="utf-8")
    offenders = [
        line.strip()
        for line in source.splitlines()
        if re.search(r"""["']docs["']\s*,""", line)
        or re.search(r"""["']docs/[^"']*\.md["']""", line)
    ]
    assert not offenders, (
        "score_validation_cohorts.py builds a docs/ path:\n  " + "\n  ".join(offenders)
    )


def test_the_historical_docs_log_is_not_tracked_again():
    """It was tracked once (328202f) and removed (b6c9d50). Keep it removed."""
    probe = _git("ls-files", "--error-unmatch", "docs/stage3_results_log.md")
    if probe.returncode == 128 and "not a git repository" in probe.stderr.lower():
        pytest.skip("git unavailable")
    assert probe.returncode != 0, (
        "docs/stage3_results_log.md is tracked again; it carries unbound "
        "experimental metrics and must not be committed"
    )


def test_the_source_scan_would_notice_a_docs_write():
    """Anti-vacuity: the regex above must actually match the pattern it hunts."""
    sample = 'log_path = os.path.join(PROJECT_ROOT, "docs", "stage3_results_log.md")'
    assert re.search(r"""["']docs["']\s*,""", sample)
