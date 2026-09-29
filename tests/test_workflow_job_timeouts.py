"""Every workflow job must declare its own timeout-minutes.

GitHub's default job timeout is 360 minutes. A job that hangs - a network read
with no deadline, a prompt nobody answers, a wedged runner - therefore occupies
a runner for six hours before it is killed. Four of the workflows this test
first bounded - dco, doc_commit_refs, doc_line_citations and pr-review-check -
back required status contexts at the time of writing, so a hang in any of them
leaves a pull request un-mergeable for that whole window rather than failing
fast and telling the author why.

The bound is not a performance target and should not be tuned like one. It is
the point past which the job is definitely not working any more. Values were
set from observed run durations: for each workflow, the 12 most recent
completed runs created before 2026-09-23 18:22 UTC, read with
`gh run list --json startedAt,updatedAt`, each taken as updatedAt minus
startedAt (run wall-clock, queue included) and rounded half up to 0.1 minute:

    lock_manifest_coverage      median 0.2m   max 0.2m
    dependency-lockfile-check   median 0.2m   max 2.9m
    pr-review-check             median 0.2m   max 1.3m
    affiliation_claims          median 0.3m   max 3.2m
    dco                         median 0.3m   max 1.6m
    dependency-review           median 0.3m   max 1.0m
    doc_commit_refs             median 0.3m   max 0.9m
    doc_line_citations          median 0.3m   max 2.3m
    pii_scan                    median 0.4m   max 1.0m
    scorecard                   median 0.8m   max 1.4m

The slowest of those samples is 3.2 minutes, so the 10 minutes those jobs
carry is about three times the slowest of those samples (run wall-clock, queue
included), and matches the value digest_portability.yml already used.
release.yml is the exception and is given more room. Only four runs exist: two
tag runs (v2.0.2, v2.0.3) that built, attested and published a GitHub Release
in 16 and 22 seconds of job time, and two that failed before any job started.
The workflow had grown from 131 lines at v2.0.3 to 331 when this test was
written, and its PyPI publish job has never run, so none of them bounds it.

A job that calls a reusable workflow with `uses:` cannot accept timeout-minutes,
so such jobs are exempt. None exists today; the exemption is here so that adding
one is not read as a violation.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

WORKFLOW_DIR = pathlib.Path(__file__).resolve().parents[1] / ".github" / "workflows"

# GitHub's own ceiling for a hosted runner job. A declared value at or above this
# is not a bound at all, so it is rejected rather than accepted as a large bound.
GITHUB_DEFAULT_TIMEOUT_MINUTES = 360


def _workflows() -> list[pathlib.Path]:
    return sorted(p for p in WORKFLOW_DIR.glob("*.y*ml") if p.suffix in {".yml", ".yaml"})


def _jobs(path: pathlib.Path):
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return (document.get("jobs") or {}).items()


def test_workflows_are_discovered() -> None:
    """Anti-vacuity: a rename or a bad glob must fail here, not pass everything.

    Without this, every assertion below becomes a loop over nothing and the file
    stays green while checking no workflow at all.
    """
    found = _workflows()
    assert len(found) >= 15, f"workflow discovery looks broken: {[p.name for p in found]}"
    assert any(p.name == "ci.yml" for p in found), [p.name for p in found]


@pytest.mark.parametrize("path", _workflows(), ids=lambda p: p.name)
def test_every_job_declares_a_timeout(path: pathlib.Path) -> None:
    """A job with no timeout-minutes inherits the 360-minute default."""
    offenders = [
        name
        for name, body in _jobs(path)
        if isinstance(body, dict) and "uses" not in body and "timeout-minutes" not in body
    ]
    assert not offenders, (
        f"{path.name}: these jobs inherit GitHub's {GITHUB_DEFAULT_TIMEOUT_MINUTES}-minute "
        f"default, so a hang holds a runner for six hours: {offenders}. "
        "Add a job-level timeout-minutes; see this module's docstring for how the "
        "existing values were chosen."
    )


@pytest.mark.parametrize("path", _workflows(), ids=lambda p: p.name)
def test_every_declared_timeout_is_a_real_bound(path: pathlib.Path) -> None:
    """A timeout at or above the default bounds nothing, and zero is not valid."""
    bad: list[str] = []
    for name, body in _jobs(path):
        if not isinstance(body, dict) or "timeout-minutes" not in body:
            continue
        value = body["timeout-minutes"]
        if not isinstance(value, int) or isinstance(value, bool):
            bad.append(f"{name}={value!r} (not an integer)")
        elif value < 1:
            bad.append(f"{name}={value} (must be at least 1)")
        elif value >= GITHUB_DEFAULT_TIMEOUT_MINUTES:
            bad.append(f"{name}={value} (at or above the default, so it bounds nothing)")
    assert not bad, f"{path.name}: {bad}"
