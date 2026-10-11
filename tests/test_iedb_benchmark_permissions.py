"""The monthly IEDB benchmark's issues token is held by one small job and no other.

`.github/workflows/iedb_benchmark.yml` installs the locked dependency set, an editable
copy of this package, and queries IEDB, a third-party academic service. Until
2026-10-10 the workflow granted `issues: write` at workflow level, so every one of those
steps ran holding a token that can open, edit and label issues. Only the final step,
which files a regression alert, needs it. The grant now lives on a separate `alert` job
that checks nothing out and runs no repository code: it downloads the regression message
the benchmark job uploaded and opens the issue. These tests pin that split, and the
hand-offs it adds: the output key, the artifact name and the marker file each job relies on.
"""

from __future__ import annotations

import pathlib
import re

import pytest

yaml = pytest.importorskip("yaml")

WORKFLOW = (
    pathlib.Path(__file__).resolve().parents[1] / ".github" / "workflows" / "iedb_benchmark.yml"
)


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _writes(permissions: object) -> set[str]:
    if permissions == "write-all":
        return {"*"}
    if not isinstance(permissions, dict):
        return set()
    return {scope for scope, level in permissions.items() if level == "write"}


def test_the_workflow_level_grant_is_read_only():
    assert _workflow()["permissions"] == {"contents": "read"}


def test_only_the_alert_job_can_write():
    jobs = _workflow()["jobs"]
    writers = {name: _writes(job.get("permissions")) for name, job in jobs.items()}
    assert {name for name, scopes in writers.items() if scopes} == {"alert"}
    assert jobs["alert"]["permissions"] == {"issues": "write"}


def test_the_alert_job_runs_no_repository_code():
    alert = _workflow()["jobs"]["alert"]
    uses = [step.get("uses", "") for step in alert["steps"]]
    assert not any("run" in step for step in alert["steps"]), (
        "the alert job must not run shell steps"
    )
    assert not any(u.startswith("actions/checkout@") for u in uses), (
        "the alert job must not check out"
    )
    assert [u.split("@")[0] for u in uses] == ["actions/download-artifact", "actions/github-script"]


def test_the_alert_job_runs_only_after_a_detected_regression():
    jobs = _workflow()["jobs"]
    alert = jobs["alert"]
    assert alert["needs"] == "benchmark"
    # Exact, not a substring: `always() || needs.benchmark.outputs.regression == 'true'`
    # contains the expected text and would run the writer on every benchmark run.
    assert alert["if"] == "needs.benchmark.outputs.regression == 'true'"
    output = jobs["benchmark"]["outputs"]["regression"]
    step_ids = {step.get("id") for step in jobs["benchmark"]["steps"]}
    assert output == "${{ steps.regression.outputs.detected }}"
    assert "regression" in step_ids


def test_the_benchmark_job_never_opens_an_issue():
    benchmark = _workflow()["jobs"]["benchmark"]
    for step in benchmark["steps"]:
        assert not str(step.get("uses", "")).startswith("actions/github-script@")
        assert "issues.create" not in str(step.get("with", {}).get("script", ""))


# The hand-offs between the two jobs. No pull request runs this workflow, so a broken link
# here (a key the job output never reads, an artifact name the alert job never finds, a
# file the issue step never opens) would surface only on a monthly run that found a
# regression, and then as a missing alert.


def _regression_handoff() -> tuple[str, dict, dict, dict, dict]:
    """The job output's key, the `regression` step, its upload, the download, the script."""
    jobs = _workflow()["jobs"]
    benchmark_steps = jobs["benchmark"]["steps"]
    match = re.fullmatch(
        r"\$\{\{ steps\.regression\.outputs\.([A-Za-z0-9_-]+) \}\}",
        jobs["benchmark"]["outputs"]["regression"],
    )
    assert match, "the job output `regression` must read one output of the `regression` step"
    position = [step.get("id") for step in benchmark_steps].index("regression")
    uploads = [
        step
        for step in benchmark_steps[position + 1 :]
        if str(step.get("uses", "")).startswith("actions/upload-artifact@")
    ]
    assert len(uploads) == 1, "exactly one upload must follow the `regression` step"
    download, script = jobs["alert"]["steps"]
    return match.group(1), benchmark_steps[position], uploads[0], download, script


def test_the_regression_step_writes_the_key_the_job_output_reads():
    key, record, upload, _, _ = _regression_handoff()
    marker = upload["with"]["path"]
    lines = [
        line.strip()
        for line in record["run"].splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert lines == [
        f"if [ -f {marker} ]; then",
        f'echo "{key}=true" >> "$GITHUB_OUTPUT"',
        "else",
        f'echo "{key}=false" >> "$GITHUB_OUTPUT"',
        "fi",
    ]


def test_the_regression_message_is_uploaded_only_when_a_regression_was_detected():
    key, _, upload, _, _ = _regression_handoff()
    # Unconditional, this step would fail every clean run: the marker is absent then and
    # the upload says `if-no-files-found: error`.
    assert upload.get("if") == f"steps.regression.outputs.{key} == 'true'"


def test_the_alert_job_downloads_the_artifact_the_benchmark_job_uploaded():
    _, _, upload, download, _ = _regression_handoff()
    assert download["uses"].startswith("actions/download-artifact@")
    assert download["with"]["name"] == upload["with"]["name"]
    # No `path`: the file lands in the workspace, where the issue step reads it by name.
    assert "path" not in download["with"]


def test_the_issue_step_reads_the_file_that_was_uploaded():
    _, _, upload, _, script = _regression_handoff()
    reads = re.findall(r"fs\.readFileSync\('([^']+)'", script["with"]["script"])
    assert reads == [pathlib.PurePosixPath(upload["with"]["path"]).name]
