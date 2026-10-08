"""The release gate must reject a CITATION.cff date that is not YYYY-MM-DD.

`.github/workflows/release.yml` runs an embedded Python script in its
"Verify tag matches package version" step. docs/releasing.md documents the
`date-released:` requirement as "an ISO calendar date (`YYYY-MM-DD`)", but the
script validated it with `datetime.date.fromisoformat` alone. On Python 3.11+
that call also accepts the ISO basic form and ISO week dates: measured on
3.11.15 and on 3.13.15 (the minor version the workflow sets up),
`fromisoformat("20260923")` returns 2026-09-23, `"2026-W38-3"` returns
2026-09-16 and `"2026-W38"` returns 2026-09-14. So a release tagged with any of
those in CITATION.cff passed the gate while contradicting the documented format.

The script is not importable - it lives in a `python - <<'PY'` heredoc inside
the workflow - so this test extracts it by YAML parse and runs it unmodified in
a temporary directory holding a crafted pyproject.toml and CITATION.cff, the
only two files it reads, with TAG_NAME set the way the step's `env:` sets it.
Running the real text rather than a copy of the regex means the test cannot
pass against a gate that has drifted from what it asserts.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release.yml"
STEP_NAME = "Verify tag matches package version"
VERSION = "9.9.9"


def _gate_script() -> str:
    workflow = yaml.safe_load(RELEASE_WORKFLOW.read_text(encoding="utf-8"))
    matches = [
        step
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if step.get("name") == STEP_NAME
    ]
    assert len(matches) == 1, f"expected one {STEP_NAME!r} step, found {len(matches)}"
    lines = matches[0]["run"].splitlines()
    # The heredoc delimiter is quoted, so the shell passes the body through
    # verbatim; fail loudly if the step stops being exactly that shape.
    assert lines[0] == "python - <<'PY'", lines[0]
    assert lines[-1] == "PY", lines[-1]
    body = "\n".join(lines[1:-1]) + "\n"
    assert "${{" not in body, "the script body now carries a workflow expression"
    return body


FIXTURE_COMMIT_DAY = "2020-01-01"


def _run_gate(
    tmp_path: Path, date_released: str, *, commit_date: str | None = FIXTURE_COMMIT_DAY
) -> subprocess.CompletedProcess[str]:
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nversion = "{VERSION}"\n', encoding="utf-8"
    )
    (tmp_path / "CITATION.cff").write_text(
        f"cff-version: 1.2.0\nversion: {VERSION}\ndate-released: {date_released}\n",
        encoding="utf-8",
    )
    env = dict(os.environ, TAG_NAME=f"v{VERSION}")
    # The workflow records this in a step before the gate; the gate fails closed without it.
    env.pop("COMMIT_DATE", None)
    if commit_date is not None:
        env["COMMIT_DATE"] = commit_date
    return subprocess.run(
        [sys.executable, "-"],
        input=_gate_script(),
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("date_released", ["2026-09-23", "2020-01-31"])
def test_extended_calendar_date_is_accepted(tmp_path, date_released):
    result = _run_gate(tmp_path, date_released)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"date-released={date_released}" in result.stdout


@pytest.mark.parametrize(
    "date_released",
    [
        "20260923",  # ISO basic form of an extended date the gate accepts
        "2026-W38-3",  # ISO week date with weekday
        "2026-W38",  # ISO week date without weekday
    ],
)
def test_other_iso_forms_are_rejected(tmp_path, date_released):
    result = _run_gate(tmp_path, date_released)
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"::error::CITATION.cff date-released {date_released!r}" in result.stdout
    # Every value above resolves to a date no later than 2026-09-23, so a
    # rejection must come from the format check, never from the future check.
    assert "in the future" not in result.stdout


def test_impossible_calendar_date_is_still_rejected(tmp_path):
    # Right shape, no such day: the format check must not replace the parse.
    result = _run_gate(tmp_path, "2026-02-30")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "::error::CITATION.cff date-released '2026-02-30'" in result.stdout


def test_future_date_is_still_rejected(tmp_path):
    result = _run_gate(tmp_path, "2999-01-01")
    assert result.returncode == 1, result.stdout + result.stderr
    expected = "::error::CITATION.cff date-released 2999-01-01 is in the future"
    assert expected in result.stdout


def test_a_date_before_the_tagged_commit_is_rejected(tmp_path):
    """The bound this gate gained: a release cannot predate the code it ships.

    Measured when it was added: v2.0.2 and v2.0.3 both shipped date-released 2026-06-12
    against commits of 2026-06-16 and 2026-06-17, so the future check alone let a stale
    date through on two published tags.
    """
    result = _run_gate(tmp_path, "2019-12-31", commit_date="2020-01-01")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "precedes the commit it tags (2020-01-01)" in result.stdout
    assert "in the future" not in result.stdout


def test_the_commits_own_day_is_accepted(tmp_path):
    """The boundary is inclusive: releasing on the day of the commit is normal."""
    result = _run_gate(tmp_path, "2020-01-01", commit_date="2020-01-01")
    assert result.returncode == 0, result.stdout + result.stderr


def test_a_missing_commit_date_fails_closed(tmp_path):
    """If the recording step is dropped, the gate must refuse rather than skip the bound."""
    result = _run_gate(tmp_path, "2020-06-01", commit_date=None)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "COMMIT_DATE" in result.stdout


def test_a_malformed_commit_date_fails_closed(tmp_path):
    """The value is dashed on purpose: an eight-digit form would be read as a commit
    SHA by scripts/check_doc_commit_refs.py and fail that gate as a dead citation."""
    result = _run_gate(tmp_path, "2020-06-01", commit_date="2020-1-1")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "COMMIT_DATE" in result.stdout


def test_the_workflow_records_the_commit_date_before_the_gate():
    """The gate reads an env var, so the step that sets it has to exist and precede it."""
    workflow = yaml.safe_load(RELEASE_WORKFLOW.read_text(encoding="utf-8"))
    for job in workflow["jobs"].values():
        names = [step.get("name") for step in job.get("steps", [])]
        if STEP_NAME not in names:
            continue
        recorder = [
            i for i, step in enumerate(job["steps"]) if "COMMIT_DATE=" in str(step.get("run", ""))
        ]
        assert len(recorder) == 1, f"expected one COMMIT_DATE recorder, found {recorder}"
        assert recorder[0] < names.index(STEP_NAME), "it must run before the gate"
        assert "%cs" in str(job["steps"][recorder[0]]["run"]), "record the committer DATE"
        return
    raise AssertionError(f"no job carries a {STEP_NAME!r} step")
