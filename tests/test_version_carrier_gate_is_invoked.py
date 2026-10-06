"""The version-carrier gate is invoked by something that runs, not just shipped.

`tools/check_version_carriers.py` fails when a tracked current-version carrier (a badge,
a citation file, a CLI example, a model-card field, the API's source-run fallback)
disagrees with `pyproject.toml`. It landed with its own unit tests and NO caller: measured
on `origin/main` before this unit, `git grep -l check_version_carriers` returned only
`tests/test_check_version_carriers.py`, so no workflow, hook, Makefile target or release
step ran it. A gate nothing invokes is documentation, which is the defect class
`.claude/rules/deletion-safety-ci-gates.md` records against two other gates in this repo.

It is now a step in `ci.yml`'s `test` job. That job is deliberate rather than convenient:
it is already a required merge context, so the gate becomes merge-blocking without anyone
having to add a context to the branch ruleset, and it runs on every pull request instead of
only at a version tag. A tag-time check would fire after the disagreement had shipped,
which is the "fires too late" shape `.claude/rules/deletion-safety.md` rule 4 records.

What these tests do NOT cover, stated so the coverage is not overread: they read the
workflow as data and do not prove GitHub schedules the job, do not check the ruleset's
required-context list (that is an API question and is re-measured, never quoted), and do
not run the gate itself - `tests/test_check_version_carriers.py` owns its behaviour.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CI = REPO_ROOT / ".github" / "workflows" / "ci.yml"
MAKEFILE = REPO_ROOT / "Makefile"
GATE = "tools/check_version_carriers.py"
JOB = "test"


@pytest.fixture(scope="module")
def ci_workflow() -> dict:
    return yaml.safe_load(CI.read_text(encoding="utf-8"))


def _gate_steps(workflow: dict) -> list[dict]:
    return [step for step in workflow["jobs"][JOB]["steps"] if GATE in str(step.get("run", ""))]


def test_the_gate_exists_and_is_executable_as_written() -> None:
    assert (REPO_ROOT / GATE).is_file(), f"{GATE} is gone; this unit's premise is void"


def test_the_required_test_job_runs_the_gate(ci_workflow: dict) -> None:
    steps = _gate_steps(ci_workflow)
    assert len(steps) == 1, f"expected exactly one {GATE} step in the {JOB} job, found {len(steps)}"


def test_the_new_step_did_not_swallow_its_neighbour(ci_workflow: dict) -> None:
    """A dropped `- name:` line merges this step's run into the one above it.

    YAML then sees two `run:` keys in one mapping and keeps the LAST, so the gate appears
    present while the sibling library-coverage gate silently vanishes. Measured: that exact
    edit takes the job from 17 steps to 16 and leaves the coverage step running this gate's
    command under the coverage step's name. A mutant doing it survived every other assertion
    here, so both gates are asserted by their own steps.
    """
    steps = ci_workflow["jobs"][JOB]["steps"]
    owners = {}
    for sibling in ("tools/check_library_coverage.py", GATE):
        matching = [i for i, step in enumerate(steps) if sibling in str(step.get("run", ""))]
        assert len(matching) == 1, f"{sibling} should run in exactly one step, found {matching}"
        owners[sibling] = matching[0]
    assert len(set(owners.values())) == 2, f"two gates share one step: {owners}"
    for sibling, index in owners.items():
        assert steps[index].get("name"), f"{sibling} runs in a step with no name"


def test_the_step_cannot_fail_silently(ci_workflow: dict) -> None:
    step = _gate_steps(ci_workflow)[0]
    assert "continue-on-error" not in step, "a continue-on-error gate reports nothing"
    assert "if" not in step, "a conditional step can skip the gate without saying so"
    assert "shell" not in step, "a custom shell may drop -e and swallow the exit status"
    run = str(step["run"])
    assert "--dry-run" not in run
    assert "||" not in run, "an || fallback turns a failure into a pass"
    assert "set +e" not in run


def test_the_gate_runs_before_the_expensive_test_step(ci_workflow: dict) -> None:
    """It is a seconds-long consistency check; running it after the suite wastes a CI run."""
    names = [str(step.get("name", "")) for step in ci_workflow["jobs"][JOB]["steps"]]
    runs = [str(step.get("run", "")) for step in ci_workflow["jobs"][JOB]["steps"]]
    gate_index = next(i for i, run in enumerate(runs) if GATE in run)
    # The suite is the INVOCATION, not the step that pip-installs pytest-cov. Matching a bare
    # "pytest" picks up that install step, which runs before the gate and made this test fail
    # on its first draft.
    suite_index = next(i for i, run in enumerate(runs) if re.search(r"-m\s+pytest\b", run))
    assert gate_index < suite_index, f"gate at {gate_index} runs after the suite at {suite_index}"
    assert names[gate_index], "the step needs a name so a red X identifies it"


def test_the_makefile_mirror_carries_the_gate() -> None:
    """The Makefile mirrors CI's gates one target per gate; an absent target hides one."""
    text = MAKEFILE.read_text(encoding="utf-8")
    assert re.search(r"^version-carriers:\n\tpython " + re.escape(GATE), text, re.M), (
        "Makefile has no version-carriers target running the gate"
    )
    assert re.search(r"^\.PHONY:.*version-carriers|^\s+hash-pins.*version-carriers", text, re.M), (
        "version-carriers is missing from .PHONY"
    )
    aggregate = re.search(r"^ci:(?:[^\n]*\\\n)*[^\n]*$", text, re.M)
    assert aggregate is not None, "the Makefile has no ci aggregate"
    assert "version-carriers" in aggregate.group(0), "the ci aggregate skips the gate"
