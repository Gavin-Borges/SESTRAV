"""The Dependabot render-lockfile guard runs when the freshness check before it fails.

`.github/workflows/dependency-lockfile-check.yml` runs two checks in one job: the freshness
check (`tools/check_lockfile_freshness.py --check`), then a guard step that fails any pull
request authored by `dependabot[bot]` that edits `environments/requirements-ci-render.txt`.
GitHub applies a default status check of `success()` to an `if:` that contains no status
check function, so a guard whose `if:` only selected Dependabot pull requests was SKIPPED
whenever the freshness check failed - and the freshness check fails on the Dependabot pip
pull requests the guard exists for. On the latest run of the workflow on each Dependabot
pip pull request open on 2026-10-25, the freshness step failed and the guard was skipped,
including on one that edits the render lockfile.

These tests pin the condition that fixes it: `!cancelled()` joined by `&&` to the two
conditions that select a Dependabot pull request, and nothing else. They also pin how the
guard's script gets the base branch: through the step's `env:`, never by a `${{ }}`
expansion inside `run:`, which is substituted into the script text before the shell
parses it and which zizmor reports as template injection. They read the YAML; nothing
here runs the workflow. The mutant tables show each checker refusing the listed wrong
inputs. They are regression ratchets, not an exhaustive model of GitHub's expression
language.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

WORKFLOW = (
    Path(__file__).resolve().parents[1] / ".github" / "workflows" / "dependency-lockfile-check.yml"
)
JOB = "lockfile-freshness"
RENDER_LOCK = "environments/requirements-ci-render.txt"
FRESHNESS_COMMAND = "tools/check_lockfile_freshness.py --check"

RUNS_UNLESS_CANCELLED = "!cancelled()"
SELECTS_DEPENDABOT = (
    "github.event_name == 'pull_request'",
    "github.event.pull_request.user.login == 'dependabot[bot]'",
)
FIXED_CONDITION = "${{ !cancelled() && " + " && ".join(SELECTS_DEPENDABOT) + " }}"

_STATUS_FUNCTION = re.compile(r"\b(?:success|failure|always|cancelled)\s*\(\s*\)")
# An opening parenthesis that is not the empty argument list of a function call.
_GROUPING = re.compile(r"\((?!\s*\))")


def _steps() -> list[dict]:
    document = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return document["jobs"][JOB]["steps"]


def _only_index(steps: list[dict], needle: str) -> int:
    hits = [i for i, step in enumerate(steps) if needle in str(step.get("run", ""))]
    assert len(hits) == 1, f"expected exactly one step whose run: holds {needle!r}, found {hits}"
    return hits[0]


def _conjuncts(condition: str) -> tuple[list[str], list[str]]:
    """Split an `if:` into its top-level `&&` terms, or say why it is not one plain conjunction."""
    text = " ".join(condition.split())
    problems: list[str] = []
    if "${{" in text:
        if not (text.startswith("${{") and text.endswith("}}") and text.count("${{") == 1):
            problems.append(f"the condition is not a single ${{{{ }}}} expression: {text!r}")
            return [], problems
        text = text[3:-2].strip()
    if "||" in text:
        problems.append("an || lets the condition hold without every term below")
    if _GROUPING.search(text):
        problems.append("grouping parentheses hide terms from this check")
    if problems:
        return [], problems
    terms = [re.sub(r"!\s+", "!", term.strip()) for term in text.split("&&")]
    return terms, problems


def guard_condition_problems(condition: object) -> list[str]:
    """Why a guard `if:` could skip the guard after a failed freshness check, or widen it.

    Empty when the condition is exactly `!cancelled()` and the two Dependabot selectors,
    joined by `&&`, in any order and with or without a `${{ }}` wrapper.
    """
    if not isinstance(condition, str) or not condition.strip():
        return [
            "the guard has no if:, so it runs on human pull requests and, carrying the "
            "default success(), is still skipped after a failed freshness check"
        ]
    terms, problems = _conjuncts(condition)
    if problems:
        return problems
    statuses = [term for term in terms if _STATUS_FUNCTION.search(term)]
    if not statuses:
        problems.append(
            "no status check function, so GitHub applies success() and the guard is "
            "skipped whenever the freshness step before it fails"
        )
    elif statuses != [RUNS_UNLESS_CANCELLED]:
        problems.append(
            f"status check {statuses} is not exactly {RUNS_UNLESS_CANCELLED}: success() is "
            "the default that skips the guard, failure() skips it when the freshness step "
            "passes, always() also runs it in a cancelled run, cancelled() runs it only then"
        )
    for selector in SELECTS_DEPENDABOT:
        if selector not in terms:
            problems.append(f"missing {selector!r}, so the guard is no longer Dependabot-only")
    extra = [t for t in terms if t not in statuses and t not in SELECTS_DEPENDABOT]
    if extra:
        problems.append(f"unexpected terms narrow when the guard runs: {extra}")
    return problems


def test_the_guard_runs_after_a_failed_freshness_check() -> None:
    steps = _steps()
    guard = steps[_only_index(steps, RENDER_LOCK)]
    assert guard_condition_problems(guard.get("if")) == []


def test_the_guard_still_selects_dependabot_pull_requests_and_nothing_wider() -> None:
    steps = _steps()
    condition = steps[_only_index(steps, RENDER_LOCK)].get("if")
    assert isinstance(condition, str), "the guard has no if:"
    terms, problems = _conjuncts(condition)
    assert problems == []
    selectors = {t for t in terms if not _STATUS_FUNCTION.search(t)}
    assert selectors == set(SELECTS_DEPENDABOT)


def test_neither_check_can_be_skipped_or_ignored_another_way() -> None:
    """The fix is the guard's status check, not a disarming of the freshness step.

    A `continue-on-error` on the freshness step would also let the guard run, by turning
    a stale lockfile green; one on the guard would turn a render-lockfile edit green; and a
    guard placed before the freshness step would make that step the one skipped.
    """
    document = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    job = document["jobs"][JOB]
    steps = job["steps"]
    freshness = _only_index(steps, FRESHNESS_COMMAND)
    guard = _only_index(steps, RENDER_LOCK)
    assert guard > freshness
    assert "if" not in steps[freshness], steps[freshness].get("if")
    for index in (freshness, guard):
        assert "continue-on-error" not in steps[index], steps[index].get("name")
    assert "continue-on-error" not in job


_WRONG_CONDITIONS = {
    "no status function (origin/main before this fix)": " && ".join(SELECTS_DEPENDABOT),
    "success()": "${{ success() && " + " && ".join(SELECTS_DEPENDABOT) + " }}",
    "failure()": "${{ failure() && " + " && ".join(SELECTS_DEPENDABOT) + " }}",
    "always()": "${{ always() && " + " && ".join(SELECTS_DEPENDABOT) + " }}",
    "cancelled()": "${{ cancelled() && " + " && ".join(SELECTS_DEPENDABOT) + " }}",
    "an || before the selectors": "${{ !cancelled() || " + " && ".join(SELECTS_DEPENDABOT) + " }}",
    "the login selector dropped": "${{ !cancelled() && " + SELECTS_DEPENDABOT[0] + " }}",
    "the event selector dropped": "${{ !cancelled() && " + SELECTS_DEPENDABOT[1] + " }}",
    "text outside the expression": "${{ !cancelled() }} && " + " && ".join(SELECTS_DEPENDABOT),
    "grouped selectors": "${{ !cancelled() && (" + " && ".join(SELECTS_DEPENDABOT) + ") }}",
    "an extra narrowing term": FIXED_CONDITION[:-3] + " && github.run_attempt == 1 }}",
    "no if at all": None,
}


@pytest.mark.parametrize("condition", list(_WRONG_CONDITIONS.values()), ids=list(_WRONG_CONDITIONS))
def test_the_checker_refuses_each_listed_wrong_condition(condition: object) -> None:
    assert guard_condition_problems(condition) != []


@pytest.mark.parametrize(
    "condition",
    [
        FIXED_CONDITION,
        "${{ !cancelled() &&\n" + " &&\n".join(SELECTS_DEPENDABOT) + " }}",
        "${{ " + " && ".join(reversed(SELECTS_DEPENDABOT)) + " && ! cancelled() }}",
        "!cancelled() && " + " && ".join(SELECTS_DEPENDABOT),
    ],
    ids=["one line", "folded over lines", "reordered, spaced bang", "unwrapped"],
)
def test_the_checker_accepts_the_fixed_condition(condition: str) -> None:
    assert guard_condition_problems(condition) == []


# Pull-request refs reach a script through env, never by expansion into its text. Both
# contexts name a branch the pull request's author picks; the head ref is arbitrary text.
_PR_REF = re.compile(r"\bgithub\.(?:base_ref|head_ref|event\.pull_request\.(?:base|head)\.ref)\b")
_EXPRESSION = re.compile(r"\$\{\{(.*?)\}\}", re.DOTALL)


def run_block_ref_expansions(document: dict) -> list[str]:
    """Every `${{ }}` expression inside a `run:` block that reads a pull-request ref."""
    found = []
    for job_name, job in (document.get("jobs") or {}).items():
        for index, step in enumerate(job.get("steps") or []):
            for expression in _EXPRESSION.findall(str(step.get("run", ""))):
                if _PR_REF.search(expression):
                    found.append(f"{job_name} step {index}: {expression.strip()}")
    return found


def test_pull_request_refs_reach_run_blocks_through_env_only() -> None:
    document = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert run_block_ref_expansions(document) == []
    steps = document["jobs"][JOB]["steps"]
    guard = steps[_only_index(steps, RENDER_LOCK)]
    # The guard must still diff against the base it is given. With the variable renamed or
    # missing, `git diff` fails on "origin/...HEAD", `grep -qx` sees no input, and the step
    # prints its PASSED line and exits 0.
    assert (guard.get("env") or {}).get("BASE_REF") == "${{ github.base_ref }}"
    assert '"origin/${BASE_REF}...HEAD"' in guard["run"]


def _one_step(**step: object) -> dict:
    return {"jobs": {"job": {"steps": [{"name": "step", **step}]}}}


_EXPANDING_STEPS = {
    "base_ref": _one_step(run='git diff "origin/${{ github.base_ref }}...HEAD"'),
    "head_ref, no spaces": _one_step(run="echo ${{github.head_ref}}"),
    "head_ref inside a function call": _one_step(
        run="echo \"${{ format('{0}', github.head_ref) }}\""
    ),
    "the event payload's head ref": _one_step(run="echo ${{ github.event.pull_request.head.ref }}"),
    "on a later line": _one_step(run="set -e\necho ${{ github.base_ref }}\n"),
}


@pytest.mark.parametrize("document", list(_EXPANDING_STEPS.values()), ids=list(_EXPANDING_STEPS))
def test_the_ref_checker_refuses_each_listed_expansion(document: dict) -> None:
    assert run_block_ref_expansions(document) != []


_SAFE_STEPS = {
    "through env": _one_step(env={"REF": "${{ github.head_ref }}"}, run='echo "$REF"'),
    "in the step's if": _one_step(**{"if": "github.base_ref == 'main'"}, run="echo ok"),
    "a shell variable": _one_step(run='git diff "origin/${BASE_REF}...HEAD"'),
    "another context": _one_step(run="echo ${{ github.sha }}"),
}


@pytest.mark.parametrize("document", list(_SAFE_STEPS.values()), ids=list(_SAFE_STEPS))
def test_the_ref_checker_accepts_each_listed_safe_step(document: dict) -> None:
    assert run_block_ref_expansions(document) == []
