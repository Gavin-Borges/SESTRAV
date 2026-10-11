"""Every actions/checkout step in .github/workflows drops the job token after checkout.

actions/checkout defaults `persist-credentials` to true, which leaves the job's
GITHUB_TOKEN configured for git in the checkout; since v6 the action's README says
it is stored in a separate file under $RUNNER_TEMP rather than in .git/config. Any
later step, and any dependency a later step runs, can read it. Every workflow here
opts out. The last one that did not was
dco.yml, whose only network call after checkout is a `git fetch` of the base
branch; it opted out on 2026-10-09.

A checkout that genuinely needs the token for a later git push must set
`persist-credentials: true` explicitly and be listed in ALLOWED_TO_PERSIST, with
the reason, so the exception is a reviewed decision rather than a default.
"""

from __future__ import annotations

from pathlib import Path

import yaml

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"

# "<workflow file>:<job id>" -> why that checkout must keep its token. Empty today.
ALLOWED_TO_PERSIST: dict[str, str] = {}


def _checkout_steps() -> list[tuple[str, str, dict]]:
    steps = []
    for path in sorted([*WORKFLOWS.glob("*.yml"), *WORKFLOWS.glob("*.yaml")]):
        workflow = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for job_id, job in (workflow.get("jobs") or {}).items():
            for step in job.get("steps") or []:
                if str(step.get("uses", "")).startswith("actions/checkout@"):
                    steps.append((path.name, job_id, step))
    return steps


def test_the_scan_sees_the_checkout_steps() -> None:
    # 29 on 2026-10-09. A floor, not a pin: it only has to prove the walk is not empty.
    assert len(_checkout_steps()) >= 20


def test_every_checkout_drops_the_job_token() -> None:
    offenders = [
        f"{name}:{job_id}"
        for name, job_id, step in _checkout_steps()
        if (step.get("with") or {}).get("persist-credentials") is not False
        and f"{name}:{job_id}" not in ALLOWED_TO_PERSIST
    ]
    assert not offenders, (
        "these actions/checkout steps leave GITHUB_TOKEN configured for git; add "
        f"`persist-credentials: false` or justify them in ALLOWED_TO_PERSIST: {offenders}"
    )


def test_every_allowed_exception_exists_and_says_true() -> None:
    steps = {f"{name}:{job_id}": step for name, job_id, step in _checkout_steps()}
    stale = sorted(set(ALLOWED_TO_PERSIST) - set(steps))
    assert not stale, f"ALLOWED_TO_PERSIST names checkouts that no longer exist: {stale}"
    implicit = sorted(
        key
        for key in ALLOWED_TO_PERSIST
        if (steps[key].get("with") or {}).get("persist-credentials") is not True
    )
    assert not implicit, (
        f"these allowed checkouts must set `persist-credentials: true` explicitly: {implicit}"
    )
