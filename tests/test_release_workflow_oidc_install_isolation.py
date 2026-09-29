"""Release jobs with OIDC authority must not resolve Python dependencies."""

from pathlib import Path

import yaml


WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "release.yml"


def _jobs() -> dict:
    document = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return document["jobs"]


def _has_oidc_write(job: dict) -> bool:
    return job.get("permissions", {}).get("id-token") == "write"


def _run_script(job: dict) -> str:
    return "\n".join(str(step.get("run", "")) for step in job.get("steps", []))


def test_token_holding_jobs_do_not_run_pip_install() -> None:
    token_jobs = {name: job for name, job in _jobs().items() if _has_oidc_write(job)}

    assert set(token_jobs) == {"release", "publish"}
    for name, job in token_jobs.items():
        assert "pip install" not in _run_script(job), name


def _has_checkout(job: dict) -> bool:
    return any(
        str(step.get("uses", "")).startswith("actions/checkout@") for step in job.get("steps", [])
    )


def test_gh_steps_in_jobs_without_a_checkout_name_the_repository() -> None:
    """gh resolves the repository from the checkout's git remote, or from GH_REPO.

    It does not read GITHUB_REPOSITORY, so in a job with no checkout a bare
    `gh release create` fails with "not a git repository". Splitting release.yml
    into build, release and publish jobs created exactly such a job.
    """
    checked = 0
    for name, job in _jobs().items():
        if _has_checkout(job):
            continue
        for step in job.get("steps", []):
            script = str(step.get("run", ""))
            if not any(line.lstrip().startswith("gh ") for line in script.splitlines()):
                continue
            checked += 1
            env = step.get("env", {}) or {}
            assert "GH_REPO" in env or "--repo" in script, (
                f"job {name!r}, step {step.get('name')!r} runs gh with no checkout "
                "and neither sets GH_REPO nor passes --repo"
            )
    assert checked >= 1, "no gh step in a checkout-less job was found; the guard is vacuous"


def test_dependency_checks_run_before_token_holding_jobs() -> None:
    jobs = _jobs()

    assert not _has_oidc_write(jobs["build"])
    assert "pip install" in _run_script(jobs["build"])
    assert jobs["release"]["needs"] == "build"
    assert jobs["publish"]["needs"] == "release"
    assert jobs["smoke"]["needs"] == "publish"
    assert not _has_oidc_write(jobs["smoke"])
    assert "pip install" in _run_script(jobs["smoke"])
