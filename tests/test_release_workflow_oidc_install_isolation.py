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


def test_dependency_checks_run_before_token_holding_jobs() -> None:
    jobs = _jobs()

    assert not _has_oidc_write(jobs["build"])
    assert "pip install" in _run_script(jobs["build"])
    assert jobs["release"]["needs"] == "build"
    assert jobs["publish"]["needs"] == "release"
    assert jobs["smoke"]["needs"] == "publish"
    assert not _has_oidc_write(jobs["smoke"])
    assert "pip install" in _run_script(jobs["smoke"])
