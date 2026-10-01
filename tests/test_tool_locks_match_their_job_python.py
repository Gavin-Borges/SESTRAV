"""Each lockfile is compiled for the Python of the CI jobs that install it.

`uv pip compile --python-version X` resolves for interpreter X only. A lock
compiled for one Python and installed on another with `--require-hashes` can
carry a package the job's interpreter never needs, or omit one it does, and
nothing reports it until an install or an import fails. Measured 2026-09-30:
seven managed tool locks named a different `python_version` in
tools/update_dependencies.py than the jobs installing them used (the semgrep
lock, for example, was compiled for 3.12 and installed by two 3.11 jobs). Six
recompiled to identical pins for their job's Python and the seventh dropped only
tomli, which Python 3.13 does not need.

The runtime and ci locks are installed across a Python matrix on purpose, so for
them the compile target must be one of the matrix versions, not all of them.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from tools.update_dependencies import LOCK_SPECS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = PROJECT_ROOT / ".github" / "workflows"

# Installed by jobs on several Pythons by design (ci.yml's compat matrix and the
# 3.13 test job install the same two files).
MULTI_PYTHON_SPECS = frozenset({"runtime", "ci"})


def job_pythons(job: dict) -> set[str]:
    """Every Python version a job's setup-python steps select, matrix expanded."""
    matrix = (job.get("strategy") or {}).get("matrix") or {}
    versions: set[str] = set()
    for step in job.get("steps") or []:
        value = (step.get("with") or {}).get("python-version")
        if value is None:
            continue
        value = str(value)
        key = re.search(r"matrix\.([\w-]+)", value)
        if key:
            listed = matrix.get(key.group(1), [])
            versions |= {str(v) for v in (listed if isinstance(listed, list) else [listed])}
            versions |= {
                str(entry[key.group(1)])
                for entry in matrix.get("include") or []
                if isinstance(entry, dict) and key.group(1) in entry
            }
        else:
            versions.add(value)
    return versions


_PIP_INSTALL = re.compile(r"\bpip(?:3(?:\.\d+)?)?\s+install\b")


def installs(job: dict, lock: str) -> bool:
    """Whether any `pip install` in the job's run steps takes `lock` as -r/--requirement."""
    runs = "\n".join(str(step.get("run", "")) for step in job.get("steps") or [])
    runs = re.sub(r"\\\s*\n\s*", " ", runs)
    requirement = re.compile(
        r"(?:-r|--requirement)(?:\s+|=)?[\"']?" + re.escape(lock) + r"[\"']?(?=\s|$)"
    )
    return any(
        _PIP_INSTALL.search(line) and requirement.search(line) for line in runs.splitlines()
    )


def installing_pythons(lock: str) -> dict[str, set[str]]:
    """Map workflow:job -> Pythons, for every job that installs `lock`."""
    found = {}
    for path in sorted(WORKFLOWS.glob("*.y*ml")):
        workflow = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for name, job in (workflow.get("jobs") or {}).items():
            if installs(job, lock):
                found[f"{path.name}:{name}"] = job_pythons(job)
    return found


@pytest.mark.parametrize("spec", LOCK_SPECS, ids=lambda s: s.name)
def test_lock_is_compiled_for_the_python_of_its_installing_jobs(spec) -> None:
    jobs = installing_pythons(spec.output)
    if not jobs:
        pytest.skip(f"no workflow job installs {spec.output}")
    for where, pythons in jobs.items():
        assert pythons, f"{where} installs {spec.output} but selects no Python"
    if spec.name in MULTI_PYTHON_SPECS:
        matrix = set().union(*jobs.values())
        assert spec.python_version in matrix, (spec.name, spec.python_version, jobs)
    else:
        wrong = {w: p for w, p in jobs.items() if p != {spec.python_version}}
        assert not wrong, f"{spec.name} is compiled for {spec.python_version}: {wrong}"


def test_most_specs_are_checked_against_a_single_python() -> None:
    """Guard against the parametrized test skipping its way to green."""
    checked = [s for s in LOCK_SPECS if s.name not in MULTI_PYTHON_SPECS and installing_pythons(s.output)]
    assert len(checked) >= 7, [s.name for s in checked]


def test_job_pythons_expands_a_matrix_and_reads_a_literal() -> None:
    job = {
        "strategy": {"matrix": {"python-version": ["3.11", "3.12"]}},
        "steps": [
            {"uses": "actions/setup-python@x", "with": {"python-version": "${{ matrix.python-version }}"}},
            {"uses": "actions/setup-python@x", "with": {"python-version": "3.13"}},
        ],
    }
    assert job_pythons(job) == {"3.11", "3.12", "3.13"}


def test_installs_matches_the_lock_path_not_a_prefix() -> None:
    job = {"steps": [{"run": "pip install --require-hashes \\\n  -r environments/requirements-ci.txt"}]}
    assert installs(job, "environments/requirements-ci.txt")
    assert not installs(job, "environments/requirements-ci")


@pytest.mark.parametrize(
    "run",
    [
        "pip install -r environments/requirements-ci.txt",
        "pip3 install -r environments/requirements-ci.txt",
        "python -m pip install --require-hashes -r environments/requirements-ci.txt",
        'pip install -r "environments/requirements-ci.txt"',
        "pip install --requirement environments/requirements-ci.txt",
        "pip install --requirement=environments/requirements-ci.txt",
        "pip install -renvironments/requirements-ci.txt",
    ],
)
def test_installs_reads_every_requirement_spelling(run: str) -> None:
    assert installs({"steps": [{"run": run}]}, "environments/requirements-ci.txt")


def test_job_pythons_reads_matrix_include_entries() -> None:
    job = {
        "strategy": {"matrix": {"include": [{"python-version": "3.11"}, {"python-version": "3.13"}]}},
        "steps": [{"with": {"python-version": "${{ matrix.python-version }}"}}],
    }
    assert job_pythons(job) == {"3.11", "3.13"}
