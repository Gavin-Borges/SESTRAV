"""No workflow job may introduce a NEW cross-file pin divergence.

A single CI job often installs several hash-pinned requirement files into ONE
environment, in sequence. Each file is internally consistent and
``--require-hashes`` verifies it, but nothing checks the LAYERING: when two files
in the same job pin the same package to different versions, the last install
silently wins and the job ends on a version that satisfies only one of them.

The motivating case, measured 2026-09-23. ``ci.yml``'s ``test`` job installs five
files, and ``packaging`` walks ``26.3 -> 25.0 -> 26.2 -> 26.3`` across them. The
pytest run and both ``snakemake --dry-run`` gates sit in the ``26.2`` window,
while ``snakemake==9.27.0`` declares ``packaging<26,>=24.0`` in its own wheel
metadata - so that job runs snakemake against a version its metadata forbids.
``pip`` prints the conflict (steps without ``--no-deps`` run the check) and then
returns 0, so the step passes and the log line sits inside a collapsed section.

No gate saw it, and the reason is structural rather than an oversight:
``tools/check_hash_pins.py`` has no version logic at all, and
``tools/check_lockfile_freshness.py`` compares each ``.in`` against its OWN
compiled ``.txt`` - vertical, never horizontal. ``tools/update_dependencies.py``
compiles each file independently with no shared constraints file, which is the
root cause. There is no ``pip check`` anywhere in the repository.

Severity of the motivating case is metadata-only: the DAG dry-run exits 0 in a
venv holding exactly ``packaging 26.2`` with ``snakemake 9.27.0``, snakemake's five
``packaging`` imports are all function-local and use only ``packaging.version``, and
three of the five sit in its conda and singularity deployment modules. So this test
is regression insurance recorded while green, the same footing as
``tests/test_dependency_floors_match_pins.py``, whose docstring says as much.

**This is a RATCHET, not a clean-slate assertion.** Fourteen packages already
diverge, in five of the eight co-install groups; failing on those today would
only add a red test nobody can clear. ``KNOWN_DIVERGENCES`` records them, and
the two tests below pin the boundary in BOTH directions: a package that starts
diverging fails, and a package that stops diverging also fails, so the baseline
can only be changed deliberately. Versions are deliberately NOT recorded -
Dependabot moves them constantly, and a version bump inside an already-known
divergence is not new information, while a newly diverging PACKAGE always is.

Scope limits, stated because they bound what a pass means. Only ``-r FILE``
installs inside a ``run:`` block that mentions ``pip install`` are seen, so a
dependency installed by bare name is invisible here. A job that creates its own
virtualenv mid-run would be treated as one environment and is not modelled; no
job in this repository does that today.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml", reason="needs PyYAML to parse the workflows")

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s\\;]+)")
_REQ = re.compile(r"-r\s+(\S+\.(?:txt|lock))")

# Measured 2026-09-23 and re-derived independently 2026-09-24. Every entry is a
# package pinned to two or more versions within one (workflow, job) group. To
# change this baseline, re-run the derivation and say in the commit message which
# direction moved and why.
KNOWN_DIVERGENCES: dict[str, frozenset[str]] = {
    "ci.yml:compat": frozenset({
        "annotated-types", "certifi", "charset-normalizer", "idna",
        "packaging", "typing-extensions",
    }),
    "ci.yml:test": frozenset({
        "annotated-types", "certifi", "charset-normalizer",
        "fastjsonschema", "idna", "nbformat", "packaging", "platformdirs",
        "pygments", "rpds-py", "traitlets", "typing-extensions",
    }),
    "fuzzing.yml:fuzz": frozenset({
        "annotated-types", "certifi", "charset-normalizer", "idna",
        "packaging", "typing-extensions",
    }),
    "security.yml:pip-audit": frozenset({
        "certifi", "charset-normalizer", "filelock", "packaging", "platformdirs",
    }),
    "sestrav_verify_benchmarking.yml:verify-benchmark": frozenset({
        "annotated-types", "certifi", "charset-normalizer", "idna",
        "packaging", "typing-extensions",
    }),
}

# Anti-vacuity anchors. Without these, a parser that silently stops matching
# would make every assertion below trivially true: no groups found means no
# divergence found means green. These are the shapes that must keep resolving.
MIN_GROUPS = 6
LARGEST_GROUP = "ci.yml:test"
LARGEST_GROUP_MIN_FILES = 5


def _normalise(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def _pins(rel: str) -> dict[str, str]:
    path = REPO_ROOT / rel
    if not path.is_file():
        return {}
    pins: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _PIN.match(stripped)
        if match:
            pins.setdefault(_normalise(match.group(1)), match.group(2))
    return pins


def _requirement_files(run: str) -> list[str]:
    """Requirement files a `run:` block installs, in order, de-duplicated.

    Backslash continuations are joined first: CI writes `pip install ... \\` then
    `-r file` on the next line, and a naive per-line scan would either miss that
    or pick up a `-r` belonging to some other command.
    """
    joined = run.replace("\\\n", " ")
    found: list[str] = []
    for line in joined.splitlines():
        if "pip install" not in line:
            continue
        for match in _REQ.finditer(line):
            rel = match.group(1).lstrip("./")
            if rel not in found:
                found.append(rel)
    return found


def _coinstall_groups() -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for workflow in sorted(WORKFLOWS.glob("*.yml")):
        document = yaml.safe_load(workflow.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            continue
        for job_name, job in (document.get("jobs") or {}).items():
            if not isinstance(job, dict):
                continue
            files: list[str] = []
            for step in job.get("steps") or []:
                if not isinstance(step, dict):
                    continue
                run = step.get("run")
                if not isinstance(run, str) or "pip install" not in run:
                    continue
                for rel in _requirement_files(run):
                    if rel not in files:
                        files.append(rel)
            if len(files) >= 2:
                groups[f"{workflow.name}:{job_name}"] = files
    return groups


def _measured_divergences() -> dict[str, frozenset[str]]:
    out: dict[str, frozenset[str]] = {}
    for group, files in _coinstall_groups().items():
        versions: dict[str, set[str]] = {}
        for rel in files:
            for name, version in _pins(rel).items():
                versions.setdefault(name, set()).add(version)
        diverging = {n for n, v in versions.items() if len(v) > 1}
        if diverging:
            out[group] = frozenset(diverging)
    return out


def test_group_discovery_is_not_vacuous() -> None:
    groups = _coinstall_groups()
    assert len(groups) >= MIN_GROUPS, (
        f"only {len(groups)} co-install group(s) found, expected at least "
        f"{MIN_GROUPS}; the workflow parser has probably stopped matching, which "
        "would make every other assertion in this file vacuously true."
    )
    assert LARGEST_GROUP in groups, (
        f"{LARGEST_GROUP} was not discovered; it is the job that layers the most "
        "requirement files and is the reason this test exists."
    )
    assert len(groups[LARGEST_GROUP]) >= LARGEST_GROUP_MIN_FILES, (
        f"{LARGEST_GROUP} resolved to {len(groups[LARGEST_GROUP])} file(s), "
        f"expected at least {LARGEST_GROUP_MIN_FILES}: "
        f"{groups[LARGEST_GROUP]}"
    )


def test_no_new_cross_file_pin_divergence() -> None:
    measured = _measured_divergences()
    new: list[str] = []
    for group, packages in sorted(measured.items()):
        known = KNOWN_DIVERGENCES.get(group, frozenset())
        for package in sorted(packages - known):
            new.append(f"{group} -> {package}")
    assert not new, (
        "NEW cross-file pin divergence. Each entry names a package pinned to two "
        "different versions among the requirement files one CI job installs into "
        "the same environment, so the last install silently wins:\n  "
        + "\n  ".join(new)
        + "\nFix the pin in whichever file is behind, or add it to "
        "KNOWN_DIVERGENCES with a note saying why it is acceptable."
    )


def test_known_divergences_have_no_stale_entries() -> None:
    measured = _measured_divergences()
    stale: list[str] = []
    for group, packages in sorted(KNOWN_DIVERGENCES.items()):
        found = measured.get(group)
        if found is None:
            stale.append(f"{group} -> entire group no longer diverges")
            continue
        for package in sorted(packages - found):
            stale.append(f"{group} -> {package}")
    assert not stale, (
        "KNOWN_DIVERGENCES records divergences that no longer exist. That is "
        "good news, but the baseline must shrink deliberately rather than drift, "
        "so remove these entries:\n  " + "\n  ".join(stale)
    )
