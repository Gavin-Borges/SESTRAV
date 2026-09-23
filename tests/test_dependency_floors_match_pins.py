"""Every requirements.in pin must satisfy the floor pyproject.toml declares for it.

The two files serve different audiences and nothing checked that they agree.
pyproject.toml declares open floors (">=X") for a consumer installing from PyPI;
requirements.in declares exact pins ("==Y") for the reproducible CI stack. If a
floor is ever raised above its pin, a consumer and CI install different, mutually
incompatible versions and no existing gate notices.

That gap was confirmed by exhaustion at d064dbc:

  - The three tests that read pyproject.toml dependency lists all strip the
    specifier and compare NAMES only (test_predict_path_dependencies_declared.py,
    test_dev_extra_runs_the_test_suite.py,
    test_scripts_extra_declares_script_imports.py).
  - tools/check_lockfile_freshness.py is the only tool that compares VERSIONS,
    and it never opens pyproject.toml. Its ">=" handling applies to floors
    declared in environments/requirements-lock.in, not to pyproject's.
  - tools/check_hash_pins.py has no version logic at all.

At the time of writing all pins satisfied all floors, so this is regression
insurance rather than a repair. That is deliberate: it is far cheaper to install
the gate while it is green than to discover the drift from a consumer's failed
install. A test is used rather than a tools/check_*.py script so that it runs in
the existing CI pytest matrix without a new workflow.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
REQUIREMENTS_IN = PROJECT_ROOT / "requirements.in"


def _canonical(name: str) -> str:
    """PEP 503 name normalisation, so scikit_learn and PyYAML match their pins."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _pyproject_floors() -> dict[str, list[tuple[str, SpecifierSet]]]:
    """Map canonical package name -> [(where it was declared, its specifier)].

    A package may be declared in several extras (fastapi appears in three). Each
    declaration is kept so a failure can name the exact one that disagrees.

    `[build-system].requires` is read alongside `[project]` because leaving it out
    made this gate blind to the one package it most needed to see. `setuptools`
    has a floor there and ONLY there, carrying a CVE justification in its own
    comment, and a pin in `requirements.in`. A reader would reasonably assume the
    gate covered it; it did not, and the omission flattered the coverage count by
    exactly the package whose comment describes the floor-versus-pin relationship
    this module exists to police.
    """
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project = data.get("project", {})

    declarations: list[tuple[str, str]] = [
        ("project.dependencies", spec) for spec in project.get("dependencies", [])
    ]
    for extra, specs in project.get("optional-dependencies", {}).items():
        declarations.extend((f"optional-dependencies.{extra}", spec) for spec in specs)
    declarations.extend(
        ("build-system.requires", spec)
        for spec in data.get("build-system", {}).get("requires", [])
    )

    floors: dict[str, list[tuple[str, SpecifierSet]]] = {}
    for where, spec in declarations:
        req = Requirement(spec)
        floors.setdefault(_canonical(req.name), []).append((where, req.specifier))
    return floors


def _requirements_pins() -> dict[str, tuple[int, Version]]:
    """Map canonical package name -> (line number, pinned version)."""
    pins: dict[str, tuple[int, Version]] = {}
    for lineno, raw in enumerate(REQUIREMENTS_IN.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        # Drop any environment marker: "nvidia-nccl-cu12==2.30.7; platform_system == ..."
        line = line.split(";", 1)[0].strip()
        if "==" not in line:
            continue
        req = Requirement(line)
        pinned = [s for s in req.specifier if s.operator == "=="]
        if len(pinned) != 1:
            continue
        pins[_canonical(req.name)] = (lineno, Version(pinned[0].version))
    return pins


def test_both_manifests_are_readable_and_non_trivial() -> None:
    """Guard against this whole module passing vacuously on a parse failure.

    Without this, a rename or a parser change that made either loader return {}
    would turn every assertion below into a loop over nothing, and the file would
    stay green while checking absolutely nothing.
    """
    floors = _pyproject_floors()
    pins = _requirements_pins()

    assert len(floors) >= 10, f"pyproject floors look under-parsed: {sorted(floors)}"
    assert len(pins) >= 10, f"requirements.in pins look under-parsed: {sorted(pins)}"

    overlap = set(floors) & set(pins)
    assert len(overlap) >= 8, f"too few packages declared in both files: {sorted(overlap)}"


def test_every_pin_satisfies_every_floor_declared_for_it() -> None:
    floors = _pyproject_floors()
    pins = _requirements_pins()

    violations: list[str] = []
    for name, (lineno, version) in sorted(pins.items()):
        for where, specifier in floors.get(name, []):
            # prereleases=True so a pinned rc is judged against the floor rather
            # than silently excluded by packaging's default prerelease filtering.
            if not specifier.contains(version, prereleases=True):
                violations.append(
                    f"{name}=={version} (requirements.in:{lineno}) "
                    f"violates floor '{specifier}' declared in pyproject.toml [{where}]"
                )

    assert not violations, "pin/floor disagreement:\n  " + "\n  ".join(violations)


def test_a_package_declared_in_several_extras_uses_one_floor() -> None:
    """fastapi, pydantic and uvicorn are each declared in more than one extra.

    pyproject.toml's own comments say the duplicates carry identical floors on
    purpose, to keep the resolver from having to reconcile them. Pin that intent:
    a future edit that raises the floor in only one extra is a resolver conflict
    waiting to happen, and it would otherwise pass unnoticed.
    """
    inconsistent: list[str] = []
    for name, declarations in sorted(_pyproject_floors().items()):
        distinct = {str(spec) for _, spec in declarations}
        if len(distinct) > 1:
            places = ", ".join(f"{where}='{spec}'" for where, spec in declarations)
            inconsistent.append(f"{name}: {places}")

    assert not inconsistent, "one package, several different floors:\n  " + "\n  ".join(
        inconsistent
    )


@pytest.mark.parametrize(
    ("floor", "pinned", "expected"),
    [
        (">=1.3.0", "1.8.0", True),
        (">=1.3.0", "1.6.0", True),
        (">=2.13.0", "2.13.0", True),
        # A local version, e.g. the CUDA build torch==2.13.0+cu130, satisfies both
        # forms even though Version equality against the bare release is False.
        ("==2.13.0", "2.13.0+cu130", True),
        (">=2.13.0", "2.13.0+cu130", True),
        # The case this gate exists to catch: a floor raised above its pin.
        (">=2.0.0", "1.8.0", False),
        (">=3.15.1", "3.15.0", False),
    ],
)
def test_the_comparison_itself_behaves(floor: str, pinned: str, expected: bool) -> None:
    """The gate is only as good as its comparator, so exercise it directly.

    A version check that silently answered True would make the test above pass
    on any input. These cases are drawn from real pins in this repo.
    """
    assert SpecifierSet(floor).contains(Version(pinned), prereleases=True) is expected
