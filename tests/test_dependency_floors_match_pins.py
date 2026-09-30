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


def _requirements_pins(path: Path = REQUIREMENTS_IN) -> dict[str, list[tuple[str, Version]]]:
    """Map canonical package name -> [("<file>:<line>", pinned version)], one per pin.

    A list rather than a single pin because a package may be pinned more than once
    under different environment markers, and every one of those pins has to clear
    the floor. Keying on the name alone let a later pin replace an earlier one, so
    the earlier pin was never compared at all.

    The location carries the name of the file actually read, so a violation
    reported by _floor_violations points at that file rather than assuming it
    was requirements.in.

    Every requirement line must be exactly one '==' pin. Anything else ('===',
    '~=', a range, a bare name, a URL) raises instead of being skipped: a line
    this parser cannot read is a pin this gate cannot check.
    """
    pins: dict[str, list[tuple[str, Version]]] = {}
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        # Drop any environment marker: "nvidia-nccl-cu12==2.30.7; platform_system == ..."
        line = line.split(";", 1)[0].strip()
        req = Requirement(line)
        pinned = [s for s in req.specifier if s.operator == "=="]
        if len(pinned) != 1:
            raise ValueError(
                f"{path.name}:{lineno}: {line!r} is not a single '==' pin, so it cannot "
                "be compared against a pyproject.toml floor. Pin it with exactly one "
                "'==', or teach this parser the new form."
            )
        pins.setdefault(_canonical(req.name), []).append(
            (f"{path.name}:{lineno}", Version(pinned[0].version))
        )
    return pins


def _floor_violations(
    floors: dict[str, list[tuple[str, SpecifierSet]]],
    pins: dict[str, list[tuple[str, Version]]],
) -> list[str]:
    """One message per (pin, floor) pair where the pin falls outside the floor."""
    violations: list[str] = []
    for name, pinned in sorted(pins.items()):
        for location, version in pinned:
            for where, specifier in floors.get(name, []):
                # prereleases=True so a pinned rc is judged against the floor rather
                # than silently excluded by packaging's default prerelease filtering.
                if not specifier.contains(version, prereleases=True):
                    violations.append(
                        f"{name}=={version} ({location}) "
                        f"violates floor '{specifier}' declared in pyproject.toml [{where}]"
                    )
    return violations


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
    violations = _floor_violations(_pyproject_floors(), _requirements_pins())

    assert not violations, "pin/floor disagreement:\n  " + "\n  ".join(violations)


# A synthetic floor, so the tests below that call _floor_violations exercise the
# parser and the comparison without depending on whatever numpy floor
# pyproject.toml happens to carry.
_SYNTHETIC_FLOORS = {"numpy": [("project.dependencies", SpecifierSet(">=2.0"))]}


@pytest.mark.parametrize(
    ("lines", "bad_lineno"),
    [
        pytest.param(
            [
                'numpy==1.0.0; platform_system == "Windows"',
                'numpy==2.4.6; platform_system == "Linux"',
            ],
            1,
            id="violating-pin-first",
        ),
        pytest.param(
            [
                'numpy==2.4.6; platform_system == "Linux"',
                'numpy==1.0.0; platform_system == "Windows"',
            ],
            2,
            id="violating-pin-second",
        ),
    ],
)
def test_every_pin_of_a_package_is_compared_not_only_the_last(
    tmp_path: Path, lines: list[str], bad_lineno: int
) -> None:
    """A package pinned twice under different markers must have BOTH pins checked.

    requirements.in already pins one package under a platform marker, so a second
    marker-split pin for the same name is a realistic edit. A parser that keys on
    the name alone keeps only the LAST pin, and a violating earlier pin is never
    compared. Both orders are run so the result cannot depend on which one wins.
    """
    requirements = tmp_path / "requirements.in"
    requirements.write_text("\n".join(lines) + "\n", encoding="utf-8")

    violations = _floor_violations(_SYNTHETIC_FLOORS, _requirements_pins(requirements))

    assert len(violations) == 1, f"expected the numpy==1.0.0 pin to be reported, got {violations}"
    assert f"numpy==1.0.0 (requirements.in:{bad_lineno})" in violations[0]


def test_a_violation_names_the_file_the_pin_was_read_from(tmp_path: Path) -> None:
    """The violation must name the file the parser read, as its ValueError does.

    Any other name sends a reader to a file and line that do not hold the pin.
    """
    requirements = tmp_path / "constraints.in"
    requirements.write_text("numpy==1.0.0\n", encoding="utf-8")

    violations = _floor_violations(_SYNTHETIC_FLOORS, _requirements_pins(requirements))

    assert len(violations) == 1, f"expected one violation, got {violations}"
    assert "numpy==1.0.0 (constraints.in:1) " in violations[0], violations[0]


@pytest.mark.parametrize(
    "line",
    [
        # '==' is a substring of '===', so it passed the old parser's substring filter
        # and then matched no '==' specifier: the case that motivated this test.
        "numpy===2.4.6",
        "numpy~=2.4",
        "numpy>=2.4",
        "numpy<3",
        "numpy!=2.4.5",
        "numpy",
        "numpy==2.4.6,==2.4.7",
        "numpy @ https://example.invalid/numpy-2.4.6.tar.gz",
    ],
)
def test_a_line_that_is_not_a_single_exact_pin_is_refused(tmp_path: Path, line: str) -> None:
    """A line the parser cannot read as one '==' pin must fail, never be skipped.

    Skipping it leaves the package out of the comparison entirely, so the gate
    stays green while blind to that package: the same vacuous pass that
    test_both_manifests_are_readable_and_non_trivial guards against, one line
    at a time instead of the whole file.
    """
    requirements = tmp_path / "requirements.in"
    requirements.write_text(f"scipy==1.17.1\n{line}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"requirements\.in:2: .*not a single '==' pin"):
        _requirements_pins(requirements)


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
