"""Declared floors must exclude releases with a fixed advisory or no CI-Python wheel.

Companion to tests/test_declared_floors_exclude_incompatible_releases.py, which
covers [project].dependencies against releases this package cannot run on. This
file covers the floors raised on 2026-10-12, in the core list, in the extras and in
[build-system].requires. A floor is what a plain `pip install .` or a downstream
consumer resolves against; the locks pin newer versions and say nothing about it.

Each excluded release below is named with its reason, which is one of four:

- A fixed advisory. The release sits inside the structured affected range of an
  OSV record (https://api.osv.dev/v1/query) that gives a fixed release, read from
  the record's ranges rather than its prose. Each release named is the highest
  non-yanked release that record covers.
- No wheel. The release publishes no CPython 3.11, 3.12 or 3.13 wheel (the CI
  Pythons) for Linux x86_64 or win_amd64, read from the PyPI Simple API's file
  list, so installing it there means building the sdist. For pydantic the wheel
  that matters is the pydantic-core release it pins exactly.
- jsonschema 4.0.1 and 4.17.3 were run on CPython 3.14.4, which requires-python
  admits, and each fails 15 of 27 tests across the three dataset-schema test
  files with TypeError: unhashable type: 'dict'; 4.18.0 passes all 27.
- fastapi 0.116.1 declares starlette<0.48.0, and 0.48.0 is the first starlette
  whose status module has HTTP_422_UNPROCESSABLE_CONTENT. That case applies only
  while code under api/ uses the name, so the test scans for it rather than
  assuming it, and stops asking for the floor if that code ever goes.

Every release named is a SENTINEL, not a range: each case asks whether one release
is admitted. A plain `>=` floor that excludes a sentinel also excludes everything
below it, but a specifier that carves out only the sentinels does not. Measured
against this file, not reasoned from it: `requests>=2.31.0,!=2.31.0,!=2.32.3,!=2.32.5`
passes every requests case below while admitting 2.32.2, which OSV lists as
affected by CVE-2024-47081 and CVE-2026-25645.
"""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
API_TREE = PROJECT_ROOT / "api"

CORE = ("project.dependencies",)
GNN = ("optional-dependencies.gnn",)
DEV = ("optional-dependencies.dev",)
DEMO = ("optional-dependencies.demo",)
# fastapi and pydantic are declared in all three, at matching floors.
FASTAPI_EXTRAS = (
    "optional-dependencies.dev",
    "optional-dependencies.api",
    "optional-dependencies.demo",
)
BUILD = ("build-system.requires",)

# (where declared, package, release every such floor must exclude, why).
EXCLUDED_RELEASES = [
    pytest.param(
        CORE,
        "requests",
        "2.31.0",
        "is affected by CVE-2024-35195 (GHSA-9wx4-h78v-vm56), fixed in 2.32.0",
        id="requests-2.31.0-cve-2024-35195",
    ),
    pytest.param(
        CORE,
        "requests",
        "2.32.3",
        "is affected by CVE-2024-47081 (GHSA-9hjg-9r4m-mvj7), fixed in 2.32.4",
        id="requests-2.32.3-cve-2024-47081",
    ),
    pytest.param(
        CORE,
        "requests",
        "2.32.5",
        "is affected by CVE-2026-25645 (GHSA-gc5v-m9x4-r6x2), fixed in 2.33.0",
        id="requests-2.32.5-cve-2026-25645",
    ),
    pytest.param(
        CORE,
        "jsonschema",
        "4.0.1",
        "raises TypeError: unhashable type: 'dict' on CPython 3.14 in jsonschema.validate",
        id="jsonschema-4.0.1-fails-on-python-3.14",
    ),
    pytest.param(
        CORE,
        "jsonschema",
        "4.17.3",
        "raises TypeError: unhashable type: 'dict' on CPython 3.14 in jsonschema.validate",
        id="jsonschema-4.17.3-fails-on-python-3.14",
    ),
    pytest.param(
        CORE,
        "biopython",
        "1.86",
        "is affected by CVE-2025-68463 (PYSEC-2026-1221 fixed 1.87; "
        "GHSA-x3vf-39hj-gxr4 last_affected 1.86)",
        id="biopython-1.86-cve-2025-68463",
    ),
    pytest.param(
        CORE,
        "PyYAML",
        "6.0",
        "publishes no CPython 3.12 or 3.13 wheel for Linux x86_64 or win_amd64",
        id="pyyaml-6.0-no-cp312-cp313-wheel",
    ),
    pytest.param(
        CORE,
        "PyYAML",
        "6.0.1",
        "publishes no CPython 3.13 wheel for Linux x86_64 or win_amd64",
        id="pyyaml-6.0.1-no-cp313-wheel",
    ),
    pytest.param(
        GNN,
        "transformers",
        "5.9.0",
        "is affected by CVE-2026-9856 (GHSA-xrqw-3rrv-vx5w), fixed in 5.10.0",
        id="transformers-5.9.0-cve-2026-9856",
    ),
    pytest.param(
        DEV,
        "pytest",
        "9.0.2",
        "is affected by CVE-2025-71176 (GHSA-6w46-j5rx-g56g), fixed in 9.0.3",
        id="pytest-9.0.2-cve-2025-71176",
    ),
    pytest.param(
        FASTAPI_EXTRAS,
        "fastapi",
        "0.109.0",
        "is affected by CVE-2024-24762 (PYSEC-2024-38), fixed in 0.109.1",
        id="fastapi-0.109.0-cve-2024-24762",
    ),
    pytest.param(
        FASTAPI_EXTRAS,
        "pydantic",
        "2.3.0",
        "is affected by CVE-2024-3772 (GHSA-mr82-8j83-vxmv), fixed in 2.4.0",
        id="pydantic-2.3.0-cve-2024-3772",
    ),
    pytest.param(
        FASTAPI_EXTRAS,
        "pydantic",
        "2.7.4",
        "pins pydantic-core==2.18.4, which publishes no CPython 3.13 wheel for "
        "Linux x86_64 or win_amd64",
        id="pydantic-2.7.4-core-has-no-cp313-wheel",
    ),
    pytest.param(
        DEMO,
        "streamlit",
        "1.36.0",
        "is affected by CVE-2024-42474 (GHSA-rxff-vr5r-8cj5), fixed in 1.37.0",
        id="streamlit-1.36.0-cve-2024-42474",
    ),
    pytest.param(
        DEMO,
        "streamlit",
        "1.53.0",
        "is affected by CVE-2026-10804 (GHSA-vqwp-45wm-r9r5), fixed in 1.53.1",
        id="streamlit-1.53.0-cve-2026-10804",
    ),
    pytest.param(
        DEMO,
        "streamlit",
        "1.53.1",
        "is affected by CVE-2026-33682 (GHSA-7p48-42j8-8846), fixed in 1.54.0",
        id="streamlit-1.53.1-cve-2026-33682",
    ),
    pytest.param(
        BUILD,
        "wheel",
        "0.45.1",
        "is affected by CVE-2026-24049 (GHSA-8rrh-rw8j-w5fx), fixed in 0.46.2",
        id="wheel-0.45.1-cve-2026-24049",
    ),
]

# The attribute api/main.py reads from fastapi's re-export of starlette.status.
STARLETTE_422_NAME = "HTTP_422_UNPROCESSABLE_CONTENT"


def _canonical(name: str) -> str:
    """PEP 503 name normalisation, so PyYAML and pyyaml match."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _declarations() -> dict[str, dict[str, SpecifierSet]]:
    """Map section -> canonical name -> specifier, for every section pyproject declares."""
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    sections: dict[str, list[str]] = {
        "project.dependencies": data["project"]["dependencies"],
        "build-system.requires": data["build-system"]["requires"],
    }
    for extra, specs in data["project"]["optional-dependencies"].items():
        sections[f"optional-dependencies.{extra}"] = specs
    out: dict[str, dict[str, SpecifierSet]] = {}
    for section, specs in sections.items():
        table: dict[str, SpecifierSet] = {}
        for spec in specs:
            req = Requirement(spec)
            name = _canonical(req.name)
            assert name not in table, f"{name} is declared twice in {section}"
            table[name] = req.specifier
        out[section] = table
    return out


def _specifier(section: str, package: str) -> SpecifierSet:
    declared = _declarations()
    assert section in declared, f"pyproject.toml has no {section}"
    name = _canonical(package)
    assert name in declared[section], f"{package} is not declared in pyproject.toml {section}"
    return declared[section][name]


def name_attribute_lines(source: str, attribute: str) -> list[int]:
    """Lines where `source` reads `<anything>.<attribute>` as an attribute.

    Parsed, not grepped, so a comment or a string holding the name does not count.
    """
    return sorted(
        node.lineno
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Attribute) and node.attr == attribute
    )


def _api_sites_using_422_content() -> list[str]:
    sites: list[str] = []
    for path in sorted(API_TREE.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        for lineno in name_attribute_lines(path.read_text(encoding="utf-8"), STARLETTE_422_NAME):
            sites.append(f"{path.relative_to(PROJECT_ROOT).as_posix()}:{lineno}")
    return sites


def test_every_section_this_file_checks_is_declared() -> None:
    """Guard against the cases below passing vacuously on a renamed extra."""
    declared = _declarations()
    for section in CORE + GNN + DEV + DEMO + FASTAPI_EXTRAS + BUILD:
        assert declared.get(section), f"pyproject.toml declares nothing in {section}"


def test_attribute_scanner_counts_code_and_ignores_comments_and_strings() -> None:
    """The fastapi case below is only as good as this detector, so pin its policy."""
    source = (
        "from fastapi import status\n"  # 1
        "a = status.HTTP_422_UNPROCESSABLE_CONTENT\n"  # 2  counted
        "# status.HTTP_422_UNPROCESSABLE_CONTENT\n"  # 3  comment, ignored
        'b = "status.HTTP_422_UNPROCESSABLE_CONTENT"\n'  # 4  string, ignored
        "c = status.HTTP_422_UNPROCESSABLE_ENTITY\n"  # 5  other name, ignored
        "d = f(x).HTTP_422_UNPROCESSABLE_CONTENT\n"  # 6  counted
    )
    assert name_attribute_lines(source, STARLETTE_422_NAME) == [2, 6]


@pytest.mark.parametrize(("sections", "package", "release", "reason"), EXCLUDED_RELEASES)
def test_floor_excludes_release(
    sections: tuple[str, ...], package: str, release: str, reason: str
) -> None:
    for section in sections:
        specifier = _specifier(section, package)
        assert not specifier.contains(release, prereleases=True), (
            f"pyproject.toml {section} declares {package}{specifier}, which admits "
            f"{package} {release}; that release {reason}"
        )


def test_fastapi_floor_admits_a_starlette_with_the_422_content_name_while_api_uses_it() -> None:
    sites = _api_sites_using_422_content()
    if not sites:
        pytest.skip(f"nothing under api/ reads {STARLETTE_422_NAME}, so no starlette floor is needed")
    for section in FASTAPI_EXTRAS:
        specifier = _specifier(section, "fastapi")
        assert not specifier.contains("0.116.1", prereleases=True), (
            f"pyproject.toml {section} declares fastapi{specifier}, which admits fastapi "
            f"0.116.1; that release declares starlette<0.48.0, and starlette added "
            f"{STARLETTE_422_NAME} in 0.48.0, which {len(sites)} site(s) read: {', '.join(sites)}"
        )
