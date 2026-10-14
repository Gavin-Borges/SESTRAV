"""The core floors in pyproject.toml must exclude releases known not to work here.

A floor is what a plain `pip install .` or a downstream consumer resolves against;
the locks pin far newer versions and say nothing about it. So a floor that admits
a release this package cannot run on is invisible to every lock-driven gate:
tests/test_dependency_floors_match_pins.py only checks that each pin clears its
floor, never what the floor itself lets in.

Each excluded release below is named with the reason it cannot serve this
package, read from its own published metadata or wheel:

- numpy 1.26.4, but only while packaged code reaches `numpy._core` as an
  attribute of the numpy module. That release does ship `numpy/_core/`, but its
  `__init__.py` is a docstring-only stub that imports none of its submodules, so
  the chain resolves only once something has imported `numpy._core.multiarray`
  by name, which neither `import numpy` nor `import torch` does. Which attribute
  is reported missing depends on the process: `numpy._core` itself in a bare
  one, and `.multiarray` once `numpy._core` but not its multiarray submodule has
  been imported, which is what `import torch` leaves behind. All four sites
  import torch before the lookup. The test scans for the code rather than
  assuming it, so it stops demanding numpy 2 if that code ever goes.
- pandas 2.2.1, scikit-learn 1.4.1.post1, scipy 1.12.0 and matplotlib 3.8.3 each
  cap numpy below 2.0 in their metadata, so none installs beside a numpy 2 floor.
- scikit-learn 1.4.2 is affected by CVE-2024-5206 (GHSA-jw8x-6495-233v), which
  OSV lists as fixed in 1.5.0.
- Every mhcflurry 2.1.x release requires TensorFlow, and `import mhcflurry`
  reaches a module-level `from pkg_resources import ...` in its downloads
  module, which setuptools has not shipped since 82.0.0. That series is 2.1.0,
  2.1.1, 2.1.2, 2.1.4 and 2.1.5; there is no 2.1.3. Only its two ends are named
  below, as sentinels, and naming sentinels is weaker than excluding the series:
  a floor admitting only releases between them, such as `>=2.1.2,<2.1.5`, passes
  this test. That is measured against this file, not reasoned from it.
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
# The three trees [tool.setuptools.packages.find] ships (sestrav*, src*, functions*).
PACKAGED_TREES = ("sestrav", "src", "functions")

# (package, release the core floor must exclude, why).
EXCLUDED_RELEASES = [
    pytest.param(
        "pandas",
        "2.2.1",
        "declares numpy<2 for every Python version",
        id="pandas-2.2.1-caps-numpy-below-2",
    ),
    pytest.param(
        "scikit-learn",
        "1.4.1.post1",
        "declares numpy<2.0",
        id="scikit-learn-1.4.1.post1-caps-numpy-below-2",
    ),
    pytest.param(
        "scikit-learn",
        "1.4.2",
        "is affected by CVE-2024-5206 (GHSA-jw8x-6495-233v), fixed in 1.5.0",
        id="scikit-learn-1.4.2-cve-2024-5206",
    ),
    pytest.param(
        "scipy",
        "1.12.0",
        "declares numpy<1.29.0",
        id="scipy-1.12.0-caps-numpy-below-2",
    ),
    pytest.param(
        "matplotlib",
        "3.8.3",
        "declares numpy<2",
        id="matplotlib-3.8.3-caps-numpy-below-2",
    ),
    pytest.param(
        "mhcflurry",
        "2.1.0",
        "requires TensorFlow and imports pkg_resources, absent from setuptools>=82.0.0",
        id="mhcflurry-2.1.0-needs-pkg-resources",
    ),
    pytest.param(
        "mhcflurry",
        "2.1.5",
        "requires TensorFlow and imports pkg_resources, absent from setuptools>=82.0.0",
        id="mhcflurry-2.1.5-needs-pkg-resources",
    ),
]


def _canonical(name: str) -> str:
    """PEP 503 name normalisation, so scikit_learn and scikit-learn match."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _core_specifiers() -> dict[str, SpecifierSet]:
    """Map canonical name -> specifier for every [project].dependencies entry."""
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    specifiers: dict[str, SpecifierSet] = {}
    for spec in data["project"]["dependencies"]:
        req = Requirement(spec)
        name = _canonical(req.name)
        assert name not in specifiers, f"{name} is declared twice in [project].dependencies"
        specifiers[name] = req.specifier
    return specifiers


def numpy_core_attribute_lines(source: str) -> list[int]:
    """Lines where `source` reaches numpy._core as an attribute of the numpy module.

    Two spellings count: `<alias>._core` and `getattr(<alias>, "_core")`, where
    <alias> is any name an `import numpy` or `import numpy as <alias>` statement
    anywhere in the module binds (function-local imports included). Comments and
    strings never count, because the source is parsed rather than grepped.

    Importing the submodule (`import numpy._core.multiarray`,
    `from numpy._core import ...`) deliberately does NOT count: numpy 1.26.4 ships
    a `numpy/_core/` shim, so that spelling imports there and is no evidence that
    numpy 2 is required.
    """
    tree = ast.parse(source)
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "numpy":
                    aliases.add(alias.asname or "numpy")
    lines: list[int] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "_core"
            and isinstance(node.value, ast.Name)
            and node.value.id in aliases
        ):
            lines.append(node.lineno)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in aliases
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == "_core"
        ):
            lines.append(node.lineno)
    return sorted(lines)


def _packaged_numpy_core_sites() -> list[str]:
    sites: list[str] = []
    for tree_name in PACKAGED_TREES:
        root = PROJECT_ROOT / tree_name
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            for lineno in numpy_core_attribute_lines(path.read_text(encoding="utf-8")):
                sites.append(f"{path.relative_to(PROJECT_ROOT).as_posix()}:{lineno}")
    return sites


def test_scanner_finds_both_spellings_and_ignores_the_rest() -> None:
    """The scan below is only as good as this detector, so pin its policy.

    Without this, a detector that silently found nothing would turn the numpy
    check into a skip and nobody would notice.
    """
    source = (
        "import numpy as np\n"  # 1
        "import numpy\n"  # 2
        "def load():\n"  # 3
        "    import numpy as xp\n"  # 4
        "    a = np._core.multiarray.scalar\n"  # 5  attribute, top-level alias
        '    b = getattr(getattr(np, "_core"), "multiarray")\n'  # 6  getattr form
        "    c = xp._core\n"  # 7  function-local alias
        "    d = numpy._core\n"  # 8  bare module name
        "    # np._core in a comment\n"  # 9  ignored
        '    e = "np._core in a string"\n'  # 10 ignored
        "    f = other._core\n"  # 11 not numpy, ignored
        "    import numpy._core.multiarray\n"  # 12 submodule import, ignored
        "    from numpy._core import multiarray\n"  # 13 submodule import, ignored
    )
    assert numpy_core_attribute_lines(source) == [5, 6, 7, 8]


def test_numpy_floor_excludes_numpy_1_while_packaged_code_reaches_numpy_core() -> None:
    sites = _packaged_numpy_core_sites()
    if not sites:
        pytest.skip(
            "no packaged module reaches numpy._core as an attribute, so nothing "
            "here requires numpy 2"
        )
    numpy_spec = _core_specifiers()["numpy"]
    assert not numpy_spec.contains("1.26.4", prereleases=True), (
        f"[project].dependencies declares numpy{numpy_spec}, which admits numpy 1.26.4, "
        f"but {len(sites)} packaged site(s) reach numpy._core as an attribute, which "
        f"numpy 1.26.4 does not set in a fresh process: {', '.join(sites)}"
    )


@pytest.mark.parametrize(("package", "release", "reason"), EXCLUDED_RELEASES)
def test_core_floor_excludes_release(package: str, release: str, reason: str) -> None:
    specifier = _core_specifiers()[_canonical(package)]
    assert not specifier.contains(release, prereleases=True), (
        f"[project].dependencies declares {package}{specifier}, which admits "
        f"{package} {release}; that release {reason}"
    )
