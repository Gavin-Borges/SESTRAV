"""PR-time guard for non-Python files required by the installed evaluator.

``src/verify/structural_gnn.py`` reads ``mhc_pseudo_sequences.json`` beside
itself at runtime. Only ``pyproject.toml``'s ``[tool.setuptools.package-data]``
entry puts that file (and ``targets.json``) into the wheel and sdist; delete
the entry and both archives lose them, while a test that reads the source tree
(``tests/test_hla_pseudo_sequence_tables.py``) still passes.
The tag-triggered release workflow re-checks the built archives, but that runs
only on a ``v*`` tag, so this test is the check that fires on a pull request.
"""

import glob
import os
from pathlib import Path
import tomllib


REPO_ROOT = Path(__file__).parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
PACKAGE = "src.verify"
PACKAGE_DIR = REPO_ROOT / "src" / "verify"
REQUIRED = {
    "src/verify/mhc_pseudo_sequences.json",
    "src/verify/targets.json",
}


def test_required_verify_data_is_declared_as_package_data():
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    package_data = data.get("tool", {}).get("setuptools", {}).get("package-data", {})
    # Resolve the patterns the way setuptools does: the "*" (every package) key
    # plus the package's own key, each globbed recursively RELATIVE TO THE
    # PACKAGE DIRECTORY against the real files (build_py.find_data_files). A
    # right-anchored name match would accept "src/verify/*.json", which makes
    # setuptools search src/verify/src/verify/ and ship neither file.
    patterns = [*package_data.get("*", []), *package_data.get(PACKAGE, [])]
    matched = {
        Path(hit).relative_to(REPO_ROOT).as_posix()
        for pattern in patterns
        for hit in glob.glob(os.path.join(PACKAGE_DIR, pattern), recursive=True)
        if os.path.isfile(hit)
    } & REQUIRED
    print(f"source-tree guard matched {len(matched)}/{len(REQUIRED)} required members")
    assert matched == REQUIRED, (
        f"pyproject.toml [tool.setuptools.package-data] does not ship these files "
        f"from {PACKAGE}: {sorted(REQUIRED - matched)}"
    )
