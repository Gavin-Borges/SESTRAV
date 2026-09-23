"""The merged-IEDB generator must emit the sidecar its artifact already carries.

PR #553 hand-corrected `data/iedb_negatives_v5_merged_provenance.json` to drop an
unresolvable `data\\iedb` entry and its Windows separators. It changed the ARTIFACT
only. The generator that writes that artifact was left emitting the old shape, so
the next `merge()` run would have silently restored both halves and nothing would
have caught it: the only tracked reference to that filename is `.gitattributes`.

These tests pin the generator to the corrected artifact, so the two cannot drift
apart again without a red test.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = REPO_ROOT / "scripts" / "merge_iedb_api_negatives.py"
ARTIFACT = REPO_ROOT / "data" / "iedb_negatives_v5_merged_provenance.json"


def _load_generator():
    """Import the generator without executing its CLI.

    It is a `scripts/` module rather than a package, and it imports `_ssl_fix`
    and `_dataset_utils` from its own directory, so the directory has to be on
    the path before the spec is executed.
    """
    import sys

    scripts_dir = str(GENERATOR.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("merge_iedb_api_negatives", GENERATOR)
    if spec is None or spec.loader is None:  # pragma: no cover - import plumbing
        pytest.skip("cannot load the generator module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sources_use_posix_separators_not_the_host_platform_s():
    """A tracked sidecar read from a clone must not carry Windows separators."""
    mod = _load_generator()
    sources = mod.provenance_sources(
        Path("data") / "iedb",
        Path("data") / "iedb_negatives_v5_merged.csv",
        net_new_api_rows=7,
    )
    assert sources == ["data/iedb", "data/iedb_negatives_v5_merged.csv"]
    for entry in sources:
        assert "\\" not in entry, f"non-portable separator in sources entry: {entry!r}"


def test_an_api_dir_that_contributed_nothing_is_not_named_as_a_source():
    """The idempotent re-merge case, which is what the tracked artifact records.

    Naming an untracked directory that supplied zero rows declares an
    unresolvable path as the source of rows it never provided.
    """
    mod = _load_generator()
    sources = mod.provenance_sources(
        Path("data") / "iedb",
        Path("data") / "iedb_negatives_v5_merged.csv",
        net_new_api_rows=0,
    )
    assert sources == ["data/iedb_negatives_v5_merged.csv"]


def test_an_api_dir_that_did_contribute_is_still_named():
    """Non-vacuity partner for the test above.

    Without this, the policy could be implemented as "never record api_dir" and
    the zero-row test would still pass while a real input went undeclared.
    """
    mod = _load_generator()
    sources = mod.provenance_sources(
        Path("data") / "iedb",
        Path("data") / "iedb_negatives_v5_merged.csv",
        net_new_api_rows=1,
    )
    assert "data/iedb" in sources


def test_the_generator_reproduces_the_tracked_artifact_s_sources():
    """The anchor: generator output must equal what the tracked sidecar holds.

    The sidecar records `net_new_api_rows: 0`, so the generator must produce
    exactly its `sources` for that case. This is the assertion that would have
    gone red had the artifact been corrected while the generator was not.
    """
    mod = _load_generator()
    recorded = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    net_new = recorded["net_new_api_rows"]

    regenerated = mod.provenance_sources(
        Path("data") / "iedb",
        Path(recorded["sources"][-1]),
        net_new_api_rows=net_new,
    )
    assert regenerated == recorded["sources"], (
        "the generator no longer emits the sources its own tracked artifact "
        f"carries: generator {regenerated!r} vs artifact {recorded['sources']!r}"
    )
