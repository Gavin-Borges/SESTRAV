"""Tests for scripts/build_binding_matrix_v5.py.

All 16 tests run without requiring a real MHCflurry model. Tests that exercise
code paths that would call the model mock mhcflurry in sys.modules.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_binding_matrix_v5 import find_new_peptides, main, merge_matrices

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------


def _make_dataset(
    tmp_path: Path, peptides: list[str], quarantined: list[bool] | None = None
) -> Path:
    data: dict[str, list] = {"peptide": peptides}
    if quarantined is not None:
        data["is_quarantined"] = quarantined
    p = tmp_path / "dataset.csv"
    pd.DataFrame(data).to_csv(p, index=False)
    return p


def _make_matrix(tmp_path: Path, peptides: list[str], filename: str = "matrix.csv") -> Path:
    data = {
        "peptide": peptides,
        "bind_A0101": [0.1] * len(peptides),
        "bind_A0201": [0.2] * len(peptides),
    }
    p = tmp_path / filename
    pd.DataFrame(data).to_csv(p, index=False)
    return p


@pytest.fixture()
def mock_mhcflurry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch mhcflurry into sys.modules so the lazy import in main() succeeds
    without a real model installation."""
    from unittest.mock import MagicMock

    def _predict(peptides: list[str], alleles: list[str], verbose: bool = False) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "peptide": list(peptides),
                "presentation_score": [0.5] * len(peptides),
            }
        )

    mock_predictor = MagicMock()
    mock_predictor.predict.side_effect = _predict

    mock_module = MagicMock()
    mock_module.Class1PresentationPredictor.load.return_value = mock_predictor

    monkeypatch.setitem(sys.modules, "mhcflurry", mock_module)


# ---------------------------------------------------------------------------
# 1. find_new_peptides: basic set difference
# ---------------------------------------------------------------------------


def test_find_new_peptides_returns_set_difference() -> None:
    active = {"AAAAAAAAA", "BBBBBBBBB", "CCCCCCCCC"}
    existing = {"AAAAAAAAA"}
    result = find_new_peptides(active, existing)
    assert result == {"BBBBBBBBB", "CCCCCCCCC"}


# ---------------------------------------------------------------------------
# 2. find_new_peptides: empty when all active peptides are covered
# ---------------------------------------------------------------------------


def test_find_new_peptides_empty_when_all_covered() -> None:
    active = {"AAAAAAAAA", "BBBBBBBBB"}
    existing = {"AAAAAAAAA", "BBBBBBBBB", "CCCCCCCCC"}
    result = find_new_peptides(active, existing)
    assert result == set()


# ---------------------------------------------------------------------------
# 3. merge_matrices: concat and deduplicate on peptide column
# ---------------------------------------------------------------------------


def test_merge_matrices_concat_and_dedup() -> None:
    existing = pd.DataFrame({"peptide": ["AAAAAAAAA", "BBBBBBBBB"], "bind_A0101": [0.1, 0.2]})
    new_df = pd.DataFrame({"peptide": ["BBBBBBBBB", "CCCCCCCCC"], "bind_A0101": [0.9, 0.3]})
    result = merge_matrices(existing, new_df)

    assert set(result["peptide"]) == {"AAAAAAAAA", "BBBBBBBBB", "CCCCCCCCC"}
    dup_row = result.loc[result["peptide"] == "BBBBBBBBB", "bind_A0101"]
    assert dup_row.iloc[0] == pytest.approx(0.2), "first occurrence (from existing) must be kept"


# ---------------------------------------------------------------------------
# 4. merge_matrices: output is sorted alphabetically by peptide
# ---------------------------------------------------------------------------


def test_merge_matrices_sort_by_peptide() -> None:
    existing = pd.DataFrame({"peptide": ["CCCCCCCCC"], "bind_A0101": [0.3]})
    new_df = pd.DataFrame({"peptide": ["AAAAAAAAA", "BBBBBBBBB"], "bind_A0101": [0.1, 0.2]})
    result = merge_matrices(existing, new_df)
    assert list(result["peptide"]) == ["AAAAAAAAA", "BBBBBBBBB", "CCCCCCCCC"]


# ---------------------------------------------------------------------------
# 5. main --dry-run: exits 0 and writes no output file
# ---------------------------------------------------------------------------


def test_main_dry_run_no_files_written(tmp_path: Path, mock_mhcflurry: None) -> None:
    dataset = _make_dataset(tmp_path, ["AAAAAAAAA", "BBBBBBBBB", "CCCCCCCCC"])
    matrix = _make_matrix(tmp_path, ["AAAAAAAAA"])
    output = tmp_path / "v5_matrix.csv"

    rc = main(
        [
            "--dataset",
            str(dataset),
            "--existing-matrix",
            str(matrix),
            "--output",
            str(output),
            "--dry-run",
        ]
    )

    assert rc == 0
    assert not output.exists()


# ---------------------------------------------------------------------------
# 6. main: missing --dataset returns 1
# ---------------------------------------------------------------------------


def test_main_missing_dataset_exits_1(tmp_path: Path) -> None:
    matrix = _make_matrix(tmp_path, ["AAAAAAAAA"])
    output = tmp_path / "out.csv"

    rc = main(
        [
            "--dataset",
            str(tmp_path / "nonexistent_dataset.csv"),
            "--existing-matrix",
            str(matrix),
            "--output",
            str(output),
        ]
    )

    assert rc == 1


# ---------------------------------------------------------------------------
# 7. main: missing --existing-matrix returns 1
# ---------------------------------------------------------------------------


def test_main_missing_existing_matrix_exits_1(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path, ["AAAAAAAAA"])
    output = tmp_path / "out.csv"

    rc = main(
        [
            "--dataset",
            str(dataset),
            "--existing-matrix",
            str(tmp_path / "nonexistent_matrix.csv"),
            "--output",
            str(output),
        ]
    )

    assert rc == 1


# ---------------------------------------------------------------------------
# 8. main: when active peptides are a subset of existing, copies existing
# ---------------------------------------------------------------------------


def test_main_no_new_peptides_copies_existing(tmp_path: Path, mock_mhcflurry: None) -> None:
    peptides = ["AAAAAAAAA", "BBBBBBBBB", "CCCCCCCCC"]
    dataset = _make_dataset(tmp_path, peptides)
    matrix = _make_matrix(tmp_path, peptides)
    output = tmp_path / "v5_matrix.csv"

    rc = main(
        [
            "--dataset",
            str(dataset),
            "--existing-matrix",
            str(matrix),
            "--output",
            str(output),
        ]
    )

    assert rc == 0
    assert output.exists()
    result = pd.read_csv(output)
    assert set(result["peptide"]) == set(peptides)


# ---------------------------------------------------------------------------
# 9. The provenance sidecar's key set is pinned
# ---------------------------------------------------------------------------


# Until this test existed the sidecar's shape was asserted nowhere, so a key could
# be dropped or renamed without any gate noticing. Asserted with == rather than a
# subset check, deliberately: a NEW key must come here and be described, the same
# way tests/test_binding_coverage_report.py pins its call-site count exactly.
EXPECTED_PROVENANCE_KEYS = {
    "timestamp",
    "git_sha",
    "source_dataset",
    "source_dataset_sha256",
    "existing_matrix",
    "new_peptide_count",
    "total_peptide_count",
    "active_peptide_count",
    "already_covered_count",
    "existing_matrix_coverage_of_active",
    "alleles",
}


def _build(tmp_path: Path, dataset_peps: list[str], matrix_peps: list[str]) -> dict:
    """Run a full build and return the parsed provenance sidecar."""
    dataset = _make_dataset(tmp_path, dataset_peps)
    matrix = _make_matrix(tmp_path, matrix_peps)
    output = tmp_path / "v5_matrix.csv"

    rc = main(
        [
            "--dataset",
            str(dataset),
            "--existing-matrix",
            str(matrix),
            "--output",
            str(output),
        ]
    )
    assert rc == 0

    sidecar = output.with_suffix(".provenance.json")
    assert sidecar.exists(), f"no provenance sidecar written at {sidecar}"
    return json.loads(sidecar.read_text(encoding="utf-8"))


def test_provenance_sidecar_carries_exactly_the_expected_keys(
    tmp_path: Path, mock_mhcflurry: None
) -> None:
    prov = _build(tmp_path, ["AAAAAAAAA", "BBBBBBBBB"], ["AAAAAAAAA"])
    assert set(prov) == EXPECTED_PROVENANCE_KEYS


# ---------------------------------------------------------------------------
# 10. The coverage numerator the build computes is PERSISTED, not discarded
# ---------------------------------------------------------------------------


def test_provenance_records_build_time_coverage(tmp_path: Path, mock_mhcflurry: None) -> None:
    """Every count here must be DISTINCT, or the test cannot tell them apart.

    The obvious fixture (matrix is a subset of the dataset) makes
    active == total == len(merged) and already_covered == len(existing_peps), so
    transposing those arguments still passes. A mutation run confirmed that: three
    one-line wrong implementations passed the first version of this test.

    So the matrix here carries two peptides the dataset does NOT contain. That
    separates all four quantities:
        active_peptide_count   = 4   (dataset, quarantine-filtered)
        already_covered_count  = 2   (the overlap)
        len(existing_peps)     = 4   (matrix rows, != already_covered)
        new_peptide_count      = 2
        total_peptide_count    = 6   (merged union, != active)
    """
    dataset_peps = ["AAAAAAAAA", "BBBBBBBBB", "CCCCCCCCC", "DDDDDDDDD"]
    matrix_peps = ["AAAAAAAAA", "BBBBBBBBB", "EEEEEEEEE", "FFFFFFFFF"]

    prov = _build(tmp_path, dataset_peps, matrix_peps)

    assert prov["active_peptide_count"] == 4
    assert prov["already_covered_count"] == 2
    assert prov["existing_matrix_coverage_of_active"] == 0.5
    assert prov["new_peptide_count"] == 2
    assert prov["total_peptide_count"] == 6


def test_coverage_counts_exclude_quarantined_rows(
    tmp_path: Path, mock_mhcflurry: None
) -> None:
    """active_peptide_count must follow the quarantine filter, not the raw row count.

    Without this, recording len(ds) instead of len(active_peps) would pass every
    other test in this file, since no other fixture quarantines anything.
    """
    dataset = _make_dataset(
        tmp_path,
        ["AAAAAAAAA", "BBBBBBBBB", "CCCCCCCCC", "DDDDDDDDD"],
        quarantined=[False, False, True, True],
    )
    matrix = _make_matrix(tmp_path, ["AAAAAAAAA"])
    output = tmp_path / "v5_matrix.csv"

    rc = main(
        [
            "--dataset",
            str(dataset),
            "--existing-matrix",
            str(matrix),
            "--output",
            str(output),
        ]
    )
    assert rc == 0

    prov = json.loads(output.with_suffix(".provenance.json").read_text(encoding="utf-8"))
    assert prov["active_peptide_count"] == 2, "quarantined rows must not be counted"
    assert prov["already_covered_count"] == 1
    assert prov["existing_matrix_coverage_of_active"] == 0.5


def test_no_new_peptides_path_also_records_coverage(
    tmp_path: Path, mock_mhcflurry: None
) -> None:
    """The early-return branch writes its own sidecar and is easy to leave unwired.

    The matrix is a strict SUPERSET of the dataset so the branch is taken while
    active_peptide_count (2) still differs from len(existing_df) (3). Using an
    equal set here would let the early return record the matrix row count and pass.
    """
    prov = _build(tmp_path, ["AAAAAAAAA", "BBBBBBBBB"], ["AAAAAAAAA", "BBBBBBBBB", "CCCCCCCCC"])

    assert prov["new_peptide_count"] == 0
    assert prov["total_peptide_count"] == 3
    assert prov["active_peptide_count"] == 2
    assert prov["already_covered_count"] == 2
    assert prov["existing_matrix_coverage_of_active"] == 1.0


# ---------------------------------------------------------------------------
# 11. The source digest is what lets a later reader detect corpus drift
# ---------------------------------------------------------------------------


def test_source_dataset_sha256_matches_the_file_it_names(
    tmp_path: Path, mock_mhcflurry: None
) -> None:
    dataset_peps = ["AAAAAAAAA", "BBBBBBBBB"]
    dataset = _make_dataset(tmp_path, dataset_peps)
    matrix = _make_matrix(tmp_path, ["AAAAAAAAA"])
    output = tmp_path / "v5_matrix.csv"

    rc = main(
        [
            "--dataset",
            str(dataset),
            "--existing-matrix",
            str(matrix),
            "--output",
            str(output),
        ]
    )
    assert rc == 0

    prov = json.loads(output.with_suffix(".provenance.json").read_text(encoding="utf-8"))
    expected = hashlib.sha256(dataset.read_bytes()).hexdigest()
    assert prov["source_dataset_sha256"] == expected


def test_source_dataset_sha256_changes_when_the_corpus_changes(
    tmp_path: Path, mock_mhcflurry: None
) -> None:
    """A corpus that changes under a FIXED path must still be detected.

    Both builds use the same directory and the same dataset filename, so the digest
    has to come from the file's CONTENT. An earlier version of this test built into
    two different tmp directories, which a digest of the path STRING would also have
    passed - measured, not assumed.

    The drift this guards against is real for the shipped artifact: the tracked v5
    matrix was built 2026-07-04 (sidecar timestamp 2026-07-04T02:35:56Z) and the
    corpus was committed again about thirteen hours later, then moved five more
    times - six corpus commits after the build under default history simplification,
    nine for the same range under --full-history. What the sidecar could NOT pin was
    an uncommitted edit, and it recorded git_sha as a 7-character abbreviation.
    """
    matrix = _make_matrix(tmp_path, ["AAAAAAAAA"])
    output = tmp_path / "v5_matrix.csv"
    sidecar = output.with_suffix(".provenance.json")

    def _run(peptides: list[str]) -> str:
        dataset = _make_dataset(tmp_path, peptides)  # same path every time
        rc = main(
            [
                "--dataset",
                str(dataset),
                "--existing-matrix",
                str(matrix),
                "--output",
                str(output),
            ]
        )
        assert rc == 0
        return json.loads(sidecar.read_text(encoding="utf-8"))["source_dataset_sha256"]

    first = _run(["AAAAAAAAA", "BBBBBBBBB"])
    second = _run(["AAAAAAAAA", "BBBBBBBBB", "CCCCCCCCC"])

    assert first != second, "a changed corpus at the same path must change the digest"


def test_provenance_values_are_actually_populated(
    tmp_path: Path, mock_mhcflurry: None
) -> None:
    """The key-set test above passes even if every new value is None.

    That was measured: hardcoding the new fields to None satisfied the key-set
    assertion. This pins the types and rules out a null-filled sidecar.
    """
    prov = _build(tmp_path, ["AAAAAAAAA", "BBBBBBBBB"], ["AAAAAAAAA"])

    assert isinstance(prov["source_dataset_sha256"], str)
    assert len(prov["source_dataset_sha256"]) == 64
    assert isinstance(prov["active_peptide_count"], int)
    assert isinstance(prov["already_covered_count"], int)
    assert isinstance(prov["existing_matrix_coverage_of_active"], float)
    assert prov["source_dataset"].endswith("dataset.csv")


# ---------------------------------------------------------------------------
# 16. The digest is resolvable BY THE GATE, not merely present
# ---------------------------------------------------------------------------


def test_digest_is_resolvable_by_check_digest_portability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mock_mhcflurry: None
) -> None:
    """Run the real gate's own pairing logic against the real payload shape.

    This exists because a comment claiming the sidecar avoids that gate's MISSING
    bucket was FALSE while it was written. `_paired_path` resolves candidates in
    the order [stem, stem_path, stem_file] with next(), so it reads
    "source_dataset" FIRST; an earlier version left a bare filename there and
    added "source_dataset_path" beside it, which the gate never reached. The
    digest landed in MISSING, and --strict does not fail on MISSING, so it failed
    silently.

    THE FIXTURE IS THE LOAD-BEARING PART. A first version of this test put the
    dataset directly under tmp_path, which is OUTSIDE the real PROJECT_ROOT, so
    the writer's `relative_to` raised and it fell back to the bare name anyway.
    The assertion was then trivially true and a mutation reintroducing the bare
    filename SURVIVED it, measured. PROJECT_ROOT is therefore repointed at
    tmp_path and the dataset placed in a SUBDIRECTORY, so the repo-relative form
    ("inputs/dataset.csv") and the bare name ("dataset.csv") differ and the
    regression is detectable.
    """
    import importlib.util

    import scripts.build_binding_matrix_v5 as builder

    gate_path = PROJECT_ROOT / "scripts" / "check_digest_portability.py"
    spec = importlib.util.spec_from_file_location("_cdp_under_test", gate_path)
    assert spec is not None and spec.loader is not None
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)

    monkeypatch.setattr(builder, "PROJECT_ROOT", tmp_path)

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    dataset = _make_dataset(inputs, ["AAAAAAAAA", "BBBBBBBBB"])
    matrix = _make_matrix(tmp_path, ["AAAAAAAAA"])
    output = tmp_path / "v5_matrix.csv"

    rc = main(
        [
            "--dataset",
            str(dataset),
            "--existing-matrix",
            str(matrix),
            "--output",
            str(output),
        ]
    )
    assert rc == 0

    prov = json.loads(output.with_suffix(".provenance.json").read_text(encoding="utf-8"))

    # The recorded value must be the repo-relative path, NOT the bare basename.
    assert prov["source_dataset"] == "inputs/dataset.csv", (
        f"source_dataset is {prov['source_dataset']!r}; a bare basename here "
        "resolves against no tracked path and sends the digest to the gate's "
        "MISSING bucket, which --strict does not fail on."
    )

    paired = gate._paired_path(prov, "source_dataset_sha256")
    assert paired == "inputs/dataset.csv", (
        f"the gate paired the digest with {paired!r}. Check _paired_path's "
        "candidate ORDER [stem, stem_path, stem_file] before changing keys."
    )
