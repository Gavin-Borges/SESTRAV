"""Tests for scripts/build_binding_matrix_v5.py.

All 8 tests run without requiring a real MHCflurry model. Tests that exercise
code paths that would call the model mock mhcflurry in sys.modules.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_binding_matrix_v5 import find_new_peptides, main, merge_matrices


# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------


def _make_dataset(tmp_path: Path, peptides: list[str], quarantined: list[bool] | None = None) -> Path:
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
    existing = pd.DataFrame(
        {"peptide": ["AAAAAAAAA", "BBBBBBBBB"], "bind_A0101": [0.1, 0.2]}
    )
    new_df = pd.DataFrame(
        {"peptide": ["BBBBBBBBB", "CCCCCCCCC"], "bind_A0101": [0.9, 0.3]}
    )
    result = merge_matrices(existing, new_df)

    assert set(result["peptide"]) == {"AAAAAAAAA", "BBBBBBBBB", "CCCCCCCCC"}
    dup_row = result.loc[result["peptide"] == "BBBBBBBBB", "bind_A0101"]
    assert dup_row.iloc[0] == pytest.approx(0.2), "first occurrence (from existing) must be kept"


# ---------------------------------------------------------------------------
# 4. merge_matrices: output is sorted alphabetically by peptide
# ---------------------------------------------------------------------------


def test_merge_matrices_sort_by_peptide() -> None:
    existing = pd.DataFrame({"peptide": ["CCCCCCCCC"], "bind_A0101": [0.3]})
    new_df = pd.DataFrame(
        {"peptide": ["AAAAAAAAA", "BBBBBBBBB"], "bind_A0101": [0.1, 0.2]}
    )
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
            "--dataset", str(dataset),
            "--existing-matrix", str(matrix),
            "--output", str(output),
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
            "--dataset", str(tmp_path / "nonexistent_dataset.csv"),
            "--existing-matrix", str(matrix),
            "--output", str(output),
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
            "--dataset", str(dataset),
            "--existing-matrix", str(tmp_path / "nonexistent_matrix.csv"),
            "--output", str(output),
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
            "--dataset", str(dataset),
            "--existing-matrix", str(matrix),
            "--output", str(output),
        ]
    )

    assert rc == 0
    assert output.exists()
    result = pd.read_csv(output)
    assert set(result["peptide"]) == set(peptides)
