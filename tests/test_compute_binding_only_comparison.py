from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.compute_binding_only_comparison import _load_aligned, compute_table


def _write_oof(path: Path, *, feature_mode: int, scores: list[float]) -> None:
    viruses = ["CMV"] * 7 + ["Orthopoxvirus vaccinia"]
    labels = [1, 1, 1, 1, 0, 0, 0, 0]
    origins = [None, None, None, None, "iedb_api", "iedb_api", "tested_negative", "tested_negative"]
    pd.DataFrame(
        {
            "peptide": [f"PEPTIDE{i}" for i in range(8)],
            "virus": viruses,
            "strain": [None] * 8,
            "protein": [f"PROTEIN{i}" for i in range(8)],
            "negative_origin": origins,
            "hla_allele": ["HLA-A*02:01"] * 8,
            "label": labels,
            "score": scores,
            "fold": [1, 2, 3, 4, 1, 2, 3, 4],
            "method": ["RandomForest"] * 8,
            "feature_mode": [feature_mode] * 8,
        }
    ).to_csv(path, index=False)


def test_compute_table_keeps_def_a_and_def_b_distinct(tmp_path: Path) -> None:
    mode31 = tmp_path / "mode31.csv"
    mode10 = tmp_path / "mode10.csv"
    _write_oof(mode31, feature_mode=31, scores=[0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])
    _write_oof(mode10, feature_mode=10, scores=[0.8, 0.7, 0.6, 0.5, 0.45, 0.35, 0.25, 0.15])

    result = compute_table(mode31, mode10, n_resamples=20, seed=42).set_index("partition")

    assert result.loc["pooled", ["n", "n_pos", "n_neg"]].tolist() == [8, 4, 4]
    assert result.loc["def_a", ["n", "n_pos", "n_neg"]].tolist() == [6, 4, 2]
    assert result.loc["def_b", ["n", "n_pos", "n_neg"]].tolist() == [7, 4, 3]
    assert result["valid_bootstrap_resamples"].between(1, 20).all()


def test_load_aligned_rejects_fold_drift(tmp_path: Path) -> None:
    mode31 = tmp_path / "mode31.csv"
    mode10 = tmp_path / "mode10.csv"
    scores = [0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1]
    _write_oof(mode31, feature_mode=31, scores=scores)
    _write_oof(mode10, feature_mode=10, scores=scores)
    drifted = pd.read_csv(mode10)
    drifted.loc[0, "fold"] = 5
    drifted.to_csv(mode10, index=False)

    with pytest.raises(ValueError, match="do not align row-for-row"):
        _load_aligned(mode31, mode10)


# The four tests below each pin one input guard of _load_aligned. Every input is
# built so that, if its guard were deleted, no LATER guard would raise the same
# message: a dropped "method" column is read by no other check, a mixed
# feature_mode column is not an alignment key, and the row-count test matches its
# own message because a short frame would otherwise fall through to the alignment
# guard and still raise ValueError. Before these tests, only the alignment guard
# had a test of its own (test_load_aligned_rejects_fold_drift above).


def _paired_oof(tmp_path: Path) -> tuple[Path, Path]:
    mode31 = tmp_path / "mode31.csv"
    mode10 = tmp_path / "mode10.csv"
    scores = [0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1]
    _write_oof(mode31, feature_mode=31, scores=scores)
    _write_oof(mode10, feature_mode=10, scores=scores)
    return mode31, mode10


def test_load_aligned_refuses_a_frame_missing_a_required_column(tmp_path: Path) -> None:
    """A frame without a required column is refused, naming the frame and the column."""
    mode31, mode10 = _paired_oof(tmp_path)
    pd.read_csv(mode10).drop(columns=["method"]).to_csv(mode10, index=False)

    with pytest.raises(ValueError, match=r"mode10 OOF is missing columns: \['method'\]"):
        _load_aligned(mode31, mode10)


def test_load_aligned_refuses_frames_of_different_lengths(tmp_path: Path) -> None:
    """Frames of unequal length are refused by the row-count guard itself."""
    mode31, mode10 = _paired_oof(tmp_path)
    pd.read_csv(mode10).iloc[:-1].to_csv(mode10, index=False)

    with pytest.raises(ValueError, match=r"OOF row-count mismatch: mode31=8, mode10=7"):
        _load_aligned(mode31, mode10)


def test_load_aligned_refuses_a_mode31_frame_with_another_mode(tmp_path: Path) -> None:
    """A row from another feature mode in the mode-31 frame is refused."""
    mode31, mode10 = _paired_oof(tmp_path)
    mixed = pd.read_csv(mode31)
    mixed.loc[0, "feature_mode"] = 10
    mixed.to_csv(mode31, index=False)

    with pytest.raises(ValueError, match="mode31 OOF does not contain feature_mode=31 exclusively"):
        _load_aligned(mode31, mode10)


def test_load_aligned_refuses_a_mode10_frame_with_another_mode(tmp_path: Path) -> None:
    """A row from another feature mode in the mode-10 frame is refused."""
    mode31, mode10 = _paired_oof(tmp_path)
    mixed = pd.read_csv(mode10)
    mixed.loc[0, "feature_mode"] = 31
    mixed.to_csv(mode10, index=False)

    with pytest.raises(ValueError, match="mode10 OOF does not contain feature_mode=10 exclusively"):
        _load_aligned(mode31, mode10)
