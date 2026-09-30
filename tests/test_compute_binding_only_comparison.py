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
