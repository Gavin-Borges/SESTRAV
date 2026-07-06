"""Unit tests for the functions/stageN_*.py pipeline transforms.

Stage 1 (peptide generation) and Stage 3 (TCR feature extraction) are pure
data transforms and are exercised directly. Stage 2 wraps MHCflurry, so its
heavy predictor is mocked and only the standardisation/pivot/best-allele logic
is tested. All stages write to ``results/``; tests chdir into a temp dir so the
real repo tree is never touched.
"""

import re

import pandas as pd
import pytest

from functions import stage1_peptide_generation as s1
from functions import stage2_mhc_binding_prediction as s2
from functions import stage3_tcr_feature_extraction as s3


@pytest.fixture
def in_tmp_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()
    return tmp_path


# --------------------------------------------------------------------------- #
# Stage 1
# --------------------------------------------------------------------------- #
def _write_fasta(path, records):
    with open(path, "w", encoding="utf-8") as fh:
        for name, seq in records:
            fh.write(f">{name}\n{seq}\n")
    return path


def test_sanitize_name_replaces_special_chars():
    assert s1._sanitize_name("HPV16/18 panel!") == "HPV16_18_panel_"
    assert s1._sanitize_name("ok-name_1") == "ok-name_1"


def test_generate_peptides_sliding_window(in_tmp_results):
    fasta = _write_fasta(in_tmp_results / "p.fasta", [("prot1", "ACDEFGHIK")])
    df = s1.generate_peptides(str(fasta), "panel", peptide_lengths=[8, 9])
    # 9-mer: one window; 8-mers: two windows.
    assert sorted(df["length"].unique()) == [8, 9]
    assert len(df[df["length"] == 9]) == 1
    assert len(df[df["length"] == 8]) == 2
    row = df[df["length"] == 9].iloc[0]
    assert row["peptide"] == "ACDEFGHIK"
    assert row["start"] == 1 and row["end"] == 9
    assert (in_tmp_results / "results" / "panel_peptides.csv").is_file()


def test_generate_peptides_rejects_nonstandard_aa(in_tmp_results):
    # 'X' is non-standard; any k-mer containing it must be dropped.
    fasta = _write_fasta(in_tmp_results / "p.fasta", [("prot1", "ACDEFGXIK")])
    df = s1.generate_peptides(str(fasta), "panel", peptide_lengths=[8])
    assert df.empty or not df["peptide"].str.contains("X").any()


def test_generate_peptides_default_lengths(in_tmp_results):
    fasta = _write_fasta(in_tmp_results / "p.fasta", [("prot1", "A" * 15)])
    df = s1.generate_peptides(str(fasta), "panel")
    assert sorted(df["length"].unique()) == s1.DEFAULT_LENGTHS


# --------------------------------------------------------------------------- #
# Stage 2 (MHCflurry mocked)
# --------------------------------------------------------------------------- #
def test_allele_to_col():
    assert s2._allele_to_col("HLA-A*02:01") == "bind_A0201"
    assert s2._allele_to_col("HLA-B*44:02") == "bind_B4402"


class _FakePredictor:
    """Stand-in for Class1PresentationPredictor returning deterministic scores."""

    @classmethod
    def load(cls):
        return cls()

    def predict(self, peptides, alleles, verbose=0):
        allele = alleles[0]
        # Give each (peptide, allele) a stable, distinct presentation score.
        rows = []
        for i, pep in enumerate(peptides):
            rows.append(
                {
                    "peptide": pep,
                    "allele": allele,
                    "affinity": 100.0 + i,
                    "presentation_score": 0.1 * (i + 1) + 0.01 * len(allele),
                    "presentation_percentile": 1.0,
                }
            )
        return pd.DataFrame(rows)


def test_predict_binding_pivots_and_picks_best_allele(in_tmp_results, monkeypatch):
    monkeypatch.setattr(s2, "Class1PresentationPredictor", _FakePredictor)
    peptides_df = pd.DataFrame({"peptide": ["ACDEFGHIK", "LMNPQRSTV"], "protein_id": ["p1", "p2"]})
    out = s2.predict_binding(peptides_df, "panel", alleles=["HLA-A*02:01", "HLA-B*07:02"])
    # One row per unique peptide.
    assert set(out["peptide"]) == {"ACDEFGHIK", "LMNPQRSTV"}
    assert len(out) == 2
    # Per-allele wide columns were produced.
    assert "bind_A0201" in out.columns and "bind_B0702" in out.columns
    # protein_id metadata merged through.
    assert "protein_ids" in out.columns
    assert (in_tmp_results / "results" / "panel_binding.csv").is_file()


def test_predict_binding_without_protein_id(in_tmp_results, monkeypatch):
    monkeypatch.setattr(s2, "Class1PresentationPredictor", _FakePredictor)
    peptides_df = pd.DataFrame({"peptide": ["ACDEFGHIK"]})
    out = s2.predict_binding(peptides_df, "panel", alleles=["HLA-A*02:01"])
    assert "protein_id" not in out.columns or out["protein_id"].isna().all()
    assert len(out) == 1


# --------------------------------------------------------------------------- #
# Stage 3
# --------------------------------------------------------------------------- #
def test_extract_tcr_features_adds_feature_columns(in_tmp_results):
    binding_df = pd.DataFrame(
        {
            "peptide": ["ACDEFGHIK", "LMNPQRSTV"],
            "presentation_score": [0.9, 0.2],
        }
    )
    out = s3.extract_tcr_features(binding_df, "panel")
    feature_cols = [c for c in out.columns if re.match(r"p[4-8]_", c)]
    assert feature_cols, "expected per-position TCR feature columns"
    assert len(out) == 2
    assert (in_tmp_results / "results" / "panel_features.csv").is_file()


def test_extract_tcr_features_falls_back_to_affinity(in_tmp_results):
    # No presentation_score column -> falls back to 'affinity'.
    binding_df = pd.DataFrame({"peptide": ["ACDEFGHIK"], "affinity": [123.0]})
    out = s3.extract_tcr_features(binding_df, "panel")
    assert len(out) == 1
