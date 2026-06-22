"""Tests for scripts/sample_tsnadb_cohort.py - TSNAdb cross-domain cohort sampling."""

import os
import sys

import pandas as pd

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts")
sys.path.insert(0, SCRIPTS_DIR)
from sample_tsnadb_cohort import DEEP_IMM_MIN, MHCF_RANK_MAX, build_cohort

SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data", "immunogenicity_dataset_v4_schema.json",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_TSV_COLS = [
    "Type", "Tissue", "Mutation", "HLA", "Peptide",
    "Deep_bind", "Deep_imm", "MHCf_rank (%)", "Net4_aff (nM)", "Net4_rank (%)",
]


def _make_raw(tmp_path, rows: list[list]) -> str:
    df = pd.DataFrame(rows, columns=_TSV_COLS)
    path = tmp_path / "snv.txt"
    df.to_csv(str(path), sep="\t", index=False)
    return str(path)


def _row(hla="HLA-A02:01", peptide="LLWTLVVLL", deep_imm=0.9, mhcf=0.5):
    return ["SNV", "Lung", "GENE_V1A_9", hla, peptide, 0.99, deep_imm, mhcf, 50.0, 0.2]


# ---------------------------------------------------------------------------
# Filter tests
# ---------------------------------------------------------------------------
def test_non_canonical_allele_excluded(tmp_path):
    rows = [_row(hla="HLA-A02:01"), _row(hla="HLA-C07:02", peptide="GILGFVFTL")]
    raw = _make_raw(tmp_path, rows)
    df = build_cohort(raw, sample_n=100, seed=42)
    assert len(df) == 1
    assert df.iloc[0]["hla_allele"] == "HLA-A*02:01"


def test_low_deep_imm_excluded(tmp_path):
    rows = [_row(deep_imm=0.9), _row(deep_imm=DEEP_IMM_MIN - 0.01, peptide="GILGFVFTL")]
    raw = _make_raw(tmp_path, rows)
    df = build_cohort(raw, sample_n=100, seed=42)
    assert len(df) == 1
    assert df.iloc[0]["peptide"] == "LLWTLVVLL"


def test_high_mhcf_rank_excluded(tmp_path):
    rows = [_row(mhcf=1.0), _row(mhcf=MHCF_RANK_MAX + 0.01, peptide="GILGFVFTL")]
    raw = _make_raw(tmp_path, rows)
    df = build_cohort(raw, sample_n=100, seed=42)
    assert len(df) == 1


def test_short_peptide_excluded(tmp_path):
    rows = [_row(peptide="LLWTLVVLL"), _row(peptide="ACDE")]  # 4-mer invalid
    raw = _make_raw(tmp_path, rows)
    df = build_cohort(raw, sample_n=100, seed=42)
    assert len(df) == 1
    assert len(df.iloc[0]["peptide"]) == 9


def test_long_peptide_excluded(tmp_path):
    rows = [_row(peptide="LLWTLVVLL"), _row(peptide="LLWTLVVLLABC")]  # 12-mer
    raw = _make_raw(tmp_path, rows)
    df = build_cohort(raw, sample_n=100, seed=42)
    assert len(df) == 1


def test_invalid_amino_acid_excluded(tmp_path):
    rows = [_row(peptide="LLWTLVVLL"), _row(peptide="LLWTLXVLL")]  # X invalid
    raw = _make_raw(tmp_path, rows)
    df = build_cohort(raw, sample_n=100, seed=42)
    assert len(df) == 1


def test_duplicate_peptide_hla_deduplicated(tmp_path):
    rows = [_row(), _row()]  # exact duplicate row
    raw = _make_raw(tmp_path, rows)
    df = build_cohort(raw, sample_n=100, seed=42)
    assert len(df) == 1


# ---------------------------------------------------------------------------
# Schema and label tests
# ---------------------------------------------------------------------------
def test_all_labels_positive(tmp_path):
    rows = [_row(), _row(peptide="GILGFVFTL")]
    raw = _make_raw(tmp_path, rows)
    df = build_cohort(raw, sample_n=100, seed=42)
    assert (df["label"] == 1).all()
    assert df["source_type"].eq("Tumor").all()
    assert df["database_source"].eq("TSNAdb").all()


def test_hla_asterisk_normalised(tmp_path):
    raw = _make_raw(tmp_path, [_row(hla="HLA-A02:01")])
    df = build_cohort(raw, sample_n=100, seed=42)
    assert df.iloc[0]["hla_allele"] == "HLA-A*02:01"


# ---------------------------------------------------------------------------
# Determinism test
# ---------------------------------------------------------------------------
_UNIQUE_9MERS = [
    "ACDEFGHIK", "ACDEFGHIL", "ACDEFGHIM", "ACDEFGHIN", "ACDEFGHIP",
    "ACDEFGHIQ", "ACDEFGHIR", "ACDEFGHIS", "ACDEFGHIT", "ACDEFGHIV",
    "ACDEFGHIW", "ACDEFGHIY", "CDEFGHIKL", "CDEFGHILM", "CDEFGHIMN",
    "CDEFGHINP", "CDEFGHIPQ", "CDEFGHIQR", "CDEFGHIRS", "CDEFGHIST",
]


def test_sample_deterministic(tmp_path):
    rows = [_row(peptide=p) for p in _UNIQUE_9MERS]
    raw = _make_raw(tmp_path, rows)
    df1 = build_cohort(raw, sample_n=10, seed=42)
    df2 = build_cohort(raw, sample_n=10, seed=42)
    pd.testing.assert_frame_equal(
        df1.reset_index(drop=True), df2.reset_index(drop=True)
    )


def test_different_seeds_differ(tmp_path):
    rows = [_row(peptide=p) for p in _UNIQUE_9MERS]
    raw = _make_raw(tmp_path, rows)
    df42 = build_cohort(raw, sample_n=10, seed=42)
    df99 = build_cohort(raw, sample_n=10, seed=99)
    # Two seeds over 20 rows sampling 10 should rarely agree on exact order
    # (not a statistical test - just confirms seeds are wired through)
    assert not df42["peptide"].tolist() == df99["peptide"].tolist()


# ---------------------------------------------------------------------------
# Sample-size cap test
# ---------------------------------------------------------------------------
def test_sample_capped_at_n(tmp_path):
    rows = [_row(peptide=p) for p in _UNIQUE_9MERS]
    raw = _make_raw(tmp_path, rows)
    df = build_cohort(raw, sample_n=5, seed=42)
    assert len(df) == 5


def test_no_sample_when_under_n(tmp_path):
    rows = [_row(peptide=p) for p in _UNIQUE_9MERS[:3]]
    raw = _make_raw(tmp_path, rows)
    df = build_cohort(raw, sample_n=100, seed=42)
    assert len(df) == 3
