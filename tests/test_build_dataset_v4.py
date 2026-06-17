"""Tests for scripts/build_dataset_v4.py — v4 merge-and-validate logic.

All tests use synthetic CSV files so no network access or real data is needed.
This validates the pipeline that is on the critical path for v4 dataset build.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from build_dataset_v4 import build_dataset_v4  # noqa: E402

SCHEMA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "immunogenicity_dataset_v4_schema.json"
)


def _v3_csv(path, rows=None):
    if rows is None:
        rows = [
            {"peptide": "GILGFVFTL", "label": 1, "virus": "Flu", "protein": "M"},
            {"peptide": "NLVPMVATV", "label": 0, "virus": "CMV", "protein": "pp65"},
        ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _v4_csv(path, rows=None):
    if rows is None:
        rows = [
            {
                "peptide": "KLGGALQAK", "label": 1, "virus": "CMV", "protein": "IE1",
                "strain": "Unknown", "hla_allele": "HLA-A*03:01",
                "source_type": "Virus", "database_source": "VDJdb",
            }
        ]
    pd.DataFrame(rows).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_build_merges_all_sources(tmp_path):
    v3 = tmp_path / "v3.csv"
    vdjdb = tmp_path / "vdjdb.csv"
    _v3_csv(v3)
    _v4_csv(vdjdb)
    out = tmp_path / "v4.csv"
    build_dataset_v4(str(v3), str(vdjdb), None, None, SCHEMA_PATH, str(out))
    df = pd.read_csv(out)
    assert len(df) == 3
    assert set(df.columns) >= {"peptide", "label", "source_type"}
    assert (tmp_path / "v4_provenance.json").exists()


def test_build_missing_optional_sources_still_succeeds(tmp_path):
    v3 = tmp_path / "v3.csv"
    _v3_csv(v3)
    out = tmp_path / "v4.csv"
    # vdjdb, tsnadb, decoys all absent — should warn but succeed
    build_dataset_v4(str(v3), str(tmp_path / "nope.csv"), None, None, SCHEMA_PATH, str(out))
    df = pd.read_csv(out)
    assert len(df) == 2


def test_build_defaults_missing_string_columns(tmp_path):
    """v3 rows without optional string cols get defaults: hla_allele='Unknown',
    source_type='Virus' (the v3/IEDB hard-code), database_source='IEDB'."""
    v3 = tmp_path / "v3.csv"
    pd.DataFrame([{"peptide": "SIINFEKL", "label": 1}]).to_csv(v3, index=False)
    out = tmp_path / "v4.csv"
    build_dataset_v4(str(v3), None, None, None, SCHEMA_PATH, str(out))
    df = pd.read_csv(out)
    assert df["hla_allele"].iloc[0] == "Unknown"
    assert df["source_type"].iloc[0] == "Virus"       # v3 path always sets 'Virus'
    assert df["database_source"].iloc[0] == "IEDB"


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def test_build_deduplicates_on_peptide_hla(tmp_path):
    rows = [
        {"peptide": "GILGFVFTL", "label": 1, "virus": "Flu", "protein": "M",
         "strain": "X", "hla_allele": "HLA-A*02:01", "source_type": "Virus",
         "database_source": "VDJdb"},
        {"peptide": "GILGFVFTL", "label": 1, "virus": "Flu", "protein": "M",
         "strain": "X", "hla_allele": "HLA-A*02:01", "source_type": "Virus",
         "database_source": "IEDB"},  # same peptide+allele → duplicate
    ]
    v4 = tmp_path / "a.csv"
    pd.DataFrame(rows).to_csv(v4, index=False)
    out = tmp_path / "v4.csv"
    build_dataset_v4(str(tmp_path / "nope.csv"), str(v4), None, None, SCHEMA_PATH, str(out))
    df = pd.read_csv(out)
    assert len(df) == 1


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_build_no_sources_raises(tmp_path):
    out = tmp_path / "v4.csv"
    with pytest.raises(ValueError, match="No data sources found"):
        build_dataset_v4(
            str(tmp_path / "nope1.csv"), str(tmp_path / "nope2.csv"),
            None, None, SCHEMA_PATH, str(out),
        )


def test_build_missing_required_columns_raises(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame([{"col1": "x"}]).to_csv(bad, index=False)
    out = tmp_path / "v4.csv"
    with pytest.raises(ValueError, match="missing required"):
        build_dataset_v4(str(bad), None, None, None, SCHEMA_PATH, str(out))


# ---------------------------------------------------------------------------
# Normalisation + schema validation
# ---------------------------------------------------------------------------

def test_build_drops_non_standard_residues(tmp_path):
    rows = [
        {"peptide": "GILGFVFTL", "label": 1},
        {"peptide": "GIL2FVFTL", "label": 1},  # digit → dropped by normalize_peptides
    ]
    v3 = tmp_path / "v3.csv"
    pd.DataFrame(rows).to_csv(v3, index=False)
    out = tmp_path / "v4.csv"
    build_dataset_v4(str(v3), None, None, None, SCHEMA_PATH, str(out))
    df = pd.read_csv(out)
    assert len(df) == 1
    assert df["peptide"].iloc[0] == "GILGFVFTL"
