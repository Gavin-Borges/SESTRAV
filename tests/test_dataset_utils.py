"""Regression tests for scripts/_dataset_utils.py (v4 ingestion helpers)."""

import json
import os
import sys

import jsonschema
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from _dataset_utils import normalize_peptides, validate_against_schema, write_provenance

SCHEMA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "immunogenicity_dataset_v4_schema.json"
)


def test_normalize_peptides_uppercases_strips_and_filters():
    # Use 9-mer peptides (valid MHC Class I length 8-11)
    df = pd.DataFrame(
        {
            "peptide": ["GILGFVFTL", "gilgfvftl", " GILGFVFTL ", "GILG2VFTL", "GILGXVFTL"],
            "hla_allele": ["HLA-A*02:01"] * 5,
        }
    )
    out = normalize_peptides(df)
    # lowercase + whitespace normalized; digit/X rows dropped; all 3 remaining are valid 9-mers
    assert list(out["peptide"]) == ["GILGFVFTL", "GILGFVFTL", "GILGFVFTL"]


def test_normalize_peptides_rejects_out_of_range_length():
    """Peptides outside the 8-11mer MHC Class I range must be dropped."""
    df = pd.DataFrame(
        {
            "peptide": ["ACDEFGHI", "GILGFVFTL", "NLVPMVATVA", "LPEPLPQGQLTAY"],  # 8, 9, 10, 13-mer
            "label": [1, 1, 1, 1],
        }
    )
    out = normalize_peptides(df)
    assert len(out) == 3  # 13-mer dropped
    assert "LPEPLPQGQLTAY" not in out["peptide"].values


def test_normalize_peptides_all_invalid_returns_empty():
    df = pd.DataFrame({"peptide": ["123", "xyz!"], "hla_allele": ["a", "b"]})
    assert len(normalize_peptides(df)) == 0


def test_validate_against_schema_accepts_valid_records():
    # Use a valid 9-mer peptide (GILGFVFTL = IAV M1 58-66, canonical HLA-A*02:01 epitope)
    df = pd.DataFrame({"peptide": ["GILGFVFTL"], "label": [1], "source_type": ["Virus"]})
    validate_against_schema(df, SCHEMA_PATH)  # must not raise


@pytest.mark.parametrize(
    "bad",
    [
        {
            "peptide": ["gilgfvftl"],
            "label": [1],
            "source_type": ["Virus"],
        },  # lowercase -> schema rejects
        {"peptide": ["GILGFVFTL"], "label": [2], "source_type": ["Virus"]},  # label out of enum
        {"peptide": ["GILGFVFTL"], "label": [1]},  # missing required source_type
        {"peptide": ["ACDEF"], "label": [1], "source_type": ["Virus"]},  # 5-mer < minLength:8
        {
            "peptide": ["GILGFVFTLNLVP"],
            "label": [1],
            "source_type": ["Virus"],
        },  # 13-mer > maxLength:11
    ],
)
def test_validate_against_schema_rejects_invalid_records(bad):
    with pytest.raises(jsonschema.exceptions.ValidationError):
        validate_against_schema(pd.DataFrame(bad), SCHEMA_PATH)


def test_write_provenance_writes_sidecar(tmp_path):
    out_csv = tmp_path / "x.csv"
    write_provenance(str(out_csv), sources=["s1", "s2"], row_count=5, extra={"seed": 42})
    prov = json.loads((tmp_path / "x_provenance.json").read_text())
    assert prov["row_count"] == 5
    assert prov["sources"] == ["s1", "s2"]
    assert prov["seed"] == 42
    assert "generated_utc" in prov and "git_sha" in prov
