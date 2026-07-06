"""Tests for scripts/ingest_vdjdb.py - VDJdb ingestion logic."""

import csv
import os
import sys
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from ingest_vdjdb import ingest_vdjdb  # noqa: E402

SCHEMA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "immunogenicity_dataset_v4_schema.json"
)


def _write_vdjdb_tsv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Epitope", "MHC A", "Epitope species", "Epitope gene"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


_ROWS = [
    {
        "Epitope": "GILGFVFTL",
        "MHC A": "HLA-A*02:01",
        "Epitope species": "Influenza A",
        "Epitope gene": "M",
    },
    {
        "Epitope": "NLVPMVATV",
        "MHC A": "HLA-A*02:01",
        "Epitope species": "CMV",
        "Epitope gene": "pp65",
    },
    {
        "Epitope": "KLGGALQAK",
        "MHC A": "HLA-A*03:01",
        "Epitope species": "CMV",
        "Epitope gene": "IE1",
    },
]


def test_ingest_vdjdb_happy_path(tmp_path):
    tsv = tmp_path / "vdjdb.txt"
    _write_vdjdb_tsv(tsv, _ROWS)
    out_csv = tmp_path / "out.csv"
    ingest_vdjdb(str(tsv), str(out_csv), SCHEMA_PATH)

    df = pd.read_csv(out_csv)
    assert len(df) == 3
    assert set(df.columns) >= {"peptide", "label", "hla_allele", "source_type"}
    assert (df["label"] == 1).all()
    assert (df["source_type"] == "Virus").all()
    assert (df["database_source"] == "VDJdb").all()
    assert (tmp_path / "out_provenance.json").exists()


def test_ingest_vdjdb_drops_low_resolution_alleles(tmp_path):
    rows = [
        {
            "Epitope": "GILGFVFTL",
            "MHC A": "HLA-A*02:01",
            "Epitope species": "Flu",
            "Epitope gene": "M",
        },
        {
            "Epitope": "NLVPMVATV",
            "MHC A": "HLA-A02",  # low-res → dropped
            "Epitope species": "CMV",
            "Epitope gene": "pp65",
        },
    ]
    tsv = tmp_path / "vdjdb.txt"
    _write_vdjdb_tsv(tsv, rows)
    out_csv = tmp_path / "out.csv"
    ingest_vdjdb(str(tsv), str(out_csv), SCHEMA_PATH)
    df = pd.read_csv(out_csv)
    assert len(df) == 1
    assert df["hla_allele"].iloc[0] == "HLA-A*02:01"


def test_ingest_vdjdb_deduplicates(tmp_path):
    rows = _ROWS + [
        {
            "Epitope": "GILGFVFTL",
            "MHC A": "HLA-A*02:01",
            "Epitope species": "Flu",
            "Epitope gene": "M",
        },
    ]
    tsv = tmp_path / "vdjdb.txt"
    _write_vdjdb_tsv(tsv, rows)
    out_csv = tmp_path / "out.csv"
    ingest_vdjdb(str(tsv), str(out_csv), SCHEMA_PATH)
    df = pd.read_csv(out_csv)
    assert len(df) == 3


def test_ingest_vdjdb_missing_columns_raises(tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_text("col1\tcol2\nfoo\tbar\n")
    with pytest.raises(ValueError, match="missing expected columns"):
        ingest_vdjdb(str(bad), str(tmp_path / "out.csv"), SCHEMA_PATH)


def test_ingest_vdjdb_missing_file_raises(tmp_path):
    with patch("ingest_vdjdb.fetch_vdjdb", return_value=None):
        with pytest.raises(FileNotFoundError):
            ingest_vdjdb(
                str(tmp_path / "nonexistent.txt"),
                str(tmp_path / "out.csv"),
                SCHEMA_PATH,
            )


def test_ingest_vdjdb_drops_non_hla_alleles(tmp_path):
    rows = [
        {
            "Epitope": "GILGFVFTL",
            "MHC A": "HLA-A*02:01",
            "Epitope species": "Flu",
            "Epitope gene": "M",
        },
        {
            "Epitope": "NLVPMVATV",
            "MHC A": "H-2Kb",  # non-HLA → filtered
            "Epitope species": "CMV",
            "Epitope gene": "pp65",
        },
    ]
    tsv = tmp_path / "vdjdb.txt"
    _write_vdjdb_tsv(tsv, rows)
    out_csv = tmp_path / "out.csv"
    ingest_vdjdb(str(tsv), str(out_csv), SCHEMA_PATH)
    df = pd.read_csv(out_csv)
    assert len(df) == 1
