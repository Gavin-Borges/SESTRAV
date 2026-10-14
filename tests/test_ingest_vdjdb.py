"""Tests for scripts/ingest_vdjdb.py - VDJdb ingestion logic."""

import csv
import json
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


def _write_vdjdb_tsv_with_cdr3(path, rows):
    """Write a legacy-format TSV carrying CDR3.alpha and CDR3.beta columns."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Epitope",
                "MHC A",
                "Epitope species",
                "Epitope gene",
                "CDR3.alpha",
                "CDR3.beta",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def test_provenance_cdr3_counts_describe_the_written_artifact(tmp_path):
    """The sidecar's CDR3 counts must count the rows the sidecar's row_count counts.

    They were taken before the dropna, the HLA-resolution filter and the
    deduplication that build the artifact, so a release whose dropped rows
    carried CDR3s produced a sidecar claiming more populated rows than the file
    has rows: the shipped data/vdjdb_v5_provenance.json records 1098 alpha and
    1283 beta against a row_count of 1066.

    The third row below is dropped by the low-resolution allele filter and
    carries both CDR3s, so the pre-filter counts are 3 and the artifact's are 2.
    """
    rows = [
        {
            "Epitope": "GILGFVFTL",
            "MHC A": "HLA-A*02:01",
            "Epitope species": "Influenza A",
            "Epitope gene": "M",
            "CDR3.alpha": "CAVRDSNYQLIW",
            "CDR3.beta": "CASSIRSSYEQYF",
        },
        {
            "Epitope": "NLVPMVATV",
            "MHC A": "HLA-A*02:01",
            "Epitope species": "CMV",
            "Epitope gene": "pp65",
            "CDR3.alpha": "CAASIGPLGTGTASKLTF",
            "CDR3.beta": "CASSPQETQYF",
        },
        {
            # Low-resolution allele: dropped, and it carries both CDR3s.
            "Epitope": "KLGGALQAK",
            "MHC A": "HLA-A*03",
            "Epitope species": "CMV",
            "Epitope gene": "IE1",
            "CDR3.alpha": "CAVNFGGGKLIF",
            "CDR3.beta": "CASSLAPGATNEKLFF",
        },
    ]
    tsv = tmp_path / "vdjdb.txt"
    _write_vdjdb_tsv_with_cdr3(tsv, rows)
    out_csv = tmp_path / "out.csv"
    ingest_vdjdb(str(tsv), str(out_csv), SCHEMA_PATH)

    with open(tmp_path / "out_provenance.json") as f:
        prov = json.load(f)
    df = pd.read_csv(out_csv)

    # The dropped row is gone, so the artifact carries two rows.
    assert prov["row_count"] == len(df) == 2

    # The invariant that the shipped sidecar violates: a count of populated rows
    # cannot exceed the number of rows.
    assert prov["tcr_alpha_cdr3_populated"] <= prov["row_count"]
    assert prov["tcr_beta_cdr3_populated"] <= prov["row_count"]

    # Stronger: each count equals what the written file actually holds.
    assert prov["tcr_alpha_cdr3_populated"] == int(df["tcr_alpha_cdr3"].notna().sum()) == 2
    assert prov["tcr_beta_cdr3_populated"] == int(df["tcr_beta_cdr3"].notna().sum()) == 2


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
