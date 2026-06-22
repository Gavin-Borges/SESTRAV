"""Tests for scripts/ingest_tsnadb.py - TSNAdb ingestion logic."""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from ingest_tsnadb import ingest_tsnadb  # noqa: E402

SCHEMA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "immunogenicity_dataset_v4_schema.json"
)


def test_ingest_tsnadb_peptide_hla_columns(tmp_path):
    tsv = tmp_path / "tsna.tsv"
    tsv.write_text("Peptide\tHLA\tGene\tCancer_type\nSIINFEKL\tHLA-A*02:01\tTP53\tLung\n")
    out_csv = tmp_path / "out.csv"
    ingest_tsnadb(str(tsv), str(out_csv), SCHEMA_PATH)

    df = pd.read_csv(out_csv)
    assert len(df) == 1
    assert df["peptide"].iloc[0] == "SIINFEKL"
    assert df["source_type"].iloc[0] == "Tumor"
    assert df["database_source"].iloc[0] == "TSNAdb"
    assert df["label"].iloc[0] == 1
    assert (tmp_path / "out_provenance.json").exists()


def test_ingest_tsnadb_mutant_peptide_and_hla_type_columns(tmp_path):
    tsv = tmp_path / "tsna.tsv"
    tsv.write_text("Mutant_Peptide\tHLA_type\nKLGGALQAK\tHLA-B*07:02\n")
    out_csv = tmp_path / "out.csv"
    ingest_tsnadb(str(tsv), str(out_csv), SCHEMA_PATH)

    df = pd.read_csv(out_csv)
    assert len(df) == 1
    assert df["peptide"].iloc[0] == "KLGGALQAK"


def test_ingest_tsnadb_deduplicates_peptide_allele(tmp_path):
    tsv = tmp_path / "tsna.tsv"
    tsv.write_text(
        "Peptide\tHLA\n"
        "SIINFEKL\tHLA-A*02:01\n"
        "SIINFEKL\tHLA-A*02:01\n"  # duplicate
        "NLVPMVATV\tHLA-A*02:01\n"
    )
    out_csv = tmp_path / "out.csv"
    ingest_tsnadb(str(tsv), str(out_csv), SCHEMA_PATH)
    df = pd.read_csv(out_csv)
    assert len(df) == 2


def test_ingest_tsnadb_normalizes_lowercase(tmp_path):
    tsv = tmp_path / "tsna.tsv"
    tsv.write_text("Peptide\tHLA\nsiinfekl\tHLA-A*02:01\n")
    out_csv = tmp_path / "out.csv"
    ingest_tsnadb(str(tsv), str(out_csv), SCHEMA_PATH)
    df = pd.read_csv(out_csv)
    assert df["peptide"].iloc[0] == "SIINFEKL"


def test_ingest_tsnadb_drops_invalid_peptides(tmp_path):
    tsv = tmp_path / "tsna.tsv"
    tsv.write_text(
        "Peptide\tHLA\n"
        "SIINFEKL\tHLA-A*02:01\n"
        "SII2FEKL\tHLA-A*02:01\n"  # digit → dropped
    )
    out_csv = tmp_path / "out.csv"
    ingest_tsnadb(str(tsv), str(out_csv), SCHEMA_PATH)
    df = pd.read_csv(out_csv)
    assert len(df) == 1


def test_ingest_tsnadb_missing_columns_raises(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("col1,col2\nfoo,bar\n")
    with pytest.raises(ValueError):
        ingest_tsnadb(str(bad), str(tmp_path / "out.csv"), SCHEMA_PATH)


