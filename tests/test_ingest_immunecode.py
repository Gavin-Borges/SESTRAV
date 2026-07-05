"""Tests for scripts/ingest_immunecode.py."""

import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.ingest_immunecode import (
    _normalize_allele,
    _detect_column,
    _require_column,
    _extract_negatives,
    _build_output_df,
    SCHEMA_COLUMNS,
    _VIRUS_TAXON_ID,
    _REFERENCE_PMID,
    main,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _mira_df(
    peptides,
    alleles,
    amino_acids,
    pep_col="Peptide",
    allele_col="HLA Restrictions",
    tcr_col="Amino Acids",
) -> pd.DataFrame:
    """Minimal MIRA-format DataFrame for unit tests."""
    return pd.DataFrame({
        pep_col: peptides,
        allele_col: alleles,
        tcr_col: amino_acids,
    })


# Two SARS-CoV-2 negatives (no TCR data) and one positive (has TCR).
_MINIMAL_MIRA_CSV = (
    "Peptide,HLA Restrictions,Amino Acids\n"
    "YLQPRTFLL,HLA-A*02:01,\n"
    "LTDEMIAQY,HLA-A*02:01,CASSIRSSYEQYF\n"
    "KIADYNYKL,HLA-B*07:02,\n"
)

# All rows have TCR data - no negatives can be extracted.
_ALL_POSITIVE_MIRA_CSV = (
    "Peptide,HLA Restrictions,Amino Acids\n"
    "YLQPRTFLL,HLA-A*02:01,CASSIRSSYEQYF\n"
    "LTDEMIAQY,HLA-A*02:01,CASSIGGYYEQYF\n"
)


# ---------------------------------------------------------------------------
# _normalize_allele
# ---------------------------------------------------------------------------

def test_normalize_allele_canonical_hla_a_unchanged():
    assert _normalize_allele("HLA-A*02:01") == "HLA-A*02:01"


def test_normalize_allele_eight_digit_no_colon_gains_colon():
    assert _normalize_allele("HLA-A*0201") == "HLA-A*02:01"


def test_normalize_allele_missing_hla_prefix_with_colon():
    assert _normalize_allele("A*02:01") == "HLA-A*02:01"


def test_normalize_allele_missing_prefix_eight_digit():
    assert _normalize_allele("A*0201") == "HLA-A*02:01"


def test_normalize_allele_hla_b_missing_prefix_eight_digit():
    assert _normalize_allele("B*0702") == "HLA-B*07:02"


def test_normalize_allele_class_ii_drb1_returns_none():
    assert _normalize_allele("HLA-DRB1*01:01") is None


def test_normalize_allele_class_ii_dpb1_returns_none():
    assert _normalize_allele("HLA-DPB1*04:01") is None


def test_normalize_allele_non_string_returns_none():
    assert _normalize_allele(None) is None  # type: ignore[arg-type]
    assert _normalize_allele(42) is None    # type: ignore[arg-type]


def test_normalize_allele_empty_string_returns_none():
    assert _normalize_allele("") is None


def test_normalize_allele_strips_surrounding_whitespace():
    assert _normalize_allele("  HLA-A*02:01  ") == "HLA-A*02:01"


# ---------------------------------------------------------------------------
# _detect_column and _require_column
# ---------------------------------------------------------------------------

def test_detect_column_returns_first_candidate_present():
    df = pd.DataFrame({"Peptide": [], "HLA Restrictions": []})
    assert _detect_column(df, ["Peptide", "peptide"]) == "Peptide"


def test_detect_column_returns_second_candidate_when_first_absent():
    df = pd.DataFrame({"peptide": [], "HLA Restrictions": []})
    assert _detect_column(df, ["Peptide", "peptide"]) == "peptide"


def test_detect_column_returns_none_when_no_candidate_present():
    df = pd.DataFrame({"SomeCol": []})
    assert _detect_column(df, ["Peptide", "peptide"]) is None


def test_require_column_returns_matching_column_name():
    df = pd.DataFrame({"Peptide": []})
    assert _require_column(df, ["Peptide", "peptide"], "peptide") == "Peptide"


def test_require_column_raises_value_error_when_all_candidates_absent():
    df = pd.DataFrame({"SomeCol": []})
    with pytest.raises(ValueError, match="peptide"):
        _require_column(df, ["Peptide", "peptide", "Antigen Sequence"], "peptide")


# ---------------------------------------------------------------------------
# _extract_negatives
# ---------------------------------------------------------------------------

def test_extract_negatives_zero_tcr_row_becomes_negative():
    df = _mira_df(["YLQPRTFLL"], ["HLA-A*02:01"], [None])
    records = _extract_negatives(df)
    assert len(records) == 1
    assert records[0]["peptide"] == "YLQPRTFLL"
    assert records[0]["hla_allele"] == "HLA-A*02:01"


def test_extract_negatives_tcr_matched_row_excluded():
    df = _mira_df(["YLQPRTFLL"], ["HLA-A*02:01"], ["CASSIRSSYEQYF"])
    records = _extract_negatives(df)
    assert records == []


def test_extract_negatives_group_with_any_tcr_hit_fully_excluded():
    # Same (peptide, allele) pair appears twice: once with TCR, once without.
    # Group sum = 1, so the pair is treated as positive and excluded entirely.
    df = _mira_df(
        ["YLQPRTFLL", "YLQPRTFLL"],
        ["HLA-A*02:01", "HLA-A*02:01"],
        ["CASSIRSSYEQYF", None],
    )
    records = _extract_negatives(df)
    assert records == []


def test_extract_negatives_semicolon_pool_zero_tcr_expands_members():
    # A zero-hit pool has both members extracted as individual negatives.
    df = _mira_df(["YLQPRTFLL;LTDEMIAQY"], ["HLA-A*02:01"], [None])
    records = _extract_negatives(df)
    peptides = {r["peptide"] for r in records}
    assert "YLQPRTFLL" in peptides
    assert "LTDEMIAQY" in peptides


def test_extract_negatives_semicolon_pool_with_tcr_hit_yields_no_members():
    # A pool with any TCR expansion is excluded - ambiguous which member responded.
    df = _mira_df(["YLQPRTFLL;LTDEMIAQY"], ["HLA-A*02:01"], ["CASSIRSSYEQYF"])
    records = _extract_negatives(df)
    assert records == []


def test_extract_negatives_seven_mer_filtered_out():
    df = _mira_df(["AAAAAAA"], ["HLA-A*02:01"], [None])
    records = _extract_negatives(df)
    assert records == []


def test_extract_negatives_twelve_mer_filtered_out():
    df = _mira_df(["AAAAAAAAAAAA"], ["HLA-A*02:01"], [None])
    records = _extract_negatives(df)
    assert records == []


def test_extract_negatives_8mer_and_11mer_both_accepted():
    df = _mira_df(
        ["AAAAAAAA", "AAAAAAAAAAA"],
        ["HLA-A*02:01", "HLA-A*02:01"],
        [None, None],
    )
    records = _extract_negatives(df)
    assert len(records) == 2


def test_extract_negatives_nonstandard_amino_acids_filtered():
    df = _mira_df(["YLQPRTFBX"], ["HLA-A*02:01"], [None])
    records = _extract_negatives(df)
    assert records == []


def test_extract_negatives_no_tcr_column_returns_empty_list():
    df = pd.DataFrame({
        "Peptide": ["YLQPRTFLL"],
        "HLA Restrictions": ["HLA-A*02:01"],
    })
    records = _extract_negatives(df)
    assert records == []


def test_extract_negatives_class_ii_allele_row_excluded():
    df = _mira_df(["YLQPRTFLL"], ["HLA-DRB1*01:01"], [None])
    records = _extract_negatives(df)
    assert records == []


def test_extract_negatives_normalizes_eight_digit_allele():
    df = _mira_df(["YLQPRTFLL"], ["A*02:01"], [None])
    records = _extract_negatives(df)
    assert len(records) == 1
    assert records[0]["hla_allele"] == "HLA-A*02:01"


def test_extract_negatives_semicolon_allele_field_expands_to_two_records():
    df = _mira_df(["YLQPRTFLL"], ["HLA-A*02:01;HLA-B*07:02"], [None])
    records = _extract_negatives(df)
    alleles = {r["hla_allele"] for r in records}
    assert alleles == {"HLA-A*02:01", "HLA-B*07:02"}


# ---------------------------------------------------------------------------
# _build_output_df
# ---------------------------------------------------------------------------

def test_build_output_df_empty_records_returns_dataframe_with_schema_columns():
    df = _build_output_df([])
    assert list(df.columns) == SCHEMA_COLUMNS
    assert len(df) == 0


def test_build_output_df_has_exactly_23_schema_columns():
    records = [{"peptide": "YLQPRTFLL", "hla_allele": "HLA-A*02:01"}]
    df = _build_output_df(records)
    assert set(df.columns) == set(SCHEMA_COLUMNS)
    assert len(df.columns) == 23


def test_build_output_df_all_labels_are_zero():
    records = [
        {"peptide": "YLQPRTFLL", "hla_allele": "HLA-A*02:01"},
        {"peptide": "KIADYNYKL", "hla_allele": "HLA-B*07:02"},
    ]
    df = _build_output_df(records)
    assert (df["label"] == 0).all()


def test_build_output_df_is_quarantined_is_false():
    records = [{"peptide": "YLQPRTFLL", "hla_allele": "HLA-A*02:01"}]
    df = _build_output_df(records)
    assert (df["is_quarantined"] == False).all()  # noqa: E712


def test_build_output_df_virus_is_sars_cov2():
    records = [{"peptide": "YLQPRTFLL", "hla_allele": "HLA-A*02:01"}]
    df = _build_output_df(records)
    assert (df["virus"] == "SARS-CoV-2").all()


def test_build_output_df_source_type_is_virus():
    records = [{"peptide": "YLQPRTFLL", "hla_allele": "HLA-A*02:01"}]
    df = _build_output_df(records)
    assert (df["source_type"] == "Virus").all()


def test_build_output_df_virus_family_is_coronaviridae():
    records = [{"peptide": "YLQPRTFLL", "hla_allele": "HLA-A*02:01"}]
    df = _build_output_df(records)
    assert (df["virus_family"] == "Coronaviridae").all()


def test_build_output_df_virus_taxon_id_matches_sars_cov2():
    records = [{"peptide": "YLQPRTFLL", "hla_allele": "HLA-A*02:01"}]
    df = _build_output_df(records)
    assert (df["virus_taxon_id"] == _VIRUS_TAXON_ID).all()


def test_build_output_df_reference_pmid_matches_nolan_2020():
    records = [{"peptide": "YLQPRTFLL", "hla_allele": "HLA-A*02:01"}]
    df = _build_output_df(records)
    assert (df["reference_pmid"] == _REFERENCE_PMID).all()


def test_build_output_df_duplicate_peptide_allele_pairs_deduplicated():
    records = [
        {"peptide": "YLQPRTFLL", "hla_allele": "HLA-A*02:01"},
        {"peptide": "YLQPRTFLL", "hla_allele": "HLA-A*02:01"},
    ]
    df = _build_output_df(records)
    assert len(df) == 1


def test_build_output_df_database_source_is_immunecode():
    records = [{"peptide": "YLQPRTFLL", "hla_allele": "HLA-A*02:01"}]
    df = _build_output_df(records)
    assert (df["database_source"] == "ImmuneCODE").all()


# ---------------------------------------------------------------------------
# CLI / main()
# ---------------------------------------------------------------------------

def test_main_happy_path_returns_zero_and_writes_csv(tmp_path):
    infile = tmp_path / "mira.csv"
    infile.write_text(_MINIMAL_MIRA_CSV)
    outfile = tmp_path / "negatives.csv"

    ret = main(["--input", str(infile), "--output", str(outfile)])
    assert ret == 0
    assert outfile.exists()


def test_main_output_contains_all_schema_columns(tmp_path):
    infile = tmp_path / "mira.csv"
    infile.write_text(_MINIMAL_MIRA_CSV)
    outfile = tmp_path / "negatives.csv"

    main(["--input", str(infile), "--output", str(outfile)])
    df = pd.read_csv(outfile)
    assert set(df.columns) == set(SCHEMA_COLUMNS)


def test_main_all_output_rows_have_label_zero(tmp_path):
    infile = tmp_path / "mira.csv"
    infile.write_text(_MINIMAL_MIRA_CSV)
    outfile = tmp_path / "negatives.csv"

    main(["--input", str(infile), "--output", str(outfile)])
    df = pd.read_csv(outfile)
    assert (df["label"] == 0).all()


def test_main_is_quarantined_column_written_as_false(tmp_path):
    infile = tmp_path / "mira.csv"
    infile.write_text(_MINIMAL_MIRA_CSV)
    outfile = tmp_path / "negatives.csv"

    main(["--input", str(infile), "--output", str(outfile)])
    df = pd.read_csv(outfile)
    assert "is_quarantined" in df.columns
    # After CSV round-trip pandas may read as bool or object; compare by string
    assert (df["is_quarantined"].astype(str).str.strip() == "False").all()


def test_main_virus_column_is_sars_cov2(tmp_path):
    infile = tmp_path / "mira.csv"
    infile.write_text(_MINIMAL_MIRA_CSV)
    outfile = tmp_path / "negatives.csv"

    main(["--input", str(infile), "--output", str(outfile)])
    df = pd.read_csv(outfile)
    assert (df["virus"] == "SARS-CoV-2").all()


def test_main_dry_run_does_not_write_csv(tmp_path):
    infile = tmp_path / "mira.csv"
    infile.write_text(_MINIMAL_MIRA_CSV)
    outfile = tmp_path / "negatives.csv"

    ret = main(["--input", str(infile), "--output", str(outfile), "--dry-run"])
    assert ret == 0
    assert not outfile.exists()


def test_main_dry_run_does_not_write_provenance(tmp_path):
    infile = tmp_path / "mira.csv"
    infile.write_text(_MINIMAL_MIRA_CSV)
    outfile = tmp_path / "negatives.csv"
    prov = tmp_path / "negatives_provenance.json"

    main(["--input", str(infile), "--output", str(outfile), "--dry-run"])
    assert not prov.exists()


def test_main_provenance_sidecar_written_alongside_csv(tmp_path):
    infile = tmp_path / "mira.csv"
    infile.write_text(_MINIMAL_MIRA_CSV)
    outfile = tmp_path / "negatives.csv"
    prov = tmp_path / "negatives_provenance.json"

    ret = main(["--input", str(infile), "--output", str(outfile)])
    assert ret == 0
    assert prov.exists()


def test_main_provenance_contains_required_keys(tmp_path):
    infile = tmp_path / "mira.csv"
    infile.write_text(_MINIMAL_MIRA_CSV)
    outfile = tmp_path / "negatives.csv"
    prov = tmp_path / "negatives_provenance.json"

    main(["--input", str(infile), "--output", str(outfile)])
    with open(prov) as f:
        data = json.load(f)
    for key in ("row_count", "database", "virus", "reference_pmid", "generated_utc"):
        assert key in data, f"Missing provenance key: {key}"


def test_main_provenance_row_count_matches_output_csv(tmp_path):
    infile = tmp_path / "mira.csv"
    infile.write_text(_MINIMAL_MIRA_CSV)
    outfile = tmp_path / "negatives.csv"
    prov = tmp_path / "negatives_provenance.json"

    main(["--input", str(infile), "--output", str(outfile)])
    df = pd.read_csv(outfile)
    with open(prov) as f:
        data = json.load(f)
    assert data["row_count"] == len(df)


def test_main_missing_input_path_returns_nonzero(tmp_path):
    outfile = tmp_path / "negatives.csv"
    ret = main(["--input", str(tmp_path / "nonexistent.csv"), "--output", str(outfile)])
    assert ret != 0


def test_main_all_tcr_matched_input_returns_nonzero(tmp_path):
    infile = tmp_path / "mira.csv"
    infile.write_text(_ALL_POSITIVE_MIRA_CSV)
    outfile = tmp_path / "negatives.csv"

    ret = main(["--input", str(infile), "--output", str(outfile)])
    assert ret != 0
