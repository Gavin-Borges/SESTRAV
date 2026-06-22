"""Unit tests for src/iedb_data_loader.py.

Covers the pure validators / metadata-inference helpers and the CSV file
loaders, including the end-to-end ``load_and_clean_iedb`` orchestrator driven
off ``tmp_path`` fixtures. The loader natively accepts ``.csv`` (not just
``.xlsx``), so every code path below is reachable without Excel binaries.

Complements the filename-helper tests in tests/test_coverage_units.py
(``_label_from_filename`` / ``_virus_from_filename``); those are not repeated
here.
"""
import numpy as np
import pandas as pd
import pytest

from src.iedb_data_loader import (
    GOLD_STANDARD_EPITOPES,
    _detect_format,
    _infer_protein_gene,
    _infer_strain,
    _load_epitope_table,
    is_mhc_class_i,
    is_valid_peptide,
    load_and_clean_iedb,
    load_iedb_file,
    load_schmidt_2021,
    map_label,
    split_gold_standard,
    standardize_allele,
)

# ---------------------------------------------------------------------------
# standardize_allele
# ---------------------------------------------------------------------------


def test_standardize_allele_adds_prefix():
    assert standardize_allele("A*02:01") == "HLA-A*02:01"


def test_standardize_allele_keeps_existing_prefix():
    assert standardize_allele("HLA-B*07:02") == "HLA-B*07:02"


def test_standardize_allele_strips_whitespace():
    assert standardize_allele("  A*01:01  ") == "HLA-A*01:01"


def test_standardize_allele_nan_is_none():
    assert standardize_allele(np.nan) is None


# ---------------------------------------------------------------------------
# map_label
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "measure, expected",
    [
        ("Positive", 1),
        ("Positive-High", 1),
        ("Positive-Intermediate", 1),
        ("Negative", 0),
        ("  negative ", 0),
        ("Inconclusive", None),
    ],
)
def test_map_label(measure, expected):
    assert map_label(measure) == expected


def test_map_label_nan_is_none():
    assert map_label(np.nan) is None


# ---------------------------------------------------------------------------
# is_valid_peptide
# ---------------------------------------------------------------------------


def test_is_valid_peptide_accepts_standard_9mer():
    assert is_valid_peptide("SLLMWITQV") is True


def test_is_valid_peptide_rejects_too_short():
    assert is_valid_peptide("ACDEFGH") is False  # 7mer


def test_is_valid_peptide_rejects_too_long():
    assert is_valid_peptide("ACDEFGHIKLMN") is False  # 12mer


def test_is_valid_peptide_rejects_nonstandard_aa():
    assert is_valid_peptide("SLLMWITQX") is False  # X not standard


def test_is_valid_peptide_nan_is_false():
    assert is_valid_peptide(np.nan) is False


def test_is_valid_peptide_normalizes_case_and_whitespace():
    assert is_valid_peptide("  sllmwitqv ") is True


# ---------------------------------------------------------------------------
# is_mhc_class_i
# ---------------------------------------------------------------------------


def test_is_mhc_class_i_accepts_class_i():
    assert is_mhc_class_i("HLA-A*02:01") is True


@pytest.mark.parametrize("allele", ["HLA-DRB1*15:01", "HLA-DPA1*01:03", "HLA-DQB1*06:02"])
def test_is_mhc_class_i_rejects_class_ii(allele):
    assert is_mhc_class_i(allele) is False


def test_is_mhc_class_i_nan_is_false():
    assert is_mhc_class_i(np.nan) is False


# ---------------------------------------------------------------------------
# _infer_protein_gene
# ---------------------------------------------------------------------------


def test_infer_protein_gene_none_input():
    assert _infer_protein_gene(None, "EBV") is None


def test_infer_protein_gene_hpv_matches_gene_token():
    assert _infer_protein_gene("Protein E7", "HPV16") == "E7"


def test_infer_protein_gene_hpv_no_match_returns_name():
    assert _infer_protein_gene("Capsid blob", "HPV16") == "Capsid blob"


@pytest.mark.parametrize(
    "antigen, expected",
    [
        ("Latent membrane protein 2", "LMP2A"),
        ("LMP1", "LMP1"),
        ("EBNA-3A", "EBNA3A"),
        ("BZLF1 transactivator", "BZLF1"),
        ("gp350", "GP350"),
    ],
)
def test_infer_protein_gene_ebv_map(antigen, expected):
    assert _infer_protein_gene(antigen, "EBV") == expected


def test_infer_protein_gene_ebv_nuclear_antigen_disambiguation():
    # "nuclear antigen" maps to None, then suffix '3a' resolves to EBNA3A.
    assert _infer_protein_gene("Nuclear antigen 3A", "EBV") == "EBNA3A"


def test_infer_protein_gene_ebv_nuclear_antigen_no_suffix():
    assert _infer_protein_gene("Nuclear antigen", "EBV") == "EBNA"


def test_infer_protein_gene_unknown_virus_returns_name():
    assert _infer_protein_gene("Some Protein", "CMV") == "Some Protein"


# ---------------------------------------------------------------------------
# _infer_strain
# ---------------------------------------------------------------------------


def test_infer_strain_none_input():
    assert _infer_strain(None, "EBV") is None


@pytest.mark.parametrize(
    "organism, expected",
    [
        ("Human herpesvirus 4 strain B95-8", "B95-8"),
        ("EBV isolate GD1", "GD1"),
        ("Human herpesvirus 4 AG876", "AG876"),
        ("Akata strain", "Akata"),
        ("Human herpesvirus 4", None),
    ],
)
def test_infer_strain_ebv(organism, expected):
    assert _infer_strain(organism, "EBV") == expected


@pytest.mark.parametrize(
    "organism, expected",
    [
        ("Human papillomavirus type 16", "HPV16"),
        ("Human papillomavirus type 18", "HPV18"),
        ("Human papillomavirus type 11", "HPV11"),
        ("Human papillomavirus", None),
    ],
)
def test_infer_strain_hpv(organism, expected):
    assert _infer_strain(organism, "HPV16") == expected


def test_infer_strain_unknown_virus_is_none():
    assert _infer_strain("Human herpesvirus 4 strain B95-8", "CMV") is None


# ---------------------------------------------------------------------------
# split_gold_standard
# ---------------------------------------------------------------------------


def test_split_gold_standard_partitions():
    gold = GOLD_STANDARD_EPITOPES[0]
    df = pd.DataFrame({"peptide": [gold, "AAAAAAAA"], "label": [1, 0]})
    train, gs = split_gold_standard(df)
    assert list(train["peptide"]) == ["AAAAAAAA"]
    assert list(gs["peptide"]) == [gold]


# ---------------------------------------------------------------------------
# _detect_format (CSV paths)
# ---------------------------------------------------------------------------


def test_detect_format_csv_tcell_assay(tmp_path):
    p = tmp_path / "assay.csv"
    p.write_text("Description,Qualitative Measure\nSLLMWITQV,Positive\n")
    assert _detect_format(str(p)) == ("tcell_assay", None)


def test_detect_format_csv_epitope_table(tmp_path):
    p = tmp_path / "epi.csv"
    p.write_text("c0,c1,Name\nx,y,SLLMWITQV\n")
    assert _detect_format(str(p)) == ("epitope_table", False)


# ---------------------------------------------------------------------------
# _load_epitope_table (CSV)
# ---------------------------------------------------------------------------


def test_load_epitope_table_extracts_peptide_antigen_organism(tmp_path):
    p = tmp_path / "epi.csv"
    p.write_text(
        "c0,c1,Name,c3,c4,c5,Antigen,c7,Organism\n"
        "x,y,SLLMWITQV,a,b,c,Protein E7,g,Human papillomavirus type 16\n"
    )
    records = _load_epitope_table(str(p), has_subheader=False)
    assert len(records) == 1
    rec = records[0]
    assert rec["peptide"] == "SLLMWITQV"
    assert rec["antigen_name"] == "Protein E7"
    assert rec["organism_name"] == "Human papillomavirus type 16"


# ---------------------------------------------------------------------------
# load_iedb_file (CSV header heuristics)
# ---------------------------------------------------------------------------


def test_load_iedb_file_plain_header(tmp_path):
    p = tmp_path / "plain.csv"
    p.write_text("Description,Qualitative Measure\nSLLMWITQV,Positive\n")
    df = load_iedb_file(str(p))
    assert "Description" in df.columns
    assert len(df) == 1


def test_load_iedb_file_dash_in_first_row(tmp_path):
    # A header containing ' - ' triggers the default single-header read.
    p = tmp_path / "dash.csv"
    p.write_text("MHC restriction - Name,Description\nHLA-A*02:01,SLLMWITQV\n")
    df = load_iedb_file(str(p))
    assert "Description" in df.columns


# ---------------------------------------------------------------------------
# load_schmidt_2021
# ---------------------------------------------------------------------------


def test_load_schmidt_missing_file_returns_empty(tmp_path):
    df = load_schmidt_2021(str(tmp_path / "nope.csv"))
    assert df.empty
    assert list(df.columns) == ["peptide", "hla", "label", "hard_negative_flag"]


def test_load_schmidt_renames_and_flags(tmp_path):
    p = tmp_path / "schmidt.csv"
    pd.DataFrame(
        {
            "Epitope": ["SLLMWITQV", "ACDEFGHX"],  # second is invalid (X)
            "MHC": ["A*02:01", "A*01:01"],
            "Immunogenicity": [1, 0],
        }
    ).to_csv(p, index=False)
    df = load_schmidt_2021(str(p))
    assert list(df.columns) == ["peptide", "hla", "label", "hard_negative_flag"]
    assert list(df["peptide"]) == ["SLLMWITQV"]  # invalid filtered out
    assert df["hla"].iloc[0] == "HLA-A*02:01"
    assert (df["hard_negative_flag"] == 1).all()


def test_load_schmidt_missing_label_defaults_zero(tmp_path):
    p = tmp_path / "schmidt_nolabel.csv"
    pd.DataFrame({"peptide": ["SLLMWITQV"], "hla": ["A*02:01"]}).to_csv(p, index=False)
    df = load_schmidt_2021(str(p))
    assert df["label"].iloc[0] == 0


def test_load_schmidt_missing_hla_uses_default(tmp_path):
    p = tmp_path / "schmidt_nohla.csv"
    pd.DataFrame({"peptide": ["SLLMWITQV"], "label": [1]}).to_csv(p, index=False)
    df = load_schmidt_2021(str(p))
    assert df["hla"].iloc[0] == "HLA-A*02:01"


def test_load_schmidt_missing_peptide_raises(tmp_path):
    p = tmp_path / "schmidt_nopep.csv"
    pd.DataFrame({"hla": ["A*02:01"], "label": [1]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="missing required column: peptide"):
        load_schmidt_2021(str(p))


def test_load_schmidt_excludes_gold_standard(tmp_path):
    gold = GOLD_STANDARD_EPITOPES[0]
    p = tmp_path / "schmidt_gold.csv"
    pd.DataFrame(
        {"peptide": [gold, "AAAAAAAA"], "hla": ["A*02:01", "A*01:01"], "label": [1, 0]}
    ).to_csv(p, index=False)
    df = load_schmidt_2021(str(p))
    assert gold not in set(df["peptide"])


# ---------------------------------------------------------------------------
# load_and_clean_iedb (end-to-end on CSV fixtures)
# ---------------------------------------------------------------------------


def test_load_and_clean_epitope_table(tmp_path):
    (tmp_path / "HPV16_T-cell_positive.csv").write_text(
        "c0,c1,Name,c3,c4,c5,Antigen,c7,Organism\n"
        "x,y,SLLMWITQV,a,b,c,Protein E7,g,Human papillomavirus type 16\n"
        "x,y,FLPSDFFPSV,a,b,c,Protein E6,g,Human papillomavirus type 16\n"
    )
    df = load_and_clean_iedb(str(tmp_path))
    assert len(df) == 2
    assert set(df["peptide"]) == {"SLLMWITQV", "FLPSDFFPSV"}
    assert (df["label"] == 1).all()
    assert (df["virus"] == "HPV16").all()
    assert set(df["protein"]) == {"E6", "E7"}
    assert (df["strain"] == "HPV16").all()


def test_load_and_clean_tcell_assay(tmp_path):
    (tmp_path / "EBV_tcell_assay.csv").write_text(
        "Description,Qualitative Measure,Allele\n"
        "SLLMWITQV,Positive,A*02:01\n"
        "AAAAAAAA,Negative,A*01:01\n"
    )
    df = load_and_clean_iedb(str(tmp_path))
    assert len(df) == 2
    assert (df["virus"] == "EBV").all()
    labels = dict(zip(df["peptide"], df["label"]))
    assert labels["SLLMWITQV"] == 1
    assert labels["AAAAAAAA"] == 0
    assert "allele" in df.columns


def test_load_and_clean_tcell_assay_drops_class_ii(tmp_path):
    (tmp_path / "EBV_tcell_assay.csv").write_text(
        "Description,Qualitative Measure,Allele\n"
        "SLLMWITQV,Positive,A*02:01\n"
        "FLPSDFFPSV,Positive,DRB1*15:01\n"  # class II -> dropped
    )
    df = load_and_clean_iedb(str(tmp_path))
    assert list(df["peptide"]) == ["SLLMWITQV"]


def test_load_and_clean_tcell_derives_virus_from_organism(tmp_path):
    # No virus token in the filename -> virus inferred per-row from organism,
    # also exercising the Antigen Name / Organism column extraction.
    (tmp_path / "tcell_assay_export.csv").write_text(
        "Description,Qualitative Measure,Allele,Antigen Name,Organism\n"
        "SLLMWITQV,Positive,A*02:01,Latent membrane protein 2,Human herpesvirus 4 strain B95-8\n"
        "FLPSDFFPSV,Positive,A*01:01,Protein E7,Human papillomavirus type 16\n"
        "CLGGLLTMVK,Negative,A*03:01,Unknown,Some unrelated organism\n"
    )
    df = load_and_clean_iedb(str(tmp_path))
    by_pep = df.set_index("peptide")
    # EBV row resolved with strain + protein metadata
    assert by_pep.loc["SLLMWITQV", "virus"] == "EBV"
    assert by_pep.loc["SLLMWITQV", "protein"] == "LMP2A"
    assert by_pep.loc["SLLMWITQV", "strain"] == "B95-8"
    # HPV16 row resolved
    assert by_pep.loc["FLPSDFFPSV", "virus"] == "HPV16"
    assert by_pep.loc["FLPSDFFPSV", "protein"] == "E7"
    # Unknown-organism row is dropped (virus undeterminable)
    assert "CLGGLLTMVK" not in by_pep.index


def test_load_and_clean_excludes_hpv11_by_default(tmp_path):
    (tmp_path / "HPV11_T-cell_positive.csv").write_text(
        "c0,c1,Name\nx,y,SLLMWITQV\n"
    )
    df = load_and_clean_iedb(str(tmp_path))
    assert df.empty


def test_load_and_clean_includes_hpv11_when_requested(tmp_path):
    (tmp_path / "HPV11_T-cell_positive.csv").write_text(
        "c0,c1,Name\nx,y,SLLMWITQV\n"
    )
    df = load_and_clean_iedb(str(tmp_path), include_hpv11=True)
    assert list(df["peptide"]) == ["SLLMWITQV"]
    assert (df["virus"] == "HPV11").all()


def test_load_and_clean_empty_dir_returns_empty(tmp_path):
    df = load_and_clean_iedb(str(tmp_path))
    assert df.empty


def test_load_and_clean_resolves_duplicate_labels_by_majority(tmp_path):
    # Same peptide appears positive twice, negative once -> majority positive.
    (tmp_path / "EBV_tcell_assay.csv").write_text(
        "Description,Qualitative Measure,Allele\n"
        "SLLMWITQV,Positive,A*02:01\n"
        "SLLMWITQV,Positive,A*02:01\n"
        "SLLMWITQV,Negative,A*02:01\n"
    )
    df = load_and_clean_iedb(str(tmp_path))
    assert len(df) == 1
    assert df["label"].iloc[0] == 1
