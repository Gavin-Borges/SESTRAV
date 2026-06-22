"""Targeted unit tests for src.iedb_data_loader CSV-branch paths.

Covers _detect_format (CSV branch), _load_epitope_table (CSV branch),
load_iedb_file (all three CSV branches), standardize_allele, map_label,
is_valid_peptide, is_mhc_class_i, _infer_protein_gene, and _infer_strain.
All tests use in-memory CSV files via tmp_path — no openpyxl or model required.
"""
import pandas as pd

from src.iedb_data_loader import (
    _detect_format,
    _infer_protein_gene,
    _infer_strain,
    _load_epitope_table,
    is_mhc_class_i,
    is_valid_peptide,
    load_iedb_file,
    map_label,
    standardize_allele,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _csv(tmp_path, rows, name="test.csv"):
    p = tmp_path / name
    p.write_text("\n".join(",".join(str(c) for c in row) for row in rows) + "\n")
    return str(p)


# ---------------------------------------------------------------------------
# _detect_format — CSV branch
# ---------------------------------------------------------------------------

def test_detect_format_csv_no_qualitative_returns_epitope_table(tmp_path):
    path = _csv(tmp_path, [["ID", "Epitope", "Name"], ["1", "EBV", "CLGGLLTMV"]])
    fmt, has_sub = _detect_format(path)
    assert fmt == "epitope_table"
    assert has_sub is False


def test_detect_format_csv_qualitative_in_row0_returns_tcell_assay(tmp_path):
    path = _csv(tmp_path, [
        ["Description", "Qualitative Measure", "Allele"],
        ["CLGGLLTMV", "Positive", "HLA-A*02:01"],
    ])
    fmt, has_sub = _detect_format(path)
    assert fmt == "tcell_assay"
    assert has_sub is None


def test_detect_format_csv_qualitative_in_row1_returns_tcell_assay(tmp_path):
    path = _csv(tmp_path, [
        ["ID", "something", "other"],
        ["Qualitative Measure", "col2", "col3"],
        ["CLGGLLTMV", "Positive", "HLA-A*02:01"],
    ])
    fmt, has_sub = _detect_format(path)
    assert fmt == "tcell_assay"
    assert has_sub is None


# ---------------------------------------------------------------------------
# _load_epitope_table — CSV branch
# ---------------------------------------------------------------------------

def _epitope_csv(tmp_path, has_subheader, peptide="CLGGLLTMV",
                 antigen="LMP1 protein", organism="Human herpesvirus 4 strain B95-8",
                 name="ep.csv"):
    """Minimal 9-column IEDB Epitope Table CSV (col 2=peptide, 6=antigen, 8=organism)."""
    header = "c0,c1,Name,c3,c4,c5,Antigen Name,c7,Organism Name"
    data = f"1,x,{peptide},,,, {antigen},,{organism}"
    if has_subheader:
        lines = f"{header}\nIEDB,,,,,,,,\n{data}\n"
    else:
        lines = f"{header}\n{data}\n"
    p = tmp_path / name
    p.write_text(lines)
    return str(p)


def test_load_epitope_table_csv_with_subheader_extracts_peptide(tmp_path):
    path = _epitope_csv(tmp_path, has_subheader=True)
    records = _load_epitope_table(path, has_subheader=True)
    assert len(records) == 1
    assert records[0]["peptide"] == "CLGGLLTMV"


def test_load_epitope_table_csv_with_subheader_extracts_antigen_organism(tmp_path):
    path = _epitope_csv(tmp_path, has_subheader=True,
                        antigen="LMP1 protein",
                        organism="Human herpesvirus 4 strain B95-8")
    records = _load_epitope_table(path, has_subheader=True)
    assert records[0]["antigen_name"] == "LMP1 protein"
    assert records[0]["organism_name"] == "Human herpesvirus 4 strain B95-8"


def test_load_epitope_table_csv_without_subheader(tmp_path):
    path = _epitope_csv(tmp_path, has_subheader=False, peptide="RAKFKQLL",
                        antigen="E7", organism="Human papillomavirus type 16")
    records = _load_epitope_table(path, has_subheader=False)
    assert len(records) == 1
    assert records[0]["peptide"] == "RAKFKQLL"


def test_load_epitope_table_csv_few_columns_no_antigen_or_organism(tmp_path):
    """Files with fewer than 7 columns yield antigen_name=None, organism_name=None."""
    p = tmp_path / "short.csv"
    p.write_text("c0,c1,Name\n1,x,GLCTLVAML\n")
    records = _load_epitope_table(str(p), has_subheader=False)
    assert records[0]["peptide"] == "GLCTLVAML"
    assert records[0]["antigen_name"] is None
    assert records[0]["organism_name"] is None


def test_load_epitope_table_csv_strips_and_uppercases_peptide(tmp_path):
    path = _epitope_csv(tmp_path, has_subheader=False, peptide=" clgglltmv ")
    records = _load_epitope_table(path, has_subheader=False)
    assert records[0]["peptide"] == "CLGGLLTMV"


# ---------------------------------------------------------------------------
# load_iedb_file — CSV branches
# ---------------------------------------------------------------------------

def test_load_iedb_file_csv_default_header(tmp_path):
    """Standard CSV (no ' - ' in row0, no 'qualitative' in row1) uses header=0."""
    p = tmp_path / "standard.csv"
    p.write_text("Description,Qualitative Measure\nCLGGLLTMV,Positive\n")
    df = load_iedb_file(str(p))
    assert "Description" in df.columns
    assert "Qualitative Measure" in df.columns
    assert len(df) == 1


def test_load_iedb_file_csv_dash_in_row0_uses_default_header(tmp_path):
    """Row 0 containing ' - ' triggers the dash-separator branch (header=0 default)."""
    p = tmp_path / "dash.csv"
    p.write_text("Col A - Sub,Col B\nCLGGLLTMV,Positive\n")
    df = load_iedb_file(str(p))
    assert "Col A - Sub" in df.columns
    assert len(df) == 1


def test_load_iedb_file_csv_qualitative_in_row1_uses_header1(tmp_path):
    """Row 1 containing 'qualitative' triggers header=1 (multi-row header case)."""
    p = tmp_path / "multirow.csv"
    p.write_text("Section A,Section B\nDescription,Qualitative Measure\nCLGGLLTMV,Positive\n")
    df = load_iedb_file(str(p))
    assert "Qualitative Measure" in df.columns
    assert len(df) == 1


# ---------------------------------------------------------------------------
# standardize_allele
# ---------------------------------------------------------------------------

def test_standardize_allele_adds_hla_prefix():
    assert standardize_allele("A*02:01") == "HLA-A*02:01"


def test_standardize_allele_leaves_existing_prefix_unchanged():
    assert standardize_allele("HLA-A*02:01") == "HLA-A*02:01"


def test_standardize_allele_strips_whitespace_then_adds_prefix():
    assert standardize_allele("  B*07:02  ") == "HLA-B*07:02"


def test_standardize_allele_nan_returns_none():
    assert standardize_allele(float("nan")) is None


def test_standardize_allele_pandas_na_returns_none():
    assert standardize_allele(pd.NA) is None


# ---------------------------------------------------------------------------
# map_label
# ---------------------------------------------------------------------------

def test_map_label_positive_exact():
    assert map_label("Positive") == 1


def test_map_label_positive_with_qualifier():
    assert map_label("positive (multiple positive assays)") == 1


def test_map_label_negative():
    assert map_label("negative") == 0


def test_map_label_nan_returns_none():
    assert map_label(float("nan")) is None


def test_map_label_inconclusive_returns_none():
    assert map_label("inconclusive") is None


# ---------------------------------------------------------------------------
# is_valid_peptide
# ---------------------------------------------------------------------------

def test_is_valid_peptide_canonical_9mer():
    assert is_valid_peptide("CLGGLLTMV") is True


def test_is_valid_peptide_8mer_boundary():
    assert is_valid_peptide("RAKFKQLL") is True


def test_is_valid_peptide_11mer_boundary():
    assert is_valid_peptide("HPVGEADYFEY") is True


def test_is_valid_peptide_too_short_returns_false():
    assert is_valid_peptide("CLGGL") is False


def test_is_valid_peptide_too_long_returns_false():
    assert is_valid_peptide("CLGGLLTMVAAK") is False


def test_is_valid_peptide_non_standard_aa_returns_false():
    assert is_valid_peptide("CLGGLLTBV") is False  # 'B' not in STANDARD_AA


def test_is_valid_peptide_nan_returns_false():
    assert is_valid_peptide(float("nan")) is False


# ---------------------------------------------------------------------------
# is_mhc_class_i
# ---------------------------------------------------------------------------

def test_is_mhc_class_i_hla_a():
    assert is_mhc_class_i("HLA-A*02:01") is True


def test_is_mhc_class_i_hla_b():
    assert is_mhc_class_i("HLA-B*07:02") is True


def test_is_mhc_class_i_drb1_returns_false():
    assert is_mhc_class_i("HLA-DRB1*01:01") is False


def test_is_mhc_class_i_dp_returns_false():
    assert is_mhc_class_i("HLA-DPA1*01:03") is False


def test_is_mhc_class_i_dq_returns_false():
    assert is_mhc_class_i("HLA-DQB1*05:01") is False


def test_is_mhc_class_i_nan_returns_false():
    assert is_mhc_class_i(float("nan")) is False


# ---------------------------------------------------------------------------
# _infer_protein_gene
# ---------------------------------------------------------------------------

def test_infer_protein_gene_hpv16_e7():
    assert _infer_protein_gene("Protein E7", "HPV16") == "E7"


def test_infer_protein_gene_hpv16_e6():
    assert _infer_protein_gene("E6 early protein", "HPV16") == "E6"


def test_infer_protein_gene_hpv16_l1():
    assert _infer_protein_gene("Major capsid protein L1", "HPV16") == "L1"


def test_infer_protein_gene_hpv16_unknown_returns_name_verbatim():
    assert _infer_protein_gene("Unknown structural protein XY", "HPV16") == "Unknown structural protein XY"


def test_infer_protein_gene_ebv_lmp1():
    assert _infer_protein_gene("LMP1", "EBV") == "LMP1"


def test_infer_protein_gene_ebv_latent_membrane_protein_2():
    assert _infer_protein_gene("Latent membrane protein 2", "EBV") == "LMP2A"


def test_infer_protein_gene_ebv_ebna1():
    assert _infer_protein_gene("EBNA1", "EBV") == "EBNA1"


def test_infer_protein_gene_ebv_nuclear_antigen_3a_disambiguation():
    assert _infer_protein_gene("Nuclear antigen EBNA-3A", "EBV") == "EBNA3A"


def test_infer_protein_gene_none_input_returns_none():
    assert _infer_protein_gene(None, "EBV") is None


def test_infer_protein_gene_unknown_virus_returns_name_verbatim():
    assert _infer_protein_gene("Spike protein", "SARS") == "Spike protein"


# ---------------------------------------------------------------------------
# _infer_strain
# ---------------------------------------------------------------------------

def test_infer_strain_ebv_b958():
    assert _infer_strain("Human herpesvirus 4 strain B95-8", "EBV") == "B95-8"


def test_infer_strain_ebv_gd1():
    assert _infer_strain("Human herpesvirus 4 GD1 variant", "EBV") == "GD1"


def test_infer_strain_ebv_generic_returns_none():
    assert _infer_strain("Human herpesvirus 4", "EBV") is None


def test_infer_strain_hpv16():
    assert _infer_strain("Human papillomavirus type 16", "HPV16") == "HPV16"


def test_infer_strain_hpv18():
    assert _infer_strain("Human papillomavirus type 18", "HPV16") == "HPV18"


def test_infer_strain_none_input_returns_none():
    assert _infer_strain(None, "EBV") is None
