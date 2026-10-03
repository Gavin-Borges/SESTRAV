"""Characterisation tests for src/iedb_data_loader.py.

src/iedb_data_loader.py carries a __main__ guard, so .coveragerc.library omits
it and the required coverage gate never measures it. Before this file every
.xlsx branch was unexecuted under the suite, as were the entry point and seven
of the twelve `continue` skips in load_and_clean_iedb; the other five, the
HPV11 exclusions among them, already ran.

Every test here PINS CURRENT BEHAVIOUR. None asserts that the behaviour is
right; several record behaviour an owner may want to change, and say so in
their docstrings. Nothing in src/iedb_data_loader.py is edited. Every input
file is synthetic, written to tmp_path in the test itself.
"""

import subprocess
import sys
from pathlib import Path

import openpyxl
import pandas as pd
import pytest

import src.iedb_data_loader as loader

REPO_ROOT = Path(__file__).resolve().parents[1]


def _xlsx(path, rows):
    wb = openpyxl.Workbook()
    ws = wb.active
    for row in rows:
        ws.append(list(row))
    wb.save(path)
    wb.close()
    return str(path)


def _epitope_rows(
    peptides, antigen="Latent membrane protein 2", organism="Human herpesvirus 4 strain B95-8"
):
    """Nine-column Epitope Table rows: col 2 peptide, col 6 antigen, col 8 organism."""
    return [["id", "iri", pep, "x", "x", "x", antigen, "x", organism] for pep in peptides]


# ---------------------------------------------------------------------------
# .xlsx format detection and loading
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rows, expected",
    [
        (
            [["Epitope", "Epitope"], ["IEDB IRI", "Name"], ["x", "SLLMWITQV"]],
            ("epitope_table", True),
        ),
        ([["Epitope", "Epitope"], ["x", "SLLMWITQV"]], ("epitope_table", False)),
        ([["Epitope", "Qualitative Measure"], ["x", "Positive"]], ("tcell_assay", None)),
        ([["Epitope", "Assay"], ["Description", "Qualitative Measure"]], ("tcell_assay", None)),
        ([["Epitope"], ["  iedb lowercase"]], ("epitope_table", True)),
    ],
    ids=[
        "subheader",
        "no-subheader",
        "qualitative-row1",
        "qualitative-row2",
        "subheader-case-blind",
    ],
)
def test_detect_format_xlsx(tmp_path, rows, expected):
    # pins behaviour, not correctness
    assert loader._detect_format(_xlsx(tmp_path / "f.xlsx", rows)) == expected


def test_detect_format_csv_never_reports_a_subheader(tmp_path):
    """A CSV Epitope Table always gets has_subheader=False, so exactly ONE row is skipped.

    A two-row CSV header therefore reaches the peptide loop as a data row; it is
    only discarded later because its peptide cell fails is_valid_peptide.
    """
    # pins behaviour, not correctness
    path = tmp_path / "EBV positive.csv"
    path.write_text(
        "Epitope,Epitope,Epitope\nIEDB IRI,Object Type,Name\nx,y,SLLMWITQV\n", encoding="utf-8"
    )
    assert loader._detect_format(str(path)) == ("epitope_table", False)
    records = loader._load_epitope_table(str(path), False)
    assert [r["peptide"] for r in records] == ["NAME", "SLLMWITQV"]


def test_load_and_clean_reads_an_xlsx_epitope_table_with_metadata(tmp_path, capsys):
    # pins behaviour, not correctness
    rows = [["Epitope"] * 9, ["IEDB IRI"] + ["h"] * 8]
    rows += _epitope_rows(["slLMWITQV ", "CLGGLLTMV", "SHORT", 12345])
    _xlsx(tmp_path / "EBV T-cell positive.xlsx", rows)

    df = loader.load_and_clean_iedb(str(tmp_path))
    out = capsys.readouterr().out

    assert df.to_dict("records") == [
        {"peptide": "CLGGLLTMV", "label": 1, "virus": "EBV", "protein": "LMP2A", "strain": "B95-8"},
        {"peptide": "SLLMWITQV", "label": 1, "virus": "EBV", "protein": "LMP2A", "strain": "B95-8"},
    ]
    assert "Epitope Table format, label=1 (from filename), 2 valid 8-11mers" in out


def test_load_epitope_table_blank_metadata_becomes_none(tmp_path):
    # pins behaviour, not correctness
    rows = [["Epitope"] * 9, ["IEDB IRI"] + ["h"] * 8]
    rows += _epitope_rows(["SLLMWITQV"], antigen="   ", organism="   ")
    path = _xlsx(tmp_path / "f.xlsx", rows)
    assert loader._load_epitope_table(path, True) == [
        {"peptide": "SLLMWITQV", "antigen_name": None, "organism_name": None}
    ]


@pytest.mark.parametrize(
    "rows, expected_columns",
    [
        (
            [["Epitope - Name", "Assay - Qualitative Measure"], ["SLLMWITQV", "Positive"]],
            ["Epitope - Name", "Assay - Qualitative Measure"],
        ),
        (
            [
                ["Epitope", "Assay"],
                ["Description", "Qualitative Measure"],
                ["SLLMWITQV", "Positive"],
            ],
            ["Description", "Qualitative Measure"],
        ),
        ([["Description", "Label"], ["SLLMWITQV", "Positive"]], ["Description", "Label"]),
    ],
    ids=["dash-header", "subheader-promoted", "plain"],
)
def test_load_iedb_file_xlsx_header_choice(tmp_path, rows, expected_columns):
    # pins behaviour, not correctness
    df = loader.load_iedb_file(_xlsx(tmp_path / "f.xlsx", rows))
    assert list(df.columns) == expected_columns
    assert df.iloc[-1].tolist() == ["SLLMWITQV", "Positive"]


# ---------------------------------------------------------------------------
# load_and_clean_iedb: skip paths and row filters
# ---------------------------------------------------------------------------


def test_every_file_skipped_returns_a_frame_with_no_columns(tmp_path, capsys):
    """OWNER LOOK: the docstring promises peptide/label/virus/protein/strain columns,
    but when nothing survives the result has NO columns, so a caller indexing
    df["peptide"] gets a KeyError rather than an empty column.
    """
    # pins behaviour, not correctness
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
    (tmp_path / "unknown positive.csv").write_text("a,b,c\nx,y,SLLMWITQV\n", encoding="utf-8")
    (tmp_path / "EBV mixed.csv").write_text("a,b,c\nx,y,SLLMWITQV\n", encoding="utf-8")
    (tmp_path / "EBV tcell.csv").write_text(
        "Sequence Id,Qualitative Measure\n1,Positive\n", encoding="utf-8"
    )

    df = loader.load_and_clean_iedb(str(tmp_path))
    out = capsys.readouterr().out

    assert df.empty and list(df.columns) == []
    assert "SKIP unknown positive.csv: cannot determine virus from filename" in out
    assert "SKIP EBV mixed.csv: cannot determine label from filename" in out
    assert "SKIP EBV tcell.csv: T-cell Assay format but missing required columns" in out
    assert "notes.txt" not in out
    assert "WARNING: No valid records loaded from any file" in out


def test_tcell_assay_row_filters_and_optional_allele(tmp_path, capsys):
    """Rows with no label or a non-standard peptide are dropped; a row without an
    allele is kept and simply carries no allele (NaN after the merge).
    """
    # pins behaviour, not correctness
    (tmp_path / "EBV tcell.csv").write_text(
        "Description,Qualitative Measure,Allele\n"
        "SLLMWITQV,Positive,A*02:01\n"
        "CLGGLLTMV,Positive-High,\n"
        "GLCTLVAML,Unknown,A*02:01\n"
        "SLLMWIXQV,Positive,A*02:01\n"
        "RAKFKQLL,Negative,DRB1*01:01\n",
        encoding="utf-8",
    )
    df = loader.load_and_clean_iedb(str(tmp_path)).set_index("peptide")
    capsys.readouterr()

    assert sorted(df.index) == ["CLGGLLTMV", "SLLMWITQV"]
    assert df.loc["SLLMWITQV", "allele"] == "HLA-A*02:01"
    assert pd.isna(df.loc["CLGGLLTMV", "allele"])
    assert df["label"].tolist() == [1, 1]


def test_hpv11_rows_found_by_organism_follow_the_include_flag(tmp_path, capsys):
    # pins behaviour, not correctness
    (tmp_path / "mixed tcell.csv").write_text(
        "Description,Qualitative Measure,Organism\n"
        "SLLMWITQV,Positive,Human papillomavirus type 11\n"
        "CLGGLLTMV,Negative,Epstein-Barr virus\n",
        encoding="utf-8",
    )
    default = loader.load_and_clean_iedb(str(tmp_path))
    included = loader.load_and_clean_iedb(str(tmp_path), include_hpv11=True)
    capsys.readouterr()

    assert default["peptide"].tolist() == ["CLGGLLTMV"]
    assert included.set_index("peptide")["virus"].to_dict() == {
        "CLGGLLTMV": "EBV",
        "SLLMWITQV": "HPV11",
    }
    assert included.set_index("peptide").loc["SLLMWITQV", "strain"] == "HPV11"


def test_a_label_tie_resolves_to_positive(tmp_path, capsys):
    """OWNER LOOK: duplicates are resolved by int(mean >= 0.5), so a 1-1 tie
    becomes a POSITIVE, and the report counts it as a conflict.
    """
    # pins behaviour, not correctness
    (tmp_path / "EBV tcell.csv").write_text(
        "Description,Qualitative Measure\nSLLMWITQV,Positive\nSLLMWITQV,Negative\n",
        encoding="utf-8",
    )
    df = loader.load_and_clean_iedb(str(tmp_path))
    out = capsys.readouterr().out
    assert df[["peptide", "label"]].to_dict("records") == [{"peptide": "SLLMWITQV", "label": 1}]
    assert "Duplicate Label Conflict Rate      : 100.00%" in out


# ---------------------------------------------------------------------------
# Name normalisers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "antigen, expected",
    [
        ("EBNA3B", "EBNA3A"),
        ("EBNA3C", "EBNA3A"),
        ("ebna3c protein", "EBNA3A"),
        ("EBNA-3B", "EBNA3B"),
        ("Epstein-Barr nuclear antigen 3C", "EBNA3C"),
        ("Tegument protein BNRF1", "Tegument protein BNRF1"),
    ],
)
def test_ebv_gene_mapping_order_folds_unhyphenated_ebna3b_and_3c_into_3a(antigen, expected):
    """OWNER LOOK: the map is scanned in insertion order and "ebna3" (mapped to
    EBNA3A) precedes "ebna3b"/"ebna3c", so the unhyphenated spellings of EBNA3B
    and EBNA3C are recorded as EBNA3A. Hyphenated and long-form names are not.
    """
    # pins behaviour, not correctness
    assert loader._infer_protein_gene(antigen, "EBV") == expected


@pytest.mark.parametrize(
    "raw, expected",
    [("A*0201", "HLA-A*0201"), ("hla-A*02:01", "HLA-hla-A*02:01"), ("HLA-B*07:02", "HLA-B*07:02")],
)
def test_standardize_allele_only_prefixes(raw, expected):
    """The docstring's example says A*0201 -> HLA-A*02:01; no colon is inserted,
    and a lower-case prefix is prefixed a second time.
    """
    # pins behaviour, not correctness
    assert loader.standardize_allele(raw) == expected


# ---------------------------------------------------------------------------
# Schmidt 2021 loader
# ---------------------------------------------------------------------------


def test_load_schmidt_reads_xlsx_and_drops_unrecognised_columns(tmp_path, capsys):
    """OWNER LOOK: load_schmidt_2021 sets hard_negative_flag=1 on every row it keeps,
    whatever its label, so an immunogenic (label 1) row is flagged as a hard
    negative. The function's docstring calls the dataset the hard-negative
    benchmark; whether a positive row should carry the flag is an owner decision.
    """
    # pins behaviour, not correctness
    path = _xlsx(
        tmp_path / "schmidt.xlsx",
        [
            ["Sequence", "MHC", "Immunogenic", "Donor"],
            ["slLMWITQV", "A*02:01", 1, "d1"],
            ["TOOLONGPEPTIDEX", "A*02:01", 0, "d2"],
        ],
    )
    df = loader.load_schmidt_2021(path)
    capsys.readouterr()
    assert df.to_dict("records") == [
        {"peptide": "SLLMWITQV", "hla": "HLA-A*02:01", "label": 1, "hard_negative_flag": 1}
    ]


# ---------------------------------------------------------------------------
# Module entry point
# ---------------------------------------------------------------------------


def test_entry_point_missing_directory_prints_to_stdout_and_exits_1(tmp_path):
    # pins behaviour, not correctness
    missing = tmp_path / "absent_dir"
    result = subprocess.run(
        [sys.executable, "-m", "src.iedb_data_loader", str(missing)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=300,
    )
    assert result.returncode == 1
    assert f"ERROR: Directory not found: {missing}" in result.stdout
    assert "ERROR" not in result.stderr


def test_entry_point_loads_a_directory_and_prints_the_report(tmp_path):
    # pins behaviour, not correctness
    (tmp_path / "EBV tcell.csv").write_text(
        "Description,Qualitative Measure\nSLLMWITQV,Positive\n", encoding="utf-8"
    )
    result = subprocess.run(
        [sys.executable, "-m", "src.iedb_data_loader", str(tmp_path), "--include-hpv11"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=300,
    )
    assert result.returncode == 0
    assert "EBV tcell.csv: T-cell Assay format, 1 valid records" in result.stdout
    assert "Unique Peptides Ingested           : 1 (vs. 720 baseline)" in result.stdout
