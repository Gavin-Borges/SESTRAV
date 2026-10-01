"""End-to-end: the entry points below DROP a non-ASCII peptide, never fold it into another.

str.upper() maps some non-ASCII letters onto amino-acid codes, so code that folds a
raw peptide BEFORE validating it accepts residues the input never contained and, for
two of the codepoints below, a different length. The failure mode is substitution,
not rejection: corrupt input becomes a different, well-formed peptide of standard
residues and is indistinguishable from real data downstream.

Rejecting non-ASCII inside a validator is not enough on its own. Every production
caller of src.iedb_data_loader.is_valid_peptide upper-cased the value BEFORE calling
it, so the validator only ever saw the already-substituted ASCII string. These tests
therefore drive the public entry points on scratch files and assert on what they
EMIT, not on any validator's return value:

* src.iedb_data_loader.load_schmidt_2021
* src.iedb_data_loader.load_and_clean_iedb, Epitope Table and T-cell Assay formats
* src.data_bias_audit._collect_raw_records, both formats
* scripts/data_qc_gate.py, run as a subprocess, which must quarantine the rows
* scripts/extract_allele_aware_data.load_tcell_assay_files, all three formats. The
  processed-flat fixture holds raw non-ASCII, which that format's producer,
  scripts/fetch_iedb_tcell.py, drops before writing; that producer is covered by
  tests/test_fold_first_peptide_sites_drop_non_ascii.py.

These are not every place that folds before validating. Other code that does is not
covered here, so a pass here says nothing about it.

Every fixture codepoint is built with chr(). tests/test_encoding_ascii_output.py
inspects string literals by VALUE through the AST, so a literal glyph and its escaped
form are flagged alike, while chr() leaves only an int literal in this file.
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.data_bias_audit import _collect_raw_records
from src.iedb_data_loader import STANDARD_AA, load_and_clean_iedb, load_schmidt_2021

REPO_ROOT = Path(__file__).resolve().parents[1]

# (raw peptide, what str.upper() turns it into). Each folded form is a valid 8-11mer
# of standard residues, which _assert_is_a_real_substitution checks, so a row that
# survives a loader would be indistinguishable from a real peptide.
FOLDING_ROWS = [
    ("SLLMWITQ" + chr(0x0131), "SLLMWITQI"),  # dotless i -> I
    ("SLLMWITQ" + chr(0x017F), "SLLMWITQS"),  # long s -> S
    ("GILGFVFT" + chr(0x00DF), "GILGFVFTSS"),  # sharp s -> SS, 9 -> 10 residues
    ("GILGFVFT" + chr(0xFB00), "GILGFVFTFF"),  # ff ligature -> FF, 9 -> 10 residues
]
FOLDED = {folded for _, folded in FOLDING_ROWS}

# ASCII controls that must keep loading exactly as before: one upper-case, and one
# lower-case that the loaders have always accepted and upper-cased.
ASCII_CONTROLS = [("SLLMWITQV", "SLLMWITQV"), ("gilgfvftl", "GILGFVFTL")]
EXPECTED = {emitted for _, emitted in ASCII_CONTROLS}


def _assert_is_a_real_substitution():
    # Anti-vacuity for the fixtures themselves: if a folded form were not a valid
    # peptide, the loaders would drop the row for an unrelated reason and these
    # tests would pass on the broken code too.
    for raw, folded in FOLDING_ROWS:
        assert not raw.isascii()
        assert raw.upper() == folded
        assert 8 <= len(folded) <= 11
        assert set(folded) <= STANDARD_AA


def _raw_peptides():
    return [raw for raw, _ in ASCII_CONTROLS] + [raw for raw, _ in FOLDING_ROWS]


def _assert_only_controls(peptides):
    emitted = list(peptides)
    leaked = sorted(FOLDED & set(emitted))
    assert not leaked, f"non-ASCII input was folded into {leaked}"
    assert sorted(emitted) == sorted(EXPECTED)


def _write_epitope_table(directory):
    lines = ["c0,c1,Name"] + [f"x,y,{raw}" for raw in _raw_peptides()]
    path = directory / "HPV16_T-cell_positive.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_tcell_assay(directory):
    lines = ["Description,Qualitative Measure,Allele"] + [
        f"{raw},Positive,A*02:01" for raw in _raw_peptides()
    ]
    path = directory / "EBV_tcell_assay.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_load_schmidt_2021_drops_non_ascii_rows_instead_of_folding_them(tmp_path):
    _assert_is_a_real_substitution()
    path = tmp_path / "schmidt.csv"
    raws = _raw_peptides()
    pd.DataFrame(
        {"peptide": raws, "hla": ["A*02:01"] * len(raws), "label": [0] * len(raws)}
    ).to_csv(path, index=False, encoding="utf-8")

    df = load_schmidt_2021(str(path))

    _assert_only_controls(df["peptide"])
    assert len(df) == len(ASCII_CONTROLS)


@pytest.mark.parametrize("writer", [_write_epitope_table, _write_tcell_assay])
def test_load_and_clean_iedb_drops_non_ascii_rows_instead_of_folding_them(tmp_path, writer):
    _assert_is_a_real_substitution()
    writer(tmp_path)

    df = load_and_clean_iedb(str(tmp_path))

    _assert_only_controls(df["peptide"])
    assert len(df) == len(ASCII_CONTROLS)


@pytest.mark.parametrize("writer", [_write_epitope_table, _write_tcell_assay])
def test_data_bias_audit_raw_records_drop_non_ascii_rows_instead_of_folding_them(tmp_path, writer):
    _assert_is_a_real_substitution()
    writer(tmp_path)

    df = _collect_raw_records(str(tmp_path))

    _assert_only_controls(df["peptide"])
    assert len(df) == len(ASCII_CONTROLS)


def test_data_qc_gate_quarantines_non_ascii_rows_instead_of_folding_them(tmp_path):
    # Thresholds sized for a tiny corpus, so the ONLY check the folding rows can
    # trip is length_and_composition_valid. Before the fix they folded, passed every
    # row-level check, and the gate exited 0.
    _assert_is_a_real_substitution()
    config = tmp_path / "config.yaml"
    config.write_text(
        "dataset_governance:\n"
        "  qc_thresholds:\n"
        "    min_peptide_yield: 2\n"
        "    max_conflict_ratio: 0.15\n"
        "    max_null_allele_fraction: 0.50\n"
        "    class_ratio_bounds: [0.1, 10.0]\n"
        "freeze_mode: false\n",
        encoding="utf-8",
    )
    raws = _raw_peptides()
    labels = [1, 0] + [1] * len(FOLDING_ROWS)
    dataset = tmp_path / "corpus.csv"
    pd.DataFrame({"peptide": raws, "label": labels, "allele": ["HLA-A*02:01"] * len(raws)}).to_csv(
        dataset, index=False, encoding="utf-8"
    )
    quarantine = tmp_path / "quarantine.csv"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "data_qc_gate.py"),
            "--dataset",
            str(dataset),
            "--config",
            str(config),
            "--quarantine",
            str(quarantine),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=REPO_ROOT,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 1, f"gate admitted folding rows: {output}"
    assert quarantine.exists(), output
    q = pd.read_csv(quarantine, encoding="utf-8")
    assert sorted(q["peptide"]) == sorted(raw for raw, _ in FOLDING_ROWS)
    assert set(q["qc_failure_reason"]) == {"Non-ASCII characters present in sequence"}


def _write_processed_flat(directory):
    raws = _raw_peptides()
    pd.DataFrame(
        {
            "peptide": raws,
            "label": [1] * len(raws),
            "virus": ["EBV"] * len(raws),
            "hla_allele": ["HLA-A*02:01"] * len(raws),
        }
    ).to_csv(directory / "ebv_processed.csv", index=False, encoding="utf-8")


def _write_multiheader(directory):
    lines = ["Epitope,Assay,Host", "Name,Qualitative Measurement,MHC Present"] + [
        f"{raw},Positive,HLA-A*02:01" for raw in _raw_peptides()
    ]
    (directory / "ebv_multiheader.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_flat_qualitative(directory):
    lines = ["Description,Qualitative Measure,Allele"] + [
        f"{raw},Positive,HLA-A*02:01" for raw in _raw_peptides()
    ]
    (directory / "ebv_flat.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.mark.parametrize(
    "writer", [_write_processed_flat, _write_multiheader, _write_flat_qualitative]
)
def test_extract_allele_aware_data_drops_non_ascii_rows_instead_of_folding_them(tmp_path, writer):
    from scripts.extract_allele_aware_data import load_tcell_assay_files

    _assert_is_a_real_substitution()
    writer(tmp_path)

    df = load_tcell_assay_files(str(tmp_path))

    _assert_only_controls(df["peptide"])
    assert len(df) == len(ASCII_CONTROLS)
