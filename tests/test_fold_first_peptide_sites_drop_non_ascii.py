"""End-to-end: seven more entry points DROP a non-ASCII peptide, never fold it into another.

Companion to tests/test_non_ascii_peptides_are_dropped_not_folded.py, which covers
src/iedb_data_loader.py, src/data_bias_audit.py, scripts/data_qc_gate.py and
scripts/extract_allele_aware_data.py. str.upper() maps some non-ASCII letters onto
amino-acid codes, so code that upper-cases a raw peptide BEFORE checking the alphabet
accepts residues the input never contained and, for two of the codepoints below, a
different length. Each entry point here did exactly that; each now tests for
non-ASCII first and drops the row through its own existing rejection path.

The tests drive each entry point on scratch input and assert on what it EMITS:

* scripts/fetch_iedb_tcell.fetch_and_save, with the IEDB page fetch replaced by an
  in-memory page. The CSV it writes is also read back through
  scripts/extract_allele_aware_data.load_tcell_assay_files, the reader of that format.
* scripts/_dataset_utils.normalize_peptides, on an in-memory frame and on a CSV
  round trip
* scripts/filter_validation_cohorts.clean_and_curate
* src/data_curation_qc.py --check-dataset, run as a subprocess in a scratch directory
* scripts/expand_allele_matched_negatives.curate_negative_records
* scripts/ingest_immunecode.main, on a MIRA-format CSV
* scripts/prepare_netmhcpan_inputs.prepare_inputs, on a held-out TSV

These are not every place that upper-cases before validating. In particular a caller
that upper-cases BEFORE it calls normalize_peptides hands it an already-substituted
ASCII string, which no check inside normalize_peptides can catch.

Every fixture codepoint is built with chr(). tests/test_encoding_ascii_output.py
inspects string literals by VALUE through the AST, so a literal glyph and its escaped
form are flagged alike, while chr() leaves only an int literal in this file.
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# (raw peptide, what str.upper() turns it into). Each folded form is a valid 8-11mer
# of standard residues, which _assert_is_a_real_substitution checks, so a row that
# survives an entry point would be indistinguishable from a real peptide.
FOLDING_ROWS = [
    ("SLLMWITQ" + chr(0x0131), "SLLMWITQI"),  # dotless i -> I
    ("SLLMWITQ" + chr(0x017F), "SLLMWITQS"),  # long s -> S
    ("GILGFVFT" + chr(0x00DF), "GILGFVFTSS"),  # sharp s -> SS, 9 -> 10 residues
    ("GILGFVFT" + chr(0xFB00), "GILGFVFTFF"),  # ff ligature -> FF, 9 -> 10 residues
]
FOLDED = {folded for _, folded in FOLDING_ROWS}

# ASCII controls that must keep coming through exactly as before: one upper-case,
# and one lower-case that every entry point here has always accepted and upper-cased.
ASCII_CONTROLS = [("SLLMWITQV", "SLLMWITQV"), ("gilgfvftl", "GILGFVFTL")]
EXPECTED = {emitted for _, emitted in ASCII_CONTROLS}


def _assert_is_a_real_substitution():
    # Anti-vacuity for the fixtures themselves: if a folded form were not a valid
    # peptide, an entry point would drop the row for an unrelated reason and these
    # tests would pass on the unguarded code too.
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


def test_fetch_and_save_drops_non_ascii_rows_instead_of_folding_them(tmp_path, monkeypatch):
    from scripts import fetch_iedb_tcell
    from scripts.extract_allele_aware_data import load_tcell_assay_files

    _assert_is_a_real_substitution()
    page = [
        {
            "epitope__name": raw,
            "assay__qualitative_measurement": "Positive",
            "assay__response_measured": "IFNg release",
            "mhc_restriction__name": "HLA-A*02:01",
            "epitope__source_molecule": "BMLF1",
            "epitope__source_organism": "human gammaherpesvirus 4",
            "reference__pmid": 1,
        }
        for raw in _raw_peptides()
    ]
    # One short page ends fetch_iedb_tcell's pagination loop; nothing reaches the network.
    monkeypatch.setattr(fetch_iedb_tcell, "_fetch_page", lambda url, **kwargs: list(page))
    out = tmp_path / "iedb_ebv_tcell.csv"

    returned = fetch_iedb_tcell.fetch_and_save("EBV", str(out))

    _assert_only_controls(returned["peptide"])
    _assert_only_controls(pd.read_csv(out, encoding="utf-8")["peptide"])
    # The processed-format reader sees only what the producer wrote.
    _assert_only_controls(load_tcell_assay_files(str(tmp_path))["peptide"])


def _frame_in_memory(tmp_path):
    raws = _raw_peptides()
    return pd.DataFrame({"peptide": raws, "label": [1] * len(raws)})


def _frame_from_csv(tmp_path):
    path = tmp_path / "ingest.csv"
    _frame_in_memory(tmp_path).to_csv(path, index=False, encoding="utf-8")
    return pd.read_csv(path, encoding="utf-8")


@pytest.mark.parametrize("build", [_frame_in_memory, _frame_from_csv])
def test_normalize_peptides_drops_non_ascii_rows_instead_of_folding_them(tmp_path, build):
    from scripts._dataset_utils import normalize_peptides

    _assert_is_a_real_substitution()

    df = normalize_peptides(build(tmp_path))

    _assert_only_controls(df["peptide"])
    assert len(df) == len(ASCII_CONTROLS)


def test_clean_and_curate_drops_non_ascii_rows_instead_of_folding_them():
    from scripts.filter_validation_cohorts import clean_and_curate

    _assert_is_a_real_substitution()
    records = [
        {
            "linear_sequence": raw,
            "qualitative_measure": "Positive",
            "mhc_allele_name": "HLA-A*02:01",
            "assay_names": "IFNg ELISPOT",
        }
        for raw in _raw_peptides()
    ]

    df = clean_and_curate(records, "InfluenzaA")

    _assert_only_controls(df["peptide"])
    assert len(df) == len(ASCII_CONTROLS)


def _run_check_dataset(tmp_path, peptides, tag):
    config = tmp_path / "config.yaml"
    config.write_text("freeze_mode: false\n", encoding="utf-8")
    dataset = tmp_path / f"{tag}.csv"
    labels = [i % 2 for i in range(len(peptides))]
    pd.DataFrame({"peptide": peptides, "label": labels}).to_csv(
        dataset, index=False, encoding="utf-8"
    )
    # cwd is the scratch dir: on success the script writes results/qc/ relative to cwd.
    return subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "src" / "data_curation_qc.py"),
            "--check-dataset",
            str(dataset),
            "--config",
            str(config),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=tmp_path,
    )


def test_data_curation_qc_check_dataset_rejects_non_ascii_instead_of_folding_it(tmp_path):
    _assert_is_a_real_substitution()

    controls = _run_check_dataset(tmp_path, [raw for raw, _ in ASCII_CONTROLS], "controls")
    folding = _run_check_dataset(tmp_path, _raw_peptides(), "folding")

    # The lower-case control is still accepted, so the failure below is the folding rows.
    assert controls.returncode == 0, controls.stdout + controls.stderr
    output = folding.stdout + folding.stderr
    assert folding.returncode == 1, f"--check-dataset admitted folding rows: {output}"
    assert f"Found {len(FOLDING_ROWS)} peptides with non-canonical amino acids" in output


def test_curate_negative_records_drops_non_ascii_rows_instead_of_folding_them():
    from scripts.expand_allele_matched_negatives import curate_negative_records

    _assert_is_a_real_substitution()
    records = [
        {"linear_sequence": raw, "mhc_allele_name": "HLA-A*11:01", "assay_names": "IFNg ELISPOT"}
        for raw in _raw_peptides()
    ]

    df = curate_negative_records(records)

    _assert_only_controls(df["peptide"])
    assert len(df) == len(ASCII_CONTROLS)


def test_ingest_immunecode_main_drops_non_ascii_rows_instead_of_folding_them(tmp_path):
    from scripts.ingest_immunecode import main

    _assert_is_a_real_substitution()
    raws = _raw_peptides()
    infile = tmp_path / "mira.csv"
    # An empty TCR column marks every row as a tested, non-expanding negative.
    pd.DataFrame(
        {
            "Peptide": raws,
            "HLA Restrictions": ["HLA-A*02:01"] * len(raws),
            "Amino Acids": [None] * len(raws),
        }
    ).to_csv(infile, index=False, encoding="utf-8")
    outfile = tmp_path / "negatives.csv"

    rc = main(["--input", str(infile), "--output", str(outfile)])

    assert rc == 0
    _assert_only_controls(pd.read_csv(outfile, encoding="utf-8")["peptide"])


def test_prepare_netmhcpan_inputs_drops_non_ascii_rows_instead_of_folding_them(tmp_path):
    from scripts.prepare_netmhcpan_inputs import prepare_inputs

    _assert_is_a_real_substitution()
    test_set_dir = tmp_path / "loo_test_sets"
    test_set_dir.mkdir()
    lines = ["peptide\thla_allele\tlabel"] + [f"{raw}\tHLA-A*02:01\t1" for raw in _raw_peptides()]
    (test_set_dir / "EBV_held_out.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_dir = tmp_path / "ext_scores"

    records = prepare_inputs(test_set_dir, out_dir)

    written = []
    for pep_file in sorted((out_dir / "inputs").glob("*.pep")):
        written.extend(pep_file.read_text(encoding="utf-8").split())
    _assert_only_controls(written)
    assert [r["n_peptides"] for r in records] == [len(ASCII_CONTROLS)]
