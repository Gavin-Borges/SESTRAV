"""Tests for scripts/generate_targets.py.

This script writes the candidate shortlist, and nothing imported it before: the
window extraction, the amino-acid filter and the virus label were all untested.
The virus label is the one that was wrong. It came from the filename, so every
record in a file naming two viruses got the same label.

Written against fixtures in a temporary directory, so these run in a clone with
no proteome files present.
"""

from __future__ import annotations

from pathlib import Path

from scripts.generate_targets import slice_fasta_sequences, virus_from_record

# UniProt ids as the shipped panels carry them: sp|<accession>|<entry name>.
HPV16_ID = "sp|P03126|VE6_HPV16"
HPV18_ID = "sp|P06463|VE6_HPV18"
EBV_ID = "sp|P03211|EBNA1_EBVB9"


def _fasta(path: Path, records: list[tuple[str, str]]) -> Path:
    path.write_text(
        "".join(f">{rid}\n{seq}\n" for rid, seq in records),
        encoding="utf-8",
    )
    return path


def test_virus_comes_from_the_record_not_the_file_name() -> None:
    """Two viruses in one file must keep their own labels.

    The shipped default file names both HPV16 and HPV18, so a filename-derived
    label is wrong for half its records.

    Each fallback differs from the expected answer. Passing the answer as the
    fallback makes the assertion pass whether the record is read or not, which
    is how an earlier version of this test let a deleted branch survive.
    """
    assert virus_from_record(HPV16_ID, "Unknown") == "HPV16"
    assert virus_from_record(HPV18_ID, "HPV16") == "HPV18"
    assert virus_from_record(EBV_ID, "Unknown") == "EBV"


def test_virus_falls_back_when_the_record_names_no_virus() -> None:
    assert virus_from_record("sp|Q00000|UNKNOWN_ENTRY", "Unknown") == "Unknown"
    assert virus_from_record("", "EBV") == "EBV"


def test_virus_reads_any_hpv_type_in_any_case() -> None:
    """The type is taken from the id, so a third type is not folded into HPV16."""
    assert virus_from_record("sp|P17386|VE7_HPV31", "HPV16") == "HPV31"
    assert virus_from_record("hpv-45_e6", "HPV16") == "HPV45"


def test_slice_labels_each_record_separately(tmp_path: Path) -> None:
    """A file holding both types yields both labels, whatever name is passed in."""
    path = _fasta(
        tmp_path / "HPV16_18_panel8.fasta",
        [(HPV16_ID, "MHQKRTAMFQ"), (HPV18_ID, "ARFEDPTRRP")],
    )

    extracted = slice_fasta_sequences(str(path), "HPV16")

    by_virus: dict[str, set[str]] = {}
    for row in extracted:
        by_virus.setdefault(row["virus"], set()).add(row["source_id"])
    assert by_virus == {"HPV16": {HPV16_ID}, "HPV18": {HPV18_ID}}


def test_slice_yields_one_window_per_start_position(tmp_path: Path) -> None:
    """A sequence of length n yields n - 8 nine-mers, in order, each tagged."""
    path = _fasta(tmp_path / "ebv.fasta", [(EBV_ID, "ACDEFGHIKLM")])

    extracted = slice_fasta_sequences(str(path), "EBV")

    assert [row["peptide"] for row in extracted] == ["ACDEFGHIK", "CDEFGHIKL", "DEFGHIKLM"]
    assert {row["source_id"] for row in extracted} == {EBV_ID}


def test_slice_drops_windows_holding_a_non_standard_residue(tmp_path: Path) -> None:
    """Only the 20 standard amino acids pass, so a window covering X is dropped.

    The sequence is 14 residues with X at index 4, so it forms 6 windows and X
    falls inside the first 5. Exactly one survives, which is what separates
    filtering from dropping the record: an empty result would pass a weaker
    assertion on its own.
    """
    path = _fasta(tmp_path / "ebv.fasta", [(EBV_ID, "ACDEXGHIKLMNPQ")])

    peptides = [row["peptide"] for row in slice_fasta_sequences(str(path), "EBV")]

    assert peptides == ["GHIKLMNPQ"]


def test_slice_returns_nothing_when_the_file_is_absent(tmp_path: Path) -> None:
    assert slice_fasta_sequences(str(tmp_path / "absent.fasta"), "EBV") == []
