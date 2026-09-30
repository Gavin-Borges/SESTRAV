"""Regression and property tests for ``_protein_name_from_header`` found by fuzzing.

The atheris harness ``fuzz/fuzz_protein_name_from_header.py`` crashed on its very
first input: an empty header. ``header.split()`` returns ``[]`` and the fallback
``parts[0]`` raised ``IndexError``. ``parse_fasta`` reaches it for any FASTA line
that is a bare ``>`` (after stripping), such as a truncated download, and it
raised there before a single protein was returned.

After the fix an empty or whitespace-only header has no token and yields ``""``,
which is the same value the UniProt branch already returns for a degenerate
header such as ``"sp| desc"``.
"""

from __future__ import annotations

from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

from src.build_predig_recombinant_input import _protein_name_from_header, parse_fasta

# The harness's recorded input was already empty (0 bytes), so -minimize_crash=1
# had nothing to remove.
MINIMISED_REPRODUCER = ""


def test_minimised_fuzz_reproducer_returns_empty_name():
    assert _protein_name_from_header(MINIMISED_REPRODUCER) == ""


def test_whitespace_only_header_returns_empty_name():
    # chr() rather than a literal: tests/test_encoding_ascii_output.py rejects
    # non-ASCII string literals outside its allowlist.
    ideographic_space = chr(0x3000)
    assert _protein_name_from_header(" \t" + ideographic_space) == ""


def test_documented_behaviour_is_unchanged():
    uniprot = "sp|P03206|BZLF1_EBVB9 Lytic switch protein BZLF1 OS=Epstein-Barr virus"
    assert _protein_name_from_header(uniprot) == "BZLF1_EBVB9"
    assert _protein_name_from_header("smoke_antigen") == "smoke_antigen"
    assert _protein_name_from_header("tr_like first second") == "tr_like"
    assert _protein_name_from_header("sp| desc") == ""


def test_parse_fasta_survives_a_bare_header_line(tmp_path: Path):
    fasta = tmp_path / "bare_header.fasta"
    fasta.write_text(">\nMKT\n>sp|P1|NAME_X desc\nAAA\n", encoding="utf-8")
    proteins = parse_fasta(str(fasta))
    assert proteins == {"": ("", "MKT"), "NAME_X": ("sp|P1|NAME_X desc", "AAA")}


@given(st.text(max_size=80))
def test_protein_name_from_header_never_raises(header: str):
    name = _protein_name_from_header(header)
    assert isinstance(name, str)
    parts = header.split()
    if parts and not parts[0].startswith("sp|"):
        assert name == parts[0]
    if not parts:
        assert name == ""
