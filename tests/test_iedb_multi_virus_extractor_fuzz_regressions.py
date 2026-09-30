"""Regression and property tests for ``is_valid_peptide`` in the multi-virus extractor.

The atheris harness ``fuzz/fuzz_is_valid_peptide.py`` found that the predicate
accepted ``"RAKFKQ\\u0131L"`` (U+0131, LATIN SMALL LETTER DOTLESS I) as a valid 8-mer.
``str.upper()`` maps that letter onto ``"I"``, so the membership check ran on a
string the input never contained, and ``clean_and_pool_epitopes`` then stored the
upper-cased ``"RAKFKQIL"`` as if IEDB had returned it. Other letters do the same,
and some change the LENGTH as well: ``"\\u00df"`` (sharp s) upper-cases to ``"SS"``,
so four of them passed the 8 to 11 window as an 8-mer.

The contract these tests pin is the one the function's ``STANDARD_AA`` set
expresses: after stripping surrounding whitespace, the input must be 8 to 11
characters, each one of the 20 standard amino-acid letters in either case.

Non-ASCII test strings are built with ``chr()`` rather than written as literals,
because tests/test_encoding_ascii_output.py rejects non-ASCII string literals
outside its allowlist.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from src.verify.iedb_multi_virus_extractor import STANDARD_AA, is_valid_peptide

# The harness's recorded input, UTF-8 bytes b"RAKFKQ\xc4\xb1L". libFuzzer
# -minimize_crash=1 did not shrink it: deleting any one byte drops it below 8
# characters or breaks the two-byte U+0131. Shorter reproducers exist through other
# letters, e.g. "RAKFKQ" + U+00DF (8 bytes), which upper-cases to the 8-mer
# "RAKFKQSS".
DOTLESS_I, LONG_S, SHARP_S, LIGATURE_FF, LIGATURE_FI = (
    chr(0x0131),
    chr(0x017F),
    chr(0x00DF),
    chr(0xFB00),
    chr(0xFB01),
)
MINIMISED_REPRODUCER = "RAKFKQ" + DOTLESS_I + "L"

EITHER_CASE = frozenset(STANDARD_AA) | frozenset(aa.lower() for aa in STANDARD_AA)


def oracle(seq: str) -> bool:
    stripped = seq.strip()
    return 8 <= len(stripped) <= 11 and all(ch in EITHER_CASE for ch in stripped)


def test_minimised_fuzz_reproducer_is_rejected():
    assert is_valid_peptide(MINIMISED_REPRODUCER) is False


def test_letter_that_upper_cases_to_two_residues_is_rejected():
    assert is_valid_peptide(SHARP_S * 4) is False


def test_ascii_behaviour_is_unchanged():
    assert is_valid_peptide("SLLMWITQV") is True
    assert is_valid_peptide("  sllmwitqv ") is True
    assert is_valid_peptide("ACDEFGH") is False
    assert is_valid_peptide("ACDEFGHIKLMN") is False
    assert is_valid_peptide("SLLMWITQX") is False
    assert is_valid_peptide("") is False
    assert is_valid_peptide(np.nan) is False


@given(st.text(max_size=16))
def test_is_valid_peptide_matches_the_standard_amino_acid_oracle(seq: str):
    assert is_valid_peptide(seq) is oracle(seq)


@given(
    st.text(alphabet=sorted(EITHER_CASE), min_size=7, max_size=10),
    st.sampled_from([DOTLESS_I, LONG_S, SHARP_S, LIGATURE_FF, LIGATURE_FI]),
    st.integers(min_value=0, max_value=10),
)
def test_non_ascii_letter_inside_a_peptide_is_always_rejected(body: str, letter: str, at: int):
    at = min(at, len(body))
    assert is_valid_peptide(body[:at] + letter + body[at:]) is False
