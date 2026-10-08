"""Regression and property tests for the two parsers, found by fuzzing.

``parse_netchop_html``: the atheris harness ``fuzz/fuzz_parse_netchop_html.py``
found that one malformed residue row aborts the whole parse. The row pattern
accepts ``[0-9.]+`` in the score column, so a row such as ``"1 G . 0.1.2 pep_0"``
matches, and the unguarded ``float()`` then raised ``ValueError``.
``query_netchop`` catches only ``requests.exceptions.RequestException`` around
this call, so the error escaped the network client, bypassing its mock fallback,
and discarded every valid row in the same response. The position column had the
same shape: ``int()`` refuses a digit string longer than
``sys.get_int_max_str_digits()``. After the fix such a row is skipped, exactly
like a row whose ``pep_N`` index is out of range, and the rest of the table is
kept.

``parse_tapreg_html``: the harness's own contract treats more than 1 s of CPU on
one input as a finding. The text pattern's ``\\s+`` and unbounded ``.*?`` can both
consume the same whitespace, so a peptide followed by N spaces and no digit costs
O(N^2) backtracking - over 1 s of CPU at 16000 spaces on every host measured.
Corpus mutation did not reach this shape; it was found by hand. After the fix at
most 200 characters may sit between the whitespace after the peptide and the
score, which narrows the text pattern (a score further along the same line is no
longer found); the docstring promises only a "nearby" score.
"""

from __future__ import annotations

import sys
import time

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.external_predictors import parse_netchop_html, parse_tapreg_html

# Minimised by libFuzzer (-minimize_crash=1) from the harness's recorded input
# (590 bytes down to 14), decoded from the harness input layout: the first byte
# selects an empty peptide list and the rest is the response body. The score
# column is a lone ".", which the row pattern accepts and float() rejects.
MINIMISED_PEPTIDES: list[str] = []
MINIMISED_BODY = "2 A . . pep_1"


def test_minimised_fuzz_reproducer_does_not_raise():
    assert parse_netchop_html(MINIMISED_BODY, MINIMISED_PEPTIDES) == {}


def test_minimised_reproducer_row_is_skipped_when_its_index_is_in_range():
    result = parse_netchop_html(MINIMISED_BODY, ["AA", "AB"])
    assert result == {"AA": {"scores": [], "cleavages": []}, "AB": {"scores": [], "cleavages": []}}


def test_row_with_unconvertible_score_is_skipped_and_the_rest_kept():
    html = "  1 G .  0.1.2  pep_0\n  2 L .  0.08234  pep_0\n  3 F S  0.72312  pep_0\n"
    result = parse_netchop_html(html, ["GLF"])
    assert result == {"GLF": {"scores": [0.08234, 0.72312], "cleavages": [".", "S"]}}


def test_row_with_overlong_position_is_skipped():
    digits = "9" * (sys.get_int_max_str_digits() + 1)
    html = f"{digits} G .  0.5  pep_0\n  2 L .  0.25  pep_0\n"
    assert parse_netchop_html(html, ["GL"]) == {"GL": {"scores": [0.25], "cleavages": ["."]}}


def _parses(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


@given(st.lists(st.from_regex(r"[0-9.]{1,8}", fullmatch=True), min_size=1, max_size=6))
def test_every_score_the_row_pattern_admits_is_parsed_or_skipped(scores: list[str]):
    rows = [f"{i + 1} G . {score} pep_0" for i, score in enumerate(scores)]
    result = parse_netchop_html("\n".join(rows), ["G" * len(scores)])
    parsed = result["G" * len(scores)]["scores"]
    assert parsed == [float(s) for s in scores if _parses(s)]


@given(
    st.text(max_size=400),
    st.lists(st.text(alphabet="ACDEFGHIKLMNPQRSTVWY", min_size=1, max_size=12), max_size=4),
)
def test_parse_netchop_html_never_raises_and_keys_match(html: str, peptides: list[str]):
    result = parse_netchop_html(html, peptides)
    assert set(result) == set(peptides)
    for entry in result.values():
        assert len(entry["scores"]) == len(entry["cleavages"])


# Hand-built worst case for parse_tapreg_html, not found by the fuzzer's corpus
# mutation within its budget (see fuzz/fuzz_parse_tapreg_html.py's docstring).
SLOW_INPUT_PEPTIDE = "GLF"
SLOW_INPUT_TRAILING_SPACES = 16000
CPU_BUDGET_SECONDS = 1.0


def test_hand_built_slow_input_stays_within_the_cpu_budget():
    text = SLOW_INPUT_PEPTIDE + " " * SLOW_INPUT_TRAILING_SPACES
    start = time.process_time()
    result = parse_tapreg_html(text, [SLOW_INPUT_PEPTIDE])
    spent = time.process_time() - start
    assert result == {}
    assert spent < CPU_BUDGET_SECONDS, f"{spent:.3f} s exceeds the {CPU_BUDGET_SECONDS} s budget"


def test_parse_tapreg_html_still_finds_a_score_within_the_bounded_window():
    text = "pep_0  GLFYTRTGL  1.2345\npep_1  AAYSDQWAL  0.9876"
    scores = parse_tapreg_html(text, ["GLFYTRTGL", "AAYSDQWAL"])
    assert scores == {"GLFYTRTGL": 1.2345, "AAYSDQWAL": 0.9876}


def test_parse_tapreg_html_html_table_fallback_is_unaffected():
    html = "<table>\n<tr><td>GLFYTRTGL</td><td>1.2345</td></tr>\n</table>"
    assert parse_tapreg_html(html, ["GLFYTRTGL"]) == {"GLFYTRTGL": 1.2345}


@pytest.mark.parametrize("spaces", [0, 500, 4096, 16000, 32000])
def test_parse_tapreg_html_cpu_time_does_not_grow_with_trailing_junk(spaces: int):
    """Fixed sizes, not Hypothesis: a CPU-time assertion driven by a property engine
    is flaky under CI load. The pre-fix cost is quadratic and host-dependent: 16000
    spaces have cost 1.4 to 3.3 s of CPU across the runs measured, so the 16000 case
    alone does not reliably exceed the 3 s ceiling. The 32000 case does (5.6 to
    10.7 s pre-fix in the same runs), while the post-fix cost stayed under 0.1 s at
    every size here, so only a reintroduced unbounded backtrack would trip it.
    """
    text = "GLF" + " " * spaces
    start = time.process_time()
    result = parse_tapreg_html(text, ["GLF"])
    spent = time.process_time() - start
    assert result == {}
    assert spent < 3.0, f"{spaces} trailing spaces cost {spent:.3f} s of CPU"
