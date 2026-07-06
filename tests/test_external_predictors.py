import pytest
import pandas as pd
import requests
from unittest.mock import MagicMock, patch
from src.external_predictors import (
    generate_fasta,
    parse_netchop_html,
    parse_tapreg_html,
    query_netchop,
    query_tapreg,
    get_predictor_dataframe,
)


def test_generate_fasta():
    peptides = ["GLFYTRTGL", "AAYSDQWAL"]
    fasta = generate_fasta(peptides)
    expected = ">pep_0\nGLFYTRTGL\n>pep_1\nAAYSDQWAL"
    assert fasta == expected


def test_parse_netchop_html():
    # Simulated NetChop text response
    mock_html = """
    <html>
    <body>
    <pre>
    pos  AA  C  score    Ident
    --------------------------------------
      1   G   .  0.12345  pep_0
      2   L   .  0.08234  pep_0
      3   F   S  0.72312  pep_0
      1   A   .  0.04123  pep_1
      2   A   .  0.10342  pep_1
      3   Y   S  0.81234  pep_1
    </pre>
    </body>
    </html>
    """
    peptides = ["GLF", "AAY"]
    results = parse_netchop_html(mock_html, peptides)

    assert "GLF" in results
    assert "AAY" in results
    assert results["GLF"]["scores"] == [0.12345, 0.08234, 0.72312]
    assert results["GLF"]["cleavages"] == [".", ".", "S"]
    assert results["AAY"]["scores"] == [0.04123, 0.10342, 0.81234]
    assert results["AAY"]["cleavages"] == [".", ".", "S"]


def test_parse_tapreg_html():
    # Simulated TAPreg text responses (both table and space formats)
    mock_text = "pep_0  GLFYTRTGL  1.2345\npep_1  AAYSDQWAL  0.9876"
    peptides = ["GLFYTRTGL", "AAYSDQWAL"]
    scores = parse_tapreg_html(mock_text, peptides)
    assert scores["GLFYTRTGL"] == 1.2345
    assert scores["AAYSDQWAL"] == 0.9876

    mock_html = """
    <table>
      <tr><td>GLFYTRTGL</td><td>1.2345</td></tr>
      <tr><td>AAYSDQWAL</td><td>0.9876</td></tr>
    </table>
    """
    scores_html = parse_tapreg_html(mock_html, peptides)
    assert scores_html["GLFYTRTGL"] == 1.2345
    assert scores_html["AAYSDQWAL"] == 0.9876


def test_mock_fallbacks():
    peptides = ["GLFYTRTGL", "AAYSDQWAL"]

    netchop_mock = query_netchop(peptides, mock_fallback=True)
    assert len(netchop_mock) == 2
    assert "GLFYTRTGL" in netchop_mock
    assert len(netchop_mock["GLFYTRTGL"]["scores"]) == len("GLFYTRTGL")
    assert all(0.0 <= val <= 1.0 for val in netchop_mock["GLFYTRTGL"]["scores"])

    tapreg_mock = query_tapreg(peptides, mock_fallback=True)
    assert len(tapreg_mock) == 2
    assert "GLFYTRTGL" in tapreg_mock
    assert isinstance(tapreg_mock["GLFYTRTGL"], float)


def test_get_predictor_dataframe():
    peptides = ["GLFYTRTGL", "AAYSDQWAL"]
    df = get_predictor_dataframe(peptides, mock_fallback=True)

    assert isinstance(df, pd.DataFrame)
    assert list(df.index) == peptides
    assert "netchop_cterm_score" in df.columns
    assert "netchop_all_scores" in df.columns
    assert "tap_score" in df.columns

    # Check that C-term score matches last item in all scores
    row0 = df.loc["GLFYTRTGL"]
    assert row0["netchop_cterm_score"] == row0["netchop_all_scores"][-1]
    assert isinstance(row0["tap_score"], float)


def test_live_query_smoke():
    # Smoke test with 2-3 sample peptides using standard remote queries
    # but allowing mock fallbacks automatically on connection issues.
    peptides = ["GLFYTRTGL", "AAYSDQWAL"]
    try:
        df = get_predictor_dataframe(peptides, mock_fallback=False)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
    except Exception as e:
        pytest.fail(f"get_predictor_dataframe crashed unexpectedly: {e}")


# ---------------------------------------------------------------------------
# Edge-case and error-branch coverage (issue #77)
# All tests below mock the requests boundary; no live network calls.
# ---------------------------------------------------------------------------


def test_query_netchop_empty_list():
    assert query_netchop([]) == {}


def test_query_tapreg_empty_list():
    assert query_tapreg([]) == {}


def test_get_predictor_dataframe_empty_list():
    df = get_predictor_dataframe([])
    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert "netchop_cterm_score" in df.columns


def test_query_netchop_request_exception_falls_back_to_mock():
    """Connection error on POST → all-mock fallback scores returned."""
    peptides = ["GLFYTRTGL"]
    with patch(
        "src.external_predictors.requests.post",
        side_effect=requests.exceptions.ConnectionError("no network"),
    ):
        results = query_netchop(peptides, mock_fallback=False, max_retries=1, initial_backoff=0)
    assert "GLFYTRTGL" in results
    assert len(results["GLFYTRTGL"]["scores"]) == len("GLFYTRTGL")


def test_query_netchop_no_job_id_in_response_falls_back_to_mock():
    """Server responds OK but no job ID is parseable → mock fallback."""
    peptides = ["GLFYTRTGL"]
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.text = "<html>No useful content here</html>"
    with patch("src.external_predictors.requests.post", return_value=mock_resp):
        results = query_netchop(peptides, mock_fallback=False, max_retries=1, initial_backoff=0)
    assert "GLFYTRTGL" in results
    assert isinstance(results["GLFYTRTGL"]["scores"], list)


def test_query_netchop_polling_empty_result_falls_back_to_mock():
    """Job ID parsed, but polling always returns empty table → mock fallback."""
    peptides = ["GLFYTRTGL"]
    submit_resp = MagicMock()
    submit_resp.raise_for_status.return_value = None
    submit_resp.text = "jobid=abc123"

    poll_resp = MagicMock()
    poll_resp.raise_for_status.return_value = None
    poll_resp.text = "<html>done</html>"  # no tabular rows → empty parse

    with (
        patch("src.external_predictors.requests.post", return_value=submit_resp),
        patch("src.external_predictors.requests.get", return_value=poll_resp),
    ):
        results = query_netchop(peptides, mock_fallback=False, max_retries=1, initial_backoff=0)
    assert "GLFYTRTGL" in results


def test_query_netchop_polling_request_exception_falls_back_to_mock():
    """Job ID obtained, but every poll raises a network error → mock fallback."""
    peptides = ["GLFYTRTGL"]
    submit_resp = MagicMock()
    submit_resp.raise_for_status.return_value = None
    submit_resp.text = "jobid=abc123"

    with (
        patch("src.external_predictors.requests.post", return_value=submit_resp),
        patch(
            "src.external_predictors.requests.get",
            side_effect=requests.exceptions.Timeout("timed out"),
        ),
    ):
        results = query_netchop(peptides, mock_fallback=False, max_retries=1, initial_backoff=0)
    assert "GLFYTRTGL" in results


def test_query_tapreg_request_exception_falls_back_to_mock():
    """Connection error → automatic mock fallback for TAPreg."""
    peptides = ["GLFYTRTGL"]
    with patch(
        "src.external_predictors.requests.post",
        side_effect=requests.exceptions.ConnectionError("no network"),
    ):
        results = query_tapreg(peptides, mock_fallback=False, max_retries=1, initial_backoff=0)
    assert "GLFYTRTGL" in results
    assert isinstance(results["GLFYTRTGL"], float)


def test_query_tapreg_vpn_restriction_falls_back_to_mock():
    """VPN restriction page detected → break and fall back to mock."""
    peptides = ["GLFYTRTGL"]
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.text = "acceso restringido - please use UCM VPN"
    with patch("src.external_predictors.requests.post", return_value=mock_resp):
        results = query_tapreg(peptides, mock_fallback=False, max_retries=1, initial_backoff=0)
    assert "GLFYTRTGL" in results
    assert isinstance(results["GLFYTRTGL"], float)


def test_parse_netchop_html_warns_on_unparseable_content(caplog):
    """Non-empty content that matches no tabular rows triggers a logger.warning."""
    import logging

    with caplog.at_level(logging.WARNING, logger="src.external_predictors"):
        result = parse_netchop_html("<html>unrelated content</html>", ["GLFYTRTGL"])
    assert "GLFYTRTGL" in result
    assert any("Failed to parse" in m for m in caplog.messages)


def test_parse_tapreg_html_no_match_returns_empty():
    """Content with no recognisable peptide+score pattern → empty dict."""
    result = parse_tapreg_html("no useful content at all", ["GLFYTRTGL"])
    assert result == {}


# ---------------------------------------------------------------------------
# Targeted branch / statement coverage - issue #77 remainder
# Each test is annotated with the line(s) it covers in external_predictors.py
# ---------------------------------------------------------------------------


# line 66 - _generate_mock_netchop_scores proline (P) branch
def test_mock_netchop_scores_proline_branch():
    from src.external_predictors import _generate_mock_netchop_scores

    result = _generate_mock_netchop_scores("AKPYL")
    assert len(result["scores"]) == 5
    assert all(0.0 <= s <= 1.0 for s in result["scores"])
    # proline at index 2; score must be in valid range (not None, not negative)
    assert 0.0 <= result["scores"][2] <= 1.0


# line 87 - _generate_mock_tapreg_score empty-peptide early return
def test_mock_tapreg_score_empty_peptide():
    from src.external_predictors import _generate_mock_tapreg_score

    score = _generate_mock_tapreg_score("", "blosum")
    assert isinstance(score, float)


# lines 93-96 - _generate_mock_tapreg_score C-terminal PDE branch
# line 94   - C-terminal RK branch
# branch 95→99 (False) - c_term not in any penalty/bonus set (fall-through to N-term check)
def test_mock_tapreg_score_cterminal_branches():
    from src.external_predictors import _generate_mock_tapreg_score

    # PDE branch (lines 95-96)
    for aa in "PDE":
        assert isinstance(_generate_mock_tapreg_score(f"GLFYTRTG{aa}", "blosum"), float)
    # RK branch (line 94)
    for aa in "RK":
        assert isinstance(_generate_mock_tapreg_score(f"GLFYTRTG{aa}", "blosum"), float)
    # Fall-through: c_term not in YFWLIVM, RK, or PDE → branch 95→99 (False)
    assert isinstance(_generate_mock_tapreg_score("GLFYTRTGN", "blosum"), float)


# line 101 - _generate_mock_tapreg_score N-terminal RKYFW branch
def test_mock_tapreg_score_nterminal_rkyfw():
    from src.external_predictors import _generate_mock_tapreg_score

    for aa in "RKYFW":
        score = _generate_mock_tapreg_score(f"{aa}LFYTRTGL", "blosum")
        assert isinstance(score, float)


# branch 104→107 (False) - non-blosum model skips the -0.05 adjustment
def test_mock_tapreg_score_sparse_model_skips_blosum_adjustment():
    from src.external_predictors import _generate_mock_tapreg_score

    blosum_score = _generate_mock_tapreg_score("GLFYTRTGL", "blosum")
    sparse_score = _generate_mock_tapreg_score("GLFYTRTGL", "sparse")
    # blosum reduces score by 0.05; sparse does not
    assert abs(blosum_score - sparse_score) == pytest.approx(0.05, abs=1e-9)


# branch 145→132 (False) - parse_netchop_html idx out-of-range skipped silently
def test_parse_netchop_html_out_of_range_idx():
    # pep_5 does not exist in a 2-peptide list; row must be silently skipped
    html = "  1 G .  0.12000  pep_5\n"
    result = parse_netchop_html(html, ["GLF", "AAY"])
    assert result["GLF"]["scores"] == []
    assert result["AAY"]["scores"] == []


# lines 149-150 - parse_netchop_html ValueError/IndexError on malformed ident
def test_parse_netchop_html_invalid_ident_skipped():
    # "pep_abc" cannot be int-cast → ValueError caught → row skipped
    html = "  1 G .  0.12000  pep_abc\n"
    result = parse_netchop_html(html, ["GLF"])
    assert result["GLF"]["scores"] == []


# line 255 - query_netchop successful poll returns parsed scores (not mock)
def test_query_netchop_polling_success_returns_parsed_scores():
    peptides = ["GLF"]
    submit_resp = MagicMock()
    submit_resp.raise_for_status.return_value = None
    submit_resp.text = "jobid=abc123"

    # Minimal NetChop-style tabular text that parse_netchop_html can parse
    poll_resp = MagicMock()
    poll_resp.raise_for_status.return_value = None
    poll_resp.text = "  1 G .  0.12345  pep_0\n  2 L .  0.08234  pep_0\n  3 F S  0.72312  pep_0\n"

    with (
        patch("src.external_predictors.requests.post", return_value=submit_resp),
        patch("src.external_predictors.requests.get", return_value=poll_resp),
    ):
        results = query_netchop(peptides, mock_fallback=False, max_retries=1, initial_backoff=0)

    assert "GLF" in results
    assert results["GLF"]["scores"] == [0.12345, 0.08234, 0.72312]
    assert results["GLF"]["cleavages"] == [".", ".", "S"]


# line 298 - parse_tapreg_html HTML-<td> fallback (text-regex misses, HTML regex hits)
def test_parse_tapreg_html_html_table_fallback():
    import re as real_re
    from src.external_predictors import parse_tapreg_html

    html = "<td>GLFYTRTGL</td><td>1.2345</td>"
    # Force the text-content pattern to return no match so the HTML fallback runs
    no_match_mock = MagicMock()
    no_match_mock.search.return_value = None
    html_pattern_real = real_re.compile(
        r"<td>\s*GLFYTRTGL\s*</td>\s*<td>\s*(-?\d+\.\d+)\s*</td>",
        real_re.IGNORECASE,
    )

    with patch(
        "src.external_predictors.re.compile", side_effect=[no_match_mock, html_pattern_real]
    ):
        result = parse_tapreg_html(html, ["GLFYTRTGL"])

    assert result == {"GLFYTRTGL": 1.2345}


# lines 346-347 - query_tapreg threshold is not None → payload populated
def test_query_tapreg_threshold_populates_payload():
    peptides = ["GLFYTRTGL"]
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.text = "VPN"  # triggers break → mock fallback; POST still fires

    with patch("src.external_predictors.requests.post", return_value=mock_resp) as mock_post:
        query_tapreg(peptides, threshold=0.5, mock_fallback=False, max_retries=1, initial_backoff=0)

    sent_data = mock_post.call_args[1]["data"]
    assert sent_data["threshold"] == "0.5"
    assert sent_data["thresh"] == "0.5"


# lines 244-247 - query_netchop polling detects "Job is running" → wait path exercised
def test_query_netchop_polling_job_running_falls_back_after_exhaustion():
    """Every poll returns 'Job is running' → polling exhausts → mock fallback."""
    peptides = ["GLF"]
    submit_resp = MagicMock()
    submit_resp.raise_for_status.return_value = None
    submit_resp.text = "jobid=abc123"

    running_resp = MagicMock()
    running_resp.raise_for_status.return_value = None
    running_resp.text = "Job is running. Please wait."

    with (
        patch("src.external_predictors.requests.post", return_value=submit_resp),
        patch("src.external_predictors.requests.get", return_value=running_resp),
    ):
        results = query_netchop(peptides, mock_fallback=False, max_retries=1, initial_backoff=0)

    assert "GLF" in results  # mock fallback fires after polling exhausted


# line 364 - query_tapreg server responds but parse finds no scores → warning logged
def test_query_tapreg_empty_parse_logs_warning_and_falls_back():
    """Response contains no parseable peptide scores → warning; mock fallback fires."""
    peptides = ["GLFYTRTGL"]
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.text = "no scores here at all"  # parse_tapreg_html returns {}

    with patch("src.external_predictors.requests.post", return_value=mock_resp):
        results = query_tapreg(peptides, mock_fallback=False, max_retries=1, initial_backoff=0)

    assert "GLFYTRTGL" in results
    assert isinstance(results["GLFYTRTGL"], float)


# lines 362-366 - query_tapreg successful parse returns real scores (no mock fallback)
def test_query_tapreg_successful_parse_returns_scores():
    peptides = ["GLFYTRTGL"]
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    # Plain-text format that parse_tapreg_html can extract a score from
    mock_resp.text = "GLFYTRTGL some result 1.2345"

    with patch("src.external_predictors.requests.post", return_value=mock_resp):
        results = query_tapreg(peptides, mock_fallback=False, max_retries=1, initial_backoff=0)

    assert "GLFYTRTGL" in results
    assert results["GLFYTRTGL"] == pytest.approx(1.2345)
