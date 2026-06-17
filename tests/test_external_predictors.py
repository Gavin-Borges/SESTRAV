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
    with patch("src.external_predictors.requests.post",
               side_effect=requests.exceptions.ConnectionError("no network")):
        results = query_netchop(peptides, mock_fallback=False, max_retries=1,
                                initial_backoff=0)
    assert "GLFYTRTGL" in results
    assert len(results["GLFYTRTGL"]["scores"]) == len("GLFYTRTGL")


def test_query_netchop_no_job_id_in_response_falls_back_to_mock():
    """Server responds OK but no job ID is parseable → mock fallback."""
    peptides = ["GLFYTRTGL"]
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.text = "<html>No useful content here</html>"
    with patch("src.external_predictors.requests.post", return_value=mock_resp):
        results = query_netchop(peptides, mock_fallback=False, max_retries=1,
                                initial_backoff=0)
    assert "GLFYTRTGL" in results
    assert isinstance(results["GLFYTRTGL"]["scores"], list)


def test_query_netchop_polling_empty_result_falls_back_to_mock():
    """Job ID parsed, but polling always returns empty table → mock fallback."""
    peptides = ["GLFYTRTGL"]
    submit_resp = MagicMock()
    submit_resp.raise_for_status.return_value = None
    submit_resp.text = 'jobid=abc123'

    poll_resp = MagicMock()
    poll_resp.raise_for_status.return_value = None
    poll_resp.text = "<html>done</html>"  # no tabular rows → empty parse

    with patch("src.external_predictors.requests.post", return_value=submit_resp), \
         patch("src.external_predictors.requests.get", return_value=poll_resp):
        results = query_netchop(peptides, mock_fallback=False, max_retries=1,
                                initial_backoff=0)
    assert "GLFYTRTGL" in results


def test_query_netchop_polling_request_exception_falls_back_to_mock():
    """Job ID obtained, but every poll raises a network error → mock fallback."""
    peptides = ["GLFYTRTGL"]
    submit_resp = MagicMock()
    submit_resp.raise_for_status.return_value = None
    submit_resp.text = 'jobid=abc123'

    with patch("src.external_predictors.requests.post", return_value=submit_resp), \
         patch("src.external_predictors.requests.get",
               side_effect=requests.exceptions.Timeout("timed out")):
        results = query_netchop(peptides, mock_fallback=False, max_retries=1,
                                initial_backoff=0)
    assert "GLFYTRTGL" in results


def test_query_tapreg_request_exception_falls_back_to_mock():
    """Connection error → automatic mock fallback for TAPreg."""
    peptides = ["GLFYTRTGL"]
    with patch("src.external_predictors.requests.post",
               side_effect=requests.exceptions.ConnectionError("no network")):
        results = query_tapreg(peptides, mock_fallback=False, max_retries=1,
                               initial_backoff=0)
    assert "GLFYTRTGL" in results
    assert isinstance(results["GLFYTRTGL"], float)


def test_query_tapreg_vpn_restriction_falls_back_to_mock():
    """VPN restriction page detected → break and fall back to mock."""
    peptides = ["GLFYTRTGL"]
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.text = "acceso restringido — please use UCM VPN"
    with patch("src.external_predictors.requests.post", return_value=mock_resp):
        results = query_tapreg(peptides, mock_fallback=False, max_retries=1,
                               initial_backoff=0)
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
