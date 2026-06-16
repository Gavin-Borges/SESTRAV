import pytest
import pandas as pd
from src.external_predictors import (
    generate_fasta,
    parse_netchop_html,
    parse_tapreg_html,
    query_netchop,
    query_tapreg,
    get_predictor_dataframe
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
