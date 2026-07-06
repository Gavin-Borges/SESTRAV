"""Extended coverage tests for src/external_predictors.py.

Targets the 17 uncovered statements and 11 branch misses that remain after
the base test_external_predictors.py suite:
  - _generate_mock_netchop_scores: proline residue path (line 66)
  - _generate_mock_tapreg_score: empty pep (87), PDE c-terminus (93-96),
    RKYFW n-terminus (101), non-blosum model branch (104->107)
  - parse_netchop_html: out-of-bounds pep index (145->132)
  - query_netchop: successful poll returns results (255)
  - query_tapreg: threshold kwarg (346-347), successful parse (362-364),
    empty-parse fallback (365-366)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from src.external_predictors import (
    _generate_mock_netchop_scores,
    _generate_mock_tapreg_score,
    parse_netchop_html,
    query_netchop,
    query_tapreg,
)


# ---------------------------------------------------------------------------
# _generate_mock_netchop_scores - proline path (line 66: val -= 0.08 branch)
# ---------------------------------------------------------------------------


class TestMockNetchopScores:
    def test_proline_residue_reduces_score(self):
        result = _generate_mock_netchop_scores("GLPYTRTGL")
        assert len(result["scores"]) == 9
        p_idx = 2  # 'P' is at index 2
        assert result["scores"][p_idx] < 0.5, "Proline should push score below cleavage threshold"
        assert all(0.01 <= s <= 0.99 for s in result["scores"])

    def test_hydrophobic_residue_raises_score(self):
        result = _generate_mock_netchop_scores("YFWLIVMAG")
        scores = result["scores"]
        hydro_score = scores[0]  # Y is hydrophobic
        assert hydro_score > 0.1


# ---------------------------------------------------------------------------
# _generate_mock_tapreg_score - empty pep (87), PDE c-term (93-96),
#   RKYFW n-term (101), non-blosum branch (104->107)
# ---------------------------------------------------------------------------


class TestMockTapregScore:
    def test_empty_peptide_returns_score(self):
        score = _generate_mock_tapreg_score("", "blosum")
        assert isinstance(score, float)

    def test_pde_cterminus_penalised(self):
        score_d = _generate_mock_tapreg_score("GILGFVFTD", "blosum")
        score_l = _generate_mock_tapreg_score("GILGFVFTL", "blosum")
        assert score_d > score_l, "D c-terminus (PDE group) should raise score vs L"

    def test_rkyfw_nterminus_reduces_score(self):
        pep = "RGLFVFTLA"
        score = _generate_mock_tapreg_score(pep, "blosum")
        # pep ends with A (no c-term group), starts with R (RKYFW: -0.2), blosum: -0.05
        base = 1.5 + (hash(pep) % 10) * 0.05
        expected = round(base - 0.2 - 0.05, 5)
        assert score == pytest.approx(expected), "RKYFW n-terminus must reduce score by exactly 0.2"

    def test_sparse_model_no_blosum_adjustment(self):
        score_blosum = _generate_mock_tapreg_score("GILGFVFTL", "blosum")
        score_sparse = _generate_mock_tapreg_score("GILGFVFTL", "sparse")
        assert score_sparse != score_blosum, "sparse model skips the 0.05 blosum reduction"

    def test_blosum_string_triggers_reduction(self):
        score = _generate_mock_tapreg_score("GILGFVFTL", "DS_613 Blosum")
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# parse_netchop_html - out-of-bounds pep index (lines 145->132)
# ---------------------------------------------------------------------------


class TestParseNetchopHtmlOOB:
    _TABULAR_ROW_TEMPLATE = "  1 G . 0.10000 pep_{idx}\n  2 L . 0.20000 pep_{idx}\n"

    def test_oob_pep_index_silently_skipped(self):
        html = self._TABULAR_ROW_TEMPLATE.format(idx=99)
        result = parse_netchop_html(html, ["GLYF"])
        assert "GLYF" in result
        assert result["GLYF"]["scores"] == [], "OOB idx must be skipped - no scores assigned"

    def test_mixed_valid_and_oob(self):
        html = "  1 G . 0.10000 pep_0\n  2 L . 0.20000 pep_0\n  1 A . 0.30000 pep_99\n"
        result = parse_netchop_html(html, ["GL"])
        assert result["GL"]["scores"] == [0.10000, 0.20000]


# ---------------------------------------------------------------------------
# query_netchop - successful poll returns results (line 255)
# ---------------------------------------------------------------------------

VALID_NETCHOP_POLL_RESPONSE = (
    "  1 G . 0.10000 pep_0\n"
    "  2 L S 0.72000 pep_0\n"
    "  3 F S 0.85000 pep_0\n"
    "  4 Y S 0.91000 pep_0\n"
    "  5 T . 0.21000 pep_0\n"
    "  6 R S 0.65000 pep_0\n"
    "  7 T . 0.18000 pep_0\n"
    "  8 G . 0.12000 pep_0\n"
    "  9 L . 0.09000 pep_0\n"
)


class TestQueryNetchopSuccessfulPoll:
    def test_successful_poll_returns_parsed_results(self):
        peptides = ["GLFYTRTGL"]

        submit_resp = MagicMock()
        submit_resp.raise_for_status.return_value = None
        submit_resp.text = "jobid=abc123"

        poll_resp = MagicMock()
        poll_resp.raise_for_status.return_value = None
        poll_resp.text = VALID_NETCHOP_POLL_RESPONSE

        with (
            patch("src.external_predictors.requests.post", return_value=submit_resp),
            patch("src.external_predictors.requests.get", return_value=poll_resp),
        ):
            results = query_netchop(peptides, mock_fallback=False, max_retries=1, initial_backoff=0)

        assert "GLFYTRTGL" in results
        assert len(results["GLFYTRTGL"]["scores"]) == 9
        assert results["GLFYTRTGL"]["scores"][0] == pytest.approx(0.10000)


# ---------------------------------------------------------------------------
# query_tapreg - threshold kwarg (346-347), parse success (362-364),
#   empty-parse fallback (365-366)
# ---------------------------------------------------------------------------

_TAPREG_RESP_WITH_SCORES = "GLFYTRTGL  result  1.2345\n"
_TAPREG_RESP_NO_SCORES = "<html>No matching content</html>"


class TestQueryTapregThresholdAndParse:
    def test_threshold_kwarg_accepted_with_mock_connection_error(self):
        peptides = ["GLFYTRTGL"]
        with patch(
            "src.external_predictors.requests.post",
            side_effect=requests.exceptions.ConnectionError("no net"),
        ):
            results = query_tapreg(
                peptides,
                threshold=0.5,
                mock_fallback=False,
                max_retries=1,
                initial_backoff=0,
            )
        assert "GLFYTRTGL" in results
        assert isinstance(results["GLFYTRTGL"], float)

    def test_successful_post_with_parseable_scores_returns_early(self):
        peptides = ["GLFYTRTGL"]
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.text = _TAPREG_RESP_WITH_SCORES

        with patch("src.external_predictors.requests.post", return_value=mock_resp):
            results = query_tapreg(
                peptides,
                mock_fallback=False,
                max_retries=1,
                initial_backoff=0,
            )
        assert "GLFYTRTGL" in results
        assert results["GLFYTRTGL"] == pytest.approx(1.2345)

    def test_successful_post_empty_parse_logs_warning_and_falls_back(self, caplog):
        import logging

        peptides = ["GLFYTRTGL"]
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.text = _TAPREG_RESP_NO_SCORES

        with (
            patch("src.external_predictors.requests.post", return_value=mock_resp),
            caplog.at_level(logging.WARNING, logger="src.external_predictors"),
        ):
            results = query_tapreg(
                peptides,
                mock_fallback=False,
                max_retries=1,
                initial_backoff=0,
            )
        assert "GLFYTRTGL" in results
        assert any("no peptide scores" in m.lower() for m in caplog.messages)
