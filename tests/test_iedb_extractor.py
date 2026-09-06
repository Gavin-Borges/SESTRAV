"""Tests for src/verify/iedb_multi_virus_extractor.py.

Covers the uncovered paths:
  - query_iedb_rest (mock HTTP)
  - query_vdjdb_cached (temp TSV file)
  - clean_and_pool_epitopes edge cases (None label, empty)
  - load_proteome_peptides (FASTA parsing, missing file)
  - generate_decoys (happy path + mutation fallback)
  - process_target with mock=True
  - main() entry point
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.verify.iedb_multi_virus_extractor import (
    clean_and_pool_epitopes,
    extract_mock_data,
    generate_decoys,
    is_valid_peptide,
    load_proteome_peptides,
    process_target,
    query_iedb_rest,
    query_vdjdb_cached,
    sanitize_csv_string,
)


# ---------------------------------------------------------------------------
# sanitize_csv_string
# ---------------------------------------------------------------------------


class TestSanitizeCsvString:
    def test_empty_returns_empty(self):
        assert sanitize_csv_string("") == ""

    def test_normal_string_unchanged(self):
        assert sanitize_csv_string("GILGFVFTL") == "GILGFVFTL"

    def test_strips_whitespace(self):
        assert sanitize_csv_string("  ABC  ") == "ABC"

    @pytest.mark.parametrize("char", ["=", "+", "-", "@"])
    def test_formula_injection_prefixed(self, char):
        result = sanitize_csv_string(f"{char}cmd")
        assert result.startswith("'")


# ---------------------------------------------------------------------------
# is_valid_peptide
# ---------------------------------------------------------------------------


class TestIsValidPeptide:
    def test_valid_9mer(self):
        assert is_valid_peptide("GILGFVFTL") is True

    def test_too_short(self):
        assert is_valid_peptide("ACDE") is False

    def test_too_long(self):
        assert is_valid_peptide("ACDEFGHIKLMN") is False  # 12 chars - exceeds max

    def test_exactly_11_valid(self):
        assert is_valid_peptide("ACDEFGHIKLM") is True

    def test_invalid_residue(self):
        assert is_valid_peptide("XACDEFGHI") is False

    def test_none_input(self):
        assert is_valid_peptide(None) is False

    def test_nan_input(self):
        import numpy as np

        assert is_valid_peptide(np.nan) is False


# ---------------------------------------------------------------------------
# clean_and_pool_epitopes
# ---------------------------------------------------------------------------


class TestCleanAndPoolEpitopes:
    def _rec(self, seq="GILGFVFTL", allele="HLA-A*02:01", label="Positive", protein="M1"):
        return {
            "linear_sequence": seq,
            "mhc_allele_name": allele,
            "qualitative_measure": label,
            "source_molecule": protein,
        }

    def test_happy_path(self):
        records = [
            self._rec("GILGFVFTL", label="Positive"),
            self._rec("FMYSDFHFI", label="Negative"),
        ]
        df = clean_and_pool_epitopes(records)
        assert len(df) == 2
        assert set(df["label"].unique()) <= {0, 1}

    def test_empty_records(self):
        df = clean_and_pool_epitopes([])
        assert df.empty
        assert list(df.columns) == ["peptide", "label", "allele", "protein"]

    def test_invalid_peptide_dropped(self):
        records = [self._rec("XBADPEP", label="Positive"), self._rec("GILGFVFTL", label="Positive")]
        df = clean_and_pool_epitopes(records)
        assert len(df) == 1
        assert df["peptide"].iloc[0] == "GILGFVFTL"

    def test_ambiguous_label_dropped(self):
        records = [self._rec("GILGFVFTL", label="Unknown")]
        df = clean_and_pool_epitopes(records)
        assert df.empty

    def test_duplicate_peptide_majority_vote(self):
        records = [
            self._rec("GILGFVFTL", label="Positive"),
            self._rec("GILGFVFTL", label="Positive"),
            self._rec("GILGFVFTL", label="Negative"),
        ]
        df = clean_and_pool_epitopes(records)
        assert len(df) == 1
        assert df["label"].iloc[0] == 1  # majority positive

    def test_none_sequence_dropped(self):
        records = [
            {
                "linear_sequence": None,
                "mhc_allele_name": "HLA-A*02:01",
                "qualitative_measure": "Positive",
                "source_molecule": "M1",
            }
        ]
        df = clean_and_pool_epitopes(records)
        assert df.empty


# ---------------------------------------------------------------------------
# extract_mock_data
# ---------------------------------------------------------------------------


class TestExtractMockData:
    def test_known_virus_returns_records(self):
        records = extract_mock_data("InfluenzaA", 11520)
        assert len(records) >= 1
        assert "linear_sequence" in records[0]

    def test_unknown_virus_returns_default(self):
        records = extract_mock_data("UnknownVirus", 99999)
        assert len(records) >= 1

    def test_sars_cov2_records(self):
        records = extract_mock_data("SARS-CoV-2", 2697049)
        assert any(r["mhc_allele_name"] == "HLA-A*02:01" for r in records)


# ---------------------------------------------------------------------------
# query_iedb_rest (mocked HTTP)
# ---------------------------------------------------------------------------


class TestQueryIedbRest:
    def test_returns_json_on_200(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"linear_sequence": "GILGFVFTL"}]
        with patch("src.verify.iedb_multi_virus_extractor.requests.get", return_value=mock_resp):
            result = query_iedb_rest(11520, max_retries=1)
        assert result == [{"linear_sequence": "GILGFVFTL"}]

    def test_returns_empty_on_all_failures(self):
        import requests as req

        with patch(
            "src.verify.iedb_multi_virus_extractor.requests.get",
            side_effect=req.exceptions.RequestException("timeout"),
        ):
            result = query_iedb_rest(11520, max_retries=2, backoff=0.01)
        assert result == []

    def test_retries_on_non_200(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        with patch("src.verify.iedb_multi_virus_extractor.requests.get", return_value=mock_resp):
            result = query_iedb_rest(11520, max_retries=2, backoff=0.01)
        assert result == []


# ---------------------------------------------------------------------------
# query_vdjdb_cached (temp TSV file)
# ---------------------------------------------------------------------------


class TestQueryVdjdbCached:
    def _write_vdjdb(self, path: Path):
        tsv = (
            "antigen.epitope\tmhc.a\tantigen.taxId\tantigen.gene\n"
            "GILGFVFTL\tHLA-A*02:01\t11520\tM1\n"
            "FMYSDFHFI\tHLA-A*02:01\t11520\tPA\n"
            "YLQPRTFLL\tHLA-A*02:01\t2697049\tSpike\n"
        )
        path.write_text(tsv, encoding="utf-8")

    def test_returns_records_for_matching_taxid(self, tmp_path):
        cache = tmp_path / "vdjdb_slim.txt"
        self._write_vdjdb(cache)
        records = query_vdjdb_cached(11520, tmp_path)
        assert len(records) == 2
        seqs = [r["linear_sequence"] for r in records]
        assert "GILGFVFTL" in seqs

    def test_returns_empty_for_no_matching_taxid(self, tmp_path):
        cache = tmp_path / "vdjdb_slim.txt"
        self._write_vdjdb(cache)
        records = query_vdjdb_cached(99999, tmp_path)
        assert records == []

    def test_downloads_when_cache_missing(self, tmp_path):
        mock_resp = MagicMock()
        mock_resp.text = (
            "antigen.epitope\tmhc.a\tantigen.taxId\tantigen.gene\n"
            "GILGFVFTL\tHLA-A*02:01\t11520\tM1\n"
        )
        mock_resp.raise_for_status = MagicMock()
        with patch("src.verify.iedb_multi_virus_extractor.requests.get", return_value=mock_resp):
            records = query_vdjdb_cached(11520, tmp_path)
        assert len(records) == 1

    def test_returns_empty_on_download_failure(self, tmp_path):
        with patch(
            "src.verify.iedb_multi_virus_extractor.requests.get",
            side_effect=Exception("network error"),
        ):
            records = query_vdjdb_cached(11520, tmp_path)
        assert records == []


# ---------------------------------------------------------------------------
# load_proteome_peptides (FASTA parsing)
# ---------------------------------------------------------------------------


class TestLoadProteomePeptides:
    def test_missing_file_returns_empty(self, tmp_path):
        result = load_proteome_peptides(tmp_path / "nonexistent.fasta")
        assert result == []

    def test_parses_single_protein(self, tmp_path):
        fa = tmp_path / "prot.fasta"
        fa.write_text(">sp|P12345|TEST_HUMAN Test protein\nACDEFGHIKLMNP\n")
        kmers = load_proteome_peptides(fa, min_len=8, max_len=9)
        assert all(8 <= len(k) <= 9 for k in kmers)
        assert len(kmers) > 0

    def test_filters_invalid_residues(self, tmp_path):
        fa = tmp_path / "prot.fasta"
        fa.write_text(">seq1\nXXXXXXXXXXXX\n")
        kmers = load_proteome_peptides(fa, min_len=8, max_len=9)
        assert kmers == []

    def test_multi_protein_fasta(self, tmp_path):
        fa = tmp_path / "multi.fasta"
        fa.write_text(">p1\nACDEFGHIKL\n>p2\nMNPQRSTVWY\n")
        kmers = load_proteome_peptides(fa, min_len=8, max_len=9)
        assert len(kmers) > 0

    def test_no_trailing_newline(self, tmp_path):
        fa = tmp_path / "no_newline.fasta"
        fa.write_text(">seq1\nACDEFGHIKLMN")
        kmers = load_proteome_peptides(fa, min_len=8, max_len=9)
        assert len(kmers) > 0


# ---------------------------------------------------------------------------
# generate_decoys
# ---------------------------------------------------------------------------


class TestGenerateDecoys:
    def test_happy_path(self):
        pos = ["GILGFVFTL", "FMYSDFHFI"]
        proteome = ["ACDEFGHIK", "LMNPQRSTV", "WYVACDEFG", "HILKMNPQR", "STVWYACDE"]
        alleles = ["HLA-A*02:01"]
        decoys = generate_decoys(pos, proteome, alleles, decoy_ratio=1.0)
        assert len(decoys) == 2
        for pep, allele, label in decoys:
            assert label == 0
            assert allele in alleles

    def test_mutation_fallback_when_no_candidates(self):
        pos = ["GILGFVFTL"]
        # proteome overlaps with all positives
        proteome = []
        alleles = ["HLA-A*02:01"]
        decoys = generate_decoys(pos, proteome, alleles, decoy_ratio=1.0)
        assert len(decoys) == 1
        assert decoys[0][2] == 0  # label is 0

    def test_respects_decoy_ratio(self):
        pos = ["GILGFVFTL", "FMYSDFHFI", "KLVALGINA"]
        proteome = [f"ACDEFGHI{i}" for i in range(20)]
        alleles = ["HLA-A*02:01"]
        decoys = generate_decoys(pos, proteome, alleles, decoy_ratio=2.0)
        assert len(decoys) == 6  # 3 pos * 2.0 ratio

    def test_no_overlap_with_positives(self):
        """The positive must be EXCLUDED, not merely left undrawn.

        With the default decoy_ratio of 1.0 and a single positive, target_count
        is 1: the seeded shuffle puts the positive last of three candidates, so
        it is never reached and the assertion holds no matter what the exclusion
        logic does. Measured: dropping the exact-match filter, the substring
        overlap filter, or both leaves this test green.

        A decoy_ratio large enough to request every proteome entry forces each
        candidate to be considered, so the filter has to do the work. The length
        assertion is the load-bearing half: without exclusion, every candidate
        comes back.

        `GILGFVFTLL` is here to pin the SUBSTRING filter independently. The
        exact-match prefilter removes a strict subset of what the substring loop
        removes (if a candidate equals a positive then `cand in pos` is also
        true), so dropping the prefilter alone is an equivalent mutation that no
        test can fail on. A candidate that merely CONTAINS the positive is
        removed only by the substring loop, and is what makes disabling that
        loop observable.
        """
        pos = ["GILGFVFTL"]
        proteome = ["GILGFVFTL", "GILGFVFTLL", "ACDEFGHIK", "LMNPQRSTV"]
        alleles = ["HLA-A*02:01"]
        decoys = generate_decoys(pos, proteome, alleles, decoy_ratio=4.0)
        decoy_seqs = {d[0] for d in decoys}
        assert "GILGFVFTL" not in decoy_seqs, "the positive itself leaked in"
        assert "GILGFVFTLL" not in decoy_seqs, "a peptide containing the positive leaked in"
        assert len(decoys) == 2, (
            "all four candidates were requested, so exactly the positive and the "
            f"peptide containing it must have been filtered out; got {sorted(decoy_seqs)}"
        )


# ---------------------------------------------------------------------------
# process_target with mock=True
# ---------------------------------------------------------------------------


def test_process_target_mock_mode(tmp_path):
    config = {
        "taxonomy_id": 11520,
        "mhc_alleles": ["HLA-A*02:01"],
        "proteome_fasta": str(tmp_path / "nonexistent.fasta"),
        "validation_out": str(tmp_path / "output.csv"),
    }
    process_target("InfluenzaA", config, tmp_path, mock=True)
    out = tmp_path / "output.csv"
    assert out.exists()
    df = pd.read_csv(out)
    assert "peptide" in df.columns
    assert "label" in df.columns


def test_process_target_empty_records_falls_back_to_mock(tmp_path):
    config = {
        "taxonomy_id": 99999,
        "mhc_alleles": ["HLA-A*02:01"],
        "proteome_fasta": str(tmp_path / "nonexistent.fasta"),
        "validation_out": str(tmp_path / "output.csv"),
    }
    with (
        patch("src.verify.iedb_multi_virus_extractor.query_iedb_rest", return_value=[]),
        patch("src.verify.iedb_multi_virus_extractor.query_vdjdb_cached", return_value=[]),
    ):
        process_target("UnknownVirus", config, tmp_path, mock=False)
    out = tmp_path / "output.csv"
    assert out.exists()


# ---------------------------------------------------------------------------
# main() entry point (lines 305-323, 326)
# ---------------------------------------------------------------------------


def test_main_runs_with_valid_json(tmp_path):
    from src.verify.iedb_multi_virus_extractor import main

    config = {
        "viruses": {
            "InfluenzaA": {
                "taxonomy_id": 11520,
                "mhc_alleles": ["HLA-A*02:01"],
                "proteome_fasta": str(tmp_path / "nonexistent.fasta"),
                "validation_out": str(tmp_path / "flu_output.csv"),
            }
        }
    }
    cfg_file = tmp_path / "targets.json"
    cfg_file.write_text(json.dumps(config))

    with patch("sys.argv", ["iedb_multi_virus_extractor.py", str(cfg_file), "--mock"]):
        main()

    assert (tmp_path / "flu_output.csv").exists()


def test_main_exits_on_missing_file(tmp_path):
    from src.verify.iedb_multi_virus_extractor import main

    with (
        patch("sys.argv", ["iedb_multi_virus_extractor.py", str(tmp_path / "nonexistent.json")]),
        pytest.raises(SystemExit),
    ):
        main()
