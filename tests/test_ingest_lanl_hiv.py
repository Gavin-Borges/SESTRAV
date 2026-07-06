"""Tests for scripts/ingest_lanl_hiv.py."""

import json
import logging
import sys
import os

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.ingest_lanl_hiv import (
    _parse_label,
    _find_col,
    resolve_columns,
    filter_rows,
    build_output,
    load_lanl_export,
    HIV1_TAXON_ID,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_df(**kwargs) -> pd.DataFrame:
    """Build a minimal test DataFrame with given column data."""
    return pd.DataFrame(kwargs)


def _sample_raw() -> pd.DataFrame:
    """Return a small representative LANL HIV export DataFrame."""
    return pd.DataFrame(
        {
            "Optimal Sequence": [
                "SLYNTVATL",  # 9-mer, valid
                "ILKEPVHGV",  # 9-mer, valid
                "TOOLONG_PEPTIDE",  # too long, should be filtered
                "AAAA",  # too short, should be filtered
                "SLYNTVATL",  # duplicate of row 0 (same peptide+HLA+PMID)
                "KLNWASQIY",  # 9-mer, valid positive
            ],
            "HLA": [
                "HLA-A*02:01",
                "HLA-A*02:01",
                "HLA-A*02:01",
                "HLA-A*02:01",
                "HLA-A*02:01",
                "HLA-B*08:01",
            ],
            "Assay Result": [
                "Negative",
                "negative",
                "Negative",
                "Negative",
                "Negative",
                "Positive",
            ],
            "PMID": ["12345", "67890", "11111", "22222", "12345", "33333"],
            "Protein": ["Gag", "Pol", "Env", "Gag", "Gag", "Gag"],
            "Assay": [
                "IFNg ELISpot",
                "Cytotoxicity",
                "IFNg ELISpot",
                "IFNg ELISpot",
                "IFNg ELISpot",
                "IFNg ELISpot",
            ],
            "Subtype": ["B", "B", "B", "B", "B", "C"],
        }
    )


# ---------------------------------------------------------------------------
# _parse_label
# ---------------------------------------------------------------------------


class TestParseLabel:
    def test_positive_variants(self):
        for s in [
            "Positive",
            "positive",
            "POSITIVE",
            "Positive-High",
            "yes",
            "immunogenic",
            "response",
            "reactive",
            "confirmed",
        ]:
            result = _parse_label(s)
            assert result == 1, f"Expected 1 for {s!r}, got {result}"

    def test_negative_variants(self):
        for s in [
            "Negative",
            "negative",
            "neg",
            "no response",
            "non-responder",
            "not detected",
            "not immunogenic",
        ]:
            result = _parse_label(s)
            assert result == 0, f"Expected 0 for {s!r}, got {result}"

    def test_ambiguous_returns_none(self):
        for s in ["inconclusive", "borderline", "pending", "", "unknown"]:
            assert _parse_label(s) is None

    def test_startswith_positive_prefix(self):
        assert _parse_label("Positive-Low") == 1
        assert _parse_label("positive-intermediate") == 1

    def test_startswith_negative_prefix(self):
        assert _parse_label("Negative-Low") == 0


# ---------------------------------------------------------------------------
# _find_col
# ---------------------------------------------------------------------------


class TestFindCol:
    def test_exact_match(self):
        cols = ["Sequence", "HLA", "Response"]
        assert _find_col(cols, "Sequence") == "Sequence"

    def test_case_insensitive(self):
        cols = ["Optimal Sequence", "hla type"]
        assert _find_col(cols, "HLA TYPE") == "hla type"
        assert _find_col(cols, "optimal sequence") == "Optimal Sequence"

    def test_first_candidate_wins(self):
        cols = ["Sequence", "Epitope"]
        result = _find_col(cols, "Sequence", "Epitope")
        assert result == "Sequence"

    def test_fallback_to_second_candidate(self):
        cols = ["Epitope", "PMID"]
        result = _find_col(cols, "Sequence", "Epitope")
        assert result == "Epitope"

    def test_none_when_missing(self):
        cols = ["ColA", "ColB"]
        assert _find_col(cols, "Sequence", "Peptide") is None


# ---------------------------------------------------------------------------
# resolve_columns
# ---------------------------------------------------------------------------


class TestResolveColumns:
    def _logger(self):
        return logging.getLogger("test_resolve")

    def test_resolves_standard_cols(self):
        df = _sample_raw()
        mapping = resolve_columns(df, {}, self._logger())
        assert mapping["epitope"] == "Optimal Sequence"
        assert mapping["hla"] == "HLA"
        assert mapping["result"] == "Assay Result"
        assert mapping["pmid"] == "PMID"
        assert mapping["protein"] == "Protein"
        assert mapping["assay"] == "Assay"
        assert mapping["subtype"] == "Subtype"

    def test_override_respected(self):
        df = pd.DataFrame(
            {"MyPeptide": ["SLYNTVATL"], "MyHLA": ["HLA-A*02:01"], "MyResult": ["Negative"]}
        )
        overrides = {"epitope": "MyPeptide", "hla": "MyHLA", "result": "MyResult"}
        mapping = resolve_columns(df, overrides, self._logger())
        assert mapping["epitope"] == "MyPeptide"
        assert mapping["hla"] == "MyHLA"
        assert mapping["result"] == "MyResult"

    def test_missing_result_col_warns_not_errors(self):
        df = pd.DataFrame({"Optimal Sequence": ["SLYNTVATL"], "HLA": ["HLA-A*02:01"]})
        mapping = resolve_columns(df, {}, self._logger())
        assert mapping["epitope"] is not None
        assert mapping["hla"] is not None
        assert mapping["result"] is None  # warned but not fatal


# ---------------------------------------------------------------------------
# filter_rows
# ---------------------------------------------------------------------------


class TestFilterRows:
    def _mapping(self, df):
        return {
            "epitope": "Optimal Sequence",
            "hla": "HLA",
            "result": "Assay Result",
            "pmid": "PMID",
            "protein": "Protein",
            "assay": "Assay",
            "subtype": "Subtype",
        }

    def test_negatives_only_mode(self):
        df = _sample_raw()
        logger = logging.getLogger("test_filter")
        mapping = self._mapping(df)
        filtered, stats = filter_rows(df, mapping, negatives_only=True, logger=logger)
        # All filtered rows should be label=0
        assert (filtered["_label"] == 0).all()
        # The positive row (KLNWASQIY) should be excluded
        assert "KLNWASQIY" not in filtered["_peptide"].values

    def test_all_labels_mode_includes_positives(self):
        df = _sample_raw()
        logger = logging.getLogger("test_filter")
        mapping = self._mapping(df)
        filtered, stats = filter_rows(df, mapping, negatives_only=False, logger=logger)
        assert 1 in filtered["_label"].values
        assert "KLNWASQIY" in filtered["_peptide"].values

    def test_invalid_length_peptides_removed(self):
        df = _sample_raw()
        logger = logging.getLogger("test_filter")
        mapping = self._mapping(df)
        filtered, stats = filter_rows(df, mapping, negatives_only=False, logger=logger)
        for pep in filtered["_peptide"]:
            assert 8 <= len(pep) <= 11

    def test_class_ii_hla_rejected(self):
        df = pd.DataFrame(
            {
                "Optimal Sequence": ["SLYNTVATL", "LLAALVVAK"],
                "HLA": ["HLA-A*02:01", "HLA-DRB1*01:01"],
                "Assay Result": ["Negative", "Negative"],
                "PMID": ["1", "2"],
            }
        )
        mapping = {
            "epitope": "Optimal Sequence",
            "hla": "HLA",
            "result": "Assay Result",
            "pmid": "PMID",
            "protein": None,
            "assay": None,
            "subtype": None,
        }
        logger = logging.getLogger("test_filter")
        filtered, stats = filter_rows(df, mapping, negatives_only=True, logger=logger)
        assert len(filtered) == 1
        assert filtered.iloc[0]["_peptide"] == "SLYNTVATL"

    def test_intra_export_dedup_on_peptide_hla_pmid(self):
        df = _sample_raw()
        logger = logging.getLogger("test_filter")
        mapping = self._mapping(df)
        # Row 0 and Row 4 are identical (same peptide, HLA, PMID, result)
        filtered, stats = filter_rows(df, mapping, negatives_only=True, logger=logger)
        assert stats["intra_export_duplicates_removed"] >= 1
        # The deduplicated SLYNTVATL should appear only once
        count = (filtered["_peptide"] == "SLYNTVATL").sum()
        assert count == 1

    def test_no_result_col_all_positive(self):
        df = pd.DataFrame(
            {
                "Optimal Sequence": ["SLYNTVATL", "ILKEPVHGV"],
                "HLA": ["HLA-A*02:01", "HLA-B*08:01"],
            }
        )
        mapping = {
            "epitope": "Optimal Sequence",
            "hla": "HLA",
            "result": None,
            "pmid": None,
            "protein": None,
            "assay": None,
            "subtype": None,
        }
        logger = logging.getLogger("test_filter")
        filtered, stats = filter_rows(df, mapping, negatives_only=False, logger=logger)
        assert (filtered["_label"] == 1).all()


# ---------------------------------------------------------------------------
# build_output
# ---------------------------------------------------------------------------


class TestBuildOutput:
    REQUIRED_V5_COLS = {
        "peptide",
        "label",
        "virus",
        "protein",
        "strain",
        "hla_allele",
        "source_type",
        "database_source",
        "tcr_alpha_cdr3",
        "tcr_beta_cdr3",
        "virus_family",
        "negative_origin",
        "assay_type",
        "assay_quality_tier",
        "assay_quality_weight",
        "reference_pmid",
        "iedb_assay_id",
        "infection_phase",
        "antigen_latency_program",
        "assay_context",
        "cross_reactivity_tested",
        "virus_taxon_id",
        "is_quarantined",
    }

    def _make_filtered(self):
        df = _sample_raw()
        logger = logging.getLogger("test_build")
        mapping = {
            "epitope": "Optimal Sequence",
            "hla": "HLA",
            "result": "Assay Result",
            "pmid": "PMID",
            "protein": "Protein",
            "assay": "Assay",
            "subtype": "Subtype",
        }
        filtered, _ = filter_rows(df, mapping, negatives_only=False, logger=logger)
        return filtered, mapping

    def test_output_has_all_v5_columns(self):
        filtered, mapping = self._make_filtered()
        logger = logging.getLogger("test_build")
        out = build_output(filtered, mapping, logger)
        missing = self.REQUIRED_V5_COLS - set(out.columns)
        assert not missing, f"Missing v5 columns: {missing}"

    def test_virus_is_hiv1(self):
        filtered, mapping = self._make_filtered()
        logger = logging.getLogger("test_build")
        out = build_output(filtered, mapping, logger)
        assert (out["virus"] == "HIV-1").all()

    def test_virus_family_is_retroviridae(self):
        filtered, mapping = self._make_filtered()
        logger = logging.getLogger("test_build")
        out = build_output(filtered, mapping, logger)
        assert (out["virus_family"] == "Retroviridae").all()

    def test_taxon_id_is_hiv1(self):
        filtered, mapping = self._make_filtered()
        logger = logging.getLogger("test_build")
        out = build_output(filtered, mapping, logger)
        assert (out["virus_taxon_id"] == HIV1_TAXON_ID).all()

    def test_database_source_is_lanl_hiv(self):
        filtered, mapping = self._make_filtered()
        logger = logging.getLogger("test_build")
        out = build_output(filtered, mapping, logger)
        assert (out["database_source"] == "LANL-HIV").all()

    def test_negative_origin_mapped_correctly(self):
        filtered, mapping = self._make_filtered()
        logger = logging.getLogger("test_build")
        out = build_output(filtered, mapping, logger)
        neg_rows = out[out["label"] == 0]
        pos_rows = out[out["label"] == 1]
        if len(neg_rows):
            assert (neg_rows["negative_origin"] == "tested_negative").all()
        if len(pos_rows):
            assert (pos_rows["negative_origin"] == "iedb_positive").all()

    def test_is_quarantined_false(self):
        filtered, mapping = self._make_filtered()
        logger = logging.getLogger("test_build")
        out = build_output(filtered, mapping, logger)
        assert (out["is_quarantined"] == False).all()  # noqa: E712

    def test_assay_quality_tier_is_int(self):
        filtered, mapping = self._make_filtered()
        logger = logging.getLogger("test_build")
        out = build_output(filtered, mapping, logger)
        assert out["assay_quality_tier"].dtype in (int, "int64", "int32")


# ---------------------------------------------------------------------------
# load_lanl_export
# ---------------------------------------------------------------------------


class TestLoadLanlExport:
    def test_loads_tsv(self, tmp_path):
        tsv_content = (
            "Optimal Sequence\tHLA\tAssay Result\tPMID\nSLYNTVATL\tHLA-A*02:01\tNegative\t12345\n"
        )
        f = tmp_path / "test.tsv"
        f.write_text(tsv_content)
        logger = logging.getLogger("test_load")
        df = load_lanl_export(f, logger)
        assert len(df) == 1
        assert "Optimal Sequence" in df.columns

    def test_loads_csv(self, tmp_path):
        csv_content = "Sequence,HLA,Response,PMID\nSLYNTVATL,HLA-A*02:01,Negative,1\n"
        f = tmp_path / "test.csv"
        f.write_text(csv_content)
        logger = logging.getLogger("test_load")
        df = load_lanl_export(f, logger)
        assert len(df) == 1
        assert "Sequence" in df.columns

    def test_single_column_csv_retries_as_tsv(self, tmp_path):
        tsv_content = "Optimal Sequence\tHLA\tAssay Result\nSLYNTVATL\tHLA-A*02:01\tNegative\n"
        f = tmp_path / "test.csv"
        f.write_text(tsv_content)
        logger = logging.getLogger("test_load")
        df = load_lanl_export(f, logger)
        # Should auto-retry with tab separator and find multiple columns
        assert len(df.columns) > 1


# ---------------------------------------------------------------------------
# CLI integration: --dry-run
# ---------------------------------------------------------------------------


class TestCLIDryRun:
    def test_dry_run_no_output_written(self, tmp_path):
        csv_content = (
            "Optimal Sequence,HLA,Assay Result,PMID,Protein,Assay,Subtype\n"
            "SLYNTVATL,HLA-A*02:01,Negative,12345,Gag,IFNg ELISpot,B\n"
            "ILKEPVHGV,HLA-A*02:01,Positive,67890,Pol,Cytotoxicity,B\n"
        )
        infile = tmp_path / "export.csv"
        infile.write_text(csv_content)
        outfile = tmp_path / "out.csv"

        from scripts.ingest_lanl_hiv import main

        ret = main(
            [
                "--input",
                str(infile),
                "--output",
                str(outfile),
                "--dry-run",
            ]
        )
        assert ret == 0
        assert not outfile.exists()

    def test_negatives_only_excludes_positives(self, tmp_path):
        csv_content = (
            "Optimal Sequence,HLA,Assay Result,PMID,Protein,Assay,Subtype\n"
            "SLYNTVATL,HLA-A*02:01,Negative,12345,Gag,IFNg ELISpot,B\n"
            "ILKEPVHGV,HLA-A*02:01,Positive,67890,Pol,Cytotoxicity,B\n"
        )
        infile = tmp_path / "export.csv"
        infile.write_text(csv_content)
        outfile = tmp_path / "out.csv"

        from scripts.ingest_lanl_hiv import main

        ret = main(
            [
                "--input",
                str(infile),
                "--output",
                str(outfile),
                "--negatives-only",
            ]
        )
        assert ret == 0
        df = pd.read_csv(outfile)
        assert (df["label"] == 0).all()
        assert len(df) == 1

    def test_all_labels_includes_both(self, tmp_path):
        csv_content = (
            "Optimal Sequence,HLA,Assay Result,PMID,Protein,Assay,Subtype\n"
            "SLYNTVATL,HLA-A*02:01,Negative,12345,Gag,IFNg ELISpot,B\n"
            "ILKEPVHGV,HLA-A*02:01,Positive,67890,Pol,Cytotoxicity,B\n"
        )
        infile = tmp_path / "export.csv"
        infile.write_text(csv_content)
        outfile = tmp_path / "out.csv"

        from scripts.ingest_lanl_hiv import main

        ret = main(
            [
                "--input",
                str(infile),
                "--output",
                str(outfile),
            ]
        )
        assert ret == 0
        df = pd.read_csv(outfile)
        assert set(df["label"].unique()) == {0, 1}
        assert len(df) == 2

    def test_provenance_sidecar_written(self, tmp_path):
        csv_content = (
            "Optimal Sequence,HLA,Assay Result,PMID\nSLYNTVATL,HLA-A*02:01,Negative,12345\n"
        )
        infile = tmp_path / "export.csv"
        infile.write_text(csv_content)
        outfile = tmp_path / "out.csv"

        from scripts.ingest_lanl_hiv import main

        ret = main(["--input", str(infile), "--output", str(outfile)])
        assert ret == 0
        prov_path = tmp_path / "out_provenance.json"
        assert prov_path.exists()
        with open(prov_path) as f:
            prov = json.load(f)
        assert prov["source"] == "LANL HIV Molecular Immunology Database (hiv.lanl.gov)"
        assert "filter_stats" in prov
        assert "output_checksum_sha256" in prov

    def test_missing_input_returns_nonzero(self, tmp_path):
        from scripts.ingest_lanl_hiv import main

        ret = main(["--input", str(tmp_path / "nonexistent.csv")])
        assert ret != 0
