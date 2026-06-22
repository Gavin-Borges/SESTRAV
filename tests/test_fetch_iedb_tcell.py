"""Tests for scripts/fetch_iedb_tcell.py - pure logic layer, no network calls."""

import sys
import os


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from fetch_iedb_tcell import (
    _normalize_allele,
    _assign_label,
    _assay_quality,
    _build_url,
    _process_records,
    ORGANISM_MAP,
    VIRUS_DISPLAY,
    ASSAY_QUALITY_MAP,
    _DEFAULT_QUALITY,
    PAGE_SIZE,
    IEDB_API_BASE,
)

# ---------------------------------------------------------------------------
# _normalize_allele
# ---------------------------------------------------------------------------

class TestNormalizeAllele:
    def test_canonical_a_allele(self):
        assert _normalize_allele("HLA-A*02:01") == "HLA-A*02:01"

    def test_canonical_b_allele(self):
        assert _normalize_allele("HLA-B*57:01") == "HLA-B*57:01"

    def test_canonical_c_allele(self):
        assert _normalize_allele("HLA-C*07:02") == "HLA-C*07:02"

    def test_eight_digit_a(self):
        """HLA-A*0201 → HLA-A*02:01 (colon inserted at position 2 of group)."""
        assert _normalize_allele("HLA-A*0201") == "HLA-A*02:01"

    def test_eight_digit_b(self):
        assert _normalize_allele("HLA-B*0702") == "HLA-B*07:02"

    def test_supertypic_unchanged(self):
        """Supertypic shorthand returned unchanged - insufficient info to expand."""
        assert _normalize_allele("HLA-A2") == "HLA-A2"

    def test_class_i_label_unchanged(self):
        assert _normalize_allele("class I") == "class I"

    def test_empty_string(self):
        assert _normalize_allele("") == ""

    def test_none_input(self):
        assert _normalize_allele(None) == ""

    def test_whitespace_stripped(self):
        assert _normalize_allele("  HLA-A*02:01  ") == "HLA-A*02:01"


# ---------------------------------------------------------------------------
# _assign_label
# ---------------------------------------------------------------------------

class TestAssignLabel:
    def test_positive(self):
        assert _assign_label("Positive") == 1

    def test_positive_high(self):
        assert _assign_label("Positive-High") == 1

    def test_positive_intermediate(self):
        assert _assign_label("Positive-Intermediate") == 1

    def test_positive_low(self):
        """Positive-Low is still a confirmed response; label=1."""
        assert _assign_label("Positive-Low") == 1

    def test_negative(self):
        assert _assign_label("Negative") == 0

    def test_inconclusive_excluded(self):
        assert _assign_label("Inconclusive") is None

    def test_empty_string_excluded(self):
        assert _assign_label("") is None

    def test_none_excluded(self):
        assert _assign_label(None) is None

    def test_case_insensitive(self):
        """Label matching is case-insensitive for robustness across API variants."""
        assert _assign_label("positive") == 1
        assert _assign_label("negative") == 0
        assert _assign_label("POSITIVE") == 1
        assert _assign_label("NEGATIVE") == 0


# ---------------------------------------------------------------------------
# _assay_quality
# ---------------------------------------------------------------------------

class TestAssayQuality:
    def test_cytotoxicity_tier1(self):
        assert _assay_quality("cytotoxicity") == 1.0

    def test_ifng_tier1(self):
        assert _assay_quality("IFNg release") == 1.0

    def test_proliferation_tier3(self):
        assert _assay_quality("proliferation") == 0.7

    def test_binding_tier4(self):
        assert _assay_quality("qualitative binding") == 0.5

    def test_case_insensitive(self):
        assert _assay_quality("IFNG RELEASE") == _assay_quality("ifng release")

    def test_unknown_returns_default(self):
        assert _assay_quality("some novel assay") == _DEFAULT_QUALITY

    def test_none_returns_default(self):
        assert _assay_quality(None) == _DEFAULT_QUALITY

    def test_all_map_keys_valid_range(self):
        for key, val in ASSAY_QUALITY_MAP.items():
            assert 0.0 < val <= 1.0, f"{key!r} weight {val} out of (0, 1]"


# ---------------------------------------------------------------------------
# _process_records
# ---------------------------------------------------------------------------

def _make_rec(peptide="GILGFVFTL", qm="Positive", response="IFNg release",
              allele="HLA-A*02:01", organism="Human gammaherpesvirus 4",
              molecule="gp350", pmid="12345678"):
    return {
        "epitope__name": peptide,
        "assay__qualitative_measurement": qm,
        "assay__response_measured": response,
        "mhc_restriction__name": allele,
        "epitope__source_organism": organism,
        "epitope__source_molecule": molecule,
        "reference__pmid": pmid,
    }


class TestProcessRecords:
    def test_valid_record_retained(self):
        result = _process_records([_make_rec()], "EBV")
        assert len(result) == 1

    def test_output_columns_complete(self):
        required = [
            "peptide", "label", "virus", "protein", "strain", "hla_allele",
            "source_type", "database_source", "assay_type", "assay_quality_weight",
            "reference_pmid",
        ]
        result = _process_records([_make_rec()], "EBV")
        for col in required:
            assert col in result[0], f"Missing column: {col}"

    def test_source_type_is_virus(self):
        result = _process_records([_make_rec()], "EBV")
        assert result[0]["source_type"] == "Virus"

    def test_database_source_is_iedb(self):
        result = _process_records([_make_rec()], "EBV")
        assert result[0]["database_source"] == "IEDB"

    def test_label_positive(self):
        result = _process_records([_make_rec(qm="Positive")], "EBV")
        assert result[0]["label"] == 1

    def test_label_negative(self):
        result = _process_records([_make_rec(qm="Negative")], "EBV")
        assert result[0]["label"] == 0

    def test_inconclusive_excluded(self):
        result = _process_records([_make_rec(qm="Inconclusive")], "EBV")
        assert len(result) == 0

    def test_7mer_dropped(self):
        result = _process_records([_make_rec(peptide="ACDEFGH")], "EBV")
        assert len(result) == 0

    def test_12mer_dropped(self):
        result = _process_records([_make_rec(peptide="ACDEFGHIKLMN")], "EBV")
        assert len(result) == 0

    def test_8mer_kept(self):
        result = _process_records([_make_rec(peptide="ACDEFGHI")], "EBV")
        assert len(result) == 1

    def test_11mer_kept(self):
        result = _process_records([_make_rec(peptide="ACDEFGHIKLM")], "EBV")
        assert len(result) == 1

    def test_nonstandard_aa_dropped(self):
        result = _process_records([_make_rec(peptide="GILGXVFTL")], "EBV")
        assert len(result) == 0

    def test_lowercase_peptide_uppercased(self):
        result = _process_records([_make_rec(peptide="gilgfvftl")], "EBV")
        assert result[0]["peptide"] == "GILGFVFTL"

    def test_allele_normalization_applied(self):
        result = _process_records([_make_rec(allele="HLA-A*0201")], "EBV")
        assert result[0]["hla_allele"] == "HLA-A*02:01"

    def test_assay_quality_weight_populated(self):
        result = _process_records([_make_rec(response="cytotoxicity")], "EBV")
        assert result[0]["assay_quality_weight"] == 1.0

    def test_assay_quality_unknown_default(self):
        result = _process_records([_make_rec(response="mystery_assay")], "EBV")
        assert result[0]["assay_quality_weight"] == _DEFAULT_QUALITY

    def test_virus_display_name_propagated(self):
        result = _process_records([_make_rec()], "HIV-1")
        assert result[0]["virus"] == "HIV-1"

    def test_none_peptide_dropped(self):
        rec = _make_rec()
        rec["epitope__name"] = None
        result = _process_records([rec], "EBV")
        assert len(result) == 0

    def test_mixed_records(self):
        records = [
            _make_rec(peptide="GILGFVFTL", qm="Positive"),   # valid
            _make_rec(peptide="ACDEF", qm="Positive"),        # too short
            _make_rec(peptide="GILGFVFTL", qm="Negative"),    # valid negative
            _make_rec(peptide="GILGXVFTL", qm="Positive"),    # invalid AA
        ]
        result = _process_records(records, "EBV")
        assert len(result) == 2
        labels = {r["label"] for r in result}
        assert labels == {0, 1}


# ---------------------------------------------------------------------------
# _build_url
# ---------------------------------------------------------------------------

class TestBuildUrl:
    def test_url_starts_with_base(self):
        url = _build_url("gammaherpesvirus 4", 0)
        assert url.startswith(IEDB_API_BASE)

    def test_class_i_filter_present(self):
        url = _build_url("gammaherpesvirus 4", 0)
        assert "mhc_restriction__class=eq.I" in url

    def test_linear_peptide_filter_present(self):
        url = _build_url("gammaherpesvirus 4", 0)
        assert "epitope__object_type=eq.Linear" in url

    def test_organism_ilike_present(self):
        url = _build_url("gammaherpesvirus 4", 0)
        assert "ilike.*gammaherpesvirus" in url

    def test_human_host_filter_present(self):
        url = _build_url("gammaherpesvirus 4", 0)
        assert "host__name=ilike.*Homo" in url

    def test_offset_zero(self):
        url = _build_url("gammaherpesvirus 4", 0)
        assert "offset=0" in url

    def test_offset_nonzero(self):
        url = _build_url("gammaherpesvirus 4", PAGE_SIZE)
        assert f"offset={PAGE_SIZE}" in url

    def test_limit_is_page_size(self):
        url = _build_url("gammaherpesvirus 4", 0)
        assert f"limit={PAGE_SIZE}" in url

    def test_select_includes_key_fields(self):
        url = _build_url("gammaherpesvirus 4", 0)
        for field in ["epitope__name", "assay__qualitative_measurement",
                      "mhc_restriction__name", "assay__response_measured"]:
            assert field in url


# ---------------------------------------------------------------------------
# Organism / display maps
# ---------------------------------------------------------------------------

class TestOrganismMap:
    def test_all_supported_viruses_present(self):
        for key in ["EBV", "HPV16", "HPV18", "HBV", "HCV", "HIV",
                    "SARSCOV2", "IAV", "CMV"]:
            assert key in ORGANISM_MAP

    def test_organism_patterns_nonempty(self):
        for key, pattern in ORGANISM_MAP.items():
            assert pattern, f"{key} has empty pattern"

    def test_display_names_nonempty(self):
        for key, name in VIRUS_DISPLAY.items():
            assert name, f"{key} has empty display name"

    def test_ebv_organism_pattern_verified(self):
        """Pattern verified against live IEDB API 2026-06-19."""
        assert ORGANISM_MAP["EBV"] == "gammaherpesvirus 4"

    def test_hiv_organism_pattern_verified(self):
        assert ORGANISM_MAP["HIV"] == "immunodeficiency virus 1"
