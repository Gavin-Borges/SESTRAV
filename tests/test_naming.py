"""Unit tests for the naming/alias compatibility helpers (src/naming.py)."""

from src.naming import (
    canonical_output_filename,
    canonicalize_proteome_id,
    proteome_id_candidates,
    resolve_model_path,
)


def test_canonicalize_proteome_id_maps_alias_and_passthrough():
    assert canonicalize_proteome_id("HPV_8_FASTAs") == "HPV16_18_panel8"
    # An already-canonical (or unknown) id is returned unchanged.
    assert canonicalize_proteome_id("HPV16_18_panel8") == "HPV16_18_panel8"
    assert canonicalize_proteome_id("something_else") == "something_else"


def test_proteome_id_candidates_canonical_first_with_legacies():
    # EBV canonical has two legacy stems; canonical must come first, no dupes.
    cands = proteome_id_candidates("EBV_panel8_B958")
    assert cands[0] == "EBV_B95_8_panel8"
    assert "EBV_8_FASTAs" in cands
    assert len(cands) == len(set(cands))


def test_proteome_id_candidates_unknown_returns_singleton():
    assert proteome_id_candidates("novel_id") == ["novel_id"]


def test_resolve_model_path_empty_is_passthrough():
    assert resolve_model_path("") == ""


def test_resolve_model_path_returns_existing_file(tmp_path):
    f = tmp_path / "rf_30feature_integrated.joblib"
    f.write_bytes(b"x")
    assert resolve_model_path(str(f)) == str(f)


def test_resolve_model_path_falls_back_to_existing_alias(tmp_path):
    # Canonical name does not exist; a known legacy alias does.
    alias = tmp_path / "rf_30f_immunogenicity.joblib"
    alias.write_bytes(b"x")
    canonical = tmp_path / "rf_30feature_integrated.joblib"
    assert resolve_model_path(str(canonical)) == str(alias)


def test_resolve_model_path_no_match_returns_original(tmp_path):
    canonical = tmp_path / "rf_30feature_integrated.joblib"  # nothing on disk
    assert resolve_model_path(str(canonical)) == str(canonical)


def test_canonical_output_filename_format():
    out = canonical_output_filename("h2_tier_a_summary", "modeA_baseline", "IEDB-v1")
    assert out == "h2_tier_a_summary__modeA_baseline__IEDB-v1.csv"
