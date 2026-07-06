"""Unit and integration tests for scripts/ingest_published_panels.py.

Covers:
  - load_panel_csv          required-column validation
  - filter_rows             label, peptide, HLA filtering
  - build_output            biology-context columns, negative_origin, assay quality
  - main                    dry-run + full-write + provenance sidecar
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from scripts.ingest_published_panels import (
    TIER_WEIGHT,
    VALID_ASSAY_CONTEXT,
    VALID_INFECTION_PHASE,
    VALID_LATENCY_PROGRAM,
    build_output,
    filter_rows,
    load_panel_csv,
    main,
)

LOGGER = logging.getLogger("test_panels")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_panel_csv(
    tmp_path: Path,
    rows: list[dict],
    filename: str = "panel.csv",
) -> Path:
    path = tmp_path / filename
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


_VALID_ROW = {
    "peptide": "GILGFVFTL",
    "label": 1,
    "hla_allele": "HLA-A*02:01",
    "protein": "M1",
}

_VALID_NEG_ROW = {
    "peptide": "NLVPMVATV",
    "label": 0,
    "hla_allele": "HLA-A*02:01",
    "protein": "pp65",
}


# ---------------------------------------------------------------------------
# load_panel_csv
# ---------------------------------------------------------------------------


def test_load_panel_csv_valid(tmp_path: Path) -> None:
    path = _make_panel_csv(tmp_path, [_VALID_ROW])
    df = load_panel_csv(path, LOGGER)
    assert len(df) == 1
    assert "peptide" in df.columns
    assert "label" in df.columns


def test_load_panel_csv_missing_required_exits(tmp_path: Path) -> None:
    path = _make_panel_csv(tmp_path, [{"peptide": "GILGFVFTL"}])
    with pytest.raises(SystemExit):
        load_panel_csv(path, LOGGER)


def test_load_panel_csv_missing_both_exits(tmp_path: Path) -> None:
    path = _make_panel_csv(tmp_path, [{"hla_allele": "HLA-A*02:01"}])
    with pytest.raises(SystemExit):
        load_panel_csv(path, LOGGER)


# ---------------------------------------------------------------------------
# filter_rows
# ---------------------------------------------------------------------------


def test_filter_rows_drops_invalid_labels(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "peptide": ["GILGFVFTL", "NLVPMVATV", "KLGGALQAK"],
            "label": [1, 2, None],
        }
    )
    out, stats = filter_rows(df, None, LOGGER)
    # label=2 and None are dropped; only label=1 survives.
    assert len(out) == 1
    assert stats["after_label_filter"] == 1


def test_filter_rows_drops_invalid_peptides() -> None:
    df = pd.DataFrame(
        {
            "peptide": [
                "GILGFVFTL",  # valid 9mer
                "SHORT",  # too short (5)
                "TOOLONGPEPTIDESEQ",  # too long
                "XILGFVFTL",  # invalid AA
            ],
            "label": [1, 0, 0, 1],
        }
    )
    out, stats = filter_rows(df, None, LOGGER)
    assert len(out) == 1
    assert out.iloc[0]["peptide"] == "GILGFVFTL"
    assert stats["after_peptide_filter"] == 1


def test_filter_rows_hla_override_applied() -> None:
    df = pd.DataFrame({"peptide": ["GILGFVFTL", "NLVPMVATV"], "label": [1, 0]})
    out, _ = filter_rows(df, "HLA-A*02:01", LOGGER)
    assert (out["hla_allele"] == "HLA-A*02:01").all()


def test_filter_rows_hla_override_normalizes() -> None:
    # Space form should be normalized to star form.
    df = pd.DataFrame({"peptide": ["GILGFVFTL"], "label": [1]})
    out, _ = filter_rows(df, "HLA-A 02:01", LOGGER)
    assert out.iloc[0]["hla_allele"] == "HLA-A*02:01"


def test_filter_rows_hla_column_normalization() -> None:
    df = pd.DataFrame(
        {
            "peptide": ["GILGFVFTL", "NLVPMVATV"],
            "label": [1, 0],
            "hla_allele": ["HLA-A*02:01", "notanallele"],
        }
    )
    out, stats = filter_rows(df, None, LOGGER)
    # Row with bad allele is dropped.
    assert len(out) == 1
    assert out.iloc[0]["hla_allele"] == "HLA-A*02:01"
    assert stats.get("after_hla_filter", 1) == 1


def test_filter_rows_no_hla_info_produces_null_column() -> None:
    df = pd.DataFrame({"peptide": ["GILGFVFTL"], "label": [1]})
    out, _ = filter_rows(df, None, LOGGER)
    assert "hla_allele" in out.columns
    assert out.iloc[0]["hla_allele"] is None


def test_filter_rows_invalid_hla_override_exits() -> None:
    df = pd.DataFrame({"peptide": ["GILGFVFTL"], "label": [1]})
    with pytest.raises(SystemExit):
        filter_rows(df, "notanallele", LOGGER)


# ---------------------------------------------------------------------------
# build_output
# ---------------------------------------------------------------------------


def _base_df(n_pos: int = 1, n_neg: int = 1) -> pd.DataFrame:
    rows_pos = [{"peptide": "GILGFVFTL", "label": 1, "hla_allele": "HLA-A*02:01"}] * n_pos
    rows_neg = [{"peptide": "NLVPMVATV", "label": 0, "hla_allele": "HLA-A*02:01"}] * n_neg
    return pd.DataFrame(rows_pos + rows_neg).reset_index(drop=True)


def _call_build(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    defaults = dict(
        virus="HBV",
        pmid="12520014",
        assay_context="natural_infection",
        infection_phase="chronic",
        latency_program=None,
        cross_reactivity_tested=None,
        virus_taxon_id=10359,
        assay_quality_tier=1,
        assay_type="T cell IFN-gamma ELISpot",
        logger=LOGGER,
    )
    defaults.update(kwargs)
    return build_output(df, **defaults)


def test_build_output_negative_origin_label0() -> None:
    out = _call_build(_base_df(n_pos=0, n_neg=2))
    assert (out["negative_origin"] == "tested_negative").all()


def test_build_output_negative_origin_label1() -> None:
    out = _call_build(_base_df(n_pos=2, n_neg=0))
    assert out["negative_origin"].isna().all()


def test_build_output_mixed_negative_origin() -> None:
    out = _call_build(_base_df(n_pos=1, n_neg=1))
    pos_mask = out["label"] == 1
    assert out.loc[pos_mask, "negative_origin"].isna().all()
    assert (out.loc[~pos_mask, "negative_origin"] == "tested_negative").all()


def test_build_output_biology_context_set() -> None:
    out = _call_build(
        _base_df(),
        assay_context="vaccine_induced",
        infection_phase="acute",
        latency_program="lytic",
        cross_reactivity_tested=True,
        virus_taxon_id=10376,
    )
    assert (out["assay_context"] == "vaccine_induced").all()
    assert (out["infection_phase"] == "acute").all()
    assert (out["antigen_latency_program"] == "lytic").all()
    assert (out["cross_reactivity_tested"] == True).all()  # noqa: E712
    assert (out["virus_taxon_id"] == 10376).all()


def test_build_output_biology_context_null_when_not_provided() -> None:
    out = _call_build(
        _base_df(),
        assay_context=None,
        infection_phase=None,
        latency_program=None,
        cross_reactivity_tested=None,
        virus_taxon_id=None,
    )
    for col in (
        "assay_context",
        "infection_phase",
        "antigen_latency_program",
        "cross_reactivity_tested",
        "virus_taxon_id",
    ):
        assert out[col].isna().all(), f"{col} should be null"


def test_build_output_assay_quality_tier_2() -> None:
    out = _call_build(_base_df(), assay_quality_tier=2)
    assert (out["assay_quality_tier"] == 2).all()
    assert (out["assay_quality_weight"] == TIER_WEIGHT[2]).all()


def test_build_output_assay_quality_tier_3() -> None:
    out = _call_build(_base_df(), assay_quality_tier=3)
    assert (out["assay_quality_weight"] == 0.5).all()


def test_build_output_static_fields() -> None:
    out = _call_build(_base_df())
    assert (out["source_type"] == "Virus").all()
    assert (out["database_source"] == "Published").all()
    assert (out["reference_pmid"] == "12520014").all()
    assert out["iedb_assay_id"].isna().all()
    assert out["virus_family"].isna().all()
    assert (out["is_quarantined"] == False).all()  # noqa: E712


def test_build_output_virus_column() -> None:
    out = _call_build(_base_df(), virus="HPV")
    assert (out["virus"] == "HPV").all()


def test_build_output_cross_reactivity_false() -> None:
    out = _call_build(_base_df(), cross_reactivity_tested=False)
    assert (out["cross_reactivity_tested"] == False).all()  # noqa: E712


# ---------------------------------------------------------------------------
# Enum constant sanity checks
# ---------------------------------------------------------------------------


def test_valid_assay_context_values() -> None:
    assert "vaccine_induced" in VALID_ASSAY_CONTEXT
    assert "natural_infection" in VALID_ASSAY_CONTEXT
    assert "unknown" in VALID_ASSAY_CONTEXT


def test_valid_infection_phase_values() -> None:
    assert "acute" in VALID_INFECTION_PHASE
    assert "chronic" in VALID_INFECTION_PHASE


def test_valid_latency_program_values() -> None:
    assert "lytic" in VALID_LATENCY_PROGRAM
    for stage in ("latent-I", "latent-II", "latent-III"):
        assert stage in VALID_LATENCY_PROGRAM


# ---------------------------------------------------------------------------
# main: dry-run and full-write
# ---------------------------------------------------------------------------


def _write_panel_csv(path: Path) -> None:
    pd.DataFrame(
        [
            {"peptide": "GILGFVFTL", "label": 1, "hla_allele": "HLA-A*02:01", "protein": "M1"},
            {"peptide": "NLVPMVATV", "label": 0, "hla_allele": "HLA-A*02:01", "protein": "pp65"},
            {"peptide": "KLGGALQAK", "label": 1, "hla_allele": "HLA-B*07:02", "protein": "IE-1"},
        ]
    ).to_csv(path, index=False)


def test_main_dry_run(tmp_path: Path) -> None:
    panel = tmp_path / "panel.csv"
    out = tmp_path / "out.csv"
    _write_panel_csv(panel)

    rc = main(
        [
            "--input",
            str(panel),
            "--virus",
            "HBV",
            "--pmid",
            "12520014",
            "--assay-context",
            "natural_infection",
            "--infection-phase",
            "chronic",
            "--output",
            str(out),
            "--dry-run",
        ]
    )
    assert rc == 0
    assert not out.exists()


def test_main_full_write(tmp_path: Path) -> None:
    panel = tmp_path / "panel.csv"
    out = tmp_path / "out.csv"
    _write_panel_csv(panel)

    rc = main(
        [
            "--input",
            str(panel),
            "--virus",
            "HCV",
            "--pmid",
            "12682213",
            "--assay-context",
            "natural_infection",
            "--infection-phase",
            "chronic",
            "--assay-type",
            "T cell IFN-gamma ELISpot",
            "--virus-taxon-id",
            "11103",
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    assert out.exists()

    df = pd.read_csv(out)
    # All expected v5 columns present.
    expected = [
        "peptide",
        "label",
        "virus",
        "hla_allele",
        "source_type",
        "database_source",
        "negative_origin",
        "assay_context",
        "infection_phase",
        "reference_pmid",
        "is_quarantined",
    ]
    for col in expected:
        assert col in df.columns, f"Missing column: {col}"

    # virus set correctly
    assert (df["virus"] == "HCV").all()
    # reference_pmid propagated (pandas may infer numeric PMID as int64 on read-back)
    assert (df["reference_pmid"].astype(str) == "12682213").all()
    # assay_context propagated
    assert (df["assay_context"] == "natural_infection").all()
    # negative_origin for label=0 rows
    neg_mask = df["label"] == 0
    assert (df.loc[neg_mask, "negative_origin"] == "tested_negative").all()
    assert df.loc[~neg_mask, "negative_origin"].isna().all()
    # virus_taxon_id set
    assert (df["virus_taxon_id"] == 11103).all()


def test_main_provenance_sidecar_written(tmp_path: Path) -> None:
    panel = tmp_path / "panel.csv"
    out = tmp_path / "out.csv"
    _write_panel_csv(panel)

    rc = main(["--input", str(panel), "--virus", "EBV", "--pmid", "99999", "--output", str(out)])
    assert rc == 0

    sidecar = out.with_name(out.stem + "_provenance.json")
    assert sidecar.exists()
    prov = json.loads(sidecar.read_text())
    assert prov["virus"] == "EBV"
    assert prov["pmid"] == "99999"
    assert "git_sha" in prov
    assert "input_sha256" in prov
    assert "output_checksum_sha256" in prov


def test_main_cross_reactivity_flag(tmp_path: Path) -> None:
    panel = tmp_path / "panel.csv"
    out = tmp_path / "out.csv"
    _write_panel_csv(panel)

    rc = main(
        [
            "--input",
            str(panel),
            "--virus",
            "HPV",
            "--pmid",
            "22222",
            "--assay-context",
            "vaccine_induced",
            "--cross-reactivity-tested",
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    df = pd.read_csv(out)
    assert df["cross_reactivity_tested"].all()


def test_main_no_cross_reactivity_flag(tmp_path: Path) -> None:
    panel = tmp_path / "panel.csv"
    out = tmp_path / "out.csv"
    _write_panel_csv(panel)

    rc = main(
        [
            "--input",
            str(panel),
            "--virus",
            "HPV",
            "--pmid",
            "22222",
            "--no-cross-reactivity-tested",
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    df = pd.read_csv(out)
    # cross_reactivity_tested=False stored as 0.0 (bool column in CSV).
    assert not df["cross_reactivity_tested"].any()


def test_main_missing_input_returns_1(tmp_path: Path) -> None:
    rc = main(
        [
            "--input",
            str(tmp_path / "nonexistent.csv"),
            "--virus",
            "HBV",
            "--pmid",
            "99",
            "--output",
            str(tmp_path / "out.csv"),
        ]
    )
    assert rc == 1


def test_main_empty_csv_live_run_returns_1(tmp_path: Path) -> None:
    panel = tmp_path / "panel.csv"
    out = tmp_path / "out.csv"
    # 'X' is not in the standard 20 AA set, so every row fails the peptide
    # validity filter. Using a valid label avoids the label-drop path so the
    # peptide filter actually runs and leaves 0 rows.
    pd.DataFrame(
        [
            {"peptide": "XXXXXXXXX", "label": 1, "hla_allele": "HLA-A*02:01"},
            {"peptide": "XXXXXXXXX", "label": 0, "hla_allele": "HLA-A*02:01"},
        ]
    ).to_csv(panel, index=False)

    rc = main(
        [
            "--input",
            str(panel),
            "--virus",
            "HPV",
            "--pmid",
            "7538538",
            "--output",
            str(out),
        ]
    )
    assert rc == 1
    assert not out.exists()


def test_main_empty_csv_dry_run_returns_0(tmp_path: Path) -> None:
    panel = tmp_path / "panel.csv"
    pd.DataFrame([{"peptide": "XXXXXXXXX", "label": 1, "hla_allele": "HLA-A*02:01"}]).to_csv(
        panel, index=False
    )

    rc = main(
        [
            "--input",
            str(panel),
            "--virus",
            "HPV",
            "--pmid",
            "7538538",
            "--dry-run",
        ]
    )
    # dry-run always returns 0 regardless of row count
    assert rc == 0


def test_main_default_output_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    panel = tmp_path / "panel.csv"
    _write_panel_csv(panel)
    monkeypatch.chdir(tmp_path)

    rc = main(["--input", str(panel), "--virus", "HBV", "--pmid", "11111"])
    assert rc == 0
    default_out = tmp_path / "data" / "published_panel_HBV_11111_v5.csv"
    assert default_out.exists()
