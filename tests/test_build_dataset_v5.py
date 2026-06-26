"""Unit and integration tests for scripts/build_dataset_v5.py.

Covers Part 15 build logic plus the Part 16 Amendment 2/3 additions:
  - V5_COLUMNS / ensure_v5_columns   the six new nullable columns
  - normalize_hpv                    HPV16/HPV18 -> virus=HPV + strain
  - assign_virus_family              family mapping
  - apply_quarantine                 depth-threshold flagging
  - audit_label_conflicts            v4-positive / IEDB-negative collision file
  - warn_low_pmid_depth              <3 distinct PMID warning for target viruses
  - main                             dry-run end-to-end with conflict audit
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_dataset_v5 import (
    MIN_DISTINCT_PMIDS_PER_TARGET,
    V5_COLUMNS,
    apply_quarantine,
    assign_virus_family,
    audit_label_conflicts,
    ensure_v5_columns,
    main,
    normalize_hpv,
    warn_low_pmid_depth,
)

LOGGER = logging.getLogger("test_build")

NEW_COLUMNS = [
    "assay_quality_tier",
    "iedb_assay_id",
    "infection_phase",
    "antigen_latency_program",
    "assay_context",
    "cross_reactivity_tested",
    "virus_taxon_id",
]

SCHEMA_PATH = Path("data/immunogenicity_dataset_v5_schema.json")


# ---------------------------------------------------------------------------
# Schema / column wiring (Amendment 2)
# ---------------------------------------------------------------------------


def test_v5_columns_include_new() -> None:
    for col in NEW_COLUMNS:
        assert col in V5_COLUMNS
    # is_quarantined remains the final column.
    assert V5_COLUMNS[-1] == "is_quarantined"


def test_ensure_v5_columns_adds_new_with_defaults() -> None:
    df = pd.DataFrame({"peptide": ["GILGFVFTL"], "label": [0]})
    out = ensure_v5_columns(df)
    for col in NEW_COLUMNS:
        assert col in out.columns
        assert out.iloc[0][col] is None


def test_ensure_v5_columns_preserves_existing() -> None:
    df = pd.DataFrame(
        {"peptide": ["GILGFVFTL"], "label": [0], "iedb_assay_id": ["AS-1"]}
    )
    out = ensure_v5_columns(df)
    assert out.iloc[0]["iedb_assay_id"] == "AS-1"


# ---------------------------------------------------------------------------
# HPV normalization
# ---------------------------------------------------------------------------


def test_normalize_hpv() -> None:
    df = pd.DataFrame(
        {
            "virus": ["HPV16", "HPV18", "HPV", "EBV"],
            "strain": [None, None, None, None],
        }
    )
    out = normalize_hpv(df, LOGGER)
    assert list(out["virus"]) == ["HPV", "HPV", "HPV", "EBV"]
    assert out.loc[0, "strain"] == "HPV16"
    assert out.loc[1, "strain"] == "HPV18"
    assert out.loc[2, "strain"] == "HPV_generic"
    assert out.loc[3, "strain"] is None


# ---------------------------------------------------------------------------
# virus_family
# ---------------------------------------------------------------------------


def test_assign_virus_family() -> None:
    df = pd.DataFrame({"virus": ["EBV", "SARS-CoV-2", "HPV", "Nonsense"]})
    out = assign_virus_family(df, LOGGER)
    assert out.loc[0, "virus_family"] == "Herpesviridae"
    assert out.loc[1, "virus_family"] == "Coronaviridae"
    assert out.loc[2, "virus_family"] == "Papillomaviridae"
    # Unmapped virus stays null.
    assert pd.isna(out.loc[3, "virus_family"])


# ---------------------------------------------------------------------------
# Quarantine
# ---------------------------------------------------------------------------


def _virus_block(
    virus: str, n_rows: int, n_real_neg: int
) -> pd.DataFrame:
    origins = ["tested_negative"] * n_real_neg + ["self_proteome_decoy"] * (
        n_rows - n_real_neg
    )
    return pd.DataFrame(
        {
            "virus": [virus] * n_rows,
            "source_type": ["Virus"] * n_rows,
            "label": [0] * n_rows,
            "negative_origin": origins,
        }
    )


def test_apply_quarantine_flags_thin_virus() -> None:
    deep = _virus_block("EBV", n_rows=60, n_real_neg=12)
    thin = _virus_block("RSV", n_rows=5, n_real_neg=0)
    df = pd.concat([deep, thin], ignore_index=True)

    out, quarantined = apply_quarantine(df, LOGGER)
    assert "RSV" in quarantined
    assert "EBV" not in quarantined
    assert out.loc[out["virus"] == "RSV", "is_quarantined"].all()
    assert not out.loc[out["virus"] == "EBV", "is_quarantined"].any()


def test_apply_quarantine_flags_insufficient_negatives() -> None:
    # 60 rows but only 3 real negatives -> below MIN_REAL_NEGATIVES_PER_VIRUS.
    df = _virus_block("HPV", n_rows=60, n_real_neg=3)
    out, quarantined = apply_quarantine(df, LOGGER)
    assert "HPV" in quarantined
    assert out["is_quarantined"].all()


# ---------------------------------------------------------------------------
# Conflict audit (Amendment 3)
# ---------------------------------------------------------------------------


def test_audit_label_conflicts_detects_and_writes(tmp_path: Path) -> None:
    v4_pos = pd.DataFrame(
        {
            "peptide": ["GILGFVFTL", "NLVPMVATV"],
            "hla_allele": ["HLA-A*02:01", "HLA-A*02:01"],
            "virus": ["IAV", "CMV"],
            "label": [1, 1],
        }
    )
    iedb_neg = pd.DataFrame(
        {
            "peptide": ["GILGFVFTL", "CINGVCWTV"],
            "hla_allele": ["HLA-A*02:01", "HLA-A*02:01"],
            "virus": ["IAV", "HCV"],
            "label": [0, 0],
        }
    )
    out_path = tmp_path / "conflicts.csv"
    n = audit_label_conflicts(v4_pos, iedb_neg, out_path, LOGGER)

    assert n == 1  # only (GILGFVFTL, HLA-A*02:01, IAV) collides
    assert out_path.exists()
    written = pd.read_csv(out_path)
    assert set(written["conflict_source"]) == {"v4_positive", "iedb_negative"}
    assert len(written) == 2  # one row from each source


def test_audit_label_conflicts_none(tmp_path: Path) -> None:
    v4_pos = pd.DataFrame(
        {
            "peptide": ["GILGFVFTL"],
            "hla_allele": ["HLA-A*02:01"],
            "virus": ["IAV"],
            "label": [1],
        }
    )
    iedb_neg = pd.DataFrame(
        {
            "peptide": ["NLVPMVATV"],
            "hla_allele": ["HLA-A*02:01"],
            "virus": ["CMV"],
            "label": [0],
        }
    )
    out_path = tmp_path / "conflicts.csv"
    n = audit_label_conflicts(v4_pos, iedb_neg, out_path, LOGGER)
    assert n == 0
    assert out_path.exists()  # empty file still written for traceability
    assert len(pd.read_csv(out_path)) == 0


# ---------------------------------------------------------------------------
# PMID depth warning (Amendment 3)
# ---------------------------------------------------------------------------


def test_warn_low_pmid_depth(caplog: pytest.LogCaptureFixture) -> None:
    df = pd.DataFrame(
        {
            "virus": ["HCV", "HCV", "EBV", "EBV", "EBV"],
            "reference_pmid": ["111", "111", "201", "202", "203"],
            "is_quarantined": [False] * 5,
        }
    )
    with caplog.at_level(logging.WARNING):
        counts = warn_low_pmid_depth(df, LOGGER)

    assert counts["HCV"] == 1  # single distinct PMID
    assert counts["EBV"] == 3
    # HCV below threshold triggers a warning; EBV (>=3) does not.
    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("HCV" in m for m in warnings)
    assert not any("EBV" in m for m in warnings)
    assert MIN_DISTINCT_PMIDS_PER_TARGET == 3


def test_warn_low_pmid_depth_excludes_quarantined() -> None:
    df = pd.DataFrame(
        {
            "virus": ["HCV", "HCV", "HCV"],
            "reference_pmid": ["111", "222", "333"],
            "is_quarantined": [True, True, True],
        }
    )
    counts = warn_low_pmid_depth(df, LOGGER)
    # All HCV rows quarantined -> 0 counted.
    assert counts["HCV"] == 0


# ---------------------------------------------------------------------------
# main: dry-run end-to-end
# ---------------------------------------------------------------------------


def _write_base_v4(path: Path) -> None:
    pd.DataFrame(
        {
            "peptide": [
                "GILGFVFTL",
                "NLVPMVATV",
                "AAAAAAAAA",
                "CCCCCCCCC",
            ],
            "label": [1, 1, 0, 0],
            "virus": ["IAV", "CMV", "Self", "Self"],
            "protein": ["M1", "pp65", "", ""],
            "strain": [None, None, None, None],
            "hla_allele": ["HLA-A*02:01"] * 4,
            "source_type": ["Virus", "Virus", "Self", "Self"],
            "database_source": ["IEDB", "IEDB", "UniProt", "UniProt"],
        }
    ).to_csv(path, index=False)


def _write_iedb_negatives(path: Path) -> None:
    pd.DataFrame(
        {
            "peptide": ["KLGGALQAK", "CINGVCWTV"],
            "label": [0, 0],
            "virus": ["CMV", "HCV"],
            "hla_allele": ["HLA-A*02:01", "HLA-A*02:01"],
            "source_type": ["Virus", "Virus"],
            "negative_origin": ["tested_negative", "tested_negative"],
            "reference_pmid": ["555", "666"],
        }
    ).to_csv(path, index=False)


def test_main_dry_run(tmp_path: Path) -> None:
    base = tmp_path / "v4.csv"
    iedb = tmp_path / "iedb_neg.csv"
    out = tmp_path / "v5.csv"
    conflicts = tmp_path / "conflicts.csv"
    _write_base_v4(base)
    _write_iedb_negatives(iedb)

    rc = main(
        [
            "--base-dataset",
            str(base),
            "--iedb-negatives",
            str(iedb),
            "--output",
            str(out),
            "--schema",
            str(SCHEMA_PATH),
            "--conflict-audit-path",
            str(conflicts),
            "--dry-run",
        ]
    )
    assert rc == 0
    assert not out.exists()  # dry-run writes no dataset
    assert conflicts.exists()  # conflict audit always runs


def test_main_full_build(tmp_path: Path) -> None:
    base = tmp_path / "v4.csv"
    iedb = tmp_path / "iedb_neg.csv"
    out = tmp_path / "v5.csv"
    conflicts = tmp_path / "conflicts.csv"
    _write_base_v4(base)
    _write_iedb_negatives(iedb)

    rc = main(
        [
            "--base-dataset",
            str(base),
            "--iedb-negatives",
            str(iedb),
            "--output",
            str(out),
            "--schema",
            str(SCHEMA_PATH),
            "--conflict-audit-path",
            str(conflicts),
        ]
    )
    assert rc == 0
    assert out.exists()

    v5 = pd.read_csv(out)
    # All v5 columns present and in order at the front.
    assert list(v5.columns)[: len(V5_COLUMNS)] == V5_COLUMNS
    # 4 v4 rows + 2 IEDB negatives = 6 merged rows.
    assert len(v5) == 6

    sidecar = out.with_name(out.stem + "_provenance.json")
    assert sidecar.exists()


# ---------------------------------------------------------------------------
# Null virus_family quarantine
# ---------------------------------------------------------------------------


def test_assign_virus_family_new_entries() -> None:
    df = pd.DataFrame(
        {
            "virus": [
                "ZIKV",
                "CHIKV",
                "Orthopoxvirus vaccinia",
                "Influenza A virus",
                "HTLV-1",
            ]
        }
    )
    out = assign_virus_family(df, LOGGER)
    assert out.loc[0, "virus_family"] == "Flaviviridae"
    assert out.loc[1, "virus_family"] == "Togaviridae"
    assert out.loc[2, "virus_family"] == "Poxviridae"
    assert out.loc[3, "virus_family"] == "Orthomyxoviridae"
    assert out.loc[4, "virus_family"] == "Retroviridae"


def _virus_block_with_family(
    virus: str, n_rows: int, n_real_neg: int, family: str | None
) -> pd.DataFrame:
    origins = ["tested_negative"] * n_real_neg + ["self_proteome_decoy"] * (
        n_rows - n_real_neg
    )
    return pd.DataFrame(
        {
            "virus": [virus] * n_rows,
            "source_type": ["Virus"] * n_rows,
            "label": [0] * n_rows,
            "negative_origin": origins,
            "virus_family": [family] * n_rows,
        }
    )


def test_apply_quarantine_null_family_quarantined() -> None:
    deep = _virus_block_with_family("EBV", n_rows=60, n_real_neg=12, family="Herpesviridae")
    unknown = _virus_block_with_family("Unknown", n_rows=60, n_real_neg=12, family=None)
    df = pd.concat([deep, unknown], ignore_index=True)

    _out, quarantined = apply_quarantine(df, LOGGER)
    assert "Unknown" in quarantined
    assert "EBV" not in quarantined


def test_apply_quarantine_null_family_real_neg_threshold_independent() -> None:
    # Null-family rule must fire regardless of row/real-neg counts.
    df = _virus_block_with_family(
        "Mycobacterium tuberculosis", n_rows=200, n_real_neg=150, family=None
    )
    _out, quarantined = apply_quarantine(df, LOGGER)
    assert "Mycobacterium tuberculosis" in quarantined
