"""Unit and integration tests for scripts/ingest_iedb_negatives.py.

Covers the Part 16 Amendment 1 hardening:
  - normalize_hla_allele     mhcgnomes 4-digit normalization, supertype /
                             non-human / Class II rejection, space repair
  - load_iedb_export         two-row IEDB header detection
  - filter_rows              response_frequency secondary negativity signal,
                             intra-export deduplication
  - assign_quality_tier      three-tier assay quality weighting
  - build_output             iedb_assay_id provenance, v5 column set
  - main                     dry-run and full write with input_sha256 sidecar
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from scripts.ingest_iedb_negatives import (
    EXCLUDED_ASSAY_TYPES,
    TIER_WEIGHT,
    ResolvedColumns,
    _repair_allele_spacing,
    assign_quality_tier,
    assign_quality_weight,
    build_output,
    filter_rows,
    is_valid_peptide,
    load_iedb_export,
    main,
    normalize_hla_allele,
    resolve_columns,
)

LOGGER = logging.getLogger("test_ingest")

# A valid human Class I 9-mer / allele pair reused across rows.
PEP_A = "GILGFVFTL"
PEP_B = "NLVPMVATV"
PEP_C = "CINGVCWTV"


# ---------------------------------------------------------------------------
# Synthetic IEDB export helpers
# ---------------------------------------------------------------------------


def _iedb_row(
    peptide: str,
    allele: str = "HLA-A*02:01",
    measure: str = "Negative",
    host: str = "Homo sapiens",
    assay_group: str = "IFN-gamma ELISpot",
    pmid: str = "12345",
    response_frequency: str = "",
    assay_id: str = "AS-1",
) -> dict[str, str]:
    return {
        "Host Organism Name": host,
        "Qualitative Measure": measure,
        "Epitope Name": peptide,
        "Allele Name": allele,
        "Assay Group": assay_group,
        "Antigen Name": "Matrix protein 1",
        "Reference PubMed ID": pmid,
        "Description": "Influenza A virus",
        "Response Frequency As Reported": response_frequency,
        "Assay ID": assay_id,
    }


def _make_iedb_df(rows: list[dict[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _resolve(df: pd.DataFrame) -> ResolvedColumns:
    return resolve_columns(df, LOGGER)


# ---------------------------------------------------------------------------
# normalize_hla_allele
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("HLA-A*02:01", "HLA-A*02:01"),
        ("A*02:01", "HLA-A*02:01"),
        ("HLA-A 02:01", "HLA-A*02:01"),  # space form repaired
        ("HLA-A*02:01:01:01", "HLA-A*02:01"),  # truncated to 4-digit
        ("HLA-B*07:02", "HLA-B*07:02"),
        ("HLA-C*07:02", "HLA-C*07:02"),
        ("  HLA-A*02:01  ", "HLA-A*02:01"),
    ],
)
def test_normalize_hla_allele_accepts(raw: str, expected: str) -> None:
    assert normalize_hla_allele(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "HLA-A2",  # supertype / serotype
        "H-2-Kb",  # mouse, non-human
        "HLA-DRB1*04:01",  # Class II
        "HLA-E*01:01",  # Class I but not A/B/C
        "Class I",  # bare class string
        "garbage",
        "",
        "   ",
    ],
)
def test_normalize_hla_allele_rejects(raw: str) -> None:
    assert normalize_hla_allele(raw) is None


def test_normalize_hla_allele_non_string() -> None:
    assert normalize_hla_allele(None) is None  # type: ignore[arg-type]
    assert normalize_hla_allele(3.14) is None  # type: ignore[arg-type]


def test_repair_allele_spacing() -> None:
    assert _repair_allele_spacing("HLA-A 02:01") == "HLA-A*02:01"
    assert _repair_allele_spacing("HLA-B 07:02") == "HLA-B*07:02"
    # No leading-space-before-digit -> unchanged.
    assert _repair_allele_spacing("HLA-A*02:01") == "HLA-A*02:01"
    assert _repair_allele_spacing("A*02:01") == "A*02:01"


# ---------------------------------------------------------------------------
# Peptide validity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seq,valid",
    [
        ("GILGFVFTL", True),  # 9-mer
        ("SIINFEKL", True),  # 8-mer
        ("GILGFVFTLAA", True),  # 11-mer
        ("GILGFVFTLAAA", False),  # 12-mer too long
        ("GILGFVF", False),  # 7-mer too short
        ("GILGFVFTX", False),  # non-standard AA
        ("", False),
    ],
)
def test_is_valid_peptide(seq: str, valid: bool) -> None:
    assert is_valid_peptide(seq) is valid


# ---------------------------------------------------------------------------
# Assay quality tiers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "assay_group,tier",
    [
        ("IFN-gamma ELISpot", 1),
        ("MHC multimer", 1),
        ("cytotoxicity", 1),
        ("qualitative binding", 2),
        ("ELISA", 2),
        ("cultured ELISpot", 3),  # culture keyword dominates functional
        ("expanded IFN-gamma", 3),
        ("bulk culture proliferation", 3),
        ("stimulated PBMC", 3),
        (None, 2),
    ],
)
def test_assign_quality_tier(assay_group: str | None, tier: int) -> None:
    assert assign_quality_tier(assay_group) == tier
    assert assign_quality_weight(assay_group) == TIER_WEIGHT[tier]


# ---------------------------------------------------------------------------
# Two-row IEDB header detection
# ---------------------------------------------------------------------------


def test_load_iedb_export_two_row_header(tmp_path: Path) -> None:
    """Post-2023 export: group-name row 0, real field names row 1."""
    csv_path = tmp_path / "iedb_two_row.csv"
    lines = [
        "Reference,Epitope,Host,Assay,MHC Restriction",
        "Reference PubMed ID,Epitope Name,Host Organism Name,Qualitative Measure,Allele Name",
        f"12345,{PEP_A},Homo sapiens,Negative,HLA-A*02:01",
    ]
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    df = load_iedb_export(csv_path, LOGGER)
    assert "Epitope Name" in df.columns
    assert "Qualitative Measure" in df.columns
    assert len(df) == 1
    assert df.iloc[0]["Epitope Name"] == PEP_A


def test_load_iedb_export_two_row_header_assay_id_first_col(tmp_path: Path) -> None:
    """Real June-2026 IEDB export has 'Assay ID' as first group-row column."""
    csv_path = tmp_path / "iedb_assay_id_first.csv"
    lines = [
        '"Assay ID",Reference,Reference,Epitope,Host,Assay,"MHC Restriction"',
        '"IEDB IRI","Reference PubMed ID",Type,"Epitope Name","Host Organism Name","Qualitative Measure","Allele Name"',
        f"http://www.iedb.org/assay/1,12345,Literature,{PEP_A},Homo sapiens,Negative,HLA-A*02:01",
    ]
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    df = load_iedb_export(csv_path, LOGGER)
    assert "Epitope Name" in df.columns
    assert "Qualitative Measure" in df.columns
    assert len(df) == 1
    assert df.iloc[0]["Epitope Name"] == PEP_A


def test_load_iedb_export_single_row_header(tmp_path: Path) -> None:
    csv_path = tmp_path / "iedb_single_row.csv"
    df_in = _make_iedb_df([_iedb_row(PEP_A)])
    df_in.to_csv(csv_path, index=False)

    df = load_iedb_export(csv_path, LOGGER)
    assert "Epitope Name" in df.columns
    assert len(df) == 1


# ---------------------------------------------------------------------------
# filter_rows: secondary negativity + intra-export dedup
# ---------------------------------------------------------------------------


def test_filter_rows_response_frequency_secondary() -> None:
    rows = [
        _iedb_row(PEP_A, measure="Negative"),  # primary negative
        _iedb_row(PEP_B, measure="", response_frequency="0.05"),  # secondary neg
        _iedb_row(PEP_C, measure="", response_frequency="0.50"),  # not negative
        _iedb_row("SIINFEKL", measure="Positive"),  # positive, dropped
    ]
    df = _make_iedb_df(rows)
    cols = _resolve(df)
    filtered, stats = filter_rows(df, cols, LOGGER)

    assert stats["negatives_by_qualitative_measure"] == 1
    assert stats["negatives_by_response_frequency"] == 1
    # PEP_A (primary) and PEP_B (secondary) survive; PEP_C and the positive drop.
    kept = set(filtered["Epitope Name"])
    assert kept == {PEP_A, PEP_B}


def test_filter_rows_intra_export_dedup() -> None:
    rows = [
        _iedb_row(PEP_A, assay_id="AS-1"),
        # Same peptide/allele/pmid via a different assay-format row.
        _iedb_row(PEP_A, assay_group="MHC multimer", assay_id="AS-2"),
        _iedb_row(PEP_B),
    ]
    df = _make_iedb_df(rows)
    cols = _resolve(df)
    filtered, stats = filter_rows(df, cols, LOGGER)

    assert stats["intra_export_duplicates_removed"] == 1
    assert stats["after_intra_export_dedup"] == 2
    assert sorted(filtered["Epitope Name"]) == sorted([PEP_A, PEP_B])


def test_filter_rows_human_host_filter() -> None:
    rows = [
        _iedb_row(PEP_A, host="Homo sapiens"),
        _iedb_row(PEP_B, host="Mus musculus"),
    ]
    df = _make_iedb_df(rows)
    cols = _resolve(df)
    filtered, stats = filter_rows(df, cols, LOGGER)
    assert stats["after_human_host_filter"] == 1
    assert list(filtered["Epitope Name"]) == [PEP_A]


# ---------------------------------------------------------------------------
# build_output
# ---------------------------------------------------------------------------


def test_build_output_columns_and_provenance() -> None:
    rows = [_iedb_row(PEP_A, assay_id="AS-42", pmid="99999")]
    df = _make_iedb_df(rows)
    cols = _resolve(df)
    filtered, _ = filter_rows(df, cols, LOGGER)
    out = build_output(filtered, cols, LOGGER)

    assert (out["label"] == 0).all()
    assert (out["negative_origin"] == "tested_negative").all()
    assert (out["database_source"] == "IEDB").all()
    assert out.iloc[0]["hla_allele"] == "HLA-A*02:01"
    assert out.iloc[0]["iedb_assay_id"] == "AS-42"
    assert out.iloc[0]["reference_pmid"] == "99999"
    assert out.iloc[0]["assay_quality_tier"] == 1
    assert out.iloc[0]["assay_quality_weight"] == 1.0
    # v5 biology-context columns present and null at ingest stage.
    for col in (
        "infection_phase",
        "antigen_latency_program",
        "cross_reactivity_tested",
        "virus_taxon_id",
    ):
        assert col in out.columns
        assert out.iloc[0][col] is None


# ---------------------------------------------------------------------------
# main: end-to-end
# ---------------------------------------------------------------------------


def _write_existing(tmp_path: Path, peptide: str = "AAAAAAAAA") -> Path:
    """Existing (frozen) dataset for the cross-dedup step."""
    path = tmp_path / "existing_v4.csv"
    pd.DataFrame(
        {
            "peptide": [peptide],
            "hla_allele": ["HLA-A*02:01"],
            "virus": ["Influenza A virus"],
            "label": [0],
        }
    ).to_csv(path, index=False)
    return path


def test_main_dry_run(tmp_path: Path) -> None:
    input_path = tmp_path / "iedb.csv"
    _make_iedb_df([_iedb_row(PEP_A), _iedb_row(PEP_B)]).to_csv(input_path, index=False)
    existing = _write_existing(tmp_path)
    output = tmp_path / "out.csv"

    rc = main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output),
            "--existing-dataset",
            str(existing),
            "--dry-run",
        ]
    )
    assert rc == 0
    assert not output.exists()  # dry-run writes nothing


def test_main_full_write_and_sidecar(tmp_path: Path) -> None:
    input_path = tmp_path / "iedb.csv"
    _make_iedb_df([_iedb_row(PEP_A), _iedb_row(PEP_B)]).to_csv(input_path, index=False)
    existing = _write_existing(tmp_path)
    output = tmp_path / "iedb_negatives_v5.csv"

    rc = main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output),
            "--existing-dataset",
            str(existing),
        ]
    )
    assert rc == 0
    assert output.exists()

    out_df = pd.read_csv(output)
    assert len(out_df) == 2
    assert set(out_df["label"]) == {0}
    for col in ("iedb_assay_id", "assay_quality_tier", "negative_origin"):
        assert col in out_df.columns

    sidecar = output.with_name(output.stem + "_provenance.json")
    assert sidecar.exists()
    prov = json.loads(sidecar.read_text(encoding="utf-8"))
    # input_sha256 pins the exact IEDB snapshot (Amendment 1 provenance fix).
    assert "input_sha256" in prov
    assert len(prov["input_sha256"]) == 64
    assert "output_checksum_sha256" in prov
    assert prov["filter_stats"]["final_output_rows"] == 2


def test_main_cross_dedup_drops_existing(tmp_path: Path) -> None:
    # The existing dataset already contains PEP_A on the same allele/virus.
    input_path = tmp_path / "iedb.csv"
    _make_iedb_df([_iedb_row(PEP_A), _iedb_row(PEP_B)]).to_csv(input_path, index=False)
    existing = _write_existing(tmp_path, peptide=PEP_A)
    output = tmp_path / "out.csv"

    rc = main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output),
            "--existing-dataset",
            str(existing),
        ]
    )
    assert rc == 0
    out_df = pd.read_csv(output)
    # PEP_A collides with the existing row and is dropped; only PEP_B remains.
    assert list(out_df["peptide"]) == [PEP_B]


# ---------------------------------------------------------------------------
# filter_rows: assay-type exclusion (Filter 6)
# ---------------------------------------------------------------------------


def test_biological_activity_rows_excluded() -> None:
    """'biological activity' rows are excluded; ELISPOT rows survive."""
    rows = [
        _iedb_row(PEP_A, assay_group="biological activity"),
        _iedb_row(PEP_B, assay_group="biological activity"),
        _iedb_row(PEP_C, assay_group="IFN-gamma ELISpot"),
    ]
    df = _make_iedb_df(rows)
    cols = _resolve(df)
    filtered, stats = filter_rows(df, cols, LOGGER)

    assert stats["biological_activity_rows_excluded"] == 2
    assert stats["after_assay_type_filter"] == 1
    kept = set(filtered["Epitope Name"])
    assert kept == {PEP_C}


def test_excluded_assay_types_constant() -> None:
    """Sanity check: the constant contains the expected value."""
    assert "biological activity" in EXCLUDED_ASSAY_TYPES
