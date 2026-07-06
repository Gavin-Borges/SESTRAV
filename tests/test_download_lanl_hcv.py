"""Unit and integration tests for scripts/download_lanl_hcv.py.

Covers:
  - load_lanl_export        tab vs comma auto-detection
  - resolve_columns         auto-detection + CLI overrides + missing-required exit
  - filter_rows             negative filter, peptide validity, HLA class-I filter, dedup
  - build_output            static fields, assay_context, virus_taxon_id
  - main                    --inspect, dry-run, full-write, provenance sidecar
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from scripts.download_lanl_hcv import (
    build_output,
    filter_rows,
    load_lanl_export,
    main,
    resolve_columns,
)

LOGGER = logging.getLogger("test_lanl")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tsv(tmp_path: Path, rows: list[dict], filename: str = "export.tsv") -> Path:
    path = tmp_path / filename
    pd.DataFrame(rows).to_csv(path, index=False, sep="\t")
    return path


def _make_csv(tmp_path: Path, rows: list[dict], filename: str = "export.csv") -> Path:
    path = tmp_path / filename
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


_NEG_ROW = {
    "Sequence": "GILGFVFTL",
    "MHC Restriction": "HLA-A*02:01",
    "Qualitative Measure": "Negative",
    "PMID": "12682213",
    "Protein": "NS3",
    "Assay": "T cell IFN-gamma ELISpot",
}

_POS_ROW = {
    "Sequence": "NLVPMVATV",
    "MHC Restriction": "HLA-A*02:01",
    "Qualitative Measure": "Positive",
    "PMID": "12682213",
    "Protein": "NS5",
    "Assay": "T cell IFN-gamma ELISpot",
}


# ---------------------------------------------------------------------------
# load_lanl_export
# ---------------------------------------------------------------------------


def test_load_tsv(tmp_path: Path) -> None:
    path = _make_tsv(tmp_path, [_NEG_ROW])
    df = load_lanl_export(path, LOGGER)
    assert len(df) == 1
    assert "Sequence" in df.columns


def test_load_csv(tmp_path: Path) -> None:
    path = _make_csv(tmp_path, [_NEG_ROW])
    df = load_lanl_export(path, LOGGER)
    assert len(df) == 1
    assert "Sequence" in df.columns


def test_load_missing_file_exits(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        load_lanl_export(tmp_path / "nonexistent.tsv", LOGGER)


# ---------------------------------------------------------------------------
# resolve_columns
# ---------------------------------------------------------------------------


def test_resolve_columns_auto_detects(tmp_path: Path) -> None:
    df = pd.DataFrame([_NEG_ROW])
    mapping = resolve_columns(df, {}, LOGGER)
    assert mapping["epitope"] == "Sequence"
    assert mapping["hla"] == "MHC Restriction"
    assert mapping["result"] == "Qualitative Measure"
    assert mapping["pmid"] == "PMID"


def test_resolve_columns_override(tmp_path: Path) -> None:
    df = pd.DataFrame([{"Pep": "GILGFVFTL", "Allele": "HLA-A*02:01", "Outcome": "Negative"}])
    overrides = {"epitope": "Pep", "hla": "Allele", "result": "Outcome"}
    mapping = resolve_columns(df, overrides, LOGGER)
    assert mapping["epitope"] == "Pep"
    assert mapping["hla"] == "Allele"
    assert mapping["result"] == "Outcome"


def test_resolve_columns_missing_required_exits() -> None:
    df = pd.DataFrame([{"SomeCol": "value"}])
    with pytest.raises(SystemExit):
        resolve_columns(df, {}, LOGGER)


def test_resolve_columns_bad_override_falls_back(tmp_path: Path) -> None:
    df = pd.DataFrame([_NEG_ROW])
    # Override to a nonexistent column; should fall back to auto-detection.
    overrides = {"epitope": "NonExistentCol"}
    mapping = resolve_columns(df, overrides, LOGGER)
    # Falls back to "Sequence" via auto-detection.
    assert mapping["epitope"] == "Sequence"


# ---------------------------------------------------------------------------
# filter_rows
# ---------------------------------------------------------------------------


def _mapping_from(df: pd.DataFrame) -> dict:
    return resolve_columns(df, {}, LOGGER)


def test_filter_drops_positive_rows() -> None:
    df = pd.DataFrame([_NEG_ROW, _POS_ROW])
    mapping = _mapping_from(df)
    out, stats = filter_rows(df, mapping, LOGGER)
    assert stats["after_negative_filter"] == 1
    assert len(out) == 1


def test_filter_negative_case_insensitive() -> None:
    row = dict(_NEG_ROW)
    row["Qualitative Measure"] = "NEGATIVE"
    df = pd.DataFrame([row])
    mapping = _mapping_from(df)
    out, _ = filter_rows(df, mapping, LOGGER)
    assert len(out) == 1


def test_filter_drops_invalid_peptides() -> None:
    valid = dict(_NEG_ROW, **{"Sequence": "GILGFVFTL"})
    short = dict(_NEG_ROW, **{"Sequence": "GFT"})
    nonstandard = dict(_NEG_ROW, **{"Sequence": "XILGFVFTL"})
    df = pd.DataFrame([valid, short, nonstandard])
    mapping = _mapping_from(df)
    out, stats = filter_rows(df, mapping, LOGGER)
    assert len(out) == 1
    assert stats["after_peptide_filter"] == 1


def test_filter_drops_non_class_i_hla() -> None:
    classii = dict(_NEG_ROW, **{"MHC Restriction": "HLA-DRB1*01:01"})
    valid = dict(_NEG_ROW)
    df = pd.DataFrame([classii, valid])
    mapping = _mapping_from(df)
    out, stats = filter_rows(df, mapping, LOGGER)
    assert len(out) == 1
    assert stats["after_hla_filter"] == 1


def test_filter_dedup_removes_duplicate_rows() -> None:
    df = pd.DataFrame([_NEG_ROW, _NEG_ROW, _NEG_ROW])
    mapping = _mapping_from(df)
    out, stats = filter_rows(df, mapping, LOGGER)
    assert len(out) == 1
    assert stats["intra_export_duplicates_removed"] == 2


def test_filter_stats_raw_input() -> None:
    df = pd.DataFrame([_NEG_ROW, _POS_ROW])
    mapping = _mapping_from(df)
    _, stats = filter_rows(df, mapping, LOGGER)
    assert stats["raw_input"] == 2


# ---------------------------------------------------------------------------
# build_output
# ---------------------------------------------------------------------------


def _filtered_df() -> tuple[pd.DataFrame, dict]:
    df = pd.DataFrame([_NEG_ROW])
    mapping = resolve_columns(df, {}, LOGGER)
    filtered, _ = filter_rows(df, mapping, LOGGER)
    return filtered, mapping


def test_build_output_label_zero() -> None:
    filtered, mapping = _filtered_df()
    out = build_output(filtered, mapping, LOGGER)
    assert (out["label"] == 0).all()


def test_build_output_negative_origin() -> None:
    filtered, mapping = _filtered_df()
    out = build_output(filtered, mapping, LOGGER)
    assert (out["negative_origin"] == "tested_negative").all()


def test_build_output_virus_hcv() -> None:
    filtered, mapping = _filtered_df()
    out = build_output(filtered, mapping, LOGGER)
    assert (out["virus"] == "HCV").all()


def test_build_output_virus_family() -> None:
    filtered, mapping = _filtered_df()
    out = build_output(filtered, mapping, LOGGER)
    assert (out["virus_family"] == "Flaviviridae").all()


def test_build_output_taxon_id() -> None:
    filtered, mapping = _filtered_df()
    out = build_output(filtered, mapping, LOGGER)
    assert (out["virus_taxon_id"] == 11103).all()


def test_build_output_assay_context_natural_infection() -> None:
    filtered, mapping = _filtered_df()
    out = build_output(filtered, mapping, LOGGER)
    assert (out["assay_context"] == "natural_infection").all()


def test_build_output_database_source() -> None:
    filtered, mapping = _filtered_df()
    out = build_output(filtered, mapping, LOGGER)
    assert (out["database_source"] == "LANL-HCV").all()


def test_build_output_hla_normalized() -> None:
    filtered, mapping = _filtered_df()
    out = build_output(filtered, mapping, LOGGER)
    assert (out["hla_allele"] == "HLA-A*02:01").all()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _write_standard_tsv(path: Path) -> None:
    pd.DataFrame([_NEG_ROW, _POS_ROW]).to_csv(path, index=False, sep="\t")


def test_main_inspect(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    tsv = tmp_path / "export.tsv"
    _write_standard_tsv(tsv)
    rc = main(["--input", str(tsv), "--inspect"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Sequence" in out
    assert "MHC Restriction" in out


def test_main_dry_run(tmp_path: Path) -> None:
    tsv = tmp_path / "export.tsv"
    out_path = tmp_path / "out.csv"
    _write_standard_tsv(tsv)
    rc = main(["--input", str(tsv), "--output", str(out_path), "--dry-run"])
    assert rc == 0
    assert not out_path.exists()


def test_main_full_write(tmp_path: Path) -> None:
    tsv = tmp_path / "export.tsv"
    out_path = tmp_path / "out.csv"
    _write_standard_tsv(tsv)
    rc = main(["--input", str(tsv), "--output", str(out_path)])
    assert rc == 0
    assert out_path.exists()

    df = pd.read_csv(out_path)
    # Only the negative row should be in output.
    assert len(df) == 1
    assert df.iloc[0]["label"] == 0
    assert df.iloc[0]["virus"] == "HCV"
    assert df.iloc[0]["assay_context"] == "natural_infection"


def test_main_provenance_sidecar(tmp_path: Path) -> None:
    tsv = tmp_path / "export.tsv"
    out_path = tmp_path / "out.csv"
    _write_standard_tsv(tsv)
    rc = main(["--input", str(tsv), "--output", str(out_path)])
    assert rc == 0

    sidecar = out_path.with_name(out_path.stem + "_provenance.json")
    assert sidecar.exists()
    prov = json.loads(sidecar.read_text())
    assert prov["source"] == "LANL HCV Immunology Database (hcv.lanl.gov)"
    assert "column_mapping" in prov
    assert "filter_stats" in prov
    assert "input_sha256" in prov


def test_main_missing_input_returns_1(tmp_path: Path) -> None:
    rc = main(
        [
            "--input",
            str(tmp_path / "missing.tsv"),
            "--output",
            str(tmp_path / "out.csv"),
        ]
    )
    assert rc == 1


def test_main_column_override(tmp_path: Path) -> None:
    # Use non-standard column names; provide overrides via CLI.
    rows = [{"Pep": "GILGFVFTL", "Allele": "HLA-A*02:01", "Outcome": "Negative"}]
    tsv = tmp_path / "export.tsv"
    pd.DataFrame(rows).to_csv(tsv, index=False, sep="\t")
    out_path = tmp_path / "out.csv"
    rc = main(
        [
            "--input",
            str(tsv),
            "--output",
            str(out_path),
            "--col-epitope",
            "Pep",
            "--col-hla",
            "Allele",
            "--col-assay-result",
            "Outcome",
        ]
    )
    assert rc == 0
    df = pd.read_csv(out_path)
    assert len(df) == 1
    assert df.iloc[0]["peptide"] == "GILGFVFTL"
