"""
Dataset refresh + bias/skew audit utilities for SESTRAV finalization.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.artifact_guard import guard_planned_paths
from src.iedb_data_loader import (
    _detect_format,
    _infer_protein_gene,
    _infer_strain,
    _label_from_filename,
    _load_epitope_table,
    _virus_from_filename,
    is_valid_peptide,
    load_and_clean_iedb,
    load_iedb_file,
    map_label,
)


def _collect_raw_records(data_dir: str, include_hpv11: bool = False) -> pd.DataFrame:
    """Collect pre-dedup records with provenance from IEDB source files."""
    records: List[Dict] = []
    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith((".xlsx", ".csv")):
            continue
        virus = _virus_from_filename(filename)
        if virus is None:
            continue
        if virus == "HPV11" and not include_hpv11:
            continue
        filepath = os.path.join(data_dir, filename)
        fmt, has_subheader = _detect_format(filepath)
        if fmt == "epitope_table":
            label = _label_from_filename(filename)
            if label is None:
                continue
            epitope_records = _load_epitope_table(filepath, has_subheader)
            for rec in epitope_records:
                seq = rec["peptide"]
                if seq is None or not is_valid_peptide(seq):
                    continue
                records.append(
                    {
                        "peptide": seq,
                        "label": int(label),
                        "virus": virus,
                        "protein": _infer_protein_gene(rec.get("antigen_name"), virus),
                        "strain": _infer_strain(rec.get("organism_name"), virus),
                        "source_file": filename,
                        "source_format": "epitope_table",
                        "label_source": "filename",
                    }
                )
        else:
            df = load_iedb_file(filepath)
            peptide_col = None
            label_col = None
            for col in df.columns:
                cl = str(col).lower().strip()
                if peptide_col is None and (
                    cl == "description" or ("epitope" in cl and "linear" in cl)
                ):
                    peptide_col = col
                if label_col is None and "qualitative" in cl:
                    label_col = col
            if peptide_col is None or label_col is None:
                continue
            for _, row in df.iterrows():
                peptide = (
                    str(row[peptide_col]).strip().upper() if pd.notna(row[peptide_col]) else None
                )
                row_label = map_label(row[label_col])
                if peptide is None or row_label is None:
                    continue
                if not is_valid_peptide(peptide):
                    continue
                records.append(
                    {
                        "peptide": peptide,
                        "label": int(row_label),
                        "virus": virus,
                        "source_file": filename,
                        "source_format": "tcell_assay",
                        "label_source": "qualitative_measure",
                    }
                )
    return pd.DataFrame(records)


def planned_data_bias_audit_paths(provenance_csv: str, audit_csv: str, audit_md: str) -> list[str]:
    """Every path this module's write side risks clobbering, across both
    refresh_dataset and write_audit_reports.

    Deliberately excludes output_csv (data/immunogenicity_dataset_v4.csv):
    rewriting it is refresh_dataset's declared purpose, it is gitignored (not
    published), and write_audit_reports reads it back intra-run moments later.
    Including it would abort every run unconditionally, since the real dataset
    already exists on disk (Hazard A in the step-8 enumeration note).

    This combined list exists only for the __main__-level defense-in-depth
    guard, which runs before either delegate has written anything. The two
    delegates below each guard a narrower subset of it - their own writes
    only - so a union guard cannot make write_audit_reports abort because
    refresh_dataset already wrote provenance_csv earlier in the same run
    (Hazard C).
    """
    return [
        provenance_csv,
        audit_csv,
        audit_csv.replace(".csv", "_virus_label_counts.csv"),
        audit_md,
    ]


def _guard_refresh_dataset(provenance_csv: str, allow_overwrite: bool) -> None:
    """Refuse to clobber refresh_dataset's own tracked-risk write.

    Only provenance_csv is guarded here; output_csv is exempt by design (see
    planned_data_bias_audit_paths).
    """
    guard_planned_paths(
        os.path.dirname(provenance_csv) or ".",
        [provenance_csv],
        allow_overwrite,
        flag="--provenance-csv",
        api_hint="refresh_dataset(..., allow_overwrite=True)",
        scope="among this run's planned artifacts",
        remedy="Point --provenance-csv at a fresh path, ",
    )


def _guard_write_audit_reports(audit_csv: str, audit_md: str, allow_overwrite: bool) -> None:
    """Refuse to clobber write_audit_reports' own writes, including the
    derived _virus_label_counts.csv name."""
    guard_planned_paths(
        os.path.dirname(audit_md) or ".",
        [audit_csv, audit_csv.replace(".csv", "_virus_label_counts.csv"), audit_md],
        allow_overwrite,
        flag="--audit-csv/--audit-md",
        api_hint="write_audit_reports(..., allow_overwrite=True)",
        detail=(
            ": data_bias_audit.md is the one git-tracked artifact this module writes"
        ),
        scope="among this run's planned artifacts",
        remedy="Point --audit-csv and --audit-md at fresh paths, ",
    )


def _guard_data_bias_audit_cli(
    provenance_csv: str, audit_csv: str, audit_md: str, allow_overwrite: bool
) -> None:
    """__main__-only defense-in-depth: checks all 4 tracked-risk paths before
    refresh_dataset's IEDB xlsx parsing starts, rather than only discovering a
    collision after that work (and after refresh_dataset has already written
    provenance_csv) via the two narrower guards above. Safe to union here
    because __main__ calls this before either delegate has written anything.
    """
    guard_planned_paths(
        os.path.dirname(audit_md) or ".",
        planned_data_bias_audit_paths(provenance_csv, audit_csv, audit_md),
        allow_overwrite,
        flag="--provenance-csv/--audit-csv/--audit-md",
        api_hint=(
            "refresh_dataset(..., allow_overwrite=True) / "
            "write_audit_reports(..., allow_overwrite=True)"
        ),
        detail=(
            ": data_bias_audit.md is the one git-tracked artifact this module writes"
        ),
        scope="among this run's planned artifacts",
        remedy="Point --provenance-csv, --audit-csv and --audit-md at fresh paths, ",
    )


def refresh_dataset(
    source_data_dir: str,
    output_csv: str,
    provenance_csv: str,
    include_hpv11: bool = False,
    allow_overwrite: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Regenerate curated immunogenicity dataset and provenance table.

    output_csv is not guarded: rewriting it is this function's declared
    purpose (see planned_data_bias_audit_paths for why). provenance_csv is.
    """
    _guard_refresh_dataset(provenance_csv, allow_overwrite)
    refreshed = load_and_clean_iedb(source_data_dir, include_hpv11=include_hpv11)
    raw_records = _collect_raw_records(source_data_dir, include_hpv11=include_hpv11)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    refreshed.to_csv(output_csv, index=False)

    if raw_records.empty:
        provenance = pd.DataFrame(
            columns=["source_file", "source_format", "label_source", "virus", "label", "n_records"]
        )
    else:
        provenance = (
            raw_records.groupby(["source_file", "source_format", "label_source", "virus", "label"])
            .size()
            .reset_index(name="n_records")
            .sort_values(["virus", "source_file", "label"])
        )
    provenance.to_csv(provenance_csv, index=False)
    return refreshed, provenance


def audit_dataset(df: pd.DataFrame, raw_records: pd.DataFrame) -> Dict:
    """Generate audit metrics and risk flags for class/subgroup skew."""
    summary: Dict = {}
    summary["n_total"] = int(len(df))
    summary["n_positive"] = int((df["label"] == 1).sum())
    summary["n_negative"] = int((df["label"] == 0).sum())
    summary["positive_rate"] = float(np.mean(df["label"] == 1)) if len(df) else np.nan
    summary["n_unique_peptides"] = int(df["peptide"].nunique()) if "peptide" in df.columns else 0
    summary["missing_virus"] = (
        int(df["virus"].isna().sum()) if "virus" in df.columns else int(len(df))
    )
    summary["missing_strain"] = (
        int(df["strain"].isna().sum()) if "strain" in df.columns else int(len(df))
    )
    summary["missing_allele"] = (
        int(df["allele"].isna().sum()) if "allele" in df.columns else int(len(df))
    )
    summary["missing_protein"] = (
        int(df["protein"].isna().sum()) if "protein" in df.columns else int(len(df))
    )
    peptide_lengths = (
        df["peptide"].astype(str).str.len() if "peptide" in df.columns else pd.Series(dtype=int)
    )
    summary["peptide_len_min"] = int(peptide_lengths.min()) if not peptide_lengths.empty else np.nan
    summary["peptide_len_max"] = int(peptide_lengths.max()) if not peptide_lengths.empty else np.nan
    summary["peptide_len_mean"] = (
        float(peptide_lengths.mean()) if not peptide_lengths.empty else np.nan
    )

    if raw_records.empty:
        summary["raw_n_records"] = 0
        summary["duplicate_conflict_peptides"] = 0
    else:
        label_nunique = raw_records.groupby("peptide")["label"].nunique()
        summary["raw_n_records"] = int(len(raw_records))
        summary["duplicate_conflict_peptides"] = int((label_nunique > 1).sum())

    return summary


def write_audit_reports(
    dataset_csv: str,
    raw_records: pd.DataFrame,
    output_csv: str,
    output_md: str,
    allow_overwrite: bool = False,
) -> Tuple[pd.DataFrame, Dict]:
    """Write structured and markdown bias audit reports."""
    _guard_write_audit_reports(output_csv, output_md, allow_overwrite)
    df = pd.read_csv(dataset_csv)
    summary = audit_dataset(df, raw_records)
    summary_df = pd.DataFrame([summary])

    virus_breakdown = (
        df.groupby(["virus", "label"]).size().reset_index(name="n")
        if {"virus", "label"}.issubset(df.columns)
        else pd.DataFrame(columns=["virus", "label", "n"])
    )
    summary_df.to_csv(output_csv, index=False)
    virus_breakdown_path = output_csv.replace(".csv", "_virus_label_counts.csv")
    virus_breakdown.to_csv(virus_breakdown_path, index=False)

    md = f"""# SESTRAV Data Bias/Skew Audit

## Dataset summary
- Total records: `{summary["n_total"]}`
- Positives: `{summary["n_positive"]}`
- Negatives: `{summary["n_negative"]}`
- Positive rate: `{summary["positive_rate"]:.4f}`
- Unique peptides: `{summary["n_unique_peptides"]}`

## Metadata quality
- Missing virus: `{summary["missing_virus"]}`
- Missing strain: `{summary["missing_strain"]}`
- Missing allele: `{summary["missing_allele"]}`
- Missing protein: `{summary["missing_protein"]}`
- Peptide length range: `{summary["peptide_len_min"]}` to `{summary["peptide_len_max"]}` (mean `{summary["peptide_len_mean"]:.2f}`)

## Label conflict risk
- Raw source records: `{summary["raw_n_records"]}`
- Peptides with conflicting raw labels: `{summary["duplicate_conflict_peptides"]}`

## Known pipeline risk points to monitor
- Labels inferred from Epitope Table filenames for those source files.
- Duplicate peptide conflict handling still uses majority-vote collapse.
- 30-feature mode still maps missing binding rows to all-zero vectors.

## Output files
- Audit summary CSV: `{output_csv}`
- Virus/label breakdown CSV: `{virus_breakdown_path}`
"""
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(md)
    return summary_df, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refresh and audit SESTRAV training dataset")
    parser.add_argument(
        "--source-data-dir", required=True, help="Directory with raw IEDB source xlsx/csv files"
    )
    parser.add_argument("--output-csv", default="data/immunogenicity_dataset_v4.csv")
    parser.add_argument(
        "--provenance-csv",
        required=True,
        help="Path for the provenance table CSV that refresh_dataset writes. No "
        "default: it refuses to guess a destination.",
    )
    parser.add_argument(
        "--audit-csv",
        required=True,
        help="Path for the bias-audit summary CSV (plus its derived "
        "_virus_label_counts.csv) that write_audit_reports writes. No default: "
        "it refuses to guess a destination.",
    )
    parser.add_argument(
        "--audit-md",
        required=True,
        help="Path for the bias-audit markdown report that write_audit_reports "
        "writes. No default: this is the one git-tracked artifact this module "
        "writes, so it refuses to guess a destination.",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Replace provenance/audit artifacts that already exist at "
        "--provenance-csv / --audit-csv / --audit-md. Does not affect "
        "--output-csv, which refresh_dataset always rewrites by design.",
    )
    args = parser.parse_args()

    _guard_data_bias_audit_cli(
        args.provenance_csv, args.audit_csv, args.audit_md, args.allow_overwrite
    )

    os.makedirs(os.path.dirname(args.provenance_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.audit_csv) or ".", exist_ok=True)
    refreshed_df, _ = refresh_dataset(
        source_data_dir=args.source_data_dir,
        output_csv=args.output_csv,
        provenance_csv=args.provenance_csv,
        include_hpv11=False,
        allow_overwrite=args.allow_overwrite,
    )
    raw_df = _collect_raw_records(args.source_data_dir, include_hpv11=False)
    write_audit_reports(
        dataset_csv=args.output_csv,
        raw_records=raw_df,
        output_csv=args.audit_csv,
        output_md=args.audit_md,
        allow_overwrite=args.allow_overwrite,
    )
    print(f"Refreshed dataset rows: {len(refreshed_df)}")
