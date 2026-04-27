"""
Dataset refresh + bias/skew audit utilities for SESTRAV finalization.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

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
                seq = rec['peptide']
                if seq is None or not is_valid_peptide(seq):
                    continue
                records.append(
                    {
                        "peptide": seq,
                        "label": int(label),
                        "virus": virus,
                        "protein": _infer_protein_gene(rec.get('antigen_name'), virus),
                        "strain": _infer_strain(rec.get('organism_name'), virus),
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
                if peptide_col is None and (cl == "description" or ("epitope" in cl and "linear" in cl)):
                    peptide_col = col
                if label_col is None and "qualitative" in cl:
                    label_col = col
            if peptide_col is None or label_col is None:
                continue
            for _, row in df.iterrows():
                peptide = str(row[peptide_col]).strip().upper() if pd.notna(row[peptide_col]) else None
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


def refresh_dataset(
    source_data_dir: str,
    output_csv: str,
    provenance_csv: str,
    include_hpv11: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Regenerate curated immunogenicity dataset and provenance table."""
    refreshed = load_and_clean_iedb(source_data_dir, include_hpv11=include_hpv11)
    raw_records = _collect_raw_records(source_data_dir, include_hpv11=include_hpv11)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    refreshed.to_csv(output_csv, index=False)

    if raw_records.empty:
        provenance = pd.DataFrame(columns=["source_file", "source_format", "label_source", "virus", "label", "n_records"])
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
    summary["missing_virus"] = int(df["virus"].isna().sum()) if "virus" in df.columns else int(len(df))
    summary["missing_strain"] = int(df["strain"].isna().sum()) if "strain" in df.columns else int(len(df))
    summary["missing_allele"] = int(df["allele"].isna().sum()) if "allele" in df.columns else int(len(df))
    summary["missing_protein"] = int(df["protein"].isna().sum()) if "protein" in df.columns else int(len(df))
    peptide_lengths = df["peptide"].astype(str).str.len() if "peptide" in df.columns else pd.Series(dtype=int)
    summary["peptide_len_min"] = int(peptide_lengths.min()) if not peptide_lengths.empty else np.nan
    summary["peptide_len_max"] = int(peptide_lengths.max()) if not peptide_lengths.empty else np.nan
    summary["peptide_len_mean"] = float(peptide_lengths.mean()) if not peptide_lengths.empty else np.nan

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
) -> Tuple[pd.DataFrame, Dict]:
    """Write structured and markdown bias audit reports."""
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
    parser.add_argument("--source-data-dir", required=True, help="Directory with raw IEDB source xlsx/csv files")
    parser.add_argument("--output-csv", default="immunogenicity_dataset.csv")
    parser.add_argument("--provenance-csv", default="results/immunogenicity_provenance.csv")
    parser.add_argument("--audit-csv", default="results/data_bias_audit_summary.csv")
    parser.add_argument("--audit-md", default="results/data_bias_audit.md")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.provenance_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.audit_csv) or ".", exist_ok=True)
    refreshed_df, _ = refresh_dataset(
        source_data_dir=args.source_data_dir,
        output_csv=args.output_csv,
        provenance_csv=args.provenance_csv,
        include_hpv11=False,
    )
    raw_df = _collect_raw_records(args.source_data_dir, include_hpv11=False)
    write_audit_reports(
        dataset_csv=args.output_csv,
        raw_records=raw_df,
        output_csv=args.audit_csv,
        output_md=args.audit_md,
    )
    print(f"Refreshed dataset rows: {len(refreshed_df)}")
