"""scripts/filter_validation_cohorts.py
====================================
Retrieves viral Class I assay datasets from the IEDB Query API, applies standard
pipeline cleaning (length/AA filtering, majority voting, allele normalization),
preserves assay type metadata (`assay_names`) for ELISPOT vs multimer stratification,
and filters out exact and substring training overlaps using bidirectional Aho-Corasick tries.

Outputs:
  - data/external/sars2_clean.csv
  - data/external/influenza_clean.csv
  - Additional viral cohorts when run in multi-organism survey mode (--survey)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from typing import Any

import ahocorasick
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
MIN_LEN = 8
MAX_LEN = 11

TRAINING_DATA_PATHS = [
    os.path.join(PROJECT_ROOT, "data", "immunogenicity_dataset_v5.csv"),
]

SARS2_OUTPUT = os.path.join(PROJECT_ROOT, "data", "external", "sars2_clean.csv")
INFLUENZA_OUTPUT = os.path.join(PROJECT_ROOT, "data", "external", "influenza_clean.csv")

# 10 viral pathogens for multi-organism contamination survey (SCI-4)
ORGANISM_CATALOG: list[tuple[str, str, str]] = [
    ("sars2", "SARS-CoV-2", "Severe acute respiratory syndrome coronavirus 2"),
    ("influenza", "InfluenzaA", "Influenza A virus"),
    ("hiv1", "HIV-1", "Human immunodeficiency virus 1"),
    ("hcv", "HCV", "Hepatitis C virus"),
    ("hbv", "HBV", "Hepatitis B virus"),
    ("ebv", "EBV", "Epstein-Barr virus"),
    ("cmv", "CMV", "Cytomegalovirus"),
    ("hpv", "HPV", "Human papillomavirus"),
    ("denv", "DENV", "Dengue virus"),
    ("yfv", "YFV", "Yellow fever virus"),
]


def normalise_allele(raw: Any) -> str | None:
    """Normalize allele notation to HLA-A*02:01."""
    if pd.isna(raw) or not str(raw).strip():
        return None
    s = str(raw).strip()
    if not s.upper().startswith("HLA-"):
        s = "HLA-" + s
    if "*" not in s:
        for prefix in ("HLA-A", "HLA-B", "HLA-C"):
            if s.upper().startswith(prefix):
                s = s[: len(prefix)] + "*" + s[len(prefix) :]
                break
    if "*" in s and ":" not in s:
        star_pos = s.index("*")
        suffix = s[star_pos + 1 :]
        if len(suffix) >= 4:
            s = s[: star_pos + 1] + suffix[:2] + ":" + suffix[2:]
    return s


def is_mhc_class_i(allele: str | None) -> bool:
    """Filter out Class II MHCs."""
    if not allele:
        return False
    return not allele.startswith(("HLA-DR", "HLA-DP", "HLA-DQ"))


def fetch_cohort_data(
    organism_name: str,
    assay_filter: str | None = "elispot",
    select_fields: str = "linear_sequence,qualitative_measure,mhc_allele_name,assay_names",
    limit: int = 10000,
) -> list[dict[str, Any]]:
    """Query IEDB IQ-API for class I assays for a given organism.

    Args:
        organism_name: Source organism pattern for IEDB query.
        assay_filter: Substring filter for assay_names (e.g. 'elispot'). If None,
            all Class I assays are retrieved regardless of assay type.
        select_fields: Selected fields in IEDB response; widened to include assay_names.
        limit: Maximum record limit.

    Returns:
        List of raw JSON record dictionaries from IEDB.
    """
    print(f"Fetching IEDB data for organism: {organism_name}")
    base_url = "https://query-api.iedb.org/tcell_search"
    params: dict[str, str | int] = {
        "source_organism_name": f"ilike.%{organism_name}%",
        "mhc_class": "eq.I",
        "select": select_fields,
        "limit": limit,
    }
    if assay_filter:
        params["assay_names"] = f"ilike.%{assay_filter}%"

    query_str = urllib.parse.urlencode(params)
    url = f"{base_url}?{query_str}"

    try:
        # Reject non-HTTPS URLs as defense-in-depth before opening.
        if not url.lower().startswith("https://"):
            raise ValueError(f"Refusing non-HTTPS URL: {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=45) as response:  # nosec B310 - trusted HTTPS endpoint
            data = json.loads(response.read().decode("utf-8"))
        print(f"  Successfully retrieved {len(data)} raw records.")
        return data
    except Exception as e:
        print(f"  Error fetching data: {e}", file=sys.stderr)
        return []


def clean_and_curate(records: list[dict[str, Any]], virus_name: str) -> pd.DataFrame:
    """Apply SESTRAV-standard validation, mapping, and majority vote deduplication."""
    cleaned_rows = []

    for r in records:
        pep = r.get("linear_sequence")
        val_measure = r.get("qualitative_measure")
        allele_raw = r.get("mhc_allele_name")
        assay_names = r.get("assay_names", "")

        if not pep or not val_measure:
            continue

        pep = str(pep).strip().upper()
        # 1. Length & standard AA filter
        if not (MIN_LEN <= len(pep) <= MAX_LEN) or not all(aa in STANDARD_AA for aa in pep):
            continue

        # 2. Binary label mapping
        val_lower = str(val_measure).strip().lower()
        if val_lower.startswith("positive"):
            label = 1
        elif val_lower == "negative":
            label = 0
        else:
            continue

        # 3. Allele normalization & Class I check
        allele = normalise_allele(allele_raw)
        if not allele or not is_mhc_class_i(allele):
            continue

        cleaned_rows.append(
            {
                "peptide": pep,
                "label": label,
                "allele": allele,
                "virus": virus_name,
                "assay_names": str(assay_names).strip() if assay_names else "",
            }
        )

    if not cleaned_rows:
        return pd.DataFrame()

    df = pd.DataFrame(cleaned_rows)

    # 4. Deduplicate by peptide (majority voting)
    agg_spec: dict[str, Any] = {
        "mean_label": ("label", "mean"),
        "allele": ("allele", "first"),
        "virus": ("virus", "first"),
    }
    if "assay_names" in df.columns:
        agg_spec["assay_names"] = (
            "assay_names",
            lambda s: "; ".join(sorted(set(str(x) for x in s if str(x).strip()))),
        )

    agg = df.groupby("peptide").agg(**agg_spec).reset_index()

    # Resolve exact ties (0.5) by dropping, and map remaining via majority vote
    agg = agg[agg["mean_label"] != 0.5].copy()
    agg["label"] = (agg["mean_label"] > 0.5).astype(int)

    cols = ["peptide", "label", "virus", "allele"]
    if "assay_names" in agg.columns:
        cols.append("assay_names")

    return agg[cols].reset_index(drop=True)


def _training_path_rel(path: str) -> str:
    """Repo-relative POSIX path for error messages (no absolute workstation path)."""
    try:
        rel = os.path.relpath(path, PROJECT_ROOT)
    except ValueError:
        rel = os.path.basename(path)
    return rel.replace("\\", "/")


def load_training_peptides() -> list[str]:
    """Load all unique peptides from required training datasets.

    Every path in TRAINING_DATA_PATHS must exist. A missing reference is a
    hard error.
    """
    missing = [p for p in TRAINING_DATA_PATHS if not os.path.exists(p)]
    if missing:
        listed = ", ".join(_training_path_rel(p) for p in missing)
        raise FileNotFoundError(f"Required training reference corpus missing: {listed}")

    peptides: set[str] = set()
    for path in TRAINING_DATA_PATHS:
        df = pd.read_csv(path)
        if "peptide" not in df.columns:
            raise ValueError(f"Training reference {_training_path_rel(path)} has no peptide column")
        peptides.update(df["peptide"].dropna().str.strip().str.upper())
        print(f"Loaded {len(df)} rows from training dataset: {os.path.basename(path)}")
    if not peptides:
        raise ValueError("Training reference corpus contained no peptides")
    return list(peptides)


def filter_bidirectional_overlap(
    train_peptides: set[str] | list[str], eval_df: pd.DataFrame, name: str = ""
) -> pd.DataFrame:
    """Remove any peptide from eval_df that has an exact or substring match with training.

    An empty train_peptides is a HARD ERROR, not an early return.
    """
    if not train_peptides:
        raise ValueError(
            f"Refusing to filter {name or 'cohort'} against an empty training peptide set: "
            "the result would be unfiltered but indistinguishable from a contamination-filtered cohort"
        )
    if eval_df.empty:
        return eval_df

    eval_peptides = eval_df["peptide"].unique()

    # Build Aho-Corasick automaton of evaluation peptides to find E in T
    a_eval = ahocorasick.Automaton()
    for pep in eval_peptides:
        a_eval.add_word(pep, pep)
    a_eval.make_automaton()

    eval_in_train: set[str] = set()
    for train_pep in train_peptides:
        for _end_idx, eval_pep in a_eval.iter(train_pep):
            eval_in_train.add(eval_pep)

    # Build Aho-Corasick automaton of training peptides to find T in E
    a_train = ahocorasick.Automaton()
    for train_pep in train_peptides:
        a_train.add_word(train_pep, train_pep)
    a_train.make_automaton()

    train_in_eval: set[str] = set()
    for eval_pep in eval_peptides:
        for _end_idx, train_pep in a_train.iter(eval_pep):
            train_in_eval.add(eval_pep)

    contaminated = eval_in_train.union(train_in_eval)

    print(f"Contamination analysis for {name}:")
    print(f"  Total validation peptides: {len(eval_df)}")
    print(f"  Eval-in-Train overlaps: {len(eval_in_train)}")
    print(f"  Train-in-Eval overlaps: {len(train_in_eval)}")
    print(f"  Total unique contaminated excluded: {len(contaminated)}")

    clean_df = eval_df[~eval_df["peptide"].isin(contaminated)].copy()
    print(f"  Clean validation peptides: {len(clean_df)}")
    return clean_df


def run_single_cohort(
    tag: str,
    virus_name: str,
    organism_query: str,
    train_peptides: list[str],
    output_dir: str,
    assay_filter: str | None = "elispot",
    save_raw: bool = True,
) -> dict[str, Any]:
    """Process a single viral cohort and return summary metrics."""
    raw_records = fetch_cohort_data(organism_query, assay_filter=assay_filter)
    raw_count = len(raw_records)

    if save_raw and raw_records:
        raw_path = os.path.join(output_dir, f"raw_{tag}.json")
        with open(raw_path, "w", encoding="utf-8") as fh:
            json.dump(raw_records, fh, indent=2)
        print(f"  Saved raw IEDB records with assay metadata -> {raw_path}")

    if not raw_records:
        return {
            "tag": tag,
            "virus": virus_name,
            "raw_records": 0,
            "unique_valid": 0,
            "contaminated": 0,
            "clean_survivors": 0,
            "contamination_pct": 0.0,
            "clean_positives": 0,
            "clean_negatives": 0,
            "status": "No IEDB records returned",
        }

    curated_df = clean_and_curate(raw_records, virus_name)
    n_curated = len(curated_df)
    if curated_df.empty:
        return {
            "tag": tag,
            "virus": virus_name,
            "raw_records": raw_count,
            "unique_valid": 0,
            "contaminated": 0,
            "clean_survivors": 0,
            "contamination_pct": 0.0,
            "clean_positives": 0,
            "clean_negatives": 0,
            "status": "No peptides survived validation/curation",
        }

    clean_df = filter_bidirectional_overlap(train_peptides, curated_df, virus_name)
    n_clean = len(clean_df)
    n_contam = n_curated - n_clean
    contam_pct = (n_contam / n_curated) * 100.0 if n_curated > 0 else 0.0

    n_pos = int((clean_df["label"] == 1).sum()) if n_clean > 0 else 0
    n_neg = int((clean_df["label"] == 0).sum()) if n_clean > 0 else 0

    if n_clean == 0:
        status = "100% Contaminated (0 clean survivors)"
    elif n_neg == 0:
        status = f"Unscoreable ({n_pos} pos / 0 neg; AUC undefined)"
    elif n_pos == 0:
        status = f"Unscoreable (0 pos / {n_neg} neg; AUC undefined)"
    else:
        status = f"Scoreable ({n_pos} pos / {n_neg} neg)"

    # Save clean cohort CSV
    clean_csv_path = os.path.join(output_dir, f"{tag}_clean.csv")
    clean_df.to_csv(clean_csv_path, index=False)
    print(f"  [SUCCESS] Wrote clean {virus_name} cohort ({n_clean} rows) -> {clean_csv_path}")

    return {
        "tag": tag,
        "virus": virus_name,
        "raw_records": raw_count,
        "unique_valid": n_curated,
        "contaminated": n_contam,
        "clean_survivors": n_clean,
        "contamination_pct": contam_pct,
        "clean_positives": n_pos,
        "clean_negatives": n_neg,
        "status": status,
    }


def format_survey_markdown(results: list[dict[str, Any]]) -> str:
    """Render survey results into a formatted GitHub Markdown table."""
    lines = [
        "# Viral Cohort Contamination Survey (IEDB Class I Assays vs SESTRAV v5 Training Set)",
        "",
        "| Organism | Raw Records | Unique Valid | Contaminated | Clean Survivors | Contamination Rate | Pos / Neg | Scoreability |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| **{r['virus']}** | {r['raw_records']:,} | {r['unique_valid']:,} | "
            f"{r['contaminated']:,} | **{r['clean_survivors']:,}** | "
            f"**{r['contamination_pct']:.1f}%** | {r['clean_positives']} / {r['clean_negatives']} | "
            f"{r['status']} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter and survey IEDB validation cohorts against training corpus."
    )
    parser.add_argument(
        "--survey",
        action="store_true",
        help="Run multi-organism survey across all 10 viral pathogens in catalog.",
    )
    parser.add_argument(
        "--assay",
        default="elispot",
        help="Assay filter substring (default: 'elispot'). Pass 'none' or '' for all assays.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(PROJECT_ROOT, "data", "external"),
        help="Directory to write clean cohorts and raw JSONs.",
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        default=True,
        help="Save raw JSON responses with assay_names metadata (default: True).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    assay_filter = None if args.assay.lower() in ("none", "", "all") else args.assay

    print("SESTRAV Zero-Overlap Validation Cohorts Filtering")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Assay filter: {assay_filter or 'All Class I assays'}")

    os.makedirs(args.output_dir, exist_ok=True)
    train_peptides = load_training_peptides()
    print(f"Total unique training peptides loaded: {len(train_peptides):,}")
    print("-" * 60)

    if args.survey:
        print("Running Multi-Organism Contamination Survey (10 Pathogens)...")
        results = []
        for tag, virus_name, organism_query in ORGANISM_CATALOG:
            print(f"\nProcessing {virus_name} ({organism_query})...")
            res = run_single_cohort(
                tag=tag,
                virus_name=virus_name,
                organism_query=organism_query,
                train_peptides=train_peptides,
                output_dir=args.output_dir,
                assay_filter=assay_filter,
                save_raw=args.save_raw,
            )
            results.append(res)

        table_md = format_survey_markdown(results)
        print("\n" + "=" * 60)
        print(table_md)

        # Write survey report to _local/notes/
        notes_dir = os.path.join(PROJECT_ROOT, "_local", "notes")
        os.makedirs(notes_dir, exist_ok=True)
        report_path = os.path.join(notes_dir, "organism_contamination_survey_2026-09-09.md")
        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write(table_md)
        print(f"\nSaved contamination survey report -> {report_path}")

    else:
        # Canonical 2-cohort pipeline (SARS-CoV-2 and Influenza A)
        print("1. Processing SARS-CoV-2...")
        run_single_cohort(
            tag="sars2",
            virus_name="SARS-CoV-2",
            organism_query="Severe acute respiratory syndrome coronavirus 2",
            train_peptides=train_peptides,
            output_dir=args.output_dir,
            assay_filter=assay_filter,
            save_raw=args.save_raw,
        )
        print("-" * 60)

        print("2. Processing Influenza A...")
        run_single_cohort(
            tag="influenza",
            virus_name="InfluenzaA",
            organism_query="Influenza A virus",
            train_peptides=train_peptides,
            output_dir=args.output_dir,
            assay_filter=assay_filter,
            save_raw=args.save_raw,
        )

    print("=" * 60)
    print("Cohort filtering pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
