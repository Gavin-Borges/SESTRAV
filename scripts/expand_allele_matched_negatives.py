"""scripts/expand_allele_matched_negatives.py
=========================================
Generates an expanded allele-matched negative pool for the external Influenza A
validation cohort (SCI-2 / T2-1).

Context:
  In the frozen Influenza A validation cohort (n=179: 102 pos / 77 neg),
  96.1% of pairs in the pooled AUC are between-allele. The only 308 same-allele
  pairs were restricted by an upstream ELISPOT-only filter that eliminated
  negatives for HLA-A*11:01 (38 pos, 0 neg) and other strata.

This script:
  1. Loads the frozen positives from `data/external/influenza_clean.csv`.
  2. Queries IEDB for all Class I negative T-cell assay records for Influenza A
     across all assay types (ELISPOT, ICS, multimer, cytotoxicity, IFNg release).
  3. Applies length (8-11mer) and standard AA validation.
  4. Normalizes alleles to standard 4-digit notation.
  5. Filters against `data/immunogenicity_dataset_v5.csv` using bidirectional
     Aho-Corasick trie matching to ensure zero training contamination.
  6. Assembles `data/external/influenza_expanded_clean.csv` with paired positive
     and negative peptides across all cohort alleles.
  7. Emits a provenance sidecar with SHA-256 and exact pair counts.

Success Criterion:
  Same-allele positive/negative pair count >= 1,000.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.parse
import urllib.request
from typing import Any

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.filter_validation_cohorts import (
    MAX_LEN,
    MIN_LEN,
    STANDARD_AA,
    filter_bidirectional_overlap,
    load_training_peptides,
    normalise_allele,
)

FROZEN_COHORT_PATH = os.path.join(PROJECT_ROOT, "data", "external", "influenza_clean.csv")
EXPANDED_COHORT_PATH = os.path.join(PROJECT_ROOT, "data", "external", "influenza_expanded_clean.csv")
PROVENANCE_PATH = os.path.join(PROJECT_ROOT, "data", "external", "influenza_expanded_clean_provenance.json")


def fetch_iedb_negatives(limit: int = 5000) -> list[dict[str, Any]]:
    """Fetch Class I negative assay records from IEDB for Influenza A."""
    print("Querying IEDB for Influenza A Class I negative assay records...")
    base_url = "https://query-api.iedb.org/tcell_search"
    params = {
        "mhc_class": "eq.I",
        "qualitative_measure": "eq.Negative",
        "source_organism_name": "ilike.*Influenza A*",
        "select": "linear_sequence,qualitative_measure,mhc_allele_name,assay_names",
        "limit": limit,
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:  # nosec B310 - trusted HTTPS endpoint
            records = json.loads(resp.read().decode("utf-8"))
        print(f"  Successfully retrieved {len(records)} negative records from IEDB.")
        return records
    except Exception as e:
        print(f"  Error querying IEDB: {e}", file=sys.stderr)
        return []


def curate_negative_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Filter and normalize candidate negative records."""
    rows = []
    for r in records:
        pep = str(r.get("linear_sequence", "")).strip().upper()
        if not (MIN_LEN <= len(pep) <= MAX_LEN) or not all(aa in STANDARD_AA for aa in pep):
            continue

        raw_allele = r.get("mhc_allele_name")
        allele = normalise_allele(raw_allele)
        if not allele:
            continue

        assay_names = str(r.get("assay_names", "")).strip()

        rows.append(
            {
                "peptide": pep,
                "label": 0,
                "allele": allele,
                "virus": "InfluenzaA",
                "assay_names": assay_names,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["peptide", "label", "allele", "virus", "assay_names"])

    df = pd.DataFrame(rows)
    # Deduplicate on peptide and allele, aggregating assay names
    agg = (
        df.groupby(["peptide", "allele", "virus"])
        .agg({"label": "first", "assay_names": lambda s: "; ".join(sorted(set(str(x) for x in s if str(x).strip())))})
        .reset_index()
    )
    return agg[["peptide", "label", "allele", "virus", "assay_names"]]


def calculate_same_allele_pairs(df: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    """Calculate positive, negative, and same-allele pair counts per stratum."""
    ct = pd.crosstab(df["allele"], df["label"])
    if 0 not in ct.columns:
        ct[0] = 0
    if 1 not in ct.columns:
        ct[1] = 0
    ct = ct.rename(columns={0: "neg", 1: "pos"})
    ct["pairs"] = ct["pos"] * ct["neg"]
    total_pairs = int(ct["pairs"].sum())
    return total_pairs, ct.sort_values("pairs", ascending=False)


def expand_cohort(
    output_path: str = EXPANDED_COHORT_PATH,
    provenance_path: str = PROVENANCE_PATH,
) -> pd.DataFrame:
    """Execute the full allele-matched negative expansion workflow."""
    print("=" * 60)
    print("SESTRAV Allele-Matched Negative Expansion (SCI-2 / T2-1)")
    print("=" * 60)

    # 1. Load baseline frozen cohort
    if not os.path.exists(FROZEN_COHORT_PATH):
        raise FileNotFoundError(f"Frozen cohort missing at {FROZEN_COHORT_PATH}")

    baseline_df = pd.read_csv(FROZEN_COHORT_PATH)
    positives = baseline_df[baseline_df["label"] == 1].copy()
    print(f"Loaded baseline cohort: {len(baseline_df)} rows ({len(positives)} pos, {len(baseline_df) - len(positives)} neg)")

    baseline_pairs, baseline_ct = calculate_same_allele_pairs(baseline_df)
    print(f"Baseline same-allele pairs: {baseline_pairs}")

    # 2. Fetch IEDB negative records
    raw_negatives = fetch_iedb_negatives()
    if not raw_negatives:
        raise RuntimeError("Failed to retrieve any negative records from IEDB")

    curated_neg = curate_negative_records(raw_negatives)
    print(f"Curated valid negative candidates: {len(curated_neg)}")

    # 3. Load training peptides and apply bidirectional Aho-Corasick filter
    train_peps = load_training_peptides()
    print(f"Filtering {len(curated_neg)} candidate negatives against {len(train_peps):,} training peptides...")

    clean_neg = filter_bidirectional_overlap(train_peps, curated_neg, name="IEDB_Expanded_IAV_Negatives")
    print(f"Zero-overlap clean negative survivors: {len(clean_neg)}")

    # Ensure clean negatives do not contain any positive peptides from our cohort
    cohort_pos_peps = set(positives["peptide"].str.strip().str.upper())
    clean_neg = clean_neg[~clean_neg["peptide"].isin(cohort_pos_peps)].copy()
    print(f"Clean negatives after cohort positive exclusion: {len(clean_neg)}")

    # 4. Filter clean negatives to only those matching alleles present in the cohort positives
    cohort_pos_alleles = set(positives["allele"].unique())
    matched_neg = clean_neg[clean_neg["allele"].isin(cohort_pos_alleles)].copy()
    print(f"Clean negatives matched to cohort positive alleles: {len(matched_neg)}")

    # 5. Assemble expanded cohort: frozen positives + matched negatives + baseline clean negatives
    combined = pd.concat([positives, matched_neg, baseline_df[baseline_df["label"] == 0]], ignore_index=True)
    # Deduplicate on peptide
    combined = combined.drop_duplicates(subset=["peptide"]).reset_index(drop=True)

    expanded_pairs, expanded_ct = calculate_same_allele_pairs(combined)
    print("\nExpanded Cohort Allele Pair Breakdown:")
    print(expanded_ct[expanded_ct["pairs"] > 0])
    print(f"\nTotal Expanded Same-Allele Pairs: {expanded_pairs}")

    if expanded_pairs < 1000:
        raise ValueError(
            f"Expanded cohort achieved {expanded_pairs} pairs, failing criterion (>= 1,000 pairs)"
        )

    # 6. Save expanded cohort CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"\n[SUCCESS] Saved expanded cohort ({len(combined)} rows) -> {output_path}")

    # 7. Write provenance sidecar
    with open(output_path, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()

    prov = {
        "artifact": os.path.basename(output_path),
        "sha256": sha256,
        "total_rows": len(combined),
        "positives": int((combined["label"] == 1).sum()),
        "negatives": int((combined["label"] == 0).sum()),
        "same_allele_pairs": expanded_pairs,
        "baseline_same_allele_pairs": baseline_pairs,
        "success_criterion_met": expanded_pairs >= 1000,
        "strata_counts": expanded_ct.to_dict(orient="index"),
    }
    with open(provenance_path, "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2)
    print(f"Saved provenance sidecar -> {provenance_path}")

    return combined


def main() -> int:
    parser = argparse.ArgumentParser(description="Expand allele-matched negatives for cohort.")
    parser.add_argument("--output", default=EXPANDED_COHORT_PATH, help="Output CSV path.")
    parser.add_argument("--provenance", default=PROVENANCE_PATH, help="Output provenance JSON path.")
    args = parser.parse_args()

    try:
        expand_cohort(output_path=args.output, provenance_path=args.provenance)
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
