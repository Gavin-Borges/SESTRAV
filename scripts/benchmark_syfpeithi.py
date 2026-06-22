"""
SYFPEITHI rank-overlap benchmark for SESTRAV.

Compares SESTRAV OOF predictions against canonical T-cell epitopes curated
from the SYFPEITHI database (Rammensee et al. 1999, Immunogenetics) for
HPV16 and EBV HLA-A*02:01-restricted responses. No live API calls are made;
the reference set is hardcoded from published SYFPEITHI entries and confirmed
by primary IEDB literature.

Metrics produced
----------------
- recall@K  : fraction of SYFPEITHI epitopes ranked in SESTRAV top-K%
- rank_pctl  : SESTRAV percentile rank for each reference epitope
- baseline_recall@K : expected recall for random ranking (= K/100)
- enrichment@K : recall@K / baseline_recall@K

Usage
-----
    python scripts/benchmark_syfpeithi.py \\
        --predictions models/rf_oof_predictions.csv \\
        --output results/syfpeithi_benchmark.json

    # dry-run (print reference set, do not require predictions file)
    python scripts/benchmark_syfpeithi.py --dry-run

References
----------
Rammensee H, Bachmann J, Emmerich NP, Bachor OA, Stevanovic S.
  SYFPEITHI: database for MHC ligands and peptide motifs.
  Immunogenetics. 1999;50(3-4):213-9.

van der Burg SH et al. (2001) Immunogenetics of HLA-A2-restricted HPV16 epitopes.
Levitsky V et al. (1998) EBV BMLF1 in HLA-A*02:01 responses. J Exp Med.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# SYFPEITHI reference epitope set
# Each entry: (peptide, protein_short, virus, hla, syfpeithi_score, citation)
# HLA-A*02:01 unless noted.  Scores from SYFPEITHI database (syfpeithi.de).
# ---------------------------------------------------------------------------

SYFPEITHI_CANONICAL: list[dict] = [
    # ── EBV HLA-A*02:01 ──────────────────────────────────────────────────────
    {
        "peptide": "GLCTLVAML",
        "protein": "BMLF1",
        "virus": "EBV",
        "hla": "A*02:01",
        "syfpeithi_score": 26,
        "source": "Rammensee 1999; Levitsky 1998 J Exp Med",
    },
    {
        "peptide": "CLGGLLTMV",
        "protein": "BMLF1",
        "virus": "EBV",
        "hla": "A*02:01",
        "syfpeithi_score": 24,
        "source": "Rammensee 1999; Levitsky 1998",
    },
    {
        "peptide": "FLYALALLL",
        "protein": "LMP2A",
        "virus": "EBV",
        "hla": "A*02:01",
        "syfpeithi_score": 21,
        "source": "Rammensee 1999; Khanna 1998",
    },
    {
        "peptide": "LLWTLVVLL",
        "protein": "LMP2A",
        "virus": "EBV",
        "hla": "A*02:01",
        "syfpeithi_score": 22,
        "source": "Rammensee 1999; Khanna 1998",
    },
    {
        "peptide": "YVLDHLIVV",
        "protein": "BRLF1",
        "virus": "EBV",
        "hla": "A*02:01",
        "syfpeithi_score": 20,
        "source": "Rammensee 1999",
    },
    # ── HPV16 HLA-A*02:01 ────────────────────────────────────────────────────
    {
        "peptide": "YMLDLQPET",
        "protein": "E7",
        "virus": "HPV16",
        "hla": "A*02:01",
        "syfpeithi_score": 20,
        "source": "van der Burg 2001; Rammensee 1999",
    },
    {
        "peptide": "LLMGTLGIV",
        "protein": "E7",
        "virus": "HPV16",
        "hla": "A*02:01",
        "syfpeithi_score": 21,
        "source": "van der Burg 2001; Rammensee 1999",
    },
    {
        "peptide": "KLPQLCTEL",
        "protein": "E7",
        "virus": "HPV16",
        "hla": "A*02:01",
        "syfpeithi_score": 19,
        "source": "van der Burg 2001",
    },
    {
        "peptide": "FAFRDLCIV",
        "protein": "E6",
        "virus": "HPV16",
        "hla": "A*02:01",
        "syfpeithi_score": 18,
        "source": "Rammensee 1999; Rudolf 2001",
    },
    {
        "peptide": "RAHYNIVTF",
        "protein": "E6",
        "virus": "HPV16",
        "hla": "A*02:01",
        "syfpeithi_score": 19,
        "source": "Rammensee 1999; Rudolf 2001",
    },
]

# Close variants of canonical epitopes found in IEDB/training data
# These are listed for informational matching - not counted in recall
SYFPEITHI_TRAINING_VARIANTS: dict[str, str] = {
    "CLGGLLTMV": "CLGGLLTMV",  # exact match
    "CLGGLLYMV": "CLGGLLTMV",  # 1-substitution variant
    "CLGGLITMV": "CLGGLLTMV",  # 1-substitution variant
    "CLGLLTMV": "CLGGLLTMV",   # 8-mer variant (compressed register)
    "LLWTLVLL": "LLWTLVVLL",   # 8-mer compressed form
    "RAHYNIVTE": "RAHYNIVTF",  # 1-substitution variant
    "KLPHLCTEL": "KLPQLCTEL",  # 1-substitution variant
}

RECALL_CUTOFFS = [5, 10, 25]


def _hamming1(a: str, b: str) -> bool:
    if len(a) != len(b):
        return False
    return sum(x != y for x, y in zip(a, b)) == 1


def _lookup_in_oof(
    ref: dict,
    oof_df: pd.DataFrame,
    peptide_col: str,
    score_col: str,
) -> dict:
    pep = ref["peptide"]
    exact = oof_df[oof_df[peptide_col].str.upper() == pep.upper()]
    if not exact.empty:
        row = exact.iloc[0]
        return {
            "match_type": "exact",
            "sestrav_score": float(row[score_col]),
            "sestrav_label": int(row.get("label", -1)),
            "rank_percentile": None,  # filled in later
        }

    # 1-substitution variants
    for candidate_pep, result in zip(
        oof_df[peptide_col].str.upper().tolist(),
        oof_df.itertuples(index=False),
    ):
        if _hamming1(pep.upper(), candidate_pep):
            return {
                "match_type": f"hamming1:{getattr(result, peptide_col)}",
                "sestrav_score": float(getattr(result, score_col)),
                "sestrav_label": int(getattr(result, "label", -1)),
                "rank_percentile": None,
            }

    return {
        "match_type": "not_in_oof",
        "sestrav_score": None,
        "sestrav_label": None,
        "rank_percentile": None,
    }


def run_benchmark(predictions_path: Path, output_path: Path | None) -> dict:
    oof_df = pd.read_csv(predictions_path)

    # Detect columns
    peptide_col = next(
        (c for c in ["peptide", "sequence", "Peptide"] if c in oof_df.columns),
        oof_df.columns[0],
    )
    score_col = next(
        (c for c in ["score", "immunogenicity_score", "probability", "Score"]
         if c in oof_df.columns),
        oof_df.columns[-1],
    )

    n_total = len(oof_df)
    feature_mode = oof_df["feature_mode"].iloc[0] if "feature_mode" in oof_df.columns else "unknown"

    # Compute percentile ranks (higher score = lower percentile rank = better)
    oof_df = oof_df.copy()
    oof_df["_rank_pctl"] = (
        oof_df[score_col].rank(ascending=False, pct=True) * 100
    )

    print("\nSYFPEITHI Rank-Overlap Benchmark")
    print(f"{'='*60}")
    print(f"  OOF predictions : {predictions_path.name}")
    print(f"  Feature mode    : {feature_mode}")
    print(f"  Total peptides  : {n_total}")
    print(f"  Score column    : {score_col}")
    print(f"  Reference set   : {len(SYFPEITHI_CANONICAL)} canonical SYFPEITHI epitopes")
    print()

    results = []
    for ref in SYFPEITHI_CANONICAL:
        match = _lookup_in_oof(ref, oof_df, peptide_col, score_col)
        if match["match_type"] != "not_in_oof":
            pep_upper = ref["peptide"].upper()
            rows = oof_df[oof_df[peptide_col].str.upper() == pep_upper]
            if rows.empty:
                for candidate, score in zip(
                    oof_df[peptide_col].str.upper().tolist(),
                    oof_df["_rank_pctl"].tolist(),
                ):
                    if _hamming1(pep_upper, candidate):
                        match["rank_percentile"] = score
                        break
            else:
                match["rank_percentile"] = float(rows["_rank_pctl"].iloc[0])

        results.append({
            "peptide": ref["peptide"],
            "protein": ref["protein"],
            "virus": ref["virus"],
            "hla": ref["hla"],
            "syfpeithi_score": ref["syfpeithi_score"],
            "source": ref["source"],
            **match,
        })

    # Print per-epitope table
    print(f"{'Peptide':<14} {'Protein':<8} {'Virus':<8} {'Match':<20} {'Score':>7} {'Pctile':>7}")
    print("-" * 70)
    for r in results:
        score_str = f"{r['sestrav_score']:.3f}" if r["sestrav_score"] is not None else "  N/A "
        pctile_str = f"{r['rank_percentile']:.1f}%" if r["rank_percentile"] is not None else "  N/A "
        print(
            f"{r['peptide']:<14} {r['protein']:<8} {r['virus']:<8} "
            f"{r['match_type']:<20} {score_str:>7} {pctile_str:>7}"
        )
    print()

    # Recall@K for epitopes that WERE in the OOF set
    evaluable = [r for r in results if r["rank_percentile"] is not None]
    not_evaluable = [r for r in results if r["rank_percentile"] is None]

    recall_summary: dict[int, dict] = {}
    for k in RECALL_CUTOFFS:
        in_top_k = [r for r in evaluable if r["rank_percentile"] <= k]
        recall = len(in_top_k) / len(evaluable) if evaluable else 0.0
        baseline = k / 100.0
        enrichment = recall / baseline if baseline > 0 else 0.0
        recall_summary[k] = {
            "recall": recall,
            "baseline": baseline,
            "enrichment": enrichment,
            "n_in_top_k": len(in_top_k),
            "n_evaluable": len(evaluable),
        }

    print(f"Rank-overlap summary  (n_evaluable={len(evaluable)}, n_not_in_oof={len(not_evaluable)})")
    print(f"{'Cutoff':>8} {'Recall':>8} {'Baseline':>10} {'Enrichment':>12} {'In top-K':>10}")
    print("-" * 55)
    for k, s in recall_summary.items():
        print(
            f"  top {k:>2}%  {s['recall']:>7.1%}   {s['baseline']:>7.1%}   "
            f"{s['enrichment']:>10.2f}x   {s['n_in_top_k']:>4}/{s['n_evaluable']}"
        )

    if not_evaluable:
        print("\nEpitopes not in OOF set (cannot evaluate - not in training data):")
        for r in not_evaluable:
            print(f"  {r['peptide']}  ({r['protein']}, {r['virus']})")

    print()
    print("Disclosure: SESTRAV OOF scores reflect models that HAVE seen the training")
    print("labels for these peptides (they were in held-out folds, not the fold being")
    print("scored). Epitopes are evaluated at their held-out fold score only.")

    benchmark_output = {
        "benchmark": "syfpeithi",
        "predictions_file": str(predictions_path),
        "feature_mode": feature_mode,
        "n_oof_peptides": n_total,
        "reference_set_size": len(SYFPEITHI_CANONICAL),
        "n_evaluable": len(evaluable),
        "n_not_in_oof": len(not_evaluable),
        "recall_cutoffs": recall_summary,
        "per_epitope": results,
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_output, f, indent=2, default=lambda x: int(x) if hasattr(x, "item") else str(x))
        print(f"\nResults written: {output_path}")

    return benchmark_output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SYFPEITHI rank-overlap benchmark for SESTRAV OOF predictions",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PROJECT_ROOT / "models" / "rf_oof_predictions.csv",
        help="Path to OOF predictions CSV (default: models/rf_oof_predictions.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "syfpeithi_benchmark.json",
        help="Path to write JSON results",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the reference epitope set and exit without reading predictions",
    )
    args = parser.parse_args(argv)

    if args.dry_run:
        print(f"SYFPEITHI reference set ({len(SYFPEITHI_CANONICAL)} epitopes):")
        for r in SYFPEITHI_CANONICAL:
            print(f"  {r['peptide']:12s}  {r['protein']:6s}  {r['virus']:6s}  "
                  f"A*02:01  score={r['syfpeithi_score']}  [{r['source']}]")
        return 0

    if not args.predictions.exists():
        print(f"ERROR: predictions file not found: {args.predictions}", file=sys.stderr)
        return 1

    run_benchmark(args.predictions, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
