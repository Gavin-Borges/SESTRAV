"""Paired-bootstrap CI + p-value on the Tier-A AUC-PR delta (manuscript Sec 3.5).

Regenerates results/tier_a_paired_bootstrap.csv: a seeded 10,000-resample paired
bootstrap of the AUC-PR difference between SESTRAV RF (mode-31, out-of-fold) and
each comparator, on the peptide subset scored by both (n = 704; 490 pos / 214 neg).
Two comparisons are reported:

  rf_vs_bigmhc        SESTRAV RF - BigMHC          (near-tie; CI includes 0)
  rf_vs_binding_only  SESTRAV RF - MHCflurry base  (significant; CI excludes 0)

Metric is average precision (AUC-PR), matching results/table3_tier_a_metrics.csv.
The bootstrap is seeded (seed=20260719) so the CI/p-value are reproducible. RF's AP
on this paired subset reproduces table3's full-set value exactly (0.8278) since RF
has no missing scores. BigMHC's AP on this subset does NOT reproduce its table3
full-set value: pairing to RF's out-of-fold scores drops 16 label=1 peptides where
rf_oof_score is NaN in results/external_validation_merged_scores.csv, shifting
BigMHC's own AP from table3's 0.8218 (n=720, full set) down to 0.8100 (n=704, the
paired subset used here). This is a real subset-size effect, not rounding noise;
the delta/p-value below are computed correctly on the matched n=704 pairs.

Inputs (committed):
  results/external_validation_merged_scores.csv  (peptide,label,rf_oof_score,binding_max)
  data/tier_a_external_benchmarks.csv            (peptide,bigmhc_score)

--output has no default: results/tier_a_paired_bootstrap.csv is a git-tracked
artifact, so a bare invocation prints the table without writing anything rather
than silently rewriting it.

Reproduce:  python scripts/compute_tier_a_paired_bootstrap.py --output results/tier_a_paired_bootstrap.csv
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

SEED = 20260719
B = 10000
TRACKED_OUTPUT = "results/tier_a_paired_bootstrap.csv"


def paired_bootstrap(sub: pd.DataFrame, col_a: str, col_b: str, rng: np.random.Generator) -> dict:
    p = sub.dropna(subset=[col_a, col_b]).reset_index(drop=True)
    y = p["label"].to_numpy()
    sa, sb = p[col_a].to_numpy(), p[col_b].to_numpy()
    n = len(p)
    ap_a = average_precision_score(y, sa)
    ap_b = average_precision_score(y, sb)
    bd = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n)
        yi = y[idx]
        if yi.sum() in (0, len(yi)):
            idx = rng.integers(0, n, n)
            yi = y[idx]
        bd[i] = average_precision_score(yi, sa[idx]) - average_precision_score(yi, sb[idx])
    lo, hi = np.percentile(bd, [2.5, 97.5])
    p_two = float(min(1.0, 2.0 * min((bd <= 0).mean(), (bd >= 0).mean())))
    return {
        "n_paired": n,
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "ap_sestrav": round(ap_a, 4),
        "ap_comparator": round(ap_b, 4),
        "delta_auc_pr": round(ap_a - ap_b, 4),
        "ci95_low": round(lo, 4),
        "ci95_high": round(hi, 4),
        "p_two_sided": round(p_two, 4),
        "ci_includes_zero": bool(lo <= 0 <= hi),
    }


def compute_bootstrap_table() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rf = pd.read_csv("results/external_validation_merged_scores.csv")[
        ["peptide", "label", "rf_oof_score", "binding_max"]
    ]
    bm = pd.read_csv("data/tier_a_external_benchmarks.csv")[["peptide", "bigmhc_score"]]
    df = rf.merge(bm, on="peptide", how="inner")

    rows = [
        {"comparison": "rf_vs_bigmhc", "comparator": "BigMHC",
         **paired_bootstrap(df, "rf_oof_score", "bigmhc_score", rng)},
        {"comparison": "rf_vs_binding_only", "comparator": "MHCflurry binding-only",
         **paired_bootstrap(df, "rf_oof_score", "binding_max", rng)},
    ]
    out = pd.DataFrame(rows)
    out.insert(1, "n_bootstrap", B)
    out.insert(2, "seed", SEED)
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired-bootstrap CI + p-value on the Tier-A AUC-PR delta (manuscript Sec 3.5)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            f"Output CSV path (optional). No default: {TRACKED_OUTPUT} is a "
            "git-tracked artifact, so this script refuses to guess a destination "
            "- omit this flag to print the table without writing anything."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out = compute_bootstrap_table()
    print(out.to_string(index=False))
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        out.to_csv(args.output, index=False)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
