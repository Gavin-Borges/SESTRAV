"""Compute the LOO binding-confound decomposition (manuscript Table 3b).

Regenerates results/loo_binding_confound_decomposition.csv from two canonical,
already-committed source CSVs so every cell is reproducible and integrity-bound:

  Regime 1 within-virus, all negatives (real + decoy):
      results/per_virus_eval_v5_mode31.csv  column auc_roc
  Regime 2 within-virus, real negatives only:
      results/per_virus_eval_v5_mode31.csv  column auc_roc_real_neg_only
  Regime 3 cross-virus leave-one-out (real negatives, target held out of train):
      results/loo_cross_virus_v5_clean.csv  column auc_roc

Decoy inflation = R1 - R2 ; transfer gap = R2 - R3, both computed at full float
precision and rounded to 3 decimals (compute-then-round). The three zero-decoy
viruses (HBV/HCV/HPV) carry exactly 0.000 inflation, a built-in mechanism control.

--output has no default: results/loo_binding_confound_decomposition.csv is a
git-tracked artifact, so a bare invocation prints the table without writing
anything rather than silently rewriting it.

Reproduce:  python scripts/compute_loo_binding_confound.py --output results/loo_binding_confound_decomposition.csv
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

CANON = ["CMV", "DENV", "EBV", "HBV", "HCV", "HIV-1", "HPV", "IAV", "SARS-CoV-2"]
PV_SRC = "results/per_virus_eval_v5_mode31.csv"
LOO_SRC = "results/loo_cross_virus_v5_clean.csv"
TRACKED_OUTPUT = "results/loo_binding_confound_decomposition.csv"


def compute_decomposition() -> pd.DataFrame:
    pv = pd.read_csv(PV_SRC).set_index("virus")
    loo = pd.read_csv(LOO_SRC).set_index("test_virus")

    rows = []
    for v in CANON:
        r1 = float(pv.loc[v, "auc_roc"])
        r2 = float(pv.loc[v, "auc_roc_real_neg_only"])
        r3 = float(loo.loc[v, "auc_roc"])
        decoy_frac = float(pv.loc[v, "n_neg_decoy"]) / (
            float(pv.loc[v, "n_neg_real"]) + float(pv.loc[v, "n_neg_decoy"])
        )
        rows.append(
            {
                "virus": v,
                "within_all_neg": round(r1, 3),       # Regime 1
                "within_real_neg": round(r2, 3),      # Regime 2
                "loo_cross_virus": round(r3, 3),      # Regime 3
                "decoy_inflation": round(r1 - r2, 3),  # R1 - R2
                "transfer_gap": round(r2 - r3, 3),     # R2 - R3
                "decoy_frac_neg": round(decoy_frac, 3),
            }
        )

    tab = pd.DataFrame(rows)
    mean_row = {
        "virus": "Mean",
        "within_all_neg": round(tab["within_all_neg"].mean(), 3),
        "within_real_neg": round(tab["within_real_neg"].mean(), 3),
        "loo_cross_virus": round(tab["loo_cross_virus"].mean(), 3),
        "decoy_inflation": round(tab["decoy_inflation"].mean(), 3),
        "transfer_gap": round(tab["transfer_gap"].mean(), 3),
        "decoy_frac_neg": "",
    }
    return pd.concat([tab, pd.DataFrame([mean_row])], ignore_index=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the LOO binding-confound decomposition (manuscript Table 3b)."
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
    out = compute_decomposition()
    print(out.to_string(index=False))
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        out.to_csv(args.output, index=False)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
