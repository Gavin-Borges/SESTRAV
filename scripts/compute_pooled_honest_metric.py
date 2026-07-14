#!/usr/bin/env python3
"""Compute the honest pooled same-pathogen discrimination metric (Def A).

Background: the retired headline "same-pathogen AUC-ROC 0.9368" only reproduced when
easy negatives (the vaccinia hard-decoy panel and allele-matched non-binders) were
pooled in. The honest same-pathogen figure restricts negatives to real IEDB REST-API
viral negatives (negative_origin == 'iedb_api') for the 9 target viruses - this is
"Def A" in _local/notes/pooled_within_virus_metric_recompute.md and the number cited
in docs/model_evaluation_summary.md.

This script makes that number reproducible and binds it to a canonical results/ CSV so
the integrity harness can reconcile it (retiring the last unbound-orphan of its class).

Reproduce:  python scripts/compute_pooled_honest_metric.py
Output:     results/pooled_honest_same_pathogen.csv
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

TARGET_VIRUSES = {"CMV", "DENV", "EBV", "HBV", "HCV", "HIV-1", "HPV", "IAV", "SARS-CoV-2"}
# Def A real-negative origin: real IEDB REST-API viral negatives only. tested_negative is
# dominated by the vaccinia hard-decoy panel and allele_matched_nonbinder is synthetic;
# both are excluded so the metric reflects genuine same-pathogen discrimination.
REAL_NEG_ORIGIN = "iedb_api"


def compute(oof_path: Path) -> dict:
    df = pd.read_csv(oof_path)
    is_pos = df["label"] == 1
    is_real_neg = df["negative_origin"] == REAL_NEG_ORIGIN
    subset = df[df["virus"].isin(TARGET_VIRUSES) & (is_pos | is_real_neg)]
    y = subset["label"].to_numpy()
    s = subset["score"].to_numpy()
    p, r, _ = precision_recall_curve(y, s)
    return {
        "metric": "honest_same_pathogen",
        "definition": "Def A: positives + real IEDB (iedb_api) negatives, 9 target viruses",
        "auc_roc": roc_auc_score(y, s),
        "auc_pr": auc(r, p),
        "n_pos": int((subset["label"] == 1).sum()),
        "n_neg": int((subset["label"] == 0).sum()),
        "source_oof": str(oof_path).replace("\\", "/"),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oof", default="models/v5/rf_oof_predictions_mode31.csv")
    ap.add_argument("--out", default="results/pooled_honest_same_pathogen.csv")
    args = ap.parse_args()

    row = compute(Path(args.oof))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out, index=False)

    print(f"honest same-pathogen (Def A): AUC-ROC {row['auc_roc']:.4f} "
          f"(-> {row['auc_roc']:.3f}), AUC-PR {row['auc_pr']:.3f}, "
          f"n_pos={row['n_pos']} n_neg={row['n_neg']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
