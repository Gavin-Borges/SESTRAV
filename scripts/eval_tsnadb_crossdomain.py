"""Score the tumor neoantigen cross-domain benchmark.

Builds a balanced mixed pool:
  Positives (label=1): data/tsnadb_crossdomain_cohort.csv  (TSNAdb neoantigens)
  Negatives (label=0): data/hard_decoys.csv                 (self-proteome binders)

Computes MHCflurry presentation scores for the TSNAdb peptides (hard decoys are
already present in models/peptide_binding_matrix_v5.csv), writes a cohort-local
merged binding matrix, then scores the pool with the mode-31 RF at MODEL_PATH.
Reports AUC-PR / AUC-ROC / ISSR@10/25 with 2,000-resample bootstrap CIs.

WARNING - what this benchmark does and does not measure
-------------------------------------------------------
Read before quoting any number this script prints. See docs/data_registry.md
AD-4 for the full disclosure; the short form:

1. The negative arm can be training data. Every peptide in data/hard_decoys.csv
   is also a label=0 row of data/immunogenicity_dataset_v4.csv, where no
   is_quarantined column existed. Any v4-era model therefore memorized 100% of
   the negatives scored here, and the resulting AUCs measure that memorization,
   not cross-domain transfer. In v5 those rows are is_quarantined=True and are
   dropped from the pooled training path, so only a v5 model gives a held-out
   read. The figures recorded in results/tsnadb_crossdomain_benchmark.json
   (generated 2026-06-21, AUC-ROC 0.9887 / AUC-PR 0.9909) are v4-era and are
   RETRACTED on exactly this ground.

2. MODEL_PATH is untracked and gitignored, so this script cannot pin what it
   scored by the model's path alone - the file there today is byte-identical
   to the v5 model, not the v4-era artifact that produced the retracted JSON
   above, which no longer exists and was never checksummed. FIXED 2026-08-12:
   every run now records the scoring model's own sha256 (`model_sha256`) in
   both the output JSON and its `.provenance.json` sidecar via
   `src.artifact_integrity.model_provenance_fields`/`write_provenance_sidecar`,
   so a future model overwrite cannot silently invalidate a past figure the
   way it did here.

3. The inflation is NOT a binding-signal artifact. Both arms are pre-filtered to
   strong binders (positives MHCf_rank <= 2%, decoys presentation_score >= 0.5),
   so the binding channel is near chance on this pool and cannot explain a high
   AUC. Do not reach for that explanation.

BINDING_MATRIX_CACHE was v4 (the cache matching the now-retracted 0.9887/0.9909
artifact) until 2026-08-12, when it was re-pointed at
models/peptide_binding_matrix_v5.csv to regenerate honestly against the v5
model - this is the production binding matrix config.yaml points at, so it is
what the model actually sees in normal v5 scoring. The constant was renamed
from BINDING_MATRIX_V4 at the same time so its name cannot go stale again.

--output has no default: results/tsnadb_crossdomain_benchmark.json is a
git-tracked artifact, so a bare invocation runs the full benchmark and prints
the metrics table without writing anything rather than silently rewriting it.

Reproduce: python scripts/eval_tsnadb_crossdomain.py --output results/tsnadb_crossdomain_benchmark.json
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from mhcflurry import Class1PresentationPredictor
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.artifact_integrity import (
    load_verified_joblib,
    model_provenance_fields,
    write_provenance_sidecar,
)
from src.evaluate_metrics import evaluate, issr_at_k
from src.train_classifier import prepare_features_31

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TSNADB_COHORT = os.path.join(PROJECT_ROOT, "data", "tsnadb_crossdomain_cohort.csv")
HARD_DECOYS = os.path.join(PROJECT_ROOT, "data", "hard_decoys.csv")
BINDING_MATRIX_CACHE = os.path.join(PROJECT_ROOT, "models", "peptide_binding_matrix_v5.csv")
BINDING_MATRIX_OUT = os.path.join(PROJECT_ROOT, "data", "tsnadb_crossdomain_binding.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "rf_31feature_integrated.joblib")
TRACKED_OUTPUT = os.path.join(PROJECT_ROOT, "results", "tsnadb_crossdomain_benchmark.json")

# Canonical-10 alleles - must match the binding matrix column order
ALLELES = [
    "HLA-A*01:01",
    "HLA-A*02:01",
    "HLA-A*03:01",
    "HLA-A*11:01",
    "HLA-A*24:02",
    "HLA-B*07:02",
    "HLA-B*08:01",
    "HLA-B*27:05",
    "HLA-B*35:01",
    "HLA-B*44:02",
]


def predict_binding(peptides: list) -> pd.DataFrame:
    """Compute MHCflurry presentation scores for each peptide × canonical-10 alleles."""
    print(f"  Running MHCflurry for {len(peptides):,} peptides × {len(ALLELES)} alleles...")
    predictor = Class1PresentationPredictor.load()
    per_allele: dict = {}
    for allele in ALLELES:
        print(f"    {allele}...")
        pred = predictor.predict(peptides=peptides, alleles=[allele], verbose=0)
        col = "bind_" + allele.replace("HLA-", "").replace("*", "").replace(":", "")
        per_allele[col] = pred.set_index("peptide")["presentation_score"].to_dict()
    pivoted = pd.DataFrame({"peptide": peptides})
    for col, scores in per_allele.items():
        pivoted[col] = pivoted["peptide"].map(scores)
    return pivoted


def bootstrap_ci(
    y_true: np.ndarray, y_scores: np.ndarray, metric_fn, n_resamples: int = 2000
) -> tuple:
    """95% bootstrap confidence interval for a scalar metric function."""
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    n = len(y_true)
    rng = np.random.default_rng(42)
    vals = []
    for _ in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        yt, ys = y_true[idx], y_scores[idx]
        if len(np.unique(yt)) < 2:
            continue
        vals.append(metric_fn(yt, ys))
    if not vals:
        return float("nan"), float("nan")
    vals.sort()
    lo = vals[int(len(vals) * 0.025)]
    hi = vals[int(len(vals) * 0.975)]
    return lo, hi


def build_pool(tsna_path: str, decoys_path: str) -> pd.DataFrame:
    """Return a deduplicated peptide pool with unambiguous labels.

    If a peptide sequence appears in both arms (extremely unlikely but guarded
    against), it is excluded from the TSNAdb arm to preserve label integrity.
    """
    tsna = pd.read_csv(tsna_path)
    decoys = pd.read_csv(decoys_path)

    tsna_peps = tsna.drop_duplicates(subset=["peptide"])[["peptide"]].assign(label=1)
    decoy_peps = decoys.drop_duplicates(subset=["peptide"])[["peptide"]].assign(label=0)

    overlap = set(tsna_peps["peptide"]) & set(decoy_peps["peptide"])
    if overlap:
        print(f"  Warning: {len(overlap)} peptides appear in both arms - excluded from positives.")
        tsna_peps = tsna_peps[~tsna_peps["peptide"].isin(overlap)]

    pool = pd.concat([tsna_peps, decoy_peps], ignore_index=True).reset_index(drop=True)
    return pool


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    print("SESTRAV - Tumor Neoantigen Cross-Domain Benchmark")
    print("=" * 60)

    # 1. Build balanced pool
    print("\n[1] Building mixed pool...")
    pool = build_pool(TSNADB_COHORT, HARD_DECOYS)
    n_pos = int(pool["label"].sum())
    n_neg = int((pool["label"] == 0).sum())
    print(f"    Pool: {len(pool):,} unique peptides  ({n_pos:,} pos, {n_neg:,} neg)")

    # 2. Build merged binding matrix
    print("\n[2] Building binding matrix...")
    bm_cache = pd.read_csv(BINDING_MATRIX_CACHE)
    bm_peps = set(bm_cache["peptide"])
    missing = [p for p in pool["peptide"] if p not in bm_peps]
    print(
        f"    {len(missing):,} peptides need MHCflurry  "
        f"({len(pool) - len(missing):,} already cached)"
    )

    if missing:
        new_rows = predict_binding(missing)
        bm_merged = pd.concat([bm_cache, new_rows], ignore_index=True)
    else:
        bm_merged = bm_cache.copy()

    # Sanity gate: every pool peptide must be covered - guards the silent zero-vector trap
    covered = set(bm_merged["peptide"])
    uncovered = [p for p in pool["peptide"] if p not in covered]
    if uncovered:
        raise RuntimeError(
            f"{len(uncovered)} pool peptides not covered by merged binding matrix. "
            "MHCflurry computation may have failed."
        )
    bm_merged.to_csv(BINDING_MATRIX_OUT, index=False)
    print(f"    Merged binding matrix: {BINDING_MATRIX_OUT} ({len(bm_merged):,} rows)")

    # 3. Featurize (mode-31: 20 physico + 10 per-allele binding + peptide_length)
    print("\n[3] Computing mode-31 features...")
    X = prepare_features_31(pool, BINDING_MATRIX_OUT)
    print(f"    Feature matrix: {X.shape}")

    # 4. Score
    print(f"\n[4] Scoring with {os.path.basename(MODEL_PATH)}...")
    model = load_verified_joblib(MODEL_PATH, required_checksum=True)
    y_scores = model.predict_proba(X)[:, 1]
    y_true = pool["label"].values

    # 5. Metrics + bootstrap CIs
    print("\n[5] Computing metrics (2,000-resample bootstrap)...")
    metrics = evaluate(y_true, y_scores)
    ci_roc = bootstrap_ci(y_true, y_scores, roc_auc_score)
    ci_pr = bootstrap_ci(y_true, y_scores, average_precision_score)
    ci_issr10 = bootstrap_ci(y_true, y_scores, lambda yt, ys: issr_at_k(yt, ys, 10))
    ci_issr25 = bootstrap_ci(y_true, y_scores, lambda yt, ys: issr_at_k(yt, ys, 25))

    print(f"\n{'Metric':<12} {'Value':>8}   95% CI")
    print("-" * 38)
    print(f"{'AUC-ROC':<12} {metrics['auc_roc']:>8.4f}   [{ci_roc[0]:.4f}, {ci_roc[1]:.4f}]")
    print(f"{'AUC-PR':<12} {metrics['auc_pr']:>8.4f}   [{ci_pr[0]:.4f}, {ci_pr[1]:.4f}]")
    print(f"{'ISSR@10':<12} {metrics['issr_10']:>8.4f}   [{ci_issr10[0]:.4f}, {ci_issr10[1]:.4f}]")
    print(f"{'ISSR@25':<12} {metrics['issr_25']:>8.4f}   [{ci_issr25[0]:.4f}, {ci_issr25[1]:.4f}]")
    print(f"{'Recall@10':<12} {metrics['recall_10']:>8.4f}")
    print(f"{'Recall@25':<12} {metrics['recall_25']:>8.4f}")

    # 6. Write result JSON
    pool["score"] = y_scores
    pool["rank_pct"] = pool["score"].rank(ascending=False, pct=True).mul(100)

    result = {
        "benchmark": "tsnadb_crossdomain",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        **model_provenance_fields(MODEL_PATH),
        "feature_mode": 31,
        "n_pool": int(len(pool)),
        "n_positive": int(n_pos),
        "n_negative": int(n_neg),
        "prevalence": float(y_true.mean()),
        "metrics": {
            "auc_roc": {
                "value": float(metrics["auc_roc"]),
                "ci_95": [float(ci_roc[0]), float(ci_roc[1])],
            },
            "auc_pr": {
                "value": float(metrics["auc_pr"]),
                "ci_95": [float(ci_pr[0]), float(ci_pr[1])],
            },
            "issr_10": {
                "value": float(metrics["issr_10"]),
                "ci_95": [float(ci_issr10[0]), float(ci_issr10[1])],
            },
            "issr_25": {
                "value": float(metrics["issr_25"]),
                "ci_95": [float(ci_issr25[0]), float(ci_issr25[1])],
            },
            "recall_10": float(metrics["recall_10"]),
            "recall_25": float(metrics["recall_25"]),
        },
        "design": {
            "positives": "TSNAdb v2 SNV-derived (Deep_imm>=0.5, MHCf_rank<=2%, canonical-10 alleles, seed=42 sample)",
            "negatives": "hard_decoys.csv (human self-proteome high-affinity binders, label=0)",
            "rationale": (
                "Neoantigen immunogenicity is mechanistically tolerance escape from self. "
                "Scoring TSNAdb positives against self-proteome decoys tests whether SESTRAV, "
                "trained on viral epitopes, transfers the discrimination signal across domains."
            ),
            "caveat": (
                "TSNAdb neoantigens are computationally predicted (NetMHCpan/MHCflurry-family tools). "
                "SESTRAV binding features partially overlap this prediction signal; the result "
                "measures presentation + immunogenicity transfer above self-background, "
                "not independently validated immunogenicity."
            ),
        },
        "per_peptide": (
            pool[["peptide", "label", "score", "rank_pct"]]
            .round({"score": 6, "rank_pct": 2})
            .to_dict(orient="records")
        ),
    }

    maybe_write_json(result, args.output)
    if args.output:
        sidecar = write_provenance_sidecar(
            args.output,
            script="scripts/eval_tsnadb_crossdomain.py",
            extra=model_provenance_fields(MODEL_PATH),
        )
        print(f"Provenance written to {sidecar}")


def maybe_write_json(result: dict, output_path: str | None) -> None:
    """Write result to output_path, or do nothing if output_path is falsy.

    Kept as its own function (rather than inlined at the end of main()) so
    the write-or-skip decision is testable without running the full
    MHCflurry/model-scoring pipeline that produces `result`.

    Writes with newline="" so the LF json.dump produces is not rewritten to
    CRLF on Windows - otherwise the sha256 a provenance sidecar records for
    this file would be a Windows-only hash (see .gitattributes eol=lf pin).
    """
    if not output_path:
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")
    print(f"\nResults written to {output_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score the tumor neoantigen cross-domain benchmark."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            f"Output JSON path (optional). No default: {TRACKED_OUTPUT} is a "
            "git-tracked artifact, so this script refuses to guess a destination "
            "- omit this flag to run the benchmark and print metrics without "
            "writing anything."
        ),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
