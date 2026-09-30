#!/usr/bin/env python3
"""Compare tracked mode-31 and binding-only mode-10 RF OOF predictions.

The comparison reports the full active scoring pool and both honest-negative
definitions used by tracked SESTRAV code. Confidence intervals use a paired row
bootstrap, so both model scores are resampled with the same indices.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.artifact_integrity import sha256_file, write_provenance_sidecar  # noqa: E402

TARGET_VIRUSES = frozenset(
    {"CMV", "DENV", "EBV", "HBV", "HCV", "HIV-1", "HPV", "IAV", "SARS-CoV-2"}
)
DEF_B_NEG_ORIGINS = frozenset({"tested_negative", "iedb_api"})
ALIGN_COLUMNS = [
    "peptide",
    "virus",
    "strain",
    "negative_origin",
    "hla_allele",
    "label",
    "fold",
]
TRACKED_OUTPUT = "results/binding_only_comparison_mode31_vs_mode10.csv"


def _load_aligned(mode31_path: Path, mode10_path: Path) -> pd.DataFrame:
    mode31 = pd.read_csv(mode31_path)
    mode10 = pd.read_csv(mode10_path)
    required = set(ALIGN_COLUMNS) | {"score", "method", "feature_mode"}
    for name, frame in (("mode31", mode31), ("mode10", mode10)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} OOF is missing columns: {sorted(missing)}")
    if len(mode31) != len(mode10):
        raise ValueError(f"OOF row-count mismatch: mode31={len(mode31)}, mode10={len(mode10)}")
    left = mode31[ALIGN_COLUMNS].fillna("<NA>")
    right = mode10[ALIGN_COLUMNS].fillna("<NA>")
    if not left.equals(right):
        raise ValueError("OOF frames do not align row-for-row on the certified keys")
    if set(mode31["feature_mode"].astype(str)) != {"31"}:
        raise ValueError("mode31 OOF does not contain feature_mode=31 exclusively")
    if set(mode10["feature_mode"].astype(str)) != {"10"}:
        raise ValueError("mode10 OOF does not contain feature_mode=10 exclusively")
    out = mode31[ALIGN_COLUMNS].copy()
    out["mode31_score"] = mode31["score"].to_numpy()
    out["mode10_score"] = mode10["score"].to_numpy()
    return out


def _bootstrap_once(
    y: np.ndarray, mode31: np.ndarray, mode10: np.ndarray, idx: np.ndarray
) -> tuple[float, float] | None:
    sampled_y = y[idx]
    if np.unique(sampled_y).size < 2:
        return None
    sampled31 = mode31[idx]
    sampled10 = mode10[idx]
    return (
        roc_auc_score(sampled_y, sampled31) - roc_auc_score(sampled_y, sampled10),
        average_precision_score(sampled_y, sampled31)
        - average_precision_score(sampled_y, sampled10),
    )


def _paired_bootstrap(
    frame: pd.DataFrame, *, n_resamples: int, seed: int
) -> dict[str, float]:
    y = frame["label"].to_numpy()
    mode31 = frame["mode31_score"].to_numpy()
    mode10 = frame["mode10_score"].to_numpy()
    rng = np.random.default_rng(seed)
    draws = (
        delayed(_bootstrap_once)(
            y, mode31, mode10, rng.integers(0, len(frame), size=len(frame))
        )
        for _ in range(n_resamples)
    )
    results = Parallel(n_jobs=-1, batch_size="auto", pre_dispatch="2*n_jobs")(draws)
    valid = np.asarray([result for result in results if result is not None], dtype=float)
    if valid.size == 0:
        raise ValueError("No bootstrap resample contained both classes")

    def summarize(values: np.ndarray) -> tuple[float, float, float]:
        opposite_sign = min(np.count_nonzero(values <= 0), np.count_nonzero(values >= 0))
        return (
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
            float(min(1.0, 2 * (opposite_sign + 1) / (len(values) + 1))),
        )

    roc_lo, roc_hi, roc_p = summarize(valid[:, 0])
    pr_lo, pr_hi, pr_p = summarize(valid[:, 1])
    return {
        "delta_auc_roc_ci_low": roc_lo,
        "delta_auc_roc_ci_high": roc_hi,
        "delta_auc_roc_p_value": roc_p,
        "delta_auc_pr_ci_low": pr_lo,
        "delta_auc_pr_ci_high": pr_hi,
        "delta_auc_pr_p_value": pr_p,
        "valid_bootstrap_resamples": int(len(valid)),
    }


def _partition_masks(frame: pd.DataFrame) -> list[tuple[str, str, pd.Series]]:
    target = frame["virus"].isin(TARGET_VIRUSES)
    positive = frame["label"] == 1
    return [
        ("pooled", "All active scored rows", pd.Series(True, index=frame.index)),
        (
            "def_a",
            "9 target viruses: positives + iedb_api negatives",
            target & (positive | frame["negative_origin"].eq("iedb_api")),
        ),
        (
            "def_b",
            "9 target viruses: positives + tested_negative or iedb_api negatives",
            target & (positive | frame["negative_origin"].isin(DEF_B_NEG_ORIGINS)),
        ),
    ]


def compute_table(
    mode31_path: Path, mode10_path: Path, *, n_resamples: int, seed: int
) -> pd.DataFrame:
    aligned = _load_aligned(mode31_path, mode10_path)
    rows: list[dict[str, object]] = []
    for partition, definition, mask in _partition_masks(aligned):
        part = aligned.loc[mask]
        y = part["label"].to_numpy()
        mode31 = part["mode31_score"].to_numpy()
        mode10 = part["mode10_score"].to_numpy()
        mode31_roc = roc_auc_score(y, mode31)
        mode10_roc = roc_auc_score(y, mode10)
        mode31_pr = average_precision_score(y, mode31)
        mode10_pr = average_precision_score(y, mode10)
        rows.append(
            {
                "partition": partition,
                "definition": definition,
                "n": len(part),
                "n_pos": int((y == 1).sum()),
                "n_neg": int((y == 0).sum()),
                "mode31_auc_roc": mode31_roc,
                "mode10_auc_roc": mode10_roc,
                "delta_auc_roc": mode31_roc - mode10_roc,
                "mode31_auc_pr": mode31_pr,
                "mode10_auc_pr": mode10_pr,
                "delta_auc_pr": mode31_pr - mode10_pr,
                "bootstrap_resamples": n_resamples,
                "bootstrap_seed": seed,
                **_paired_bootstrap(part, n_resamples=n_resamples, seed=seed),
            }
        )
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode31-oof", default="models/v5/rf_oof_predictions_mode31.csv")
    parser.add_argument("--mode10-oof", default="models/v5/rf_oof_predictions_mode10.csv")
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    mode31_path = Path(args.mode31_oof)
    mode10_path = Path(args.mode10_oof)
    table = compute_table(
        mode31_path,
        mode10_path,
        n_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    print(table.to_string(index=False))
    if args.output is None:
        print("No --output given: nothing written.")
        return 0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False, lineterminator="\n")
    sidecar = write_provenance_sidecar(
        output,
        script="scripts/compute_binding_only_comparison.py",
        extra={
            "mode31_oof": mode31_path.as_posix(),
            "mode31_oof_sha256": sha256_file(mode31_path),
            "mode10_oof": mode10_path.as_posix(),
            "mode10_oof_sha256": sha256_file(mode10_path),
            "bootstrap_resamples": args.bootstrap_resamples,
            "bootstrap_seed": args.seed,
            "bootstrap_unit": "OOF row, paired across models",
            "alignment_columns": ALIGN_COLUMNS,
        },
    )
    print(f"wrote {output}")
    print(f"wrote {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
