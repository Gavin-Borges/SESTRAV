#!/usr/bin/env python3
"""Attribute pooled AUC-ROC to cross-coverage comparisons.

Read the pooled-population mode-31 rows of the output together, never the
overall row alone. The overall pooled delta is negative, -0.0139, while no
single virus has a negative delta: CMV +0.0511, DENV +0.2721, EBV +0.1470,
HIV-1 +0.0335, IAV +0.1115, SARS-CoV-2 +0.0287, and six viruses at exactly
zero. The sign reversal comes from between-virus pairs entering the pooled
AUC and is a composition effect, not a within-virus null.

Those six zeros are 6 of the table's 44 exactly-zero deltas (three
populations, two models). In each, every row falls in one coverage group,
so no cross-coverage pair exists: the delta is set rather than computed,
and `coverage_is_degenerate` marks it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.artifact_integrity import sha256_file, write_provenance_sidecar  # noqa: E402

TRACKED_OUTPUT = "results/binding_zerofill_attribution.csv"
KEY_COLUMNS = [
    "peptide",
    "virus",
    "strain",
    "negative_origin",
    "hla_allele",
    "label",
    "fold",
]
POPULATIONS = ("def_a", "def_b", "pooled")
TARGET_VIRUSES = {
    "CMV",
    "DENV",
    "EBV",
    "HBV",
    "HCV",
    "HIV-1",
    "HPV",
    "IAV",
    "SARS-CoV-2",
}
OUTPUT_COLUMNS = [
    "population",
    "scope",
    "virus",
    "model",
    "n",
    "n_pos",
    "n_neg",
    "covered_pos",
    "covered_neg",
    "uncovered_pos",
    "uncovered_neg",
    "pooled_auc_roc",
    "within_coverage_auc_roc",
    "cross_coverage_delta_auc_roc",
    "coverage_is_degenerate",
]


def population_mask(frame: pd.DataFrame, population: str) -> pd.Series:
    """Apply the repository's corrected def_a, def_b, and pooled definitions."""
    positive = frame["label"].eq(1)
    target = frame["virus"].isin(TARGET_VIRUSES)
    if population == "def_a":
        selected = positive | (frame["label"].eq(0) & frame["negative_origin"].eq("iedb_api"))
        return target & selected
    if population == "def_b":
        honest = frame["negative_origin"].isin(["tested_negative", "iedb_api"])
        return target & (positive | (frame["label"].eq(0) & honest))
    if population == "pooled":
        return pd.Series(True, index=frame.index)
    raise ValueError(f"unsupported population: {population}")


def _normalized_keys(frame: pd.DataFrame) -> pd.MultiIndex:
    keys = frame[KEY_COLUMNS].copy()
    for column in KEY_COLUMNS:
        if column not in {"label", "fold"}:
            keys[column] = keys[column].astype("string").fillna("<NA>")
    return pd.MultiIndex.from_frame(keys, names=KEY_COLUMNS)


def align_oof(mode31: pd.DataFrame, mode10: pd.DataFrame) -> pd.DataFrame:
    """Return row-aligned scores or fail before any attribution is computed."""
    missing31 = sorted(set(KEY_COLUMNS) - set(mode31.columns))
    missing10 = sorted(set(KEY_COLUMNS) - set(mode10.columns))
    if missing31 or missing10:
        raise ValueError(f"missing alignment columns: mode31={missing31}, mode10={missing10}")
    left = mode31.copy()
    right = mode10.copy()
    left.index = _normalized_keys(left)
    right.index = _normalized_keys(right)
    if not left.index.is_unique or not right.index.is_unique:
        raise ValueError("OOF alignment keys must be unique in both inputs")
    missing_from_10 = left.index.difference(right.index)
    missing_from_31 = right.index.difference(left.index)
    if len(missing_from_10) or len(missing_from_31):
        raise ValueError(
            "OOF key mismatch: "
            f"missing_from_mode10={len(missing_from_10)} "
            f"missing_from_mode31={len(missing_from_31)}"
        )
    right = right.loc[left.index]
    aligned = left.reset_index(drop=True)
    aligned = aligned.rename(columns={"score": "mode31_score"})
    aligned["mode10_score"] = right["score"].to_numpy()
    return aligned


def auc_attribution(
    labels: pd.Series, scores: pd.Series, covered: pd.Series
) -> dict[str, float | bool]:
    """Compute pooled AUC minus the pair-count-weighted within-coverage AUC.

    `coverage_is_degenerate` marks the rows whose delta is set rather than
    computed, so a zero produced by the early return below is distinguishable
    in the output from a zero that a comparison produced.
    """
    y = labels.to_numpy(dtype=int)
    prediction = scores.to_numpy(dtype=float)
    coverage = covered.to_numpy(dtype=bool)
    if len(np.unique(y)) != 2:
        raise ValueError("AUC-ROC requires both classes")
    pooled = float(roc_auc_score(y, prediction))
    if len(np.unique(coverage)) == 1:
        # All rows fall in one coverage group, so no cross-coverage pair
        # exists. This zero is by construction, not a measured null.
        return {
            "pooled_auc_roc": pooled,
            "within_coverage_auc_roc": pooled,
            "cross_coverage_delta_auc_roc": 0.0,
            "coverage_is_degenerate": True,
        }
    weighted_sum = 0.0
    pair_count = 0
    for group in (False, True):
        mask = coverage == group
        group_y = y[mask]
        positives = int(np.sum(group_y == 1))
        negatives = int(np.sum(group_y == 0))
        weight = positives * negatives
        if weight:
            weighted_sum += float(roc_auc_score(group_y, prediction[mask])) * weight
            pair_count += weight
    if pair_count == 0:
        raise ValueError("no within-coverage positive-negative pairs")
    within = weighted_sum / pair_count
    return {
        "pooled_auc_roc": pooled,
        "within_coverage_auc_roc": within,
        "cross_coverage_delta_auc_roc": pooled - within,
        "coverage_is_degenerate": False,
    }


def attribution_row(
    frame: pd.DataFrame,
    population: str,
    scope: str,
    virus: str,
    model: str,
) -> dict[str, object]:
    labels = frame["label"]
    covered = frame["covered"]
    metrics = auc_attribution(labels, frame[f"{model}_score"], covered)
    return {
        "population": population,
        "scope": scope,
        "virus": virus,
        "model": model,
        "n": len(frame),
        "n_pos": int(labels.eq(1).sum()),
        "n_neg": int(labels.eq(0).sum()),
        "covered_pos": int((covered & labels.eq(1)).sum()),
        "covered_neg": int((covered & labels.eq(0)).sum()),
        "uncovered_pos": int((~covered & labels.eq(1)).sum()),
        "uncovered_neg": int((~covered & labels.eq(0)).sum()),
        **metrics,
    }


def compute_table(
    mode31_path: Path,
    mode10_path: Path,
    matrix_path: Path,
    population: str | None,
) -> pd.DataFrame:
    aligned = align_oof(pd.read_csv(mode31_path), pd.read_csv(mode10_path))
    binding = pd.read_csv(matrix_path, usecols=["peptide"])
    if binding["peptide"].duplicated().any():
        raise ValueError("binding matrix peptide key must be unique")
    aligned["covered"] = aligned["peptide"].isin(set(binding["peptide"]))
    populations = (population,) if population else POPULATIONS
    rows: list[dict[str, object]] = []
    for definition in populations:
        selected = aligned.loc[population_mask(aligned, definition)].copy()
        for model in ("mode31", "mode10"):
            rows.append(attribution_row(selected, definition, "overall", "ALL", model))
        for virus, group in selected.groupby("virus", sort=True):
            if group["label"].nunique() != 2:
                continue
            for model in ("mode31", "mode10"):
                rows.append(attribution_row(group, definition, "virus", str(virus), model))
    return pd.DataFrame(rows)[OUTPUT_COLUMNS]


def validate_comparison(table: pd.DataFrame, comparison_path: Path) -> float:
    comparison = pd.read_csv(comparison_path)
    if comparison["partition"].duplicated().any():
        raise ValueError("comparison partitions must be unique")
    indexed = comparison.set_index("partition")
    maximum = 0.0
    for population in POPULATIONS:
        for model in ("mode31", "mode10"):
            measured = float(
                table.loc[
                    (table.population == population)
                    & (table.scope == "overall")
                    & (table.model == model),
                    "pooled_auc_roc",
                ].iloc[0]
            )
            expected = float(indexed.loc[population, f"{model}_auc_roc"])
            maximum = max(maximum, abs(measured - expected))
    if maximum > 1e-12:
        raise ValueError(f"comparison pooled-AUC mismatch: max_abs_diff={maximum:.12g}")
    return maximum


def git_blob(path: Path) -> str:
    """Return the blob SHA the input's own bytes hash to.

    `git hash-object --path` applies the filters and end-of-line conversion
    `git add` would apply at that path, so the result equals the blob `git add`
    would write for these bytes. It is derived from the file rather than looked
    up in a revision, so it records what was read and needs no ref to resolve.
    """
    resolved = path.resolve()
    relative = resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    return subprocess.check_output(
        ["git", "hash-object", f"--path={relative}", str(resolved)], cwd=REPO_ROOT, text=True
    ).strip()


def input_record(path: Path) -> dict[str, str]:
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "blob_sha": git_blob(path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode31-oof", default="models/v5/rf_oof_predictions_mode31.csv")
    parser.add_argument("--mode10-oof", default="models/v5/rf_oof_predictions_mode10.csv")
    parser.add_argument("--binding-matrix", default="models/peptide_binding_matrix_v5.csv")
    parser.add_argument(
        "--comparison", default="results/binding_only_comparison_mode31_vs_mode10.csv"
    )
    parser.add_argument("--population", choices=POPULATIONS)
    parser.add_argument("--output", default=TRACKED_OUTPUT)
    args = parser.parse_args(argv)
    mode31_path = Path(args.mode31_oof)
    mode10_path = Path(args.mode10_oof)
    matrix_path = Path(args.binding_matrix)
    comparison_path = Path(args.comparison)
    output = Path(args.output)
    table = compute_table(mode31_path, mode10_path, matrix_path, args.population)
    comparison_max_diff = validate_comparison(table, comparison_path) if not args.population else None
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False, lineterminator="\n", float_format="%.10g")
    sidecar = write_provenance_sidecar(
        output,
        script="scripts/compute_binding_zerofill_attribution.py",
        extra={
            "git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip(),
            "metric": "pooled AUC-ROC minus pair-count-weighted within-coverage AUC-ROC",
            "populations": list(POPULATIONS) if not args.population else [args.population],
            "comparison_max_abs_diff": comparison_max_diff,
            "inputs": [
                input_record(mode31_path),
                input_record(mode10_path),
                input_record(matrix_path),
                input_record(comparison_path),
            ],
        },
    )
    print(f"rows={len(table)}")
    print(f"comparison_max_abs_diff={comparison_max_diff}")
    print(f"wrote {output}")
    print(f"wrote {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
