#!/usr/bin/env python3
"""Measure whether mode 51's gain is explained by allele identity.

Run from a git checkout: the provenance sidecar records `git rev-parse HEAD`,
which is resolved before anything is computed or written, so outside a
repository the script fails without leaving a CSV behind.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.artifact_integrity import sha256_file, write_provenance_sidecar  # noqa: E402
from src.evaluate_metrics import evaluate  # noqa: E402
from src.features import (  # noqa: E402
    ALLELE_CONTACT_WEIGHTS,
    CONTACT_WEIGHT_COLUMNS,
    FEATURE_COLUMNS_50,
    FEATURE_COLUMNS_51,
    POPULATION_AVG_CONTACT_WEIGHTS,
)
from src.hla_supertypes import (  # noqa: E402
    HLA_FAMILY_SUPERTYPE_MAP,
    HLA_SUPERTYPE_MAP,
    get_hla_supertype,
)
from src.iedb_data_loader import GOLD_STANDARD_EPITOPES  # noqa: E402
from src.ml_utils import PeptideGroupedKFold  # noqa: E402
from src.train_classifier import (  # noqa: E402
    _cross_validate,
    _filter_quarantined,
    prepare_features_50,
)

TRACKED_OUTPUT = "results/mode51_allele_identity_discriminator.csv"
PERMUTATION_SEEDS = (7, 42, 99, 123, 1337, 2026)
MODEL_SEED = 42
OTHER = "other"
OUTPUT_COLUMNS = [
    "arm",
    "description",
    "backend",
    "variant_seed",
    "model_seed",
    "training_rows",
    "feature_count",
    "fold_auc_pr",
    "fold_auc_roc",
    "pooled_auc_pr",
    "pooled_auc_roc",
    "gain_vs_A_auc_pr",
    "gain_retained_vs_B",
    "matrix_equal_B",
    "runtime_seconds",
]


def contact_weight_frame(
    alleles: pd.Series,
    mapping: dict[str, list[float]],
) -> pd.DataFrame:
    """Return mode-51's five per-allele contact columns."""
    records = [
        dict(zip(CONTACT_WEIGHT_COLUMNS, mapping.get(allele, POPULATION_AVG_CONTACT_WEIGHTS)))
        for allele in alleles
    ]
    return pd.DataFrame(records, columns=CONTACT_WEIGHT_COLUMNS)


def build_mode51(
    base_50: pd.DataFrame,
    alleles: pd.Series,
    mapping: dict[str, list[float]],
) -> pd.DataFrame:
    weights = contact_weight_frame(alleles.reset_index(drop=True), mapping)
    return pd.concat([base_50.reset_index(drop=True), weights], axis=1)[FEATURE_COLUMNS_51]


def permuted_contact_weights(seed: int) -> dict[str, list[float]]:
    keys = list(ALLELE_CONTACT_WEIGHTS)
    tuples = [tuple(ALLELE_CONTACT_WEIGHTS[key]) for key in keys]
    order = np.random.default_rng(seed).permutation(len(tuples))
    return {key: list(tuples[index]) for key, index in zip(keys, order)}


def allele_identity_features(alleles: pd.Series) -> pd.DataFrame:
    """One-hot panel identity plus one other bucket, independent of corpus values."""
    keys = sorted(ALLELE_CONTACT_WEIGHTS)
    normalized = alleles.where(alleles.isin(keys), OTHER)
    categories = [*keys, OTHER]
    categorical = pd.Categorical(normalized, categories=categories)
    encoded = pd.get_dummies(categorical, dtype=float)
    encoded.columns = [f"allele_id_{value}" for value in categories]
    return encoded.reset_index(drop=True)


def integer_allele_feature(alleles: pd.Series) -> pd.DataFrame:
    """Encode other as zero and the sorted ten-allele panel as 1..10."""
    codes = {allele: index for index, allele in enumerate(sorted(ALLELE_CONTACT_WEIGHTS), 1)}
    return pd.DataFrame({"allele_integer_code": alleles.map(codes).fillna(0).astype(float)})


def _supertype(allele: object) -> str:
    try:
        return get_hla_supertype(str(allele))
    except ValueError:
        return OTHER


def supertype_features(alleles: pd.Series) -> pd.DataFrame:
    """One-hot the repo's declared supertype vocabulary plus other."""
    categories = [
        *sorted(set(HLA_SUPERTYPE_MAP.values()) | set(HLA_FAMILY_SUPERTYPE_MAP.values())),
        OTHER,
    ]
    values = alleles.map(_supertype)
    categorical = pd.Categorical(values, categories=categories)
    encoded = pd.get_dummies(categorical, dtype=float)
    encoded.columns = [f"hla_supertype_{value}" for value in categories]
    return encoded.reset_index(drop=True)


def within_allele_shuffled_weights(alleles: pd.Series, seed: int) -> pd.DataFrame:
    """Shuffle complete contact rows only among rows sharing the same allele."""
    keys = alleles.fillna(OTHER).astype(str).reset_index(drop=True)
    original = contact_weight_frame(alleles.reset_index(drop=True), ALLELE_CONTACT_WEIGHTS)
    shuffled = original.copy()
    rng = np.random.default_rng(seed)
    for _allele, positions in keys.groupby(keys, sort=True).groups.items():
        target = np.asarray(list(positions), dtype=int)
        source = rng.permutation(target)
        shuffled.iloc[target] = original.iloc[source].to_numpy()
    return shuffled


def target_rate_features(
    train_alleles: pd.Series,
    train_y: np.ndarray,
    heldout_alleles: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit rates without held-out labels; training rows use leave-one-out rates.

    Training rows use a leave-one-out rate, so each training row's own label
    enters its own feature; arm H is therefore not a clean control and neither
    of its rows is evidence about per-allele base rates.
    """
    keys = train_alleles.fillna(OTHER).astype(str).reset_index(drop=True)
    heldout_keys = heldout_alleles.fillna(OTHER).astype(str).reset_index(drop=True)
    labels = pd.Series(np.asarray(train_y, dtype=float)).reset_index(drop=True)
    table = pd.DataFrame({"allele": keys, "label": labels})
    stats = table.groupby("allele", sort=True)["label"].agg(["sum", "count"])
    sums = keys.map(stats["sum"]).astype(float)
    counts = keys.map(stats["count"]).astype(float)
    global_sum = float(labels.sum())
    global_count = float(len(labels))
    fallback = (global_sum - labels) / max(global_count - 1.0, 1.0)
    train_rate = ((sums - labels) / (counts - 1.0)).where(counts > 1.0, fallback)
    global_rate = global_sum / max(global_count, 1.0)
    heldout_rate = heldout_keys.map(stats["sum"] / stats["count"]).fillna(global_rate)
    return (
        pd.DataFrame({"allele_train_only_positive_rate": train_rate.astype(float)}),
        pd.DataFrame({"allele_train_only_positive_rate": heldout_rate.astype(float)}),
    )


def _model(backend: str, y: np.ndarray):
    negatives = int(np.sum(y == 0))
    positives = int(np.sum(y == 1))
    if backend == "rf":
        return RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=MODEL_SEED,
            n_jobs=-1,
        )
    if backend == "xgb":
        return XGBClassifier(
            n_estimators=200,
            scale_pos_weight=negatives / max(positives, 1),
            random_state=MODEL_SEED,
            eval_metric="aucpr",
            objective="binary:logistic",
            nthread=-1,
        )
    raise ValueError(f"unsupported backend: {backend}")


def _fold_metrics(oof: pd.DataFrame) -> tuple[list[float], list[float]]:
    auc_pr: list[float] = []
    auc_roc: list[float] = []
    for _fold, frame in oof.groupby("fold", sort=True):
        metrics = evaluate(frame["label"].to_numpy(), frame["score"].to_numpy())
        auc_pr.append(float(metrics["auc_pr"]))
        auc_roc.append(float(metrics["auc_roc"]))
    return auc_pr, auc_roc


def run_static_arm(
    features: pd.DataFrame,
    y: np.ndarray,
    metadata: pd.DataFrame,
    backend: str,
) -> dict[str, object]:
    started = time.perf_counter()
    model = _model(backend, y)
    _avg, _std, _subgroups, oof = _cross_validate(
        features,
        y,
        metadata,
        type(model),
        model.get_params(),
        n_splits=5,
        random_state=MODEL_SEED,
        subgroup_columns=[],
        cv_group_by="peptide",
    )
    runtime = time.perf_counter() - started
    pooled = evaluate(oof["label"].to_numpy(), oof["score"].to_numpy())
    fold_pr, fold_roc = _fold_metrics(oof)
    return {
        "fold_auc_pr": json.dumps(fold_pr, separators=(",", ":")),
        "fold_auc_roc": json.dumps(fold_roc, separators=(",", ":")),
        "pooled_auc_pr": float(pooled["auc_pr"]),
        "pooled_auc_roc": float(pooled["auc_roc"]),
        "runtime_seconds": runtime,
    }


def grouped_splits(
    base_50: pd.DataFrame,
    y: np.ndarray,
    metadata: pd.DataFrame,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = PeptideGroupedKFold(n_splits=5, shuffle=True, random_state=MODEL_SEED)
    return list(
        splitter.split(
            base_50,
            y,
            negative_origin=metadata.get("negative_origin"),
            hla_alleles=metadata.get("hla_allele"),
            peptides=metadata["peptide"],
        )
    )


def run_target_rate_arm(
    base_50: pd.DataFrame,
    alleles: pd.Series,
    y: np.ndarray,
    metadata: pd.DataFrame,
    backend: str,
) -> dict[str, object]:
    """Fit every held-out encoding from that outer fold's training rows only."""
    started = time.perf_counter()
    oof_parts: list[pd.DataFrame] = []
    for fold, (train_index, heldout_index) in enumerate(
        grouped_splits(base_50, y, metadata), 1
    ):
        train_rate, heldout_rate = target_rate_features(
            alleles.iloc[train_index], y[train_index], alleles.iloc[heldout_index]
        )
        train_x = pd.concat(
            [base_50.iloc[train_index].reset_index(drop=True), train_rate], axis=1
        )
        heldout_x = pd.concat(
            [base_50.iloc[heldout_index].reset_index(drop=True), heldout_rate], axis=1
        )
        model = _model(backend, y[train_index])
        model.fit(train_x, y[train_index])
        scores = model.predict_proba(heldout_x)[:, 1]
        oof_parts.append(
            pd.DataFrame({"fold": fold, "label": y[heldout_index], "score": scores})
        )
    oof = pd.concat(oof_parts, ignore_index=True)
    pooled = evaluate(oof["label"].to_numpy(), oof["score"].to_numpy())
    fold_pr, fold_roc = _fold_metrics(oof)
    return {
        "fold_auc_pr": json.dumps(fold_pr, separators=(",", ":")),
        "fold_auc_roc": json.dumps(fold_roc, separators=(",", ":")),
        "pooled_auc_pr": float(pooled["auc_pr"]),
        "pooled_auc_roc": float(pooled["auc_roc"]),
        "runtime_seconds": time.perf_counter() - started,
    }


def compute_table(data_path: Path, binding_path: Path) -> pd.DataFrame:
    frame = _filter_quarantined(pd.read_csv(data_path))
    train_pool = frame[~frame["peptide"].isin(GOLD_STANDARD_EPITOPES)].reset_index(drop=True)
    y = train_pool["label"].to_numpy()
    alleles = train_pool["hla_allele"].reset_index(drop=True)
    metadata_columns = [
        column
        for column in ("peptide", "virus", "strain", "negative_origin", "hla_allele")
        if column in train_pool.columns
    ]
    metadata = train_pool[metadata_columns].copy()
    base_50 = prepare_features_50(train_pool, binding_path)
    if list(base_50.columns) != FEATURE_COLUMNS_50:
        raise ValueError("mode-50 feature columns do not match FEATURE_COLUMNS_50")

    shipped = build_mode51(base_50, alleles, ALLELE_CONTACT_WEIGHTS)
    matrices: list[tuple[str, str, int | None, pd.DataFrame, bool | None]] = [
        ("A", "mode 50", None, base_50, None),
        ("B", "mode 51 as shipped", None, shipped, True),
        (
            "C",
            "mode 50 plus panel-allele one-hot and other",
            None,
            pd.concat([base_50.reset_index(drop=True), allele_identity_features(alleles)], axis=1),
            None,
        ),
        (
            "D",
            "mode 50 plus integer allele code",
            None,
            pd.concat([base_50.reset_index(drop=True), integer_allele_feature(alleles)], axis=1),
            None,
        ),
        (
            "E",
            "mode 50 plus repo HLA-supertype one-hot",
            None,
            pd.concat([base_50.reset_index(drop=True), supertype_features(alleles)], axis=1),
            None,
        ),
    ]
    for seed in PERMUTATION_SEEDS:
        matrices.append(
            (
                "F",
                "mode 51 with weights permuted across alleles",
                seed,
                build_mode51(base_50, alleles, permuted_contact_weights(seed)),
                None,
            )
        )
    shuffled = within_allele_shuffled_weights(alleles, MODEL_SEED)
    arm_g = pd.concat([base_50.reset_index(drop=True), shuffled], axis=1)[FEATURE_COLUMNS_51]
    g_equal = arm_g.equals(shipped)
    if not g_equal:
        raise AssertionError("within-allele row shuffle changed a constant per-allele matrix")
    matrices.append(
        ("G", "mode 51 with rows shuffled within allele", MODEL_SEED, arm_g, g_equal)
    )

    rows: list[dict[str, object]] = []
    for arm, description, seed, matrix, equal_b in matrices:
        for backend in ("rf", "xgb"):
            metrics = run_static_arm(matrix, y, metadata, backend)
            rows.append(
                {
                    "arm": arm,
                    "description": description,
                    "backend": backend,
                    "variant_seed": seed,
                    "model_seed": MODEL_SEED,
                    "training_rows": len(train_pool),
                    "feature_count": matrix.shape[1],
                    **metrics,
                    "gain_vs_A_auc_pr": np.nan,
                    "gain_retained_vs_B": np.nan,
                    "matrix_equal_B": equal_b,
                }
            )
            print(
                f"arm={arm} backend={backend} seed={seed} "
                f"pooled_auc_pr={metrics['pooled_auc_pr']:.10f}"
            )
    for backend in ("rf", "xgb"):
        metrics = run_target_rate_arm(base_50, alleles, y, metadata, backend)
        rows.append(
            {
                "arm": "H",
                "description": (
                    "mode 50 plus per-allele positive rate (held-out rows: training-fold "
                    "rate; training rows: leave-one-out rate that leaks their own label)"
                ),
                "backend": backend,
                "variant_seed": None,
                "model_seed": MODEL_SEED,
                "training_rows": len(train_pool),
                "feature_count": base_50.shape[1] + 1,
                **metrics,
                "gain_vs_A_auc_pr": np.nan,
                "gain_retained_vs_B": np.nan,
                "matrix_equal_B": None,
            }
        )
        print(f"arm=H backend={backend} pooled_auc_pr={metrics['pooled_auc_pr']:.10f}")

    table = pd.DataFrame(rows)
    for backend in ("rf", "xgb"):
        base = float(table.loc[(table.arm == "A") & (table.backend == backend), "pooled_auc_pr"].iloc[0])
        shipped_score = float(
            table.loc[(table.arm == "B") & (table.backend == backend), "pooled_auc_pr"].iloc[0]
        )
        mask = table.backend == backend
        table.loc[mask, "gain_vs_A_auc_pr"] = table.loc[mask, "pooled_auc_pr"] - base
        table.loc[mask, "gain_retained_vs_B"] = (
            table.loc[mask, "pooled_auc_pr"] - base
        ) / (shipped_score - base)
    return table[OUTPUT_COLUMNS]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/immunogenicity_dataset_v5.csv")
    parser.add_argument("--binding-matrix", default="models/peptide_binding_matrix_v5.csv")
    parser.add_argument("--output", default=TRACKED_OUTPUT)
    args = parser.parse_args(argv)
    data_path = Path(args.data)
    binding_path = Path(args.binding_matrix)
    output = Path(args.output)
    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    table = compute_table(data_path, binding_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False, lineterminator="\n", float_format="%.10g")
    sidecar = write_provenance_sidecar(
        output,
        script="scripts/compute_mode51_allele_identity_discriminator.py",
        extra={
            "training_data": data_path.as_posix(),
            "training_data_sha256": sha256_file(data_path),
            "binding_matrix": binding_path.as_posix(),
            "binding_matrix_sha256": sha256_file(binding_path),
            "git_sha": git_sha,
            "python": platform.python_version(),
            "platform": f"{platform.system()} {platform.machine()}",
            "permutation_seeds": list(PERMUTATION_SEEDS),
            "model_seed": MODEL_SEED,
            "splitter": "PeptideGroupedKFold(n_splits=5, shuffle=True, random_state=42)",
            "target_rate_guard": (
                "held-out encodings use only outer-fold training labels; training encodings "
                "are leave-one-out"
            ),
        },
    )
    print(f"wrote {output}")
    print(f"wrote {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
