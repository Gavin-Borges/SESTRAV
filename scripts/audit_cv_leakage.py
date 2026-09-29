#!/usr/bin/env python3
"""Measure peptide-level cross-validation leakage in the v5 training pipeline.

Background: `src/train_classifier.py` splits folds with `MultiStratifiedKFold`
(`src/ml_utils.py`), which accepts a `peptides=` argument but only uses it to
bin peptide length for stratification - never as a fold group. The v5 dataset
is deduplicated on `(peptide, hla_allele)`, not on `peptide`
(`scripts/build_dataset_v5.py`), and `FEATURE_COLUMNS_31` is a pure function of
the peptide string (20 physico + 10 fixed-panel binding scores + length; the
allele column never enters the vector). Two rows sharing a peptide are
therefore feature-identical, and an ungrouped splitter can and does place them
on opposite sides of a fold boundary.

This script makes that leak reproducible and binds it to a canonical results/
CSV so the claims register and the integrity harness can reconcile any number
drawn from it, instead of the finding living only as a one-off audit
measurement (docs/proposals/2026_feature_upgrade_roadmap.md cites this file).

A second, independent channel is measured by the same harness: NEAR-HOMOLOG
overlap (substring containment, Hamming-1, single-indel). Peptide grouping
closes the exact-duplicate channel by construction and leaves that one open,
so the two arms answer different questions and neither substitutes for the
other. See the near-homolog arm's header comment for the relation and its
grounds.

Reproduce:  python scripts/audit_cv_leakage.py
Output:     results/cv_leakage_audit.csv (+ .provenance.json sidecar)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.artifact_guard import guard_planned_paths, planned_paths_under  # noqa: E402
from src.artifact_integrity import sha256_file as _sha256_file  # noqa: E402
from src.evaluate_metrics import evaluate  # noqa: E402
from src.ml_utils import MultiStratifiedKFold, PeptideGroupedKFold  # noqa: E402
from src.train_classifier import (  # noqa: E402
    prepare_features_31,
    prepare_features_35,
    prepare_features_50,
)

RANDOM_STATE = 42
N_SPLITS = 5
# Matches src/train_classifier.py's rf_kwargs (n_estimators=200) exactly, so the
# "production_splitter" reproduction below is comparable to the certified
# models/v5/training_results_mode31.csv figures, not just proportionally similar.
N_ESTIMATORS = 200

# Same 9-virus target set and real-negative-origin filter as
# scripts/compute_pooled_honest_metric.py, reused here so this audit can
# directly test whether results/per_virus_eval_v5_mode31.csv (mean 0.751 as certified
# when this audit was written; RETRACTED by D15 and re-baselined to 0.658) and
# results/pooled_honest_same_pathogen.csv (0.712; likewise retracted, now 0.6015) -
# both downstream consumers of the same MultiStratifiedKFold-derived
# models/v5/rf_oof_predictions_mode31.csv - share the same leakage exposure as the
# pooled mode31_auc_pr headline.
TARGET_VIRUSES = {"CMV", "DENV", "EBV", "HBV", "HCV", "HIV-1", "HPV", "IAV", "SARS-CoV-2"}
REAL_NEG_ORIGIN = "iedb_api"
MIN_VIRUS_SIZE = 20

DATASET_PATH = PROJECT_ROOT / "data" / "immunogenicity_dataset_v5.csv"
BINDING_MATRIX_PATH = PROJECT_ROOT / "models" / "peptide_binding_matrix_v5.csv"
ANTIGEN_PROCESSING_CACHE_PATH = PROJECT_ROOT / "data" / "antigen_processing_cache.csv"
SELF_SIMILARITY_CACHE_PATH = PROJECT_ROOT / "data" / "self_similarity_cache.csv"


def _load_active(dataset_path: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_path, low_memory=False)
    quarantined = df.get("is_quarantined", pd.Series(False, index=df.index)).fillna(False)
    return df[~quarantined.astype(bool)].reset_index(drop=True)


def _fold_overlap(active: pd.DataFrame) -> list[dict]:
    """Per-fold count of test rows whose exact peptide also appears in that fold's train set."""
    y = active["label"].astype(int)
    splitter = MultiStratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(
        splitter.split(
            np.zeros((len(active), 1)),
            y,
            negative_origin=active.get("negative_origin"),
            hla_alleles=active.get("hla_allele"),
            peptides=active["peptide"],
        )
    )
    peptides = active["peptide"].to_numpy()
    rows = []
    total_test = 0
    total_leak = 0
    for fold_id, (train_idx, test_idx) in enumerate(splits):
        train_peptides = set(peptides[train_idx])
        test_peptides = peptides[test_idx]
        leaked = int(np.isin(test_peptides, list(train_peptides)).sum())
        rows.append(
            {
                "config": "production_splitter",
                "metric": f"fold{fold_id}_peptide_overlap_pct",
                "value": 100.0 * leaked / len(test_idx),
                "std": np.nan,
                "n_test": len(test_idx),
                "n_leaked": leaked,
            }
        )
        total_test += len(test_idx)
        total_leak += leaked
    rows.append(
        {
            "config": "production_splitter",
            "metric": "overall_peptide_overlap_pct",
            "value": 100.0 * total_leak / total_test,
            "std": np.nan,
            "n_test": total_test,
            "n_leaked": total_leak,
        }
    )
    return rows


def _cv_auc(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray | None) -> tuple[float, float, float, np.ndarray]:
    """5-fold AUC-PR / AUC-ROC. Peptide-grouped when groups is given, else the production splitter.

    Also returns the pooled out-of-fold (OOF) score array (aligned to X/y's row order) -
    each row's prediction from the one fold where it was held out. This reuses the same
    per-fold fits already needed for the AP/ROC means, at no extra training cost, so
    callers needing a per-virus or per-subset breakdown (see _per_virus_and_pooled_honest_ab)
    do not need to retrain.
    """
    aps: list[float] = []
    rocs: list[float] = []
    oof = np.full(len(y), np.nan)
    if groups is None:
        splitter = MultiStratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        splits = list(splitter.split(X, pd.Series(y)))
    else:
        splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        splits = list(splitter.split(X, y, groups=groups))
    for train_idx, test_idx in splits:
        clf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        )
        clf.fit(X.iloc[train_idx], y[train_idx])
        proba = clf.predict_proba(X.iloc[test_idx])[:, 1]
        oof[test_idx] = proba
        aps.append(average_precision_score(y[test_idx], proba))
        rocs.append(roc_auc_score(y[test_idx], proba))
    return float(np.mean(aps)), float(np.std(aps)), float(np.mean(rocs)), oof


def _cv_auc_production_splitter(
    X: pd.DataFrame,
    y: np.ndarray,
    splitter_cls: type,
    negative_origin: pd.Series | None,
    hla_alleles: pd.Series | None,
    peptides: pd.Series,
) -> tuple[float, float, float, np.ndarray]:
    """5-fold AUC-PR/AUC-ROC using an actual src.ml_utils splitter class.

    Unlike _cv_auc's "production_splitter" (label-only, no origin/allele/
    peptide passed) and "peptide_grouped_splitter" (StratifiedGroupKFold
    stratified on y alone) - both the original D15 measurement, kept
    unchanged in code so they keep reproducing to 4 decimal places (they are
    NOT bit-frozen: Phase 0's _bin_origin fix shifts _fold_overlap's per-fold
    percentages) - this constructs the splitter with
    the full negative_origin/hla_alleles/peptides composite-key arguments
    exactly as src/train_classifier.py's _cross_validate passes them, so it
    reflects Phase 0's harness repair (fixed _bin_origin, coarsening ladder).
    """
    splitter = splitter_cls(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(
        splitter.split(
            X,
            pd.Series(y),
            negative_origin=negative_origin,
            hla_alleles=hla_alleles,
            peptides=peptides,
        )
    )
    aps: list[float] = []
    rocs: list[float] = []
    oof = np.full(len(y), np.nan)
    for train_idx, test_idx in splits:
        clf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        )
        clf.fit(X.iloc[train_idx], y[train_idx])
        proba = clf.predict_proba(X.iloc[test_idx])[:, 1]
        oof[test_idx] = proba
        aps.append(average_precision_score(y[test_idx], proba))
        rocs.append(roc_auc_score(y[test_idx], proba))
    return float(np.mean(aps)), float(np.std(aps)), float(np.mean(rocs)), oof


def _splitter_ab(active: pd.DataFrame) -> list[dict]:
    """Production (ungrouped) vs peptide-grouped AUC-PR/AUC-ROC, mode-31 features."""
    X = prepare_features_31(active, str(BINDING_MATRIX_PATH))
    y = active["label"].astype(int).to_numpy()
    groups = active["peptide"].to_numpy()

    rows = []
    for config, grp in (("production_splitter", None), ("peptide_grouped_splitter", groups)):
        ap_mean, ap_std, roc_mean, _oof = _cv_auc(X, y, grp)
        rows.append({"config": config, "metric": "mode31_auc_pr", "value": ap_mean, "std": ap_std})
        rows.append({"config": config, "metric": "mode31_auc_roc", "value": roc_mean, "std": np.nan})

    # Phase 0 repaired arms - see _cv_auc_production_splitter's docstring for
    # why these are separate from the two D15-anchor arms above.
    negative_origin = active.get("negative_origin")
    hla_alleles = active.get("hla_allele")
    peptides = active["peptide"]
    for config, splitter_cls in (
        ("production_splitter_repaired", MultiStratifiedKFold),
        ("production_grouped_splitter", PeptideGroupedKFold),
    ):
        ap_mean, ap_std, roc_mean, _oof = _cv_auc_production_splitter(
            X, y, splitter_cls, negative_origin, hla_alleles, peptides
        )
        rows.append({"config": config, "metric": "mode31_auc_pr", "value": ap_mean, "std": ap_std})
        rows.append({"config": config, "metric": "mode31_auc_roc", "value": roc_mean, "std": np.nan})
    return rows


def _per_virus_and_pooled_honest_ab(active: pd.DataFrame) -> list[dict]:
    """Test whether the then-certified per-virus (0.751) and pooled-honest (0.712) figures -
    RETRACTED by D15 (2026-08-10) and re-baselined to 0.658 / 0.6015, and both computed
    downstream of the same MultiStratifiedKFold OOF predictions as the
    pooled mode31_auc_pr headline - share its leakage exposure, under a matched
    (n_estimators=200) production vs peptide-grouped comparison."""
    X = prepare_features_31(active, str(BINDING_MATRIX_PATH))
    y = active["label"].astype(int).to_numpy()
    groups = active["peptide"].to_numpy()
    virus = active["virus"].to_numpy()
    origin = active.get("negative_origin", pd.Series("", index=active.index)).fillna("").to_numpy()

    rows: list[dict[str, object]] = []
    for config, grp in (("production_splitter", None), ("peptide_grouped_splitter", groups)):
        _ap_mean, _ap_std, _roc_mean, oof = _cv_auc(X, y, grp)

        per_virus_rocs = []
        for v in sorted(TARGET_VIRUSES):
            mask = virus == v
            if mask.sum() < MIN_VIRUS_SIZE or len(np.unique(y[mask])) < 2:
                continue
            roc = roc_auc_score(y[mask], oof[mask])
            per_virus_rocs.append(roc)
            rows.append({"config": config, "metric": f"per_virus_auc_roc.{v}", "value": roc, "std": np.nan})
        if per_virus_rocs:
            rows.append(
                {
                    "config": config,
                    "metric": "per_virus_auc_roc.mean",
                    "value": float(np.mean(per_virus_rocs)),
                    "std": float(np.std(per_virus_rocs)),
                    "n_viruses": len(per_virus_rocs),
                }
            )

        target_mask = np.isin(virus, list(TARGET_VIRUSES))
        real_neg_mask = origin == REAL_NEG_ORIGIN
        pooled_mask = target_mask & ((y == 1) | real_neg_mask)
        if pooled_mask.sum() and len(np.unique(y[pooled_mask])) >= 2:
            rows.append(
                {
                    "config": config,
                    "metric": "pooled_honest_same_pathogen_auc_roc",
                    "value": float(roc_auc_score(y[pooled_mask], oof[pooled_mask])),
                    "std": np.nan,
                    "n_pos": int((y[pooled_mask] == 1).sum()),
                    "n_neg": int((y[pooled_mask] == 0).sum()),
                }
            )
            rows.append(
                {
                    "config": config,
                    "metric": "pooled_honest_same_pathogen_auc_pr",
                    "value": float(average_precision_score(y[pooled_mask], oof[pooled_mask])),
                    "std": np.nan,
                }
            )
    return rows


def _tier_a_ab(active: pd.DataFrame) -> list[dict]:
    """Re-measure a subset of Tier A peptides with a FRESH v5-trained model, not the certified pathway.

    What this measures: whether a v5 mode-31 RF trained today, evaluated under two different
    CV splitters, is splitter-sensitive when scoring the subset of Tier A peptides that
    happen to resolve to an active v5 row - a hypothetical "if today's model scored these
    peptides" question. This function trains fresh models on the CURRENT `active` (v5)
    corpus via `_cv_auc` below; it never reads the certified `rf_oof_score` column that
    `results/table3_tier_a_metrics.csv` actually reports, so its output is NOT a measurement
    of that certified pathway's own leakage exposure and must not be cited as one
    (`docs/claims_register.md` D22). The certified 0.828 was produced by a one-time 2026-05
    run on a different, 720-row, zero-duplicate-peptide corpus (`immunogenicity_dataset.csv`
    at `69e0e5c`, D16); the exact-duplicate leakage mechanism this function probes is a
    structural no-op on that corpus. See D22 for the investigation and for the separate,
    unquantified substring-homology risk that does apply to that corpus.

    Method: score each Tier A peptide with the OOF value from the v5 row(s) carrying that
    peptide, taking the first occurrence to mirror the drop_duplicates(keep="first") in
    `src/prepare_external_validation_inputs.py:122`, then evaluate with the repo's own shared
    `src.evaluate_metrics.evaluate` so the numbers are directly comparable to
    `results/table3_tier_a_metrics.csv`. Labels come from the Tier A field, not from v5.

    Coverage caveat recorded in the output: only the Tier A peptides that appear in the v5
    corpus can be scored this way, so n is smaller than the certified n=704 and the two
    arms are compared to each other on that common subset, not to the certified cell.
    """
    tier_a_path = PROJECT_ROOT / "results" / "external_validation_input.csv"
    if not tier_a_path.is_file():
        return [{"config": "tier_a", "metric": "status", "value": np.nan, "note": "external_validation_input.csv absent"}]

    tier_a = pd.read_csv(tier_a_path)
    # The certified field is the 704 rows flagged baseline-complete (the other 16 are
    # gold-standard holdout rows excluded from the published table).
    if "tier_a_baseline_complete" in tier_a.columns:
        tier_a = tier_a[tier_a["tier_a_baseline_complete"].astype(bool)]

    X = prepare_features_31(active, str(BINDING_MATRIX_PATH))
    y = active["label"].astype(int).to_numpy()
    groups = active["peptide"].to_numpy()

    first_row = (
        pd.DataFrame({"peptide": active["peptide"].to_numpy(), "_idx": np.arange(len(active))})
        .drop_duplicates(subset=["peptide"], keep="first")
        .set_index("peptide")["_idx"]
    )
    joined = tier_a[["peptide", "label"]].join(first_row, on="peptide", how="inner")
    n_field = len(tier_a)
    n_scored = len(joined)

    rows: list[dict[str, object]] = [
        {
            "config": "tier_a",
            "metric": "coverage_n_scored",
            "value": float(n_scored),
            "std": np.nan,
            "n_rows": n_field,
            "note": "Tier A peptides resolvable to a v5 row; certified table reports n=704",
        }
    ]
    if n_scored == 0 or joined["label"].nunique() < 2:
        return rows

    idx = joined["_idx"].to_numpy()
    y_true = joined["label"].astype(int).to_numpy()
    for config, grp in (("production_splitter", None), ("peptide_grouped_splitter", groups)):
        _ap, _apstd, _roc, oof = _cv_auc(X, y, grp)
        m = evaluate(y_true, oof[idx])
        for key in ("auc_pr", "auc_roc", "issr_10"):
            rows.append(
                {
                    "config": config,
                    "metric": f"tier_a_sestrav_rf_{key}",
                    "value": float(m[key]),
                    "std": np.nan,
                    "n_rows": n_scored,
                }
            )
    return rows


def _feature_mode_ab(active: pd.DataFrame) -> list[dict]:
    """Mode 31 vs 35 vs 50 AUC-PR under peptide-grouped CV, isolating feature-side gains."""
    y = active["label"].astype(int).to_numpy()
    groups = active["peptide"].to_numpy()
    rows = []

    X31 = prepare_features_31(active, str(BINDING_MATRIX_PATH))
    ap_mean, ap_std, roc_mean, _oof = _cv_auc(X31, y, groups)
    rows.append({"config": "peptide_grouped_splitter", "metric": "mode31_grouped_auc_pr", "value": ap_mean, "std": ap_std})

    try:
        X35 = prepare_features_35(
            active,
            str(BINDING_MATRIX_PATH),
            str(ANTIGEN_PROCESSING_CACHE_PATH),
            str(SELF_SIMILARITY_CACHE_PATH),
        )
        ap_mean, ap_std, roc_mean, _oof = _cv_auc(X35, y, groups)
        rows.append({"config": "peptide_grouped_splitter", "metric": "mode35_grouped_auc_pr", "value": ap_mean, "std": ap_std})
    except Exception as exc:  # noqa: BLE001 - record the failure, do not abort the run
        rows.append({"config": "peptide_grouped_splitter", "metric": "mode35_grouped_auc_pr", "value": np.nan, "std": np.nan, "note": str(exc)[:200]})

    X50 = prepare_features_50(active, str(BINDING_MATRIX_PATH))
    ap_mean, ap_std, roc_mean, _oof = _cv_auc(X50, y, groups)
    rows.append({"config": "peptide_grouped_splitter", "metric": "mode50_grouped_auc_pr", "value": ap_mean, "std": ap_std})
    return rows


def _vaccinia_ablation(active: pd.DataFrame) -> list[dict]:
    """Peptide-grouped mode-31 AUC-PR/AUC-ROC with the out-of-panel Orthopoxvirus vaccinia
    negative bloc excluded. Those rows are real IEDB assay negatives, not decoys (D19)."""
    mask = (active["virus"] != "Orthopoxvirus vaccinia").to_numpy()
    subset = active[mask].reset_index(drop=True)
    X = prepare_features_31(subset, str(BINDING_MATRIX_PATH))
    y = subset["label"].astype(int).to_numpy()
    groups = subset["peptide"].to_numpy()
    ap_mean, ap_std, roc_mean, _oof = _cv_auc(X, y, groups)
    return [
        {"config": "peptide_grouped_splitter_no_vaccinia", "metric": "mode31_auc_pr", "value": ap_mean, "std": ap_std, "n_rows": len(subset)},
        {"config": "peptide_grouped_splitter_no_vaccinia", "metric": "mode31_auc_roc", "value": roc_mean, "std": np.nan, "n_rows": len(subset)},
    ]


# ---------------------------------------------------------------------------
# Near-homolog leakage arm
# ---------------------------------------------------------------------------
# The relation is declared here, ahead of any model fit, and this arm reports
# EXPOSURE only: how much of the corpus is related, and how much of that
# relation survives the peptide-grouped splitter. It fits nothing and tunes
# nothing, so no threshold can be chosen after seeing the AUC it would produce.
#
# Two peptides are near-homologs when any of three edge types holds:
#
#   containment  one is a contiguous sub-window of the other
#   hamming1     equal length, differing at exactly one position
#   indel1       one is the other with exactly one residue deleted
#
# Grounds, from docs/claims_register.md D22: a pool-internal substring scan of
# the 704-peptide Tier A pool found 185 overlapping pairs over 226 distinct
# peptides (32.1%), at length differences of 1 to 3 residues and 77.8%
# same-label concordance - the signature of sliding-window / registration
# boundary variants of one underlying epitope rather than spurious short-string
# matches. D22 measured that inside the 704-peptide Tier A pool; it states no
# figure for v5, which is the corpus this arm measures. It is the CONSERVATIVE
# end of D22's evidence: D22 saw length differences up to 3 residues and only
# the 1-residue edge types are admitted here, so what this arm reports is a
# floor, not an estimate.
#
# Cost. The active v5 corpus is 8mers to 11mers (measured, not assumed - the
# sub-window scan reads its widths off the data), so all three edge types are
# bounded rather than all-pairs. Containment is a sub-window lookup against the
# distinct-peptide set, capped at the lengths that actually occur in it; indel1
# is the same lookup over each peptide's single-residue deletions; hamming1 is
# positional-wildcard bucketing, where two equal-length peptides share a bucket
# key exactly when they differ at that one position, so no candidate pair needs
# re-verification. n^2/2 never appears.


def _near_homolog_adjacency(peptides: Iterable[str]) -> dict[str, frozenset[str]]:
    """Map each distinct peptide to the distinct peptides it is a near-homolog of.

    Self-edges are never emitted and the relation is symmetric by
    construction. An exact repeat of a peptide is not a near-homolog of
    itself; that is the exact-duplicate channel `_fold_overlap` already
    measures. A peptide with no near-homolog is ABSENT from the mapping
    rather than carrying an empty set, so len() of the result is directly the
    "has at least one near-homolog" count.
    """
    distinct = sorted({str(p) for p in peptides})
    members = set(distinct)
    lengths = sorted({len(p) for p in distinct})
    adjacency: dict[str, set[str]] = defaultdict(set)

    def _link(a: str, b: str) -> None:
        if a != b:
            adjacency[a].add(b)
            adjacency[b].add(a)

    for peptide in distinct:
        n = len(peptide)
        # containment: contiguous sub-windows, only at lengths the corpus uses
        for width in lengths:
            if width >= n:
                break
            for start in range(n - width + 1):
                window = peptide[start : start + width]
                if window in members:
                    _link(peptide, window)
        # indel1: delete exactly one residue. Deleting a terminal residue is
        # also a containment edge; the union makes that overlap harmless.
        for i in range(n):
            shortened = peptide[:i] + peptide[i + 1 :]
            if shortened in members:
                _link(peptide, shortened)

    # hamming1: positional wildcard buckets. Two DISTINCT peptides share the
    # key (i, prefix, suffix) exactly when they are the same length and differ
    # at position i alone.
    buckets: dict[tuple[int, str, str], list[str]] = defaultdict(list)
    for peptide in distinct:
        for i in range(len(peptide)):
            buckets[(i, peptide[:i], peptide[i + 1 :])].append(peptide)
    for bucket in buckets.values():
        if len(bucket) < 2:
            continue
        for a_i in range(len(bucket)):
            for b_i in range(a_i + 1, len(bucket)):
                _link(bucket[a_i], bucket[b_i])

    return {p: frozenset(n) for p, n in adjacency.items() if n}


def _homology_clusters(
    distinct: Sequence[str], adjacency: dict[str, frozenset[str]]
) -> dict[str, str]:
    """Union-find the near-homolog graph; return peptide -> cluster representative.

    Reported because the audit's operative question is whether a
    homology-grouped splitter would remain CONSTRUCTIBLE: a relation that
    collapses the corpus into one giant component cannot serve as a fold
    group, however well it describes the biology.
    """
    parent = {peptide: peptide for peptide in distinct}

    def _find(x: str) -> str:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for peptide, neighbours in adjacency.items():
        for neighbour in neighbours:
            a, b = _find(peptide), _find(neighbour)
            if a != b:
                parent[a] = b

    return {peptide: _find(peptide) for peptide in distinct}


def _near_homolog_fold_rows(
    config: str,
    peptides: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    adjacency: dict[str, frozenset[str]],
) -> list[dict]:
    """Per-fold and overall share of held-out ROWS with a near-homolog in their own train fold.

    Rows, not peptides: a peptide held out in fold k contributes once per row
    carrying it, because the row is the unit every fold metric in this file is
    computed over. `overall_exact_peptide_overlap_pct` is emitted alongside as
    a control - it must be 0.0 for any genuinely peptide-grouped splitter, so a
    non-zero value there means the split was not grouped and the near-homolog
    figure beside it is answering a different question than its name claims.
    """
    rows: list[dict] = []
    total_test = 0
    total_leak = 0
    total_exact = 0
    for fold_id, (train_idx, test_idx) in enumerate(splits):
        train_peptides = set(peptides[train_idx])
        leaked = 0
        exact = 0
        for i in test_idx:
            peptide = peptides[i]
            if peptide in train_peptides:
                exact += 1
            neighbours = adjacency.get(peptide)
            if neighbours is not None and not train_peptides.isdisjoint(neighbours):
                leaked += 1
        rows.append(
            {
                "config": config,
                "metric": f"fold{fold_id}_near_homolog_row_pct",
                "value": 100.0 * leaked / len(test_idx) if len(test_idx) else np.nan,
                "std": np.nan,
                "n_test": len(test_idx),
                "n_leaked": leaked,
            }
        )
        total_test += len(test_idx)
        total_leak += leaked
        total_exact += exact
    rows.append(
        {
            "config": config,
            "metric": "overall_near_homolog_row_pct",
            "value": 100.0 * total_leak / total_test if total_test else np.nan,
            "std": np.nan,
            "n_test": total_test,
            "n_leaked": total_leak,
        }
    )
    rows.append(
        {
            "config": config,
            "metric": "overall_exact_peptide_overlap_pct",
            "value": 100.0 * total_exact / total_test if total_test else np.nan,
            "std": np.nan,
            "n_test": total_test,
            "n_leaked": total_exact,
        }
    )
    return rows


def _homology_ab(active: pd.DataFrame) -> list[dict]:
    """Near-homolog exposure of the active corpus, and what survives peptide-grouped CV.

    TWO grouped splitters are reported, because this file already carries two
    and they are not interchangeable:

      peptide_grouped_splitter     bare StratifiedGroupKFold stratified on the
                                   label alone, as `_cv_auc` constructs it
      production_grouped_splitter  src.ml_utils.PeptideGroupedKFold with the
                                   full negative_origin / hla_allele / peptide
                                   composite key, as src/train_classifier.py's
                                   _cross_validate constructs it

    Both keep every row carrying a given peptide in one fold, so both drive the
    exact-peptide control to zero; they differ in stratification, so fold
    membership - and with it the near-homolog figure - can differ between them.
    Name the arm when citing a number from here, never "the" grouped splitter.
    """
    peptides = active["peptide"].astype(str).to_numpy()
    y = active["label"].astype(int)
    distinct = sorted(set(peptides))

    started = time.perf_counter()
    adjacency = _near_homolog_adjacency(distinct)
    clusters = _homology_clusters(distinct, adjacency)
    elapsed = time.perf_counter() - started

    sizes = Counter(clusters.values())
    largest_root, largest_size = (
        sizes.most_common(1)[0] if sizes else ("", 0)
    )
    largest_rows = int(sum(1 for p in peptides if clusters.get(p) == largest_root))
    # Wall-clock is deliberately printed and NOT written to the CSV: the CSV is
    # hash-bound by the provenance sidecar and consumed as a reproducible
    # artifact, and a timing would change on every run and every machine.
    print(
        f"near-homolog relation: {len(distinct)} distinct peptides, "
        f"{len(adjacency)} with >=1 near-homolog, {len(sizes)} clusters, "
        f"built in {elapsed:.2f}s"
    )

    n_edges = sum(len(v) for v in adjacency.values()) // 2
    rows: list[dict] = [
        {"config": "homology_relation", "metric": "distinct_peptides", "value": float(len(distinct))},
        {"config": "homology_relation", "metric": "peptides_with_near_homolog", "value": float(len(adjacency))},
        {
            "config": "homology_relation",
            "metric": "peptides_with_near_homolog_pct",
            "value": 100.0 * len(adjacency) / len(distinct) if distinct else np.nan,
        },
        {"config": "homology_relation", "metric": "near_homolog_edges", "value": float(n_edges)},
        {"config": "homology_relation", "metric": "n_clusters", "value": float(len(sizes))},
        {
            "config": "homology_relation",
            "metric": "singleton_clusters",
            "value": float(sum(1 for s in sizes.values() if s == 1)),
        },
        {"config": "homology_relation", "metric": "largest_cluster_peptides", "value": float(largest_size)},
        {
            "config": "homology_relation",
            "metric": "largest_cluster_active_rows_pct",
            "value": 100.0 * largest_rows / len(peptides) if len(peptides) else np.nan,
            "n_rows": largest_rows,
        },
    ]

    bare = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows += _near_homolog_fold_rows(
        "peptide_grouped_splitter",
        peptides,
        list(bare.split(np.zeros((len(active), 1)), y.to_numpy(), groups=peptides)),
        adjacency,
    )
    production = PeptideGroupedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows += _near_homolog_fold_rows(
        "production_grouped_splitter",
        peptides,
        list(
            production.split(
                np.zeros((len(active), 1)),
                y,
                negative_origin=active.get("negative_origin"),
                hla_alleles=active.get("hla_allele"),
                peptides=active["peptide"],
            )
        ),
        adjacency,
    )
    return rows


def _dataset_shape(active: pd.DataFrame, dataset_path: Path) -> list[dict]:
    full = pd.read_csv(dataset_path, low_memory=False)
    dup_counts = active["peptide"].value_counts()
    dup_rows = int(dup_counts[dup_counts > 1].sum())
    vaccinia_rows = int((active["virus"] == "Orthopoxvirus vaccinia").sum())
    return [
        {"config": "dataset_shape", "metric": "total_rows", "value": len(full)},
        {"config": "dataset_shape", "metric": "active_rows", "value": len(active)},
        {"config": "dataset_shape", "metric": "unique_peptides", "value": active["peptide"].nunique()},
        {"config": "dataset_shape", "metric": "dup_peptide_rows", "value": dup_rows},
        {"config": "dataset_shape", "metric": "dup_peptide_rows_pct", "value": 100.0 * dup_rows / len(active)},
        {"config": "dataset_shape", "metric": "vaccinia_active_rows", "value": vaccinia_rows},
        {"config": "dataset_shape", "metric": "vaccinia_pct_of_negatives", "value": 100.0 * vaccinia_rows / int((active["label"] == 0).sum())},
    ]


def run(dataset_path: Path) -> pd.DataFrame:
    active = _load_active(dataset_path)
    rows: list[dict] = []
    rows += _dataset_shape(active, dataset_path)
    rows += _fold_overlap(active)
    rows += _splitter_ab(active)
    rows += _per_virus_and_pooled_honest_ab(active)
    rows += _tier_a_ab(active)
    rows += _feature_mode_ab(active)
    rows += _vaccinia_ablation(active)
    rows += _homology_ab(active)
    return pd.DataFrame(rows)


def _provenance_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".provenance.json")


def _planned_paths(output_path: Path) -> list[str]:
    return planned_paths_under(
        str(output_path.parent), [output_path.name, _provenance_path(output_path).name]
    )


def _feature_input_provenance() -> dict[str, dict[str, str]]:
    """Digest every feature input this audit reads, not just the corpus.

    The sidecar recorded dataset_sha256 alone, so a run was reproducible in its
    ROWS but not in its FEATURES. That gap is not hypothetical: on 2026-09-04 a
    stale binding matrix was found to reach 0.79% of the v5 corpus's
    tested_negative rows, turning "every bind_ column is zero" into a label proxy
    worth roughly -0.154 pooled AUC-PR, and no artifact of the affected runs
    recorded which matrix had been opened. The self-similarity and antigen
    processing caches carry the same exposure: both fill absent peptides with an
    in-scale value, so their coverage is a silent input to modes 33 and 35.

    Only the binding matrix is read unconditionally by run(). The two mode-35
    caches are read at the prepare_features_35 call alone, inside a try/except
    that records the failure and continues, so the audit can reach this point
    with one of them absent. self_similarity_cache.csv is additionally
    gitignored and untracked, so that is the normal state of a fresh clone, not
    an edge case. Digesting it unguarded would abort a run that had already
    completed all of its work.

    An absent input is therefore recorded as absent rather than raising, and
    rather than being omitted: a missing key would be indistinguishable from a
    schema change, while "status": "absent" says which input was not read.
    """
    provenance: dict[str, dict[str, str]] = {}
    for name, path in (
        ("binding_matrix", BINDING_MATRIX_PATH),
        ("antigen_processing_cache", ANTIGEN_PROCESSING_CACHE_PATH),
        ("self_similarity_cache", SELF_SIMILARITY_CACHE_PATH),
    ):
        record = {"path": path.resolve().relative_to(PROJECT_ROOT).as_posix()}
        if path.exists():
            record["status"] = "present"
            record["sha256"] = _sha256_file(path)
        else:
            record["status"] = "absent"
        provenance[name] = record
    return provenance


def _write_provenance(output_path: Path, dataset_path: Path, n_rows: int) -> None:
    try:
        dataset_rel = dataset_path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        dataset_rel = dataset_path.name
    try:
        out_rel = output_path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        out_rel = output_path.name
    provenance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/audit_cv_leakage.py",
        # The artifact's OWN hash and an explicit pointer to it. Recording only
        # the INPUT hash (dataset_sha256, below) leaves the integrity harness's
        # provenance check with nothing to verify - it looks for `sha256` and
        # resolves the artifact from `artifact`/`file`. Added 2026-08-12 with the
        # same fix on scripts/assess_calibration.py. The committed sidecar still
        # lacks these fields and will keep reporting SKIP (no-checksum) until
        # this audit is next regenerated, which is deliberate: regenerating it
        # means retraining, and this change exists so that regeneration does not
        # silently re-create the gap.
        "artifact": out_rel,
        "sha256": _sha256_file(output_path),
        "dataset_path": dataset_rel,
        "dataset_sha256": _sha256_file(dataset_path),
        # The corpus alone does not determine the features. See
        # _feature_input_provenance for why each of these is recorded.
        "feature_inputs": _feature_input_provenance(),
        "random_state": RANDOM_STATE,
        "n_splits": N_SPLITS,
        "n_estimators": N_ESTIMATORS,
        "n_rows_emitted": n_rows,
    }
    provenance_path = _provenance_path(output_path)
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"wrote {provenance_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DATASET_PATH))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "results" / "cv_leakage_audit.csv"))
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Replace the audit CSV and its provenance sidecar at --out if they already "
        "exist. Without this flag the run aborts before any work if it would be overwritten.",
    )
    args = parser.parse_args(argv)

    dataset_path = Path(args.dataset)
    output_path = Path(args.out)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / args.out

    guard_planned_paths(
        str(output_path.parent),
        _planned_paths(output_path),
        args.allow_overwrite,
        flag="--out",
        api_hint="main(['--allow-overwrite'])",
        scope="among this run's planned artifacts",
        remedy="Point --out at a fresh path, ",
        detail=": these numbers are cited by docs/proposals/2026_feature_upgrade_roadmap.md",
    )

    result = run(dataset_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    _write_provenance(output_path, dataset_path, len(result))
    print(f"wrote {output_path} ({len(result)} rows)")

    overall = result[result["metric"] == "overall_peptide_overlap_pct"]["value"]
    prod = result[(result["config"] == "production_splitter") & (result["metric"] == "mode31_auc_pr")]["value"]
    grp = result[(result["config"] == "peptide_grouped_splitter") & (result["metric"] == "mode31_auc_pr")]["value"]
    if len(overall) and len(prod) and len(grp):
        print(
            f"overall peptide overlap: {overall.iloc[0]:.1f}% | "
            f"production AUC-PR {prod.iloc[0]:.4f} vs peptide-grouped AUC-PR {grp.iloc[0]:.4f} "
            f"(delta {prod.iloc[0] - grp.iloc[0]:+.4f})"
        )

    hom_pep = result[
        (result["config"] == "homology_relation")
        & (result["metric"] == "peptides_with_near_homolog_pct")
    ]["value"]
    hom_rows = result[
        (result["config"] == "production_grouped_splitter")
        & (result["metric"] == "overall_near_homolog_row_pct")
    ]["value"]
    if len(hom_pep) and len(hom_rows):
        print(
            f"near-homolog peptides: {hom_pep.iloc[0]:.1f}% | held-out rows with a "
            f"near-homolog in their own train fold (production_grouped_splitter): "
            f"{hom_rows.iloc[0]:.2f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
