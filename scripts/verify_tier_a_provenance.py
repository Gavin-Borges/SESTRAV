#!/usr/bin/env python3
"""Verify which model and corpus produced the certified Tier A SESTRAV arm.

Background (docs/claims_register.md D16): `results/table3_tier_a_metrics.csv` reports
SESTRAV RF AUC-PR 0.828 / AUC-ROC 0.7255 / ISSR@10 0.8429 over an n=704 field, and
`README.md` presents that as the canonical v5 `mode_31` result, calling it "weighted OOF".
It is none of those things. This script is the evidence, so the register cites a
reproducible check rather than an assertion.

The two decisive facts do NOT depend on re-running the model:

  1. `feature_mode=31` did not exist when the column was created. `results/
     external_validation_input.csv` has exactly one commit in history, `f360b90`
     (2026-05-23); at that commit `src/train_classifier.py` declares
     `--feature-mode ... choices=[21, 30, 50, 166]` and contains no
     `prepare_features_31`. That helper first appears at `27cdc61` (2026-06-18),
     26 days later. So 0.828 cannot be a mode-31 measurement - it is mode-30.

  2. All 704 stored scores are exact multiples of 1/200, against 362/704 for 1/500.
     That fingerprints `n_estimators=200`, not the 500 the v3 model card documents.

The corpus is the 720-row root `immunogenicity_dataset.csv` at `69e0e5c`; the field is
exactly 720 minus the 16 `GOLD_STANDARD_EPITOPES` = 704. That file was deleted from the
working tree at `ec9aba0` (2026-06-03) and is not tracked at HEAD, but `69e0e5c` is an
ancestor of `main`, so it stays recoverable from a clean clone.

The reproduction arm below is corroborating, not decisive: an independent run of
(720-row corpus, prepare_features_30, StratifiedKFold seed 42, RF n_estimators=200,
class_weight=balanced, unweighted) has been reported bit-exact against all three certified
cells, while a re-run in a different environment landed close but not exact. Both are
printed so the reader sees the spread rather than a single asserted number.

The consequence worth carrying: the v5 mode-31 production model has never been evaluated
on the full Tier A field and cannot be, because 290 of its 704 peptides are unreachable in
the active v5 corpus. Stated precisely (corrected 2026-08-09): 468 of the 704 exist somewhere
in v5 and 236 exist in neither v4 nor v5 (none are v4-only); only 414 resolve to an active,
non-quarantined v5 row, the other 54 appearing solely in quarantined rows. 236 + 54 = 290.
An earlier version of this docstring, and of claims_register D16, paired "414 exist in v5"
with "236 in neither", which is arithmetically impossible (704 - 414 = 290).
Re-running Tier A under v5 produces a different, smaller field - a replacement, not a refresh.

This is a read-only verifier: it prints a comparison and writes nothing. It is deliberately
NOT an artifact generator, so it does not widen the tracked `results/` surface.

Reproduce:  python scripts/verify_tier_a_provenance.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate_metrics import evaluate  # noqa: E402
from src.train_classifier import prepare_features_30  # noqa: E402

RANDOM_STATE = 42
N_SPLITS = 5
# 200 is the fingerprinted value (all 704 stored scores are exact multiples of 1/200).
# 500 is what the v3 model card documents and is retained only to show it does NOT fit.
ESTIMATOR_SETTINGS = (200, 500)
# The corpus is the 720-row root dataset, deleted from the working tree at ec9aba0 but
# recoverable because 69e0e5c is an ancestor of main.
TIER_A_CORPUS_REV = "69e0e5c"
TIER_A_CORPUS_PATH_AT_REV = "immunogenicity_dataset.csv"

# Certified cells from results/table3_tier_a_metrics.csv (SESTRAV RF row).
CERTIFIED = {"auc_pr": 0.827767, "auc_roc": 0.725505, "issr_10": 0.842857}

V3_PATH = PROJECT_ROOT / "data" / "immunogenicity_dataset_v3.csv"
V4_PATH = PROJECT_ROOT / "data" / "immunogenicity_dataset_v4.csv"
V5_PATH = PROJECT_ROOT / "data" / "immunogenicity_dataset_v5.csv"
BINDING_MATRIX_PATH = PROJECT_ROOT / "models" / "peptide_binding_matrix.csv"
TIER_A_PATH = PROJECT_ROOT / "results" / "external_validation_input.csv"
V5_OOF_PATH = PROJECT_ROOT / "models" / "rf_oof_predictions.csv"


REPO_ROOT_GIT = PROJECT_ROOT


def _tier_a_corpus() -> pd.DataFrame | None:
    """The 720-row root corpus, read out of git history (not tracked at HEAD)."""
    import subprocess
    from io import StringIO

    try:
        blob = subprocess.run(
            ["git", "show", f"{TIER_A_CORPUS_REV}:{TIER_A_CORPUS_PATH_AT_REV}"],
            cwd=REPO_ROOT_GIT, capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return pd.read_csv(StringIO(blob))


def _peptides(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return set(pd.read_csv(path, low_memory=False)["peptide"])


def _tier_a_field() -> pd.DataFrame:
    field = pd.read_csv(TIER_A_PATH)
    if "tier_a_baseline_complete" in field.columns:
        field = field[field["tier_a_baseline_complete"].astype(bool)]
    return field.reset_index(drop=True)


def report_membership(field: pd.DataFrame) -> None:
    peptides = set(field["peptide"])
    v3, v4, v5 = _peptides(V3_PATH), _peptides(V4_PATH), _peptides(V5_PATH)
    print(f"Tier A certified field: {len(field)} rows, {len(peptides)} unique peptides")
    print(f"  resolvable against v3 : {len(peptides & v3):>4}  (tracked)")
    print(f"  resolvable against v4 : {len(peptides & v4):>4}  (gitignored)")
    print(f"  resolvable against v5 : {len(peptides & v5):>4}")
    print(f"  in NEITHER v4 nor v5  : {len(peptides - v4 - v5):>4}")
    corpus = _tier_a_corpus()
    if corpus is not None:
        print(f"  720-row root corpus @ {TIER_A_CORPUS_REV}: {len(corpus)} rows, "
              f"{corpus['peptide'].nunique()} unique peptides")
        print(f"  field == corpus minus the 16 GOLD_STANDARD_EPITOPES: {len(corpus) - 16} == {len(peptides)}")
    if v3 and peptides <= v3:
        print("  note: v3 is a superset of the field, but it is NOT the training corpus - see the")
        print("        feature-mode and estimator evidence below.")


def report_history_evidence() -> None:
    """The two facts that settle provenance without re-running the model."""
    import subprocess

    print("\n--- decisive evidence (no model run required) ---")
    try:
        commits = subprocess.run(
            ["git", "log", "--follow", "--diff-filter=A", "--format=%h %ci",
             "--", "results/external_validation_input.csv"],
            cwd=REPO_ROOT_GIT, capture_output=True, text=True, check=True,
        ).stdout.strip()
        print(f"1. external_validation_input.csv creation commit(s): {commits or '(none)'}")
        src = subprocess.run(
            ["git", "show", "f360b90:src/train_classifier.py"],
            cwd=REPO_ROOT_GIT, capture_output=True, text=True, check=True,
        ).stdout
        choices = [ln.strip() for ln in src.splitlines() if "--feature-mode" in ln and "choices" in ln]
        print(f"   at f360b90: {choices[0] if choices else '(feature-mode arg not found)'}")
        print(f"   prepare_features_31 occurrences at f360b90: {src.count('prepare_features_31')}"
              "  -> mode 31 did not exist, so 0.828 is not a mode-31 figure")
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"1. git history check unavailable: {exc}")

    field = _tier_a_field()
    stored = field["rf_oof_score"].to_numpy()
    print("2. n_estimators fingerprint (stored scores should be exact multiples of 1/n):")
    for n in (200, 500):
        exact = int((np.abs(stored * n - np.round(stored * n)) < 1e-9).sum())
        print(f"   1/{n}: {exact}/{len(stored)} exact")


def report_v5_disagreement(field: pd.DataFrame) -> None:
    if not V5_OOF_PATH.is_file():
        print("\n[v5 OOF comparison skipped: models/rf_oof_predictions.csv absent]")
        return
    oof = pd.read_csv(V5_OOF_PATH)
    if "method" in oof.columns and (oof["method"] == "RandomForest").any():
        oof = oof[oof["method"] == "RandomForest"]
    current = oof.drop_duplicates(subset=["peptide"], keep="first")[["peptide", "score"]]
    merged = field[["peptide", "rf_oof_score"]].merge(current, on="peptide", how="inner")
    if merged.empty:
        print("\nv5 OOF overlap with the Tier A field: none")
        return
    delta = (merged["rf_oof_score"] - merged["score"]).abs()
    print(f"\nStored column vs CURRENT v5 OOF: matched {len(merged)}/{len(field)} peptides")
    print(f"  identical (<1e-9): {int((delta < 1e-9).sum())}/{len(merged)}   mean abs diff: {delta.mean():.4f}")


def report_reproduction(field: pd.DataFrame) -> None:
    """Corroborating arm: re-run the era-faithful configuration and show the spread."""
    df = _tier_a_corpus()
    if df is None:
        print("\n[reproduction skipped: cannot read the 720-row corpus from git history]")
        return
    X = prepare_features_30(df, str(BINDING_MATRIX_PATH))
    y = df["label"].astype(int).to_numpy()

    for n_estimators in ESTIMATOR_SETTINGS:
        oof = np.full(len(y), np.nan)
        splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        for train_idx, test_idx in splitter.split(X, y):
            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced",
            )
            clf.fit(X.iloc[train_idx], y[train_idx])
            oof[test_idx] = clf.predict_proba(X.iloc[test_idx])[:, 1]

        scored = pd.DataFrame(
            {"peptide": df["peptide"].to_numpy(), "score": oof}
        ).drop_duplicates(subset=["peptide"], keep="first")
        merged = field[["peptide", "label", "rf_oof_score"]].merge(scored, on="peptide", how="left")
        resolved = merged.dropna(subset=["score"])
        if resolved.empty:
            print(f"\nmode-30 re-run (n_estimators={n_estimators}): no Tier A peptide resolved")
            continue

        metrics = evaluate(resolved["label"].to_numpy(), resolved["score"].to_numpy())
        delta = (resolved["rf_oof_score"] - resolved["score"]).abs()
        print(
            f"\nmode-30 re-run on the 720-row corpus, n_estimators={n_estimators}: "
            f"coverage {len(resolved)}/{len(merged)}"
        )
        for key in ("auc_pr", "auc_roc", "issr_10"):
            print(
                f"  {key:8} {metrics[key]:.4f}   certified {CERTIFIED[key]:.4f}   "
                f"delta {metrics[key] - CERTIFIED[key]:+.4f}"
            )
        print(f"  mean abs diff vs stored column: {delta.mean():.4f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-reproduction",
        action="store_true",
        help="Report field membership and the v5 disagreement only, without retraining on v3.",
    )
    args = parser.parse_args(argv)

    for required in (V3_PATH, BINDING_MATRIX_PATH, TIER_A_PATH):
        if not required.is_file():
            print(f"Missing required tracked input: {required}", file=sys.stderr)
            return 1

    report_membership(_tier_a_field())
    report_history_evidence()
    report_v5_disagreement(_tier_a_field())
    if not args.skip_reproduction:
        report_reproduction(_tier_a_field())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
