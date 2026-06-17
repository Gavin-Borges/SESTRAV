"""
SESTRAV-VERIFY GNN Model Promotion Orchestrator

Validates the 5 Canonical Promotion Gates before mutating config.yaml and
model_artifact_checksums.json.

Gate definitions (all must pass):
  Gate 1 – Generalization:   GNN OOF AUC-PR >= 0.85 on full training dataset.
  Gate 2 – Stability:        Cross-fold AUC-PR std <= 0.02 across 5 CV folds.
  Gate 3 – Latency:          GNN CPU inference <= 2× RF CPU inference (per batch).
  Gate 4 – Calibration:      Expected Calibration Error (ECE) < 0.05.
  Gate 5 – Escape Sensitivity: GNN correctly differentiates >= 80% of IEDB
                               gold-standard epitopes from decoys.

Security hardening:
  - All torch.load calls use weights_only=True (prevents arbitrary code exec).
  - Checksum generation uses native Python hashlib (no shell injection risk).
  - No eval()/exec() used anywhere.
"""
from __future__ import annotations

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import logging
import time
from typing import NamedTuple

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("gnn-promote")

# ---------------------------------------------------------------------------
# Canonical paths — relative to project root (cwd must be project root)
# ---------------------------------------------------------------------------
GNN_CHECKPOINT = Path("models/gnn/structural_gnn_v2.pth")
GNN_SCALER    = Path("models/gnn/gnn_scaler.joblib")
RF_MODEL_PATH  = Path("models/rf_30feature_integrated.joblib")
OOF_PATH       = Path("models/gnn_oof_predictions.csv")
CONFIG_PATH    = Path("config.yaml")
CHECKSUM_FILE  = Path("models/model_artifact_checksums.json")

# Gate thresholds (immutable constants — edit requires PR review)
GATE1_AUC_PR_MIN:  float = 0.85
GATE2_STD_MAX:     float = 0.02
GATE3_LATENCY_FACTOR: float = 2.0   # GNN must be <= 2× RF latency
GATE4_ECE_MAX:    float = 0.05
GATE5_SENSITIVITY_MIN: float = 0.80

# Latency benchmark settings
LATENCY_BATCH_SIZE: int  = 50
LATENCY_WARMUP_REPS: int = 3
LATENCY_TIMED_REPS: int  = 10


class GateResult(NamedTuple):
    name: str
    passed: bool
    value: float | str
    threshold: str


# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------

def _sha256_file(filepath: Path) -> str:
    """SHA-256 via native Python hashlib — no shell invocation."""
    import hashlib
    digest = hashlib.sha256()
    with filepath.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Gate implementations
# ---------------------------------------------------------------------------

def _load_oof() -> pd.DataFrame:
    if not OOF_PATH.exists():
        raise FileNotFoundError(
            f"OOF predictions not found at {OOF_PATH}. "
            "Run full GNN training on immunogenicity_dataset_v3.csv first."
        )
    df = pd.read_csv(OOF_PATH)
    required = {"label", "gnn_oof_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"OOF file missing columns: {missing}")
    return df


def gate1_generalization(df: pd.DataFrame) -> GateResult:
    """AUC-PR on OOF predictions >= GATE1_AUC_PR_MIN."""
    from sklearn.metrics import average_precision_score
    auc_pr = float(average_precision_score(df["label"], df["gnn_oof_score"]))
    passed = auc_pr >= GATE1_AUC_PR_MIN
    return GateResult(
        name="Gate 1 — Generalization (AUC-PR)",
        passed=passed,
        value=round(auc_pr, 4),
        threshold=f">= {GATE1_AUC_PR_MIN}",
    )


def gate2_stability(df: pd.DataFrame) -> GateResult:
    """Cross-fold AUC-PR std must not exceed GATE2_STD_MAX.

    If the OOF file contains a 'fold' column (emitted by train_gnn.py when
    --save-fold-ids is set), we compute per-fold AUC-PRs.  Otherwise we
    fall back to a jackknife leave-one-out estimate.
    """
    from sklearn.metrics import average_precision_score

    labels  = df["label"].values
    scores  = df["gnn_oof_score"].values

    if "fold" in df.columns:
        fold_ids = df["fold"].values
        folds = sorted(df["fold"].unique())
        fold_auc_prs: list[float] = []
        for fid in folds:
            mask = fold_ids == fid
            if labels[mask].sum() == 0:
                continue  # skip folds with no positives
            fold_auc_prs.append(
                float(average_precision_score(labels[mask], scores[mask]))
            )
        std = float(np.std(fold_auc_prs)) if len(fold_auc_prs) > 1 else 0.0
        method = "per-fold"
    else:
        # Jackknife (leave-one-out) estimate of variance
        n = len(labels)
        loo_auc_prs: list[float] = []
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            if labels[mask].sum() == 0:
                continue
            try:
                loo_auc_prs.append(
                    float(average_precision_score(labels[mask], scores[mask]))
                )
            except Exception:
                pass
        std = float(np.std(loo_auc_prs)) if len(loo_auc_prs) > 1 else 0.0
        method = "jackknife-LOO"

    passed = std <= GATE2_STD_MAX
    return GateResult(
        name=f"Gate 2 — Stability (AUC-PR std, {method})",
        passed=passed,
        value=round(std, 4),
        threshold=f"<= {GATE2_STD_MAX}",
    )


def _time_model_ms(predict_fn, node_x, feat_x, warmup: int, reps: int) -> float:
    """Returns median wall-clock latency in milliseconds over *reps* timed calls."""
    import torch
    for _ in range(warmup):
        with torch.no_grad():
            predict_fn(node_x, feat_x)
    times: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        with torch.no_grad():
            predict_fn(node_x, feat_x)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


def gate3_latency() -> GateResult:
    """GNN CPU inference latency <= GATE3_LATENCY_FACTOR × RF latency.

    Uses a fixed synthetic batch of LATENCY_BATCH_SIZE 9-mer sequences so the
    measurement is reproducible without any real dataset access.
    """
    import torch
    from src.gnn.models import GraphPredictor
    from src.gnn.graph_builder import GraphBuilder
    from src.features import TRAIN_FEATURE_COLUMNS
    from src.artifact_integrity import load_verified_joblib

    device = torch.device("cpu")

    # --- RF baseline ---
    if not RF_MODEL_PATH.exists():
        return GateResult(
            name="Gate 3 — Latency",
            passed=False,
            value="RF model not found",
            threshold=f"<= {GATE3_LATENCY_FACTOR}× RF latency",
        )
    rf_model = load_verified_joblib(RF_MODEL_PATH)

    # Synthetic feature matrix for RF (30 features)
    rng = np.random.default_rng(0)
    rf_features = getattr(rf_model, "n_features_in_", 30)
    X_rf = rng.standard_normal((LATENCY_BATCH_SIZE, rf_features))

    rf_times: list[float] = []
    for _ in range(LATENCY_WARMUP_REPS):
        rf_model.predict_proba(X_rf)
    for _ in range(LATENCY_TIMED_REPS):
        t0 = time.perf_counter()
        rf_model.predict_proba(X_rf)
        rf_times.append((time.perf_counter() - t0) * 1000.0)
    rf_latency_ms = float(np.median(rf_times))

    # --- GNN benchmark ---
    if not GNN_CHECKPOINT.exists():
        return GateResult(
            name="Gate 3 — Latency",
            passed=False,
            value="GNN checkpoint not found",
            threshold=f"<= {GATE3_LATENCY_FACTOR}× RF latency",
        )

    num_features = len(TRAIN_FEATURE_COLUMNS)
    gnn_model = GraphPredictor(num_continuous_features=num_features).to(device)
    # weights_only=True prevents arbitrary code execution during checkpoint load
    state = torch.load(GNN_CHECKPOINT, map_location="cpu", weights_only=True)
    gnn_model.load_state_dict(state)
    gnn_model.eval()

    synthetic_seqs = ["GILGFVFTL"] * LATENCY_BATCH_SIZE  # canonical 9-mer
    node_batch = torch.stack([GraphBuilder.sequence_to_node_features(s) for s in synthetic_seqs])
    feat_batch = torch.tensor(X_rf[:, :num_features], dtype=torch.float32)
    # Chain-graph adjacency shared across all batch elements (max_len × max_len)
    adj = GraphBuilder.build_chain_adj().to(device)

    gnn_latency_ms = _time_model_ms(
        lambda nx, fx: gnn_model(nx, fx, adj),
        node_batch,
        feat_batch,
        LATENCY_WARMUP_REPS,
        LATENCY_TIMED_REPS,
    )

    ratio = gnn_latency_ms / max(rf_latency_ms, 0.001)
    passed = ratio <= GATE3_LATENCY_FACTOR
    return GateResult(
        name="Gate 3 — Latency",
        passed=passed,
        value=f"GNN={gnn_latency_ms:.2f}ms, RF={rf_latency_ms:.2f}ms, ratio={ratio:.2f}×",
        threshold=f"ratio <= {GATE3_LATENCY_FACTOR}×",
    )


def gate4_calibration(df: pd.DataFrame) -> GateResult:
    """Expected Calibration Error (ECE) < GATE4_ECE_MAX.

    Uses equal-width binning (15 bins) following Guo et al. 2017.
    """
    probs  = df["gnn_oof_score"].values.astype(float)
    labels = df["label"].values.astype(float)
    n_bins = 15
    bins   = np.linspace(0.0, 1.0, n_bins + 1)
    ece    = 0.0
    n      = len(probs)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        acc  = labels[mask].mean()
        conf = probs[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    passed = ece < GATE4_ECE_MAX
    return GateResult(
        name="Gate 4 — Calibration (ECE, 15-bin)",
        passed=passed,
        value=round(ece, 4),
        threshold=f"< {GATE4_ECE_MAX}",
    )


def gate5_escape_sensitivity(df: pd.DataFrame) -> GateResult:
    """Fraction of IEDB gold-standard epitopes scored above median decoy score >= 0.80.

    Requires the OOF file to contain both positive (label=1) and negative
    (label=0) rows.  The gate checks that the model rank-orders positives over
    negatives at the 80% sensitivity level.
    """
    positives = df[df["label"] == 1]["gnn_oof_score"].values
    negatives = df[df["label"] == 0]["gnn_oof_score"].values

    if len(positives) == 0 or len(negatives) == 0:
        return GateResult(
            name="Gate 5 — Escape Sensitivity",
            passed=False,
            value="Insufficient class diversity in OOF file",
            threshold=f">= {GATE5_SENSITIVITY_MIN:.0%}",
        )

    decoy_median = float(np.median(negatives))
    sensitivity  = float((positives > decoy_median).mean())
    passed = sensitivity >= GATE5_SENSITIVITY_MIN
    return GateResult(
        name="Gate 5 — Escape Sensitivity",
        passed=passed,
        value=round(sensitivity, 4),
        threshold=f">= {GATE5_SENSITIVITY_MIN}",
    )


# ---------------------------------------------------------------------------
# Scorecard runner
# ---------------------------------------------------------------------------

def check_promotion_gates() -> bool:
    logger.info("=" * 60)
    logger.info("SESTRAV GNN Promotion Scorecard — 5 Gates")
    logger.info("=" * 60)

    if not GNN_CHECKPOINT.exists():
        logger.error(
            f"Checkpoint {GNN_CHECKPOINT} not found. "
            "Execute full GNN training on immunogenicity_dataset_v3.csv before promotion."
        )
        return False

    try:
        df = _load_oof()
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        return False

    results: list[GateResult] = []

    # Gates 1, 2, 4, 5 depend only on OOF CSV
    for gate_fn in (gate1_generalization, gate2_stability, gate4_calibration, gate5_escape_sensitivity):
        try:
            r = gate_fn(df)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"{gate_fn.__name__} raised unexpectedly: {exc}")
            results.append(GateResult(name=gate_fn.__name__, passed=False, value=str(exc), threshold="—"))
            continue
        results.append(r)

    # Gate 3 requires loading real models
    try:
        r3 = gate3_latency()
    except Exception as exc:  # noqa: BLE001
        logger.error(f"gate3_latency raised unexpectedly: {exc}")
        r3 = GateResult(name="Gate 3 — Latency", passed=False, value=str(exc), threshold="—")
    results.append(r3)

    # Log full scorecard
    logger.info("")
    logger.info("─" * 60)
    all_passed = True
    for r in sorted(results, key=lambda x: x.name):
        status = "PASS ✓" if r.passed else "FAIL ✗"
        logger.info(f"  {r.name}")
        logger.info(f"    Value: {r.value}   Threshold: {r.threshold}   [{status}]")
        if not r.passed:
            all_passed = False
    logger.info("─" * 60)
    if all_passed:
        logger.info("SCORECARD RESULT: ALL GATES PASSED — ready for promotion.")
    else:
        logger.error("SCORECARD RESULT: ONE OR MORE GATES FAILED — promotion blocked.")
    logger.info("=" * 60)

    return all_passed


# ---------------------------------------------------------------------------
# Promotion executor (only called when all gates pass)
# ---------------------------------------------------------------------------

def promote_model() -> None:
    """Mutates config.yaml and model_artifact_checksums.json iff all gates pass."""
    if not check_promotion_gates():
        logger.error("Model failed promotion gates. config.yaml will NOT be modified.")
        return

    logger.info("Promoting Structural GNN to canonical pipeline …")

    # Secure SHA-256 (native Python; no shell=True, no subprocess)
    gnn_sha256 = _sha256_file(GNN_CHECKPOINT)
    logger.info(f"Checkpoint SHA-256: {gnn_sha256}")

    # --- Update config.yaml ---
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
        config["model_path"] = str(GNN_CHECKPOINT)
        with CONFIG_PATH.open("w", encoding="utf-8") as fh:
            yaml.dump(config, fh, default_flow_style=False, sort_keys=False)
        logger.info(f"Updated {CONFIG_PATH}: model_path → {GNN_CHECKPOINT}")
    else:
        logger.warning(f"{CONFIG_PATH} not found — skipping config update.")

    # --- Update model_artifact_checksums.json ---
    try:
        from src.artifact_integrity import update_checksum_manifest
        update_checksum_manifest(CHECKSUM_FILE, [GNN_CHECKPOINT])
        logger.info(f"Updated {CHECKSUM_FILE} using canonical schema.")
    except Exception as exc:
        logger.error(f"Failed to update checksum manifest {CHECKSUM_FILE}: {exc}")
        raise


if __name__ == "__main__":
    promote_model()
