"""
SESTRAV-VERIFY GNN Model Promotion Orchestrator

Validates the 5 Canonical Promotion Gates before mutating config.yaml and
model_artifact_checksums.json.

Gate definitions (all must pass):
  Gate 1 - Generalization:   GNN OOF AUC-PR >= 0.65 on full training dataset,
                             scored under a peptide-grouped splitter
                             (re-anchored 2026-08-10; see GATE1_AUC_PR_MIN).
                             The splitter is a hard precondition, not a note:
                             see grouped_splitter_violation.
  Gate 2 - Stability:        Cross-fold AUC-PR std <= 0.02 across 5 CV folds.
  Gate 3 - Latency:          GNN CPU inference <= 2x RF CPU inference (per batch).
  Gate 4 - Calibration:      Expected Calibration Error (ECE) < 0.05.
  Gate 5 - Escape Sensitivity: >= 80% of ALL out-of-fold positives score above
                               the median out-of-fold negative, pool-wide.
                               Despite the name, this is NOT restricted to IEDB
                               gold-standard epitopes - both training entry
                               points mask them out of the pool before any CV
                               fold is built, so none reach this gate. See
                               docs/claims_register.md D26.

Run `python -m src.verify.promote_gnn --dry-run` to evaluate the scorecard
without touching config.yaml or the checksum manifest.

Standing result and re-run policy (recorded 2026-08-16; measured 2026-08-13)
---------------------------------------------------------------------------
The v5 GNN was evaluated against these gates and **returned a null result on the
pre-registered bar**, which is an AND-conjunction, so Gate 1 alone decides it:

  Gate 1 (pooled AUC-PR, peptide-grouped)  0.6458  vs >= 0.65   FAIL by 0.0042
  Gate 2 (cross-fold AUC-PR std)           0.0234  vs <= 0.02   FAIL
  Gates 3/4/5 (latency, ECE, escape)                            PASS

Underneath that null sits a real effect, and both halves must be reported
together or neither is honest: against RF mode-31 the GNN scored AUC-PR 0.6458
vs 0.6055, **delta +0.0402, 95% CI [0.0286, 0.0520], excludes zero, p < 0.0001**
(paired bootstrap, seed 20260813, 10,000 resamples, 35,555 rows matched 1:1).
So the architecture is measurably better on discrimination and still misses the
promotion bar. It is not promoted.

**Do NOT re-run this evaluation with different hyperparameters against the same
held-out set.** Tuning until a 0.0042 shortfall closes is the leakage this
project flags in every other model, and a bar that moves after seeing the result
is not a bar. This prohibition is written here, next to the gates it governs,
because it previously existed only as prose in local planning files that do not
survive a handoff - and it is exactly the rule most likely to be rationalised
away under deadline pressure.

What legitimately re-opens the track: a new bar pre-registered BEFORE the run,
evaluated on data not used to produce the result above (a fresh corpus or a
genuinely held-out cohort). A different architecture or feature set still needs
the pre-registration, because the failure mode being guarded against is
selection over repeated attempts, not any particular model.

Downstream consequence of a successful promotion - READ BEFORE PROMOTING.
Passing the five gates is necessary but NOT sufficient to ship: promotion
rewrites config.yaml's model_path to GNN_CHECKPOINT and does not relocate the
checkpoint, and the FastAPI service cannot serve that value.

  - ModelManager.load (api/main.py) passes config.model_path.name - the
    BASENAME only - to ModelRegistry.load, which calls
    ModelRegistry.resolve_model to rebuild the path as models/<basename>
    relative to cwd. The directory is discarded, so the lookup targets
    models/structural_gnn_v2.pth, which does not exist. resolve_model itself
    returns that path unconditionally (its only raise is a ValueError on
    directory escape); the FileNotFoundError comes from its caller,
    ModelRegistry.load, and propagates out of the unguarded startup lifespan
    handler. The API then fails to START - /health included. It does not
    degrade and it does not fall back, and the failure is not confined to
    /score.
  - Two further defects sit latent behind that one and would surface the
    moment the path resolved: ModelRegistry.load returns a raw state_dict for
    a .pth (no predict_proba), while _score_peptide builds a flat 31-float
    vector and GraphPredictorV2.forward expects a PyG Data/Batch carrying
    ESM-2 node features. There is no model-type dispatch anywhere in
    api/main.py, and its /model-card and ScoreResponse payloads hardcode
    RF-specific identifier strings.

Serving a promoted GNN therefore requires an api/main.py change. It is not a
config-only operation. Verified 2026-08-16; recorded here rather than in a
planning document because that is where it was lost the last time.

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

import argparse
import logging
import time
from typing import NamedTuple

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("gnn-promote")

# ---------------------------------------------------------------------------
# Canonical paths - relative to project root (cwd must be project root)
# ---------------------------------------------------------------------------
GNN_CHECKPOINT = Path("models/gnn/structural_gnn_v2.pth")
GNN_SCALER = Path("models/gnn/gnn_scaler.joblib")
GNN_CONFIG = Path("models/gnn/gnn_config.json")
RF_MODEL_PATH = Path("models/rf_31feature_integrated.joblib")
OOF_PATH = Path("models/gnn_oof_predictions.csv")
CONFIG_PATH = Path("config.yaml")
CHECKSUM_FILE = Path("models/model_artifact_checksums.json")

# Gate thresholds (immutable constants - edit requires PR review)
# Gate 1 re-anchored 2026-08-10 from 0.85 to 0.65. The 0.85 threshold was set
# against the pre-remediation ungrouped RF baseline (pooled AUC-PR 0.8312), which
# is retracted as peptide-leakage-inflated (docs/claims_register.md D15). Against
# the certified peptide-grouped RF baseline of 0.6058 it was unreachable rather
# than ambitious. A promotion candidate must be scored under a peptide-grouped
# splitter (src.ml_utils.PeptideGroupedKFold) for this comparison to be valid.
GATE1_AUC_PR_MIN: float = 0.65
GATE2_STD_MAX: float = 0.02
GATE3_LATENCY_FACTOR: float = 2.0  # GNN must be <= 2x RF latency
GATE4_ECE_MAX: float = 0.05
GATE5_SENSITIVITY_MIN: float = 0.80

# Gate 1 splitter precondition.
#
# GATE1_AUC_PR_MIN is only meaningful for a score produced under a
# peptide-grouped splitter. An OOF frame from an ungrouped run is not a weaker
# generalization estimate, it is a different quantity: mode-31 features are a
# pure function of the peptide string, so an ungrouped fold boundary leaves a
# held-out peptide's feature-identical twin in the training set. Comparing such
# a number against a threshold anchored on the peptide-grouped RF baseline
# (0.6058) is a category error, so the frame must carry positive evidence of
# its splitter. Absence of the marker fails the gate; it never waives it.
SPLITTER_COLUMN: str = "splitter"
GROUPED_SPLITTERS: frozenset[str] = frozenset({"PeptideGroupedKFold"})
FOLD_COLUMN: str = "fold"

# Latency benchmark settings
LATENCY_BATCH_SIZE: int = 50
LATENCY_WARMUP_REPS: int = 3
LATENCY_TIMED_REPS: int = 10


class GateResult(NamedTuple):
    name: str
    passed: bool
    value: float | str
    threshold: str


# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------


def _sha256_file(filepath: Path) -> str:
    """SHA-256 via native Python hashlib - no shell invocation."""
    import hashlib

    digest = hashlib.sha256()
    with filepath.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Gate implementations
# ---------------------------------------------------------------------------


def _load_oof(oof_path: Path | None = None) -> pd.DataFrame:
    """Load an OOF prediction frame, defaulting to the tracked OOF_PATH.

    `oof_path` is resolved at call time so a caller can score a scratch run
    without touching the tracked artifact. Passing None keeps the historical
    behaviour exactly, including for tests that patch the module constant.
    """
    path = OOF_PATH if oof_path is None else Path(oof_path)
    if not path.exists():
        raise FileNotFoundError(
            f"OOF predictions not found at {path}. "
            "Run full GNN training (src/train_gnn.py) on the current dataset first."
        )
    df = pd.read_csv(path)
    required = {"label", "gnn_oof_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"OOF file missing columns: {missing}")
    return df


def grouped_splitter_violation(df: pd.DataFrame) -> str | None:
    """Why this OOF frame is not demonstrably peptide-grouped, or None if it is.

    Positive evidence is required: the frame must carry a SPLITTER_COLUMN whose
    every value names a splitter in GROUPED_SPLITTERS. A frame with no marker is
    treated as ungrouped, because an unmarked frame is exactly the shape the
    pre-repair GNN track emitted.
    """
    if SPLITTER_COLUMN not in df.columns:
        return (
            f"the frame carries no '{SPLITTER_COLUMN}' column, so there is no evidence it "
            "was scored under a peptide-grouped splitter. Artifacts written before the "
            "peptide-grouping repair have exactly this shape "
            "(peptide,label,gnn_oof_score). Re-run src/train_gnn.py, which now splits "
            "with src.ml_utils.PeptideGroupedKFold and stamps every OOF row"
        )

    observed = sorted({str(v).strip() for v in df[SPLITTER_COLUMN].dropna().unique()})
    if not observed:
        return (
            f"the '{SPLITTER_COLUMN}' column is present but every value is null, which "
            "records nothing about how the folds were built"
        )

    ungrouped = [name for name in observed if name not in GROUPED_SPLITTERS]
    if ungrouped:
        accepted = ", ".join(sorted(GROUPED_SPLITTERS))
        return (
            f"the frame is marked {ungrouped}, which is not a peptide-grouped splitter "
            f"(accepted: {accepted}). Rows sharing a peptide can land on both sides of "
            "an ungrouped fold boundary, and every mode-31 feature is a pure function of "
            "the peptide string, so the resulting score is a memorization estimate "
            "(docs/claims_register.md D15)"
        )
    return None


def gate1_generalization(df: pd.DataFrame) -> GateResult:
    """AUC-PR on OOF predictions >= GATE1_AUC_PR_MIN, under a peptide-grouped splitter.

    The splitter precondition is checked FIRST and is not a warning: an
    unmarked or ungrouped frame fails the gate without an AUC-PR ever being
    reported, so a leakage-inflated number is never printed next to a threshold
    it was not measured against.
    """
    from sklearn.metrics import average_precision_score

    threshold = f">= {GATE1_AUC_PR_MIN} under {'/'.join(sorted(GROUPED_SPLITTERS))}"

    violation = grouped_splitter_violation(df)
    if violation is not None:
        logger.error(
            "Gate 1 precondition FAILED - refusing to score this OOF frame: %s.", violation
        )
        return GateResult(
            name="Gate 1 - Generalization (AUC-PR)",
            passed=False,
            value=f"NOT PEPTIDE-GROUPED: {violation}",
            threshold=threshold,
        )

    auc_pr = float(average_precision_score(df["label"], df["gnn_oof_score"]))
    passed = auc_pr >= GATE1_AUC_PR_MIN
    return GateResult(
        name="Gate 1 - Generalization (AUC-PR)",
        passed=passed,
        value=round(auc_pr, 4),
        threshold=threshold,
    )


def gate2_stability(df: pd.DataFrame) -> GateResult:
    """Cross-fold AUC-PR std across CV folds must not exceed GATE2_STD_MAX.

    Requires the per-row FOLD_COLUMN that src/train_gnn.py now writes on every
    run. An earlier version of this function documented a --save-fold-ids flag
    on train_gnn.py to explain when that column appears; no such flag ever
    existed, so the column was never present and the gate always took its
    fallback branch. That fallback measured the std of leave-one-ROW-out
    resamples of one pooled AUC-PR, which is a jackknife standard-error
    estimate of a single number, not the spread of the per-fold scores this
    gate is defined on - and it cost one full AUC-PR computation per row on a
    35k-row frame. Both are gone: without fold identity, cross-fold stability
    is not computable, and the gate says so instead of substituting a
    different, smaller statistic that happens to pass.
    """
    from sklearn.metrics import average_precision_score

    threshold = f"<= {GATE2_STD_MAX}"

    if FOLD_COLUMN not in df.columns:
        logger.error(
            "Gate 2 FAILED - the OOF frame carries no '%s' column, so per-fold AUC-PRs "
            "cannot be computed. Re-run src/train_gnn.py, which stamps fold identity on "
            "every OOF row.",
            FOLD_COLUMN,
        )
        return GateResult(
            name="Gate 2 - Stability (AUC-PR std, per-fold)",
            passed=False,
            value=(
                f"no '{FOLD_COLUMN}' column in the OOF frame; cross-fold stability is "
                "not measurable without per-row fold identity"
            ),
            threshold=threshold,
        )

    labels = df["label"].values
    scores = df["gnn_oof_score"].values
    fold_ids = df[FOLD_COLUMN].values

    fold_auc_prs: list[float] = []
    skipped: list[str] = []
    for fid in sorted(df[FOLD_COLUMN].dropna().unique()):
        mask = fold_ids == fid
        # A fold with a single class has no defined AUC-PR; record it rather
        # than dropping it silently, because dropping folds shrinks the std.
        if labels[mask].sum() == 0 or labels[mask].sum() == mask.sum():
            skipped.append(str(fid))
            continue
        fold_auc_prs.append(float(average_precision_score(labels[mask], scores[mask])))

    if len(fold_auc_prs) < 2:
        return GateResult(
            name="Gate 2 - Stability (AUC-PR std, per-fold)",
            passed=False,
            value=(
                f"only {len(fold_auc_prs)} scoreable fold(s) in the OOF frame "
                f"(single-class folds skipped: {skipped or 'none'}); a std across folds "
                "needs at least 2"
            ),
            threshold=threshold,
        )

    std = float(np.std(fold_auc_prs))
    passed = std <= GATE2_STD_MAX
    return GateResult(
        name=f"Gate 2 - Stability (AUC-PR std, per-fold over {len(fold_auc_prs)} folds)",
        passed=passed,
        value=round(std, 4),
        threshold=threshold,
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


def _time_model_ms_v2(predict_fn, batch, warmup: int, reps: int) -> float:
    """Latency timer for v2 models that accept a PyG batch object."""
    import torch

    for _ in range(warmup):
        with torch.no_grad():
            predict_fn(batch)
    times: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        with torch.no_grad():
            predict_fn(batch)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


def gate3_latency(checkpoint_path: Path | None = None) -> GateResult:
    """GNN CPU inference latency <= GATE3_LATENCY_FACTOR x RF latency.

    Uses a fixed synthetic batch of LATENCY_BATCH_SIZE 9-mer sequences so the
    measurement is reproducible without any real dataset access.
    Uses GraphPredictorV2 (GINEConv + ESM-2, 320-dim node features).

    `checkpoint_path` scores an alternative checkpoint instead of the tracked
    GNN_CHECKPOINT (see check_promotion_gates). Its sibling `gnn_config.json`
    - written alongside every checkpoint by src/train_gnn.py in the same
    --model-dir - is read from `checkpoint_path.parent / "gnn_config.json"`
    rather than the tracked GNN_CONFIG, so node_dim and num_continuous_features
    stay matched to the checkpoint actually being timed (GNN rule 8: these must
    agree with the saved state dict or this gate loads the wrong architecture).
    """
    import torch
    from torch_geometric.data import Data, Batch
    from src.gnn.models import GraphPredictorV2
    from src.gnn.graph_builder import GraphBuilder
    from src.features import TRAIN_FEATURE_COLUMNS
    from src.artifact_integrity import load_verified_joblib

    device = torch.device("cpu")

    # --- RF baseline ---
    if not RF_MODEL_PATH.exists():
        return GateResult(
            name="Gate 3 - Latency",
            passed=False,
            value="RF model not found",
            threshold=f"<= {GATE3_LATENCY_FACTOR}x RF latency",
        )
    rf_model = load_verified_joblib(RF_MODEL_PATH)

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

    # --- GNN v2.1 benchmark ---
    checkpoint = GNN_CHECKPOINT if checkpoint_path is None else Path(checkpoint_path)
    config_source = GNN_CONFIG if checkpoint_path is None else checkpoint.parent / "gnn_config.json"
    if not checkpoint.exists():
        return GateResult(
            name="Gate 3 - Latency",
            passed=False,
            value=f"GNN checkpoint not found: {checkpoint}",
            threshold=f"<= {GATE3_LATENCY_FACTOR}x RF latency",
        )

    # Read node_dim and num_continuous_features from gnn_config.json so gate3 matches
    # whatever ESM-2 variant and feature mode the checkpoint was trained with.
    import json as _json

    node_dim = 320  # default (t6 ESM-2)
    num_features = len(TRAIN_FEATURE_COLUMNS)  # default: 21 physico-only
    pooling = "mean"  # default readout (v2.1-v2.3); v2.4 may use attention
    if config_source.exists():
        with config_source.open() as _fh:
            _cfg = _json.load(_fh)
            node_dim = _cfg.get("node_dim", 320)
            num_features = _cfg.get("num_continuous_features", num_features)
            pooling = _cfg.get("pooling", "mean")

    gnn_model = GraphPredictorV2(
        num_continuous_features=num_features, node_dim=node_dim, pooling=pooling
    ).to(device)
    # weights_only=True prevents arbitrary code execution during checkpoint load
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    gnn_model.load_state_dict(state)
    gnn_model.eval()

    # Build synthetic PyG batch using the same node dim as the trained model
    ESM_DIM = node_dim
    MAX_LEN = 11
    edge_index, edge_attr = GraphBuilder.build_pyg_chain_graph(MAX_LEN)
    data_list = [
        Data(
            x=torch.zeros(MAX_LEN, ESM_DIM),
            edge_index=edge_index,
            edge_attr=edge_attr,
            physico=torch.zeros(1, num_features),
            y=torch.zeros(1),
        )
        for _ in range(LATENCY_BATCH_SIZE)
    ]
    synthetic_batch = Batch.from_data_list(data_list).to(device)

    gnn_latency_ms = _time_model_ms_v2(
        lambda b: gnn_model(b),
        synthetic_batch,
        LATENCY_WARMUP_REPS,
        LATENCY_TIMED_REPS,
    )

    ratio = gnn_latency_ms / max(rf_latency_ms, 0.001)
    passed = ratio <= GATE3_LATENCY_FACTOR
    return GateResult(
        name="Gate 3 - Latency",
        passed=passed,
        value=f"GNN={gnn_latency_ms:.2f}ms, RF={rf_latency_ms:.2f}ms, ratio={ratio:.2f}x",
        threshold=f"ratio <= {GATE3_LATENCY_FACTOR}x",
    )


def gate4_calibration(df: pd.DataFrame) -> GateResult:
    """Expected Calibration Error (ECE) < GATE4_ECE_MAX.

    Uses equal-width binning (15 bins) following Guo et al. 2017.
    """
    probs = df["gnn_oof_score"].values.astype(float)
    labels = df["label"].values.astype(float)
    n_bins = 15
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    passed = ece < GATE4_ECE_MAX
    return GateResult(
        name="Gate 4 - Calibration (ECE, 15-bin)",
        passed=passed,
        value=round(ece, 4),
        threshold=f"< {GATE4_ECE_MAX}",
    )


def gate5_escape_sensitivity(df: pd.DataFrame) -> GateResult:
    """Fraction of ALL OOF positives scored above the median OOF negative >= 0.80.

    Requires the OOF file to contain both positive (label=1) and negative
    (label=0) rows. The gate checks that the model rank-orders positives over
    negatives at the 80% sensitivity level, pool-wide - despite the gate's
    name, this is NOT restricted to IEDB gold-standard epitopes. Both GNN
    training entry points (train_gnn(), train_gnn_v2() in src/train_gnn.py)
    mask GOLD_STANDARD_EPITOPES out of the training pool before any CV fold
    is built, so no gold-standard peptide ever reaches this OOF frame. See
    docs/claims_register.md D26.
    """
    positives = df[df["label"] == 1]["gnn_oof_score"].values
    negatives = df[df["label"] == 0]["gnn_oof_score"].values

    if len(positives) == 0 or len(negatives) == 0:
        return GateResult(
            name="Gate 5 - Escape Sensitivity",
            passed=False,
            value="Insufficient class diversity in OOF file",
            threshold=f">= {GATE5_SENSITIVITY_MIN:.0%}",
        )

    decoy_median = float(np.median(negatives))
    sensitivity = float((positives > decoy_median).mean())
    passed = sensitivity >= GATE5_SENSITIVITY_MIN
    return GateResult(
        name="Gate 5 - Escape Sensitivity",
        passed=passed,
        value=round(sensitivity, 4),
        threshold=f">= {GATE5_SENSITIVITY_MIN}",
    )


# ---------------------------------------------------------------------------
# Scorecard runner
# ---------------------------------------------------------------------------


def check_promotion_gates(oof_path: Path | None = None, checkpoint_path: Path | None = None) -> bool:
    logger.info("=" * 60)
    logger.info("SESTRAV GNN Promotion Scorecard - 5 Gates")
    logger.info("=" * 60)

    checkpoint = GNN_CHECKPOINT if checkpoint_path is None else Path(checkpoint_path)
    if not checkpoint.exists():
        logger.error(
            f"Checkpoint {checkpoint} not found. "
            "Execute full GNN training (src/train_gnn.py) before promotion."
        )
        return False

    if oof_path is not None:
        logger.info("Scoring OOF frame: %s (overrides the default %s)", oof_path, OOF_PATH)
    if checkpoint_path is not None:
        logger.info(
            "Scoring checkpoint: %s (overrides the default %s)", checkpoint_path, GNN_CHECKPOINT
        )

    try:
        df = _load_oof(oof_path)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        return False

    results: list[GateResult] = []

    # Gates 1, 2, 4, 5 depend only on OOF CSV
    for gate_fn in (
        gate1_generalization,
        gate2_stability,
        gate4_calibration,
        gate5_escape_sensitivity,
    ):
        try:
            r = gate_fn(df)
        except Exception as exc:  # noqa: BLE001 - promotion gate loop; must catch all failures to log and continue
            logger.error(f"{gate_fn.__name__} raised unexpectedly: {exc}")
            results.append(
                GateResult(name=gate_fn.__name__, passed=False, value=str(exc), threshold="-")
            )
            continue
        results.append(r)

    # Gate 3 requires loading real models
    try:
        r3 = gate3_latency(checkpoint_path)
    except Exception as exc:  # noqa: BLE001 - top-level promotion gate; must catch all failures to log and exit non-zero
        logger.error(f"gate3_latency raised unexpectedly: {exc}")
        r3 = GateResult(name="Gate 3 - Latency", passed=False, value=str(exc), threshold="-")
    results.append(r3)

    # Log full scorecard
    logger.info("")
    logger.info("-" * 60)
    all_passed = True
    for r in sorted(results, key=lambda x: x.name):
        status = "PASS" if r.passed else "FAIL"
        logger.info(f"  {r.name}")
        logger.info(f"    Value: {r.value}   Threshold: {r.threshold}   [{status}]")
        if not r.passed:
            all_passed = False
    logger.info("-" * 60)
    if all_passed:
        logger.info("SCORECARD RESULT: ALL GATES PASSED - ready for promotion.")
    else:
        logger.error("SCORECARD RESULT: ONE OR MORE GATES FAILED - promotion blocked.")
    logger.info("=" * 60)

    return all_passed


# ---------------------------------------------------------------------------
# Promotion executor (only called when all gates pass)
# ---------------------------------------------------------------------------


def promote_model(
    dry_run: bool = False, oof_path: Path | None = None, checkpoint_path: Path | None = None
) -> None:
    """Mutates config.yaml and model_artifact_checksums.json iff all gates pass.

    dry_run runs the identical scorecard and reports exactly which mutations
    would follow, then returns without writing anything. It exists so the gates
    can be exercised - on a candidate, in CI, or after a gate definition
    changes - without the side effect of repointing the production model_path
    and re-stamping the checksum manifest.

    oof_path scores an alternative OOF frame, so a scratch training run can be
    put through the scorecard without first overwriting the tracked
    models/gnn_oof_predictions.csv. It selects the INPUT only; it does not
    relax any gate and does not change where a successful promotion writes.

    checkpoint_path scores an alternative checkpoint (Gate 3 latency, and the
    SHA-256 shown here) the same way oof_path scores an alternative OOF frame.
    It closes A2-gap: before this parameter existed, a real (dry_run=False)
    promotion always certified whatever file happened to already sit at
    GNN_CHECKPOINT, with nothing tying that file to whatever OOF/checkpoint had
    actually just been scored via --oof. Only meaningful combined with
    dry_run=True - combining it with dry_run=False is refused below rather than
    silently promoting a scratch checkpoint from a gitignored path, or silently
    re-certifying a stale/different file at the canonical path. To promote a
    passing scratch candidate for real: copy it to GNN_CHECKPOINT (and its
    sibling gnn_config.json/gnn_scaler.joblib) yourself, then call this
    function again with dry_run=False and no override, so it certifies
    exactly the file it just copied.
    """
    if checkpoint_path is not None and not dry_run:
        raise ValueError(
            "checkpoint_path is only valid combined with dry_run=True. A real promotion "
            "always certifies GNN_CHECKPOINT so the file that gets certified is never "
            "silently different from the file at the canonical path - copy the scored "
            "checkpoint (and its sibling gnn_config.json/gnn_scaler.joblib) to "
            f"{GNN_CHECKPOINT} yourself first, then call promote_model(dry_run=False) with "
            "no checkpoint_path override."
        )

    if not check_promotion_gates(oof_path, checkpoint_path):
        logger.error("Model failed promotion gates. config.yaml will NOT be modified.")
        return

    if dry_run:
        logger.info("DRY RUN: all gates passed. No files will be written.")
    else:
        logger.info("Promoting Structural GNN to canonical pipeline...")

    # Secure SHA-256 (native Python; no shell=True, no subprocess) - read-only.
    scored_checkpoint = GNN_CHECKPOINT if checkpoint_path is None else Path(checkpoint_path)
    gnn_sha256 = _sha256_file(scored_checkpoint)
    logger.info(f"Checkpoint SHA-256: {gnn_sha256}")

    if dry_run:
        logger.info(f"DRY RUN: would set model_path -> {GNN_CHECKPOINT} in {CONFIG_PATH}")
        logger.info(f"DRY RUN: would record {GNN_CHECKPOINT} in {CHECKSUM_FILE}")
        if checkpoint_path is not None:
            logger.info(
                f"DRY RUN: the SHA-256 above is {scored_checkpoint}'s, not the canonical "
                f"path's - a real promotion still certifies {GNN_CHECKPOINT} and would only "
                "match this SHA-256 if that file is copied there first."
            )
        logger.info("DRY RUN complete: config.yaml and the checksum manifest are unchanged.")
        return

    # --- Update config.yaml ---
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
        config["model_path"] = str(GNN_CHECKPOINT)
        with CONFIG_PATH.open("w", encoding="utf-8") as fh:
            yaml.dump(config, fh, default_flow_style=False, sort_keys=False)
        logger.info(f"Updated {CONFIG_PATH}: model_path -> {GNN_CHECKPOINT}")
    else:
        logger.warning(f"{CONFIG_PATH} not found - skipping config update.")

    # --- Update model_artifact_checksums.json ---
    try:
        from src.artifact_integrity import update_checksum_manifest

        update_checksum_manifest(CHECKSUM_FILE, [GNN_CHECKPOINT])
        logger.info(f"Updated {CHECKSUM_FILE} using canonical schema.")
    except Exception as exc:
        logger.error(f"Failed to update checksum manifest {CHECKSUM_FILE}: {exc}")
        raise


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the 5 canonical GNN promotion gates and, unless "
        "--dry-run is passed, promote the checkpoint."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate every gate and report the mutations that would follow, "
        "without modifying config.yaml or model_artifact_checksums.json.",
    )
    parser.add_argument(
        "--oof",
        type=Path,
        default=None,
        metavar="CSV",
        help="Score this OOF predictions CSV instead of the default "
        f"{OOF_PATH}. Lets a scratch run be put through the scorecard without "
        "overwriting the tracked artifact. Selects the input only - it does not "
        "relax a gate or change where a promotion writes.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        metavar="PTH",
        help="Score this checkpoint (Gate 3 latency + the displayed SHA-256) "
        f"instead of the default {GNN_CHECKPOINT}. Its sibling gnn_config.json "
        "is read from the same directory. Only valid with --dry-run - refused "
        "otherwise, so a real promotion can never silently certify a file "
        "different from the one just scored.",
    )
    return parser


if __name__ == "__main__":
    _args = _build_arg_parser().parse_args()
    promote_model(dry_run=_args.dry_run, oof_path=_args.oof, checkpoint_path=_args.checkpoint)
