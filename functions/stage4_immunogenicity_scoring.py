"""
SESTRAV Stage 4 - Immunogenicity Scoring

Production mode: loads a pre-trained classifier and scores peptides.
Auto-detects the model's expected feature count (21, 22, or 30) and
selects the matching columns from the Stage 3 output.

Supported model formats:
  - sklearn .joblib  (RF / XGBoost, 21 or 30 features)
  - PyTorch .pt      (ANN, 30 features - includes embedded scaler)

Optional post-scoring enhancements (when artifact files are present):
  - Platt calibration via platt_calibrator.joblib
  - Threshold-based binary classification via optimal_thresholds.json
  - MC Dropout uncertainty via PyTorch ANN (N=50 forward passes)

Prototype mode (no model file): trains inline on the feature data using a
RandomForestClassifier with binding-derived pseudo-labels.  NOT scientifically
valid - exists only for end-to-end pipeline testing.

Immunogenicity scores are probabilities in [0, 1] from predict_proba.
Higher score = higher predicted immunogenicity.  Rank 1 = top candidate.
"""

import json
import os
import re
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.features import (
    FEATURE_COLUMNS,
    TRAIN_FEATURE_COLUMNS,
    FEATURE_COLUMNS_30,
    FEATURE_COLUMNS_31,
    FEATURE_COLUMNS_33,
    FEATURE_COLUMNS_35,
    FEATURE_COLUMNS_50,
)
from src.naming import resolve_model_path

try:
    from src.artifact_integrity import load_verified_joblib
except ImportError:
    load_verified_joblib = None  # type: ignore[assignment]


#: Maps a trained model's fit-time feature count to the column layout that reproduces it.
#:
#: Keyed by width because ``n_features_in_`` (or an ANN checkpoint's ``n_features``) is the
#: only thing a loaded artifact reliably exposes. A caller's ``--feature-mode`` is NOT used
#: for this decision - it is advisory, and keying off it would let a mislabeled request
#: silently score against the wrong matrix.
#:
#: This table exists because the mapping used to be written out twice in this module as an
#: if/elif ladder - once in the ANN branch and once in the joblib branch - and both copies
#: omitted modes 31, 33 and 35. A 31-feature model (the config.yaml default) therefore fell
#: through to the 22-column legacy set, and the mismatch surfaced only as a third-party
#: feature-name error from inside predict_proba. One table, read by both paths, stops that
#: recurring here. Note a third, independent copy still lives in ``src/baseline_comparison``
#: and handles only 30/21/legacy-22; it is unreachable at other widths today because its
#: candidate lists offer no other artifact.
_FEATURE_LAYOUTS: dict[int, tuple[list[str], str]] = {
    len(FEATURE_COLUMNS_50): (FEATURE_COLUMNS_50, "50-feature multi-allele mode"),
    len(FEATURE_COLUMNS_35): (FEATURE_COLUMNS_35, "35-feature extended mode"),
    len(FEATURE_COLUMNS_33): (FEATURE_COLUMNS_33, "33-feature antigen-processing mode"),
    len(FEATURE_COLUMNS_31): (FEATURE_COLUMNS_31, "31-feature canonical mode"),
    len(FEATURE_COLUMNS_30): (FEATURE_COLUMNS_30, "30-feature multi-allele mode"),
    len(FEATURE_COLUMNS): (FEATURE_COLUMNS, "full legacy set"),
    len(TRAIN_FEATURE_COLUMNS): (TRAIN_FEATURE_COLUMNS, "sequence-only (binding_score excluded)"),
}


def _resolve_feature_layout(expected_n: int) -> tuple[list[str], str]:
    """Return the (column list, human label) a model of width ``expected_n`` was fit on.

    Falls back to the legacy 22-column set for an unrecognised width; callers are
    expected to verify the resulting selection actually matches ``expected_n``.
    """
    return _FEATURE_LAYOUTS.get(expected_n, (FEATURE_COLUMNS, "full legacy set"))


def _select_model_columns(expected_n: int, features_df) -> tuple[list[str], str]:
    """Select the columns a model of width ``expected_n`` needs, or refuse clearly.

    Raises ``RuntimeError`` when the pipeline cannot supply that layout instead of
    substituting a narrower one. Modes 33 and 35 are the load-bearing case: Stage 3 does
    not compute ``netchop_score``/``tap_score`` or the ``self_similarity_*`` columns at
    all, so those models are genuinely unscoreable here and must say so plainly rather
    than failing several frames deeper with a feature-name mismatch.
    """
    expected_list, label = _resolve_feature_layout(expected_n)
    model_cols = [c for c in expected_list if c in features_df.columns]
    if len(model_cols) != expected_n:
        if expected_n not in _FEATURE_LAYOUTS:
            raise RuntimeError(
                f"[Stage 4] Unrecognised feature width: no shipped layout has "
                f"{expected_n} columns (known widths: {sorted(_FEATURE_LAYOUTS)}), and the "
                f"fallback {label} supplies {len(model_cols)}. Refusing to score against a "
                "substituted feature matrix."
            )
        missing = [c for c in expected_list if c not in features_df.columns]
        raise RuntimeError(
            f"[Stage 4] Cannot score a {expected_n}-feature model: the pipeline supplied "
            f"{len(model_cols)} of the {len(expected_list)} columns for the {label}; "
            f"missing {missing}. Refusing to score against a substituted feature matrix."
        )
    return model_cols, label


def _load_torch_checkpoint(model_path, required=True):
    """Load ANN checkpoints using the safe weights-only path only.

    PyTorch 2.6+ enforces strict allowlisting for weights_only=True. SESTRAV
    checkpoints embed numpy scalars and dtypes in scaler parameters
    (scaler_mean / scaler_scale). Both are allowlisted explicitly here -
    the PyTorch-recommended approach for trusted, internally-generated
    checkpoints.
    """
    from src.artifact_integrity import verify_artifact_checksum

    verify_artifact_checksum(model_path, required=required)
    try:
        import numpy as np
        import torch.serialization

        # Allowlist numpy types used in scaler_mean/scaler_scale checkpoint arrays.
        # numpy._core is used (not deprecated numpy.core alias) to silence warnings.
        # Safe: checkpoints are generated only by SESTRAV's own training pipeline.
        torch.serialization.add_safe_globals(
            [
                np._core.multiarray.scalar,  # serialised numpy scalar values
                np.dtype,  # serialised numpy dtype objects
            ]
        )
    except Exception:  # nosec B110
        pass
    import torch

    return torch.load(model_path, map_location="cpu", weights_only=True)  # nosec B614 nosemgrep


def _load_pytorch_model(model_path, features_df, model_cols):
    """Load a PyTorch .pt checkpoint and score peptides on the 30-feature set."""
    import torch
    import torch.nn as nn

    checkpoint = _load_torch_checkpoint(model_path, required=True)
    n_features = checkpoint["n_features"]
    scaler_mean = checkpoint["scaler_mean"].numpy()
    scaler_scale = checkpoint["scaler_scale"].numpy()

    class FlexibleMLP(nn.Module):
        def __init__(self, input_dim, hidden_sizes=(64, 32), dropout=0.3):
            super().__init__()
            layers = []
            prev = input_dim
            for h in hidden_sizes:
                layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(-1)

    model = FlexibleMLP(input_dim=n_features)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    X = features_df[model_cols].values.astype(np.float64)
    X_scaled = (X - scaler_mean) / (scaler_scale + 1e-10)

    with torch.no_grad():
        logits = model(torch.tensor(X_scaled, dtype=torch.float32)).numpy()
    scores = 1.0 / (1.0 + np.exp(-logits))
    return scores, model


def _mc_dropout_predict(pt_model, X_tensor, n_passes=50):
    """Run MC Dropout: N stochastic forward passes with dropout active."""
    import torch

    pt_model.train()
    preds = []
    with torch.no_grad():
        for _ in range(n_passes):
            logits = pt_model(X_tensor).cpu().numpy()
            preds.append(1.0 / (1.0 + np.exp(-logits)))
    preds = np.array(preds)
    return preds.mean(axis=0), preds.std(axis=0)


#: The nine target viruses with genuine positive labels in the training corpus
#: (decoy-only viruses, e.g. Orthopoxvirus vaccinia, are excluded). Mirrors
#: scripts/fit_calibrator.py's TARGET_VIRUSES; duplicated rather than imported
#: because every script in this repo that needs this list declares its own
#: copy (scripts/audit_cv_leakage.py, scripts/fit_per_virus_calibrator.py,
#: scripts/compute_pooled_honest_metric.py all do the same).
#:
#: NOT READ BY THIS MODULE. Corrected 2026-08-15: this comment used to claim the
#: tuple was "used here only to decide the log label (known off-panel vs
#: unrecognised)". No such code exists or ever existed here - _apply_calibration
#: computes its scope purely by comparing the resolved path against the per-virus
#: path, and never consults this tuple. The tuple is documentation only, kept
#: because it records which viruses have genuine positive labels.
#:
#: The half of the old comment that IS true, and is the actual contract: artifact
#: ABSENCE, not list membership, triggers the global fallback, so a caller never
#: needs to keep this list in sync with whatever per-virus calibrators happen to
#: be deployed. tests/test_stage4_scoring.py pins that an off-panel virus behaves
#: identically to virus=None; do not add an allow-list check against this tuple.
TARGET_VIRUSES = (
    "CMV",
    "DENV",
    "EBV",
    "HBV",
    "HCV",
    "HIV-1",
    "HPV",
    "IAV",
    "SARS-CoV-2",
)

#: Where a promoted set of per-virus calibrators would live. Does not exist by
#: default - no per-virus calibrator has been promoted (see A1-promote in the
#: open-item register). Its absence is exactly what makes every virus "unknown/
#: off-panel" today: the resolver below falls back to the global calibrator
#: whenever the specific file is missing, with no separate on/off-panel branch.
DEFAULT_PER_VIRUS_CALIBRATION_DIR = "models/calibration/per_virus"


def _resolve_calibrator_path(model_dir, calibration_path=None, virus=None, per_virus_dir=None):
    """Return an existing calibrator path, or None if none is available.

    Preference order:
      1. an explicit ``calibration_path`` (from config), if it exists - wins
         regardless of ``virus``, matching the pre-existing override contract;
      2. a per-virus calibrator at ``per_virus_dir/<sanitized virus>.joblib``,
         if ``virus`` is given and that specific file exists. Whether ``virus``
         is one of TARGET_VIRUSES is NOT checked here - the fallback below
         fires from file absence alone, so an unrecognised name behaves
         identically to a recognised-but-not-yet-promoted one;
      3. an isotonic calibrator alongside the model (isotonic_calibrator.joblib);
      4. the legacy Platt calibrator (platt_calibrator.joblib).
    """
    if calibration_path and os.path.isfile(calibration_path):
        return calibration_path
    if virus:
        pv_dir = per_virus_dir or DEFAULT_PER_VIRUS_CALIBRATION_DIR
        pv_candidate = os.path.join(pv_dir, f"{_sanitize_name(virus)}.joblib")
        if os.path.isfile(pv_candidate):
            return pv_candidate
    for name in ("isotonic_calibrator.joblib", "platt_calibrator.joblib"):
        candidate = os.path.join(model_dir, name)
        if os.path.isfile(candidate):
            return candidate
    return None


def _apply_calibration(scores, model_dir, calibration_path=None, virus=None, per_virus_dir=None):
    """Apply an available calibrator; return (calibrated_scores, applied_bool).

    Supports two calibrator types and dispatches on the loaded object:
      - isotonic (sklearn IsotonicRegression): exposes ``.predict`` and maps the
        raw score in [0, 1] directly to a calibrated probability;
      - Platt (exposes ``.predict_proba``): maps logits of the score.

    ``virus``/``per_virus_dir`` select a per-virus calibrator when one has been
    promoted for that virus (see _resolve_calibrator_path); passing neither
    reproduces the pre-existing global-only behaviour exactly. A virus with no
    promoted calibrator - including every virus today, since none has been
    promoted yet - transparently falls back to the global calibrator.

    The raw scores are returned unchanged when no calibrator is present or the
    verified-loader is unavailable. Output is clipped to [0, 1] and NaNs are
    replaced with the corresponding raw score as a safety guard.
    """
    cal_path = _resolve_calibrator_path(model_dir, calibration_path, virus, per_virus_dir)
    if cal_path is None or load_verified_joblib is None:
        return scores, False

    pv_dir = per_virus_dir or DEFAULT_PER_VIRUS_CALIBRATION_DIR
    is_per_virus = virus and cal_path == os.path.join(pv_dir, f"{_sanitize_name(virus)}.joblib")
    scope = f"per-virus:{virus}" if is_per_virus else "global"

    calibrator = load_verified_joblib(cal_path, required_checksum=True)
    raw = np.asarray(scores, dtype=np.float64)

    if hasattr(calibrator, "predict_proba"):
        logits = np.log((raw + 1e-10) / (1 - raw + 1e-10)).reshape(-1, 1)
        calibrated = np.asarray(calibrator.predict_proba(logits))[:, 1]
        label = "Platt"
    else:
        # Isotonic (or any calibrator exposing only .predict) operates on the
        # raw probability score directly, not on logits.
        calibrated = np.asarray(calibrator.predict(raw), dtype=np.float64)
        label = "isotonic"

    # Guard against NaN / out-of-range outputs: fall back to the raw score for
    # any non-finite entry, then clip into the valid probability range.
    calibrated = np.where(np.isfinite(calibrated), calibrated, raw)
    calibrated = np.clip(calibrated, 0.0, 1.0)
    print(f"[Stage 4] Applied {label} calibration ({scope})")
    return calibrated, True


def _resolve_thresholds_path(model_dir, thresholds_path=None):
    """Return an existing optimal-thresholds path, or None if none is available.

    Preference order mirrors ``_resolve_calibrator_path``:
      1. an explicit ``thresholds_path`` (from config), if it exists;
      2. ``optimal_thresholds.json`` alongside the model.

    Before this resolver existed, ``config.yaml``'s ``thresholds_path`` key was
    read by nothing at all - the file was located purely by convention from the
    model directory, so repointing the config key was a silent no-op while
    repointing ``model_path`` silently moved the operating point with it.
    """
    if thresholds_path and os.path.isfile(thresholds_path):
        return thresholds_path
    candidate = os.path.join(model_dir, "optimal_thresholds.json")
    if os.path.isfile(candidate):
        return candidate
    return None


def _apply_thresholds(features_df, model_dir, thresholds_path=None):
    """Add binary immunogenic column using exported optimal thresholds."""
    thresh_path = _resolve_thresholds_path(model_dir, thresholds_path)
    if thresh_path is None:
        return
    with open(thresh_path) as f:
        thresholds = json.load(f)
    score_col = (
        "calibrated_score" if "calibrated_score" in features_df.columns else "immunogenicity_score"
    )
    f1_thresh = thresholds.get("threshold", thresholds.get("f1_threshold", 0.5))
    features_df["immunogenic"] = (features_df[score_col] >= f1_thresh).astype(int)
    print(f"[Stage 4] Applied F1-optimal threshold {f1_thresh:.3f}")


def _sanitize_name(name):
    """Allow only alphanumeric, underscores, and hyphens."""
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", name)


def score_immunogenicity(
    features_df,
    proteome_id,
    model_path=None,
    calibrate=True,
    mc_dropout=False,
    freeze_mode=False,
    calibration_path=None,
    thresholds_path=None,
    virus=None,
    per_virus_calibration_dir=None,
    output_dir=None,
):
    """
    Score each peptide's immunogenicity.

    Auto-detects model type (.joblib vs .pt) and feature dimensionality
    (21, 22, 30, 31, 33, 35 or 50 features) from the artifact itself.

    The two branches differ on an unsatisfiable layout, deliberately: the joblib
    path raises ``RuntimeError`` rather than score against a substituted matrix,
    while the ANN path warns and proceeds unless ``freeze_mode`` is set. A named
    ``model_path`` that does not exist always raises ``FileNotFoundError``; pass
    ``model_path=None`` to request the prototype classifier on purpose.

    Args:
        features_df: DataFrame with feature columns from Stage 3
        proteome_id: label used in output filename (sanitized for filesystem safety)
        model_path:  path to a serialized model (optional)
        calibrate:   apply calibration if a calibrator artifact exists
                     (isotonic or Platt)
        mc_dropout:  run MC Dropout uncertainty (PyTorch models only)
        freeze_mode: raise on any missing artifact or model incompatibility
        calibration_path: explicit calibrator path (from config); when unset,
                     the model directory is searched for a calibrator artifact
        thresholds_path: explicit optimal-thresholds path (from config); when
                     unset, the model directory is searched for
                     optimal_thresholds.json
        virus:       which virus this proteome represents, if known (e.g.
                     "SARS-CoV-2"). Selects a per-virus calibrator when one has
                     been promoted for it; unset, unrecognised, or off-panel
                     values all fall back to the global calibrator identically
                     - there is no separate error path for "off-panel".
        per_virus_calibration_dir: directory to search for
                     ``<virus>.joblib`` calibrators; defaults to
                     DEFAULT_PER_VIRUS_CALIBRATION_DIR, which does not exist
                     until a per-virus calibrator is promoted (see the
                     open-item register's A1-promote).
        output_dir:  directory the ranked CSV is written into; unset keeps the
                     original "results" default. Two concurrent calls for the
                     same proteome_id no longer collide as long as each is
                     given a distinct output_dir (SESTRAV-Dev C5).

    Returns:
        (ranked_df, model) tuple.
    """
    proteome_id = _sanitize_name(proteome_id)
    model = None
    if model_path:
        resolved_path = resolve_model_path(model_path)
        if resolved_path != model_path:
            print(f"[Stage 4] Using alias model path '{resolved_path}' (from '{model_path}')")
        model_path = resolved_path
    model_dir = os.path.dirname(model_path) if model_path else "models"

    # A caller who NAMED a model must get that model or an error - never the prototype.
    # Without this, a wrong path fell through to the inline prototype classifier below,
    # which trains on pseudo-labels derived from binding_score and then has its output
    # calibrated and thresholded like a real score. The resulting ranked CSV carries no
    # marker distinguishing it from a genuine run, and the only disclosure is a stdout
    # line that any redirected or backgrounded run discards. `sestrav predict` validates
    # --model itself, but pipeline.py does not: ModelRegistry.resolve_model confines a
    # path to models/ without checking that it exists. Passing model_path=None remains
    # the supported way to ask for the prototype deliberately.
    if model_path and not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"[Stage 4] Model artifact not found: {model_path!r}. Refusing to fall back to "
            "the prototype inline classifier, whose scores are not scientifically valid and "
            "are indistinguishable from a real run once written. Pass model_path=None to "
            "request the prototype deliberately."
        )

    if model_path and os.path.isfile(model_path):
        is_pytorch = model_path.endswith(".pt")

        if is_pytorch:
            checkpoint = _load_torch_checkpoint(model_path, required=False)
            expected_n = checkpoint["n_features"]

            # Same layout table as the joblib path, but this branch keeps its existing
            # warn-unless-freeze_mode semantics rather than refusing outright.
            expected_list, _ = _resolve_feature_layout(expected_n)
            model_cols = [c for c in expected_list if c in features_df.columns]

            if len(model_cols) < expected_n:
                missing = set(expected_list) - set(features_df.columns)
                missing_sorted = sorted(missing)
                msg = f"[Stage 4] Missing {len(missing)} of {expected_n} ANN features: {missing_sorted}"
                if freeze_mode:
                    raise RuntimeError(msg)
                print(f"[Stage 4] WARNING: {msg}")
            scores, model = _load_pytorch_model(model_path, features_df, model_cols)
            features_df["immunogenicity_score"] = scores
            print(f"[Stage 4] Loaded PyTorch ANN from {model_path} ({len(model_cols)} features)")

            if mc_dropout:
                import torch  # optional dependency; only needed for the MC-dropout path

                checkpoint = _load_torch_checkpoint(model_path, required=True)
                X = features_df[model_cols].values.astype(np.float64)
                X_scaled = (X - checkpoint["scaler_mean"].numpy()) / (
                    checkpoint["scaler_scale"].numpy() + 1e-10
                )
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
                mc_mean, mc_std = _mc_dropout_predict(model, X_tensor)
                features_df["mc_score"] = mc_mean
                features_df["uncertainty_std"] = mc_std
                print(
                    f"[Stage 4] MC Dropout: {(mc_std < np.median(mc_std)).sum()} "
                    f"high-confidence predictions"
                )

        elif load_verified_joblib is not None:
            model = load_verified_joblib(model_path, required_checksum=True)
            expected_n = model.n_features_in_

            model_cols, layout_label = _select_model_columns(expected_n, features_df)
            print(f"[Stage 4] Using {len(model_cols)} features ({layout_label})")

            X = features_df[model_cols].copy()
            features_df["immunogenicity_score"] = model.predict_proba(X)[:, 1]
            print(f"[Stage 4] Loaded trained model from {model_path}")
        else:
            print("[Stage 4] WARNING: joblib not available, cannot load .joblib model")

    if "immunogenicity_score" not in features_df.columns:
        if freeze_mode:
            raise RuntimeError(
                "[Stage 4] Freeze mode requires a trained model; prototype inline "
                "classifier fallback is disabled."
            )
        from sklearn.ensemble import RandomForestClassifier

        available_cols = [c for c in FEATURE_COLUMNS if c in features_df.columns]
        X = features_df[available_cols].copy()

        if "binding_score" in features_df.columns:
            median_binding = features_df["binding_score"].median()
            pseudo_labels = (features_df["binding_score"] >= median_binding).astype(int).values
        elif "presentation_score" in features_df.columns:
            median_ps = features_df["presentation_score"].median()
            pseudo_labels = (features_df["presentation_score"] >= median_ps).astype(int).values
        else:
            pseudo_labels = np.zeros(len(features_df), dtype=int)

        if np.unique(pseudo_labels).size < 2:
            # Single-class pseudo-labels -> RandomForest would return shape (n,1)
            # and [:, 1] would raise IndexError. Assign a constant score instead.
            features_df["immunogenicity_score"] = 0.0
            print(
                "[Stage 4] No score columns found - prototype degenerate case; "
                "all immunogenicity scores set to 0.0 (NOT scientifically valid)"
            )
        else:
            model = RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=42,
                n_jobs=1,
            )
            model.fit(X, pseudo_labels)
            features_df["immunogenicity_score"] = model.predict_proba(X)[:, 1]
            print(
                "[Stage 4] No trained model found - used prototype inline classifier "
                "(NOT scientifically valid)"
            )

    # immunogenicity_score is documented (module docstring) and consumed downstream
    # (calibration, thresholding, ranking, api/main.py's ScoreResponse) as a [0, 1]
    # probability. That holds for a standard sklearn/PyTorch classifier's
    # predict_proba/sigmoid output, but nothing enforces it: a model loaded here with a
    # ranking objective (e.g. XGBClassifier(objective="rank:pairwise")) returns raw
    # margins outside [0, 1] instead, and every downstream stage - calibration,
    # thresholding, the ranked CSV - silently treats that margin as a probability.
    # LRF-1: models/xgb_50feature_integrated.joblib is exactly this case; a mode-50
    # sweep was observed writing mean_score=-4.38 while reporting status=SUCCESS.
    # Fail loudly and immediately rather than propagate a value nothing downstream
    # can distinguish from a genuine low-confidence score.
    # NaN is checked explicitly rather than left to the comparisons: (x < 0) | (x > 1)
    # is False for NaN, so a bare pair of comparisons passes NaN straight through to
    # calibration and ranking - as undetectable downstream as the margin above, and
    # reachable on the PyTorch branch, where _load_pytorch_model divides by
    # (scaler_scale + 1e-10) with no finiteness check.
    score_values = features_df["immunogenicity_score"].to_numpy(dtype=float)
    non_finite = ~np.isfinite(score_values)
    out_of_range = non_finite | (score_values < 0.0) | (score_values > 1.0)
    if out_of_range.any():
        finite_bad = score_values[out_of_range & ~non_finite]
        observed = (
            f"observed finite range [{finite_bad.min():.4f}, {finite_bad.max():.4f}]"
            if finite_bad.size
            else "no finite out-of-range values"
        )
        raise RuntimeError(
            f"[Stage 4] immunogenicity_score out of the required [0, 1] range: "
            f"{out_of_range.sum()} of {len(score_values)} value(s), "
            f"{non_finite.sum()} of them non-finite, {observed}. "
            f"Model: {model_path!r}. This model's predict_proba output is not a "
            "probability - check its objective/loss (e.g. a ranking objective such as "
            "rank:pairwise returns raw margins, not [0, 1] scores)."
        )

    if calibrate:
        cal_scores, was_calibrated = _apply_calibration(
            features_df["immunogenicity_score"].values,
            model_dir,
            calibration_path,
            virus,
            per_virus_calibration_dir,
        )
        if was_calibrated:
            features_df["calibrated_score"] = cal_scores

    _apply_thresholds(features_df, model_dir, thresholds_path)

    # Use deterministic contiguous ranking (1..N) based on score ordering.
    # Prefer calibrated_score when available (Platt calibration is monotonic,
    # but boundary rounding can shift relative order for tied raw scores).
    rank_col = (
        "calibrated_score" if "calibrated_score" in features_df.columns else "immunogenicity_score"
    )
    features_df = features_df.sort_values(
        by=[rank_col, "peptide"], ascending=[False, True]
    ).reset_index(drop=True)
    features_df["rank"] = features_df.index + 1

    # SESTRAV-Dev C5: this used to be hardcoded to "results/", so two concurrent
    # scoring runs for the same proteome_id (e.g. a batch sweep across models)
    # overwrote each other's output. output_dir defaults to "results" so any
    # caller that does not pass it keeps the original behavior unchanged.
    resolved_output_dir = output_dir if output_dir is not None else "results"
    os.makedirs(resolved_output_dir, exist_ok=True)
    output_path = os.path.join(str(resolved_output_dir), f"{proteome_id}_ranked.csv")
    features_df.to_csv(output_path, index=False)
    print(f"[Stage 4] Scored and ranked {len(features_df)} peptides")

    return features_df, model


def plot_immunogenicity_scores(ranked_df, proteome_id, top_n=20):
    """Save top-N bar chart and score distribution histogram."""
    os.makedirs("results", exist_ok=True)

    top_df = ranked_df.head(top_n)
    plt.figure(figsize=(10, 6))
    plt.barh(top_df["peptide"], top_df["immunogenicity_score"], color="#4C72B0")
    plt.xlabel("Immunogenicity Score")
    plt.ylabel("Peptide")
    plt.title(f"Top {top_n} Immunogenic Peptides - {proteome_id}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"results/{proteome_id}_top{top_n}_immunogenicity.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(ranked_df["immunogenicity_score"], bins=50, color="#DD8452", alpha=0.85)
    plt.xlabel("Immunogenicity Score")
    plt.ylabel("Number of Peptides")
    plt.title(f"Immunogenicity Score Distribution - {proteome_id}")
    plt.tight_layout()
    plt.savefig(f"results/{proteome_id}_score_distribution.png", dpi=150)
    plt.close()

    print(f"[Plot] Saved plots for {proteome_id}")
