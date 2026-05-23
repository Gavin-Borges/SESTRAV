"""
SESTRAV Stage 4 — Immunogenicity Scoring

Production mode: loads a pre-trained classifier and scores peptides.
Auto-detects the model's expected feature count (21, 22, or 30) and
selects the matching columns from the Stage 3 output.

Supported model formats:
  - sklearn .joblib  (RF / XGBoost, 21 or 30 features)
  - PyTorch .pt      (ANN, 30 features — includes embedded scaler)

Optional post-scoring enhancements (when artifact files are present):
  - Platt calibration via platt_calibrator.joblib
  - Threshold-based binary classification via optimal_thresholds.json
  - MC Dropout uncertainty via PyTorch ANN (N=50 forward passes)

Prototype mode (no model file): trains inline on the feature data using a
RandomForestClassifier with binding-derived pseudo-labels.  NOT scientifically
valid — exists only for end-to-end pipeline testing.

Immunogenicity scores are probabilities in [0, 1] from predict_proba.
Higher score = higher predicted immunogenicity.  Rank 1 = top candidate.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.features import (
    FEATURE_COLUMNS, TRAIN_FEATURE_COLUMNS, FEATURE_COLUMNS_30, FEATURE_COLUMNS_50,
)
from src.naming import resolve_model_path

try:
    from joblib import load as joblib_load
except ImportError:
    joblib_load = None


def _load_pytorch_model(model_path, features_df, model_cols):
    """Load a PyTorch .pt checkpoint and score peptides on the 30-feature set."""
    import torch
    import torch.nn as nn

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    n_features = checkpoint['n_features']
    scaler_mean = checkpoint['scaler_mean'].numpy()
    scaler_scale = checkpoint['scaler_scale'].numpy()

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
    model.load_state_dict(checkpoint['model_state_dict'])
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


def _apply_calibration(scores, model_dir):
    """Apply Platt calibrator if available; return calibrated scores or originals."""
    cal_path = os.path.join(model_dir, 'platt_calibrator.joblib')
    if not os.path.isfile(cal_path) or joblib_load is None:
        return scores, False
    calibrator = joblib_load(cal_path)
    logits = np.log((scores + 1e-10) / (1 - scores + 1e-10)).reshape(-1, 1)
    calibrated = calibrator.predict_proba(logits)[:, 1]
    print("[Stage 4] Applied Platt calibration")
    return calibrated, True


def _apply_thresholds(features_df, model_dir):
    """Add binary immunogenic column using exported optimal thresholds."""
    thresh_path = os.path.join(model_dir, 'optimal_thresholds.json')
    if not os.path.isfile(thresh_path):
        return
    with open(thresh_path) as f:
        thresholds = json.load(f)
    score_col = 'calibrated_score' if 'calibrated_score' in features_df.columns else 'immunogenicity_score'
    f1_thresh = thresholds.get('threshold', thresholds.get('f1_threshold', 0.5))
    features_df['immunogenic'] = (features_df[score_col] >= f1_thresh).astype(int)
    print(f"[Stage 4] Applied F1-optimal threshold {f1_thresh:.3f}")


def score_immunogenicity(features_df, proteome_id, model_path=None,
                         calibrate=True, mc_dropout=False, freeze_mode=False):
    """
    Score each peptide's immunogenicity.

    Auto-detects model type (.joblib vs .pt) and feature dimensionality
    (21, 22, or 30 features).

    Args:
        features_df: DataFrame with feature columns from Stage 3
        proteome_id: label used in output filename
        model_path:  path to a serialized model (optional)
        calibrate:   apply Platt calibration if calibrator file exists
        mc_dropout:  run MC Dropout uncertainty (PyTorch models only)

    Returns:
        (ranked_df, model) tuple.
    """
    model = None
    if model_path:
        resolved_path = resolve_model_path(model_path)
        if resolved_path != model_path:
            print(f"[Stage 4] Using alias model path '{resolved_path}' (from '{model_path}')")
        model_path = resolved_path
    model_dir = os.path.dirname(model_path) if model_path else 'models'

    if model_path and os.path.isfile(model_path):
        is_pytorch = model_path.endswith('.pt')

        if is_pytorch:
            import torch
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            expected_n = checkpoint['n_features']
            
            if expected_n == len(FEATURE_COLUMNS_50):
                model_cols = [c for c in FEATURE_COLUMNS_50 if c in features_df.columns]
                expected_list = FEATURE_COLUMNS_50
            elif expected_n == len(FEATURE_COLUMNS_30):
                model_cols = [c for c in FEATURE_COLUMNS_30 if c in features_df.columns]
                expected_list = FEATURE_COLUMNS_30
            elif expected_n == len(TRAIN_FEATURE_COLUMNS):
                model_cols = [c for c in TRAIN_FEATURE_COLUMNS if c in features_df.columns]
                expected_list = TRAIN_FEATURE_COLUMNS
            else:
                model_cols = [c for c in FEATURE_COLUMNS if c in features_df.columns]
                expected_list = FEATURE_COLUMNS

            if len(model_cols) < expected_n:
                missing = set(expected_list) - set(features_df.columns)
                missing_sorted = sorted(missing)
                msg = f"[Stage 4] Missing {len(missing)} of {expected_n} ANN features: {missing_sorted}"
                if freeze_mode:
                    raise RuntimeError(msg)
                print(f"[Stage 4] WARNING: {msg}")
            scores, model = _load_pytorch_model(model_path, features_df, model_cols)
            features_df['immunogenicity_score'] = scores
            print(f"[Stage 4] Loaded PyTorch ANN from {model_path} ({len(model_cols)} features)")

            if mc_dropout:
                import torch
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                X = features_df[model_cols].values.astype(np.float64)
                X_scaled = (X - checkpoint['scaler_mean'].numpy()) / (checkpoint['scaler_scale'].numpy() + 1e-10)
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
                mc_mean, mc_std = _mc_dropout_predict(model, X_tensor)
                features_df['mc_score'] = mc_mean
                features_df['uncertainty_std'] = mc_std
                print(f"[Stage 4] MC Dropout: {(mc_std < np.median(mc_std)).sum()} "
                      f"high-confidence predictions")

        elif joblib_load is not None:
            model = joblib_load(model_path)
            expected_n = model.n_features_in_

            if expected_n == len(FEATURE_COLUMNS_50):
                model_cols = [c for c in FEATURE_COLUMNS_50 if c in features_df.columns]
                print(f"[Stage 4] Using {len(model_cols)} features (50-feature multi-allele mode)")
            elif expected_n == len(FEATURE_COLUMNS_30):
                model_cols = [c for c in FEATURE_COLUMNS_30 if c in features_df.columns]
                print(f"[Stage 4] Using {len(model_cols)} features (30-feature multi-allele mode)")
            elif expected_n == len(TRAIN_FEATURE_COLUMNS):
                model_cols = [c for c in TRAIN_FEATURE_COLUMNS if c in features_df.columns]
                print(f"[Stage 4] Using {len(model_cols)} sequence-only features (binding_score excluded)")
            else:
                model_cols = [c for c in FEATURE_COLUMNS if c in features_df.columns]
                print(f"[Stage 4] Using {len(model_cols)} features (full legacy set)")

            X = features_df[model_cols].copy()
            features_df['immunogenicity_score'] = model.predict_proba(X)[:, 1]
            print(f"[Stage 4] Loaded trained model from {model_path}")
        else:
            print("[Stage 4] WARNING: joblib not available, cannot load .joblib model")

    if 'immunogenicity_score' not in features_df.columns:
        if freeze_mode:
            raise RuntimeError(
                "[Stage 4] Freeze mode requires a trained model; prototype inline "
                "classifier fallback is disabled."
            )
        from sklearn.ensemble import RandomForestClassifier

        available_cols = [c for c in FEATURE_COLUMNS if c in features_df.columns]
        X = features_df[available_cols].copy()

        if 'binding_score' in features_df.columns:
            median_binding = features_df['binding_score'].median()
            pseudo_labels = (features_df['binding_score'] >= median_binding).astype(int).values
        elif 'presentation_score' in features_df.columns:
            median_ps = features_df['presentation_score'].median()
            pseudo_labels = (features_df['presentation_score'] >= median_ps).astype(int).values
        else:
            pseudo_labels = np.zeros(len(features_df), dtype=int)

        model = RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',
            random_state=42,
            n_jobs=1,
        )
        model.fit(X, pseudo_labels)
        features_df['immunogenicity_score'] = model.predict_proba(X)[:, 1]
        print("[Stage 4] No trained model found — used prototype inline classifier "
              "(NOT scientifically valid)")

    if calibrate:
        cal_scores, was_calibrated = _apply_calibration(
            features_df['immunogenicity_score'].values, model_dir
        )
        if was_calibrated:
            features_df['calibrated_score'] = cal_scores

    _apply_thresholds(features_df, model_dir)

    # Use deterministic contiguous ranking (1..N) based on score ordering.
    # Prefer calibrated_score when available (Platt calibration is monotonic,
    # but boundary rounding can shift relative order for tied raw scores).
    rank_col = 'calibrated_score' if 'calibrated_score' in features_df.columns else 'immunogenicity_score'
    features_df = features_df.sort_values(
        by=[rank_col, 'peptide'],
        ascending=[False, True]
    ).reset_index(drop=True)
    features_df['rank'] = features_df.index + 1

    output_path = f"results/{proteome_id}_ranked.csv"
    features_df.to_csv(output_path, index=False)
    print(f"[Stage 4] Scored and ranked {len(features_df)} peptides")

    return features_df, model


def plot_immunogenicity_scores(ranked_df, proteome_id, top_n=20):
    """Save top-N bar chart and score distribution histogram."""
    os.makedirs("results", exist_ok=True)

    top_df = ranked_df.head(top_n)
    plt.figure(figsize=(10, 6))
    plt.barh(top_df["peptide"], top_df["immunogenicity_score"], color='#4C72B0')
    plt.xlabel("Immunogenicity Score")
    plt.ylabel("Peptide")
    plt.title(f"Top {top_n} Immunogenic Peptides — {proteome_id}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"results/{proteome_id}_top{top_n}_immunogenicity.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(ranked_df["immunogenicity_score"], bins=50, color='#DD8452', alpha=0.85)
    plt.xlabel("Immunogenicity Score")
    plt.ylabel("Number of Peptides")
    plt.title(f"Immunogenicity Score Distribution — {proteome_id}")
    plt.tight_layout()
    plt.savefig(f"results/{proteome_id}_score_distribution.png", dpi=150)
    plt.close()

    print(f"[Plot] Saved plots for {proteome_id}")
