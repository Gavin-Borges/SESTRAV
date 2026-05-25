"""
SESTRAV Binding-Only Baseline Comparison

Demonstrates that SESTRAV's TCR-contact features add predictive value
beyond MHC binding alone by comparing four ranking strategies on
gold-standard epitope recovery:

  1. RF (SESTRAV default)  — rank by immunogenicity_score from trained RF
  2. XGB                   — rank by immunogenicity_score from trained XGBoost
  3. ANN (MLP)             — rank by immunogenicity_score from trained ANN
  4. Binding-only baseline — rank by MHCflurry presentation_score (no ML)

For each strategy, we compute:
  - Gold-standard recovery at top 10% and top 25%
  - Mean rank percentile of gold-standard epitopes
  - ISSR@10 and ISSR@25 on the gold-standard subset

Usage (after pipeline.py has run):
    python -m src.baseline_comparison --results-dir results
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.artifact_integrity import load_verified_joblib, verify_artifact_checksum
from src.features import FEATURE_COLUMNS, FEATURE_COLUMNS_30, TRAIN_FEATURE_COLUMNS
from src.gold_standard import GOLD_STANDARD, VIRUS_FILE_MAP
from src.naming import proteome_id_candidates, resolve_model_path

try:
    import torch
    from src.model import FlexibleMLP
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _select_model_columns(features_df, model):
    """Select compatible feature columns from a model's expected width."""
    expected_n = getattr(model, "n_features_in_", None)
    if expected_n == len(FEATURE_COLUMNS_30):
        return [c for c in FEATURE_COLUMNS_30 if c in features_df.columns]
    if expected_n == len(TRAIN_FEATURE_COLUMNS):
        return [c for c in TRAIN_FEATURE_COLUMNS if c in features_df.columns]
    return [c for c in FEATURE_COLUMNS if c in features_df.columns]


def _score_with_model(features_df, model, binding_matrix_path=None):
    """Score peptides using a trained model's predict_proba."""
    expected_n = getattr(model, "n_features_in_", None)
    if expected_n == len(FEATURE_COLUMNS_30) and binding_matrix_path:
        from src.train_classifier import prepare_features_30
        # If the features_df lacks the bind_ columns, we must compute them.
        if 'bind_A0101' not in features_df.columns:
            features_df = prepare_features_30(features_df, binding_matrix_path)
    cols = _select_model_columns(features_df, model)
    X = features_df[cols]
    return model.predict_proba(X)[:, 1]


def _load_torch_checkpoint(model_path):
    """Load ANN checkpoints using the safe weights-only path only."""
    verify_artifact_checksum(model_path, required=False)
    try:
        return torch.load(model_path, map_location='cpu', weights_only=True)
    except Exception:
        raise RuntimeError(
            f"[Baseline] Unable to load ANN checkpoint '{model_path}' with weights_only=True. "
            "Resave the checkpoint from a trusted training run before loading it."
        )


def _score_with_ann(features_df, model_path):
    """Score peptides using a trained ANN checkpoint.

    ANN checkpoints can be either legacy 21-feature or canonical 30-feature.
    This helper reads checkpoint metadata and aligns feature columns to the
    expected schema. Returned values are probabilities (sigmoid of logits).
    """

    checkpoint = _load_torch_checkpoint(model_path)
    n_features = checkpoint['n_features']

    if n_features == len(FEATURE_COLUMNS_30):
        cols = [c for c in FEATURE_COLUMNS_30 if c in features_df.columns]
    elif n_features == len(TRAIN_FEATURE_COLUMNS):
        cols = [c for c in TRAIN_FEATURE_COLUMNS if c in features_df.columns]
    else:
        raise ValueError(
            f"Unsupported ANN checkpoint feature width: {n_features} "
            f"(path: {model_path})"
        )
    if len(cols) != n_features:
        raise ValueError(
            f"ANN feature mismatch for {model_path}: expected {n_features}, got {len(cols)}"
        )
    X = features_df[cols].values

    scaler_mean = checkpoint['scaler_mean']
    scaler_scale = checkpoint['scaler_scale']
    if hasattr(scaler_mean, "numpy"):
        scaler_mean = scaler_mean.numpy()
    if hasattr(scaler_scale, "numpy"):
        scaler_scale = scaler_scale.numpy()

    scaler = StandardScaler()
    scaler.mean_ = scaler_mean
    scaler.scale_ = scaler_scale
    scaler.n_features_in_ = n_features
    X_scaled = scaler.transform(X)

    arch_meta = checkpoint.get('architecture', {})
    hidden = arch_meta.get('hidden', [64, 32])
    activation = arch_meta.get('activation', 'relu')
    dropout = arch_meta.get('dropout', 0.3)
    model = FlexibleMLP(
        input_dim=n_features,
        hidden_sizes=hidden,
        dropout=dropout,
        activation=activation,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        logits = model(X_tensor).numpy()
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))


def _rank_and_evaluate(df, score_col, gs_peptides, label):
    """Rank by score_col, compute gold-standard recovery metrics."""
    n = len(df)
    df = df.copy()
    df['_rank'] = df[score_col].rank(ascending=False).astype(int)

    top_10_cutoff = n * 0.10
    top_25_cutoff = n * 0.25

    gs_rows = df[df['peptide'].isin(gs_peptides)]
    n_gs = len(gs_rows)
    if n_gs == 0:
        return None

    found = n_gs
    in_top_10 = int((gs_rows['_rank'] <= top_10_cutoff).sum())
    in_top_25 = int((gs_rows['_rank'] <= top_25_cutoff).sum())
    mean_rank_pct = float((gs_rows['_rank'] / n * 100).mean())
    median_rank_pct = float((gs_rows['_rank'] / n * 100).median())

    return {
        'method': label,
        'n_peptides': n,
        'gs_found': found,
        'gs_in_top_10pct': in_top_10,
        'gs_in_top_25pct': in_top_25,
        'gs_recovery_top10': in_top_10 / found,
        'gs_recovery_top25': in_top_25 / found,
        'mean_rank_pct': mean_rank_pct,
        'median_rank_pct': median_rank_pct,
    }


def compare_methods(results_dir, model_dir='models', strict_ann_loading=False, binding_matrix_path='models/peptide_binding_matrix.csv'):
    """Run the 3-way comparison on each virus and combined."""
    def _load_optional(paths):
        for path in paths:
            if os.path.isfile(path):
                return load_verified_joblib(path, required_checksum=True), path
        return None, None

    rf_model, rf_path = _load_optional([
        resolve_model_path(os.path.join(model_dir, 'rf_30feature_integrated.joblib')),
        resolve_model_path(os.path.join(model_dir, 'rf_21feature_legacy.joblib')),
    ])
    xgb_model, xgb_path = _load_optional([
        resolve_model_path(os.path.join(model_dir, 'xgb_30feature_integrated.joblib')),
        resolve_model_path(os.path.join(model_dir, 'xgb_21feature_legacy.joblib')),
    ])

    if rf_path:
        print(f"[Baseline] Loaded RF model: {rf_path}")
    else:
        print("[Baseline] RF model not found; skipping RF comparison")
    if xgb_path:
        print(f"[Baseline] Loaded XGBoost model: {xgb_path}")
    else:
        print("[Baseline] XGBoost model not found; skipping XGBoost comparison")

    all_rows = []

    for virus, prefix in VIRUS_FILE_MAP.items():
        features_path = None
        for cand in proteome_id_candidates(prefix):
            path = os.path.join(results_dir, f"{cand}_features.csv")
            if os.path.isfile(path):
                features_path = path
                break
        if features_path is None:
            features_path = os.path.join(results_dir, f'{prefix}_features.csv')
        if not os.path.isfile(features_path):
            print(f"[Baseline] Skipping {virus}: {features_path} not found")
            continue

        df = pd.read_csv(features_path)
        gs_peptides = {gs['peptide'] for gs in GOLD_STANDARD if gs['virus'] == virus}

        ann_path = None
        if _HAS_TORCH:
            for candidate in [
                'ann_30feature_integrated.pt',
                'ann_21feature_legacy.pt',
            ]:
                resolved = resolve_model_path(os.path.join(model_dir, candidate))
                if os.path.isfile(resolved):
                    ann_path = resolved
                    break
        has_ann = ann_path is not None
        methods = []

        if rf_model is not None:
            df['rf_score'] = _score_with_model(df, rf_model, binding_matrix_path)
            methods.append(('rf_score', 'RF (SESTRAV)'))
        if xgb_model is not None:
            df['xgb_score'] = _score_with_model(df, xgb_model, binding_matrix_path)
            methods.append(('xgb_score', 'XGBoost'))
        if has_ann:
            try:
                df['ann_score'] = _score_with_ann(df, ann_path)
                methods.append(('ann_score', 'ANN (MLP)'))
            except Exception as exc:
                if strict_ann_loading:
                    raise RuntimeError(
                        f"[Baseline] Freeze mode requires ANN checkpoint compatibility: {exc}"
                    ) from exc
                print(f"[Baseline] Skipping ANN due to checkpoint load/scoring error: {exc}")

        binding_col = 'presentation_score' if 'presentation_score' in df.columns else 'binding_score'
        if binding_col in df.columns:
            methods.append((binding_col, 'Binding-only baseline'))

        for score_col, label in methods:
            row = _rank_and_evaluate(df, score_col, gs_peptides, label)
            if row:
                row['virus'] = virus
                all_rows.append(row)

    columns = [
        'method', 'virus', 'n_peptides', 'gs_found', 'gs_in_top_10pct',
        'gs_in_top_25pct', 'gs_recovery_top10', 'gs_recovery_top25',
        'mean_rank_pct', 'median_rank_pct'
    ]
    results = pd.DataFrame(all_rows, columns=columns)

    if results.empty:
        return results

    combined_rows = []
    all_methods = ['RF (SESTRAV)', 'XGBoost', 'ANN (MLP)', 'Binding-only baseline']
    for method in all_methods:
        m = results[results['method'] == method]
        if m.empty:
            continue
        combined_rows.append({
            'method': method,
            'virus': 'Combined',
            'n_peptides': int(m['n_peptides'].sum()),
            'gs_found': int(m['gs_found'].sum()),
            'gs_in_top_10pct': int(m['gs_in_top_10pct'].sum()),
            'gs_in_top_25pct': int(m['gs_in_top_25pct'].sum()),
            'gs_recovery_top10': int(m['gs_in_top_10pct'].sum()) / max(int(m['gs_found'].sum()), 1),
            'gs_recovery_top25': int(m['gs_in_top_25pct'].sum()) / max(int(m['gs_found'].sum()), 1),
            'mean_rank_pct': float(m['mean_rank_pct'].mean()),
            'median_rank_pct': float(m['median_rank_pct'].mean()),
        })

    results = pd.concat([results, pd.DataFrame(combined_rows)], ignore_index=True)
    return results


def print_comparison(results):
    """Pretty-print the comparison table."""
    print("\n" + "=" * 80)
    print("SESTRAV vs BINDING-ONLY BASELINE COMPARISON")
    print("=" * 80)

    for virus in ['EBV', 'HPV', 'Combined']:
        v_df = results[results['virus'] == virus]
        if v_df.empty:
            continue
        print(f"\n  {virus}:")
        print(f"  {'Method':<25s} {'Found':>5s} {'Top10%':>7s} {'Top25%':>7s} {'MeanRk%':>8s} {'MedRk%':>7s}")
        print("  " + "-" * 65)
        for _, row in v_df.iterrows():
            print(f"  {row['method']:<25s} "
                  f"{int(row['gs_found']):>5d} "
                  f"{int(row['gs_in_top_10pct']):>4d}   "
                  f"{int(row['gs_in_top_25pct']):>4d}   "
                  f"{row['mean_rank_pct']:>7.1f}% "
                  f"{row['median_rank_pct']:>6.1f}%")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SESTRAV baseline comparison')
    parser.add_argument('--results-dir', default='results',
                        help='Directory containing pipeline output CSVs')
    parser.add_argument('--model-dir', default='models',
                        help='Directory containing .joblib model files')
    args = parser.parse_args()

    results = compare_methods(args.results_dir, args.model_dir)

    out_path = os.path.join(args.results_dir, 'baseline_comparison.csv')
    results.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    print_comparison(results)
