"""
SESTRAV Calibration Analysis

Computes Brier scores and reliability diagrams for OOF predictions,
comparing v1 and v2 datasets to assess probability calibration quality.
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def _load_oof(path, method='RandomForest'):
    """Load OOF predictions for a given method from the standard CSV format."""
    df = pd.read_csv(path)
    if 'method' in df.columns:
        df = df[df['method'] == method]
    return df


def compute_calibration_metrics(y_true, y_prob, n_bins=10):
    """Compute Brier score and binned calibration curve."""
    brier = brier_score_loss(y_true, y_prob)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins,
                                             strategy='uniform')
    trivial_brier = brier_score_loss(y_true, np.full_like(y_prob, y_true.mean()))
    return {
        'brier_score': brier,
        'trivial_brier': trivial_brier,
        'brier_skill': 1.0 - brier / trivial_brier if trivial_brier > 0 else 0.0,
        'prob_true': prob_true,
        'prob_pred': prob_pred,
    }


def run_calibration_analysis(v2_oof_path, v1_oof_path=None,
                             output_dir='results', method='RandomForest'):
    """Run calibration analysis and produce reliability diagram + Brier scores."""
    os.makedirs(output_dir, exist_ok=True)

    v2_df = _load_oof(v2_oof_path, method)
    v2_metrics = compute_calibration_metrics(
        v2_df['label'].values, v2_df['score'].values)

    results = [{
        'dataset': 'v2',
        'method': method,
        'n_samples': len(v2_df),
        'positive_rate': v2_df['label'].mean(),
        'brier_score': v2_metrics['brier_score'],
        'trivial_brier': v2_metrics['trivial_brier'],
        'brier_skill_score': v2_metrics['brier_skill'],
    }]

    print("=" * 60)
    print("CALIBRATION ANALYSIS")
    print("=" * 60)
    print(f"  v2: Brier={v2_metrics['brier_score']:.4f}, "
          f"Trivial={v2_metrics['trivial_brier']:.4f}, "
          f"Skill={v2_metrics['brier_skill']:.4f}")

    v1_metrics = None
    if v1_oof_path and os.path.isfile(v1_oof_path):
        v1_df = _load_oof(v1_oof_path, method)
        v1_metrics = compute_calibration_metrics(
            v1_df['label'].values, v1_df['score'].values)
        results.append({
            'dataset': 'v1',
            'method': method,
            'n_samples': len(v1_df),
            'positive_rate': v1_df['label'].mean(),
            'brier_score': v1_metrics['brier_score'],
            'trivial_brier': v1_metrics['trivial_brier'],
            'brier_skill_score': v1_metrics['brier_skill'],
        })
        print(f"  v1: Brier={v1_metrics['brier_score']:.4f}, "
              f"Trivial={v1_metrics['trivial_brier']:.4f}, "
              f"Skill={v1_metrics['brier_skill']:.4f}")
    print("=" * 60)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated', alpha=0.5)
    ax.plot(v2_metrics['prob_pred'], v2_metrics['prob_true'],
            's-', label=f"v2 (Brier={v2_metrics['brier_score']:.4f})", linewidth=2)
    if v1_metrics:
        ax.plot(v1_metrics['prob_pred'], v1_metrics['prob_true'],
                'o-', label=f"v1 (Brier={v1_metrics['brier_score']:.4f})", linewidth=2)
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title(f'{method} — Reliability Diagram (OOF)')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'calibration_reliability_diagram.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Reliability diagram saved to {output_dir}/calibration_reliability_diagram.png")

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    bins = np.linspace(0, 1, 21)
    ax.hist(v2_df['score'].values, bins=bins, alpha=0.6, label='v2', edgecolor='black')
    if v1_oof_path and os.path.isfile(v1_oof_path):
        v1_df_full = _load_oof(v1_oof_path, method)
        ax.hist(v1_df_full['score'].values, bins=bins, alpha=0.4, label='v1', edgecolor='black')
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Count')
    ax.set_title(f'{method} — OOF Score Distribution')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'calibration_score_distribution.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Score distribution saved to {output_dir}/calibration_score_distribution.png")

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'calibration_metrics.csv'), index=False)
    print(f"  Metrics CSV saved to {output_dir}/calibration_metrics.csv")
    return results_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SESTRAV calibration analysis')
    parser.add_argument('--v2-oof', default='models/rf_oof_predictions.csv')
    parser.add_argument('--v1-oof', default='models/v1_backup/rf_oof_predictions.csv')
    parser.add_argument('--output-dir', default='results')
    parser.add_argument('--method', default='RandomForest')
    args = parser.parse_args()
    run_calibration_analysis(args.v2_oof, args.v1_oof, args.output_dir, args.method)
