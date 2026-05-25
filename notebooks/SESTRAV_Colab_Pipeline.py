# -*- coding: utf-8 -*-
"""
SESTRAV Colab Pipeline — Full Model Training & Evaluation
=========================================================

This script is designed to run in Google Colab. It clones the SESTRAV
repository, installs dependencies, and runs the complete model training
and evaluation pipeline including ANN architecture search, GNN benchmarks,
and ablation studies.

To use: Copy cells into a Colab notebook or upload as a .py file.

Repository: https://github.com/Gavin-Borges/SESTRAV
"""

# =============================================================================
# Cell 1: Setup & Installation
# =============================================================================

# Install core dependencies
# !pip install mhcflurry shap xgboost tqdm -q

# Install PyTorch (Colab usually has this pre-installed)
# !pip install torch -q

# Install torch-geometric for GNN benchmarks (optional)
# !pip install torch-geometric -q

# Download MHCflurry models (only needed if running Stage 2 from scratch)
# !mhcflurry-downloads fetch models_class1_presentation

# =============================================================================
# Cell 2: Clone Repository & Setup Paths
# =============================================================================

import os
import subprocess  # nosec B404
import sys

# Clone the repository (skip if already cloned)
REPO_DIR = "/content/SESTRAV"
if not os.path.isdir(REPO_DIR):
    subprocess.run(  # nosec B603 B607
        ["git", "clone", "https://github.com/Gavin-Borges/SESTRAV.git", REPO_DIR],
        check=True,
    )

# Add repo root to Python path so src/ imports work
sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)
print(f"Working directory: {os.getcwd()}")

# =============================================================================
# Cell 3: Load Data
# =============================================================================

import numpy as np
import pandas as pd

from src.iedb_data_loader import GOLD_STANDARD_EPITOPES

# Load the immunogenicity dataset
DATA_PATH = "immunogenicity_dataset.csv"
BINDING_MATRIX_PATH = "models/peptide_binding_matrix.csv"

df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} records from {DATA_PATH}")
print(f"Columns: {list(df.columns)}")
print(f"\nClass distribution:")
print(df['label'].value_counts())
print(f"\nPositive rate: {df['label'].mean():.1%}")

# Hold out gold-standard epitopes
gs_mask = df['peptide'].isin(GOLD_STANDARD_EPITOPES)
train_pool = df[~gs_mask].copy()
print(f"\nHeld out {gs_mask.sum()} gold-standard epitope records")
print(f"Training pool: {len(train_pool)} records")

# =============================================================================
# Cell 4: Prepare Feature Matrices
# =============================================================================

from src.features import (
    compute_features, TRAIN_FEATURE_COLUMNS, FEATURE_COLUMNS_30,
    PHYSICO_COLUMNS, BINDING_ALLELE_COLUMNS,
)
from src.train_classifier import prepare_features, prepare_features_30

# 21-feature matrix (legacy, sequence-only)
X_21 = prepare_features(train_pool, include_binding=False)
print(f"21-feature matrix: {X_21.shape}")

# 30-feature matrix (canonical, physico + multi-allele binding)
X_30 = prepare_features_30(train_pool, BINDING_MATRIX_PATH)
print(f"30-feature matrix: {X_30.shape}")

y = train_pool['label'].values
virus = train_pool['virus'].values if 'virus' in train_pool.columns else np.zeros(len(train_pool))
strat_key = np.array([f"{l}_{v}" for l, v in zip(y, virus)])

print(f"\nClass balance: {np.mean(y):.2%} positive, {1 - np.mean(y):.2%} negative")

# =============================================================================
# Cell 5: Train RF / XGBoost Classifiers (Production Models)
# =============================================================================

from src.train_classifier import train_models

print("=" * 70)
print("TRAINING PRODUCTION CLASSIFIERS (RF + XGBoost)")
print("=" * 70)

# 30-feature canonical track
rf_model, xgb_model, rf_avg, xgb_avg = train_models(
    DATA_PATH,
    model_dir='models',
    feature_mode=30,
    binding_matrix_path=BINDING_MATRIX_PATH,
)

print(f"\nRF  30f AUC-PR: {rf_avg['auc_pr']:.4f}")
print(f"XGB 30f AUC-PR: {xgb_avg['auc_pr']:.4f}")

# =============================================================================
# Cell 6: Train ANN — Single Architecture (Project 2 Best)
# =============================================================================

from src.ann_benchmark import train_ann

print("\n" + "=" * 70)
print("TRAINING ANN (256-128-64 ReLU d0.2, 30 features)")
print("=" * 70)

ann_model, ann_scaler, ann_avg, ann_std = train_ann(
    DATA_PATH,
    model_dir='models',
    feature_mode=30,
    binding_matrix_path=BINDING_MATRIX_PATH,
    architecture='256-128-64',
)

print(f"\nANN 30f AUC-PR: {ann_avg['auc_pr']:.4f} +/- {ann_std['auc_pr']:.4f}")

# =============================================================================
# Cell 7: Architecture Search (Optional — ~30 min on T4 GPU)
# =============================================================================

# Uncomment to run the full 14-config architecture search:
#
# from src.ann_benchmark import train_ann
#
# ann_model_search, _, search_avg, search_std = train_ann(
#     DATA_PATH,
#     model_dir='models',
#     feature_mode=30,
#     binding_matrix_path=BINDING_MATRIX_PATH,
#     search=True,
# )
#
# # Results saved to models/ann_architecture_search.csv
# search_results = pd.read_csv('models/ann_architecture_search.csv')
# print(search_results[['config', 'auc_pr_mean', 'auc_roc_mean']].to_string())

# =============================================================================
# Cell 8: GNN Benchmarks (Optional — requires torch-geometric)
# =============================================================================

try:
    from src.gnn_benchmark import run_gnn_benchmark, run_bipartite_gnn_benchmark

    print("\n" + "=" * 70)
    print("GNN BENCHMARKS (GCN + GAT)")
    print("=" * 70)

    peptides = train_pool['peptide'].values
    n_pos = int(y.sum())
    pos_weight = (len(y) - n_pos) / max(1, n_pos)

    seq_results = run_gnn_benchmark(
        peptides, y, strat_key,
        pos_weight=pos_weight,
    )
    print("\nSequence GNN Results:")
    print(seq_results.to_string(index=False))

    # Bipartite GNN
    print("\n" + "=" * 70)
    print("BIPARTITE PEPTIDE-ALLELE GNN")
    print("=" * 70)

    X_physico = X_30[PHYSICO_COLUMNS].values
    X_binding = X_30[BINDING_ALLELE_COLUMNS].values

    bi_results = run_bipartite_gnn_benchmark(
        X_physico, X_binding, y, strat_key,
        pos_weight=pos_weight,
    )
    print("\nBipartite GNN Results:")
    print(bi_results.to_string(index=False))

except ImportError:
    print("\ntorch-geometric not installed. Skipping GNN benchmarks.")
    print("To install: !pip install torch-geometric")

# =============================================================================
# Cell 9: Ablation Study
# =============================================================================

from src.ablation_study import run_ablation

print("\n" + "=" * 70)
print("ABLATION STUDY")
print("=" * 70)

ablation_results = run_ablation(
    DATA_PATH,
    BINDING_MATRIX_PATH,
    output_dir='models',
)

print("\nAblation Results Summary:")
print(ablation_results[['feature_set', 'n_features', 'auc_pr_mean', 'auc_roc_mean']].to_string(index=False))

# =============================================================================
# Cell 10: Model Comparison Summary
# =============================================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON SUMMARY")
print("=" * 70)

summary = pd.DataFrame([
    {"Model": "RF (30f)", "AUC-PR": rf_avg['auc_pr'], "AUC-ROC": rf_avg['auc_roc']},
    {"Model": "XGBoost (30f)", "AUC-PR": xgb_avg['auc_pr'], "AUC-ROC": xgb_avg['auc_roc']},
    {"Model": "ANN 256-128-64 (30f)", "AUC-PR": ann_avg['auc_pr'], "AUC-ROC": ann_avg['auc_roc']},
])
print(summary.to_string(index=False))
print("\n(Higher AUC-PR = better; primary metric for imbalanced data)")

# =============================================================================
# Cell 11: Save Results to Google Drive (Optional)
# =============================================================================

# Uncomment to mount Google Drive and copy results:
#
# from google.colab import drive
# drive.mount('/content/drive')
#
# DRIVE_DIR = '/content/drive/MyDrive/SESTRAV_Results'
# os.makedirs(DRIVE_DIR, exist_ok=True)
#
# import shutil
# for f in ['models/training_results.csv', 'models/ann_architecture_search.csv',
#           'models/ablation_study_results.csv', 'models/ann_30feature_integrated.pt',
#           'models/rf_30feature_integrated.joblib', 'models/xgb_30feature_integrated.joblib']:
#     if os.path.isfile(f):
#         shutil.copy2(f, DRIVE_DIR)
#         print(f"Copied {f} -> {DRIVE_DIR}")
#
# print(f"\nAll results saved to {DRIVE_DIR}")
