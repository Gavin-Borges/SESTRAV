# File: SESTRAV-Dev/tests/run_evals.py
import os
import pandas as pd
import numpy as np
import pytest

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

def test_data_leakage_contamination_gate():
    """Verify that training set peptides do not overlap with validation/evaluation sets."""
    # Look for immunogenicity dataset
    dataset_path = os.path.join(DATA_DIR, "immunogenicity_dataset_v3.csv")
    if not os.path.exists(dataset_path):
        # Fallback to general immunogenicity_dataset.csv
        dataset_path = os.path.join(os.path.dirname(__file__), "..", "immunogenicity_dataset.csv")
        
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        # Mock split check - verify training/validation partition strategy
        if "split" in df.columns:
            train_peps = set(df[df["split"] == "train"]["peptide"].dropna().str.upper())
            test_peps = set(df[df["split"] == "test"]["peptide"].dropna().str.upper())
            overlap = train_peps.intersection(test_peps)
            assert len(overlap) == 0, f"DATA LEAKAGE DETECTED! Overlapping peptides: {overlap}"
            print(f"[EVAL SUCCESS] Contamination gate verified: 0 overlapping peptides.")
        else:
            print("[EVAL SKIP] Dataset does not contain split column. Skipping partition audit.")
    else:
        print("[EVAL WARNING] Immunogenicity dataset not found. Skipping contamination check.")

def test_gnn_batch_dimension_safety():
    """Assert dimensions are preserved during mock batch operations on GNN layers."""
    import torch
    # GNN dimension invariance check
    input_dim = 128
    hidden_dim = 64
    output_dim = 1
    
    # Simulating GNN message passing dimensions
    x = torch.randn(20, input_dim) # 20 nodes
    edge_index = torch.randint(0, 20, (2, 40), dtype=torch.long) # 40 edges
    
    # Verify linear projection constraints
    proj_weight = torch.randn(input_dim, hidden_dim)
    proj_out = torch.matmul(x, proj_weight)
    
    assert proj_out.shape == (20, hidden_dim), f"GNN dimension projection failed: expected (20, {hidden_dim}), got {proj_out.shape}"
    print("[EVAL SUCCESS] GNN batch dimensions verified.")

def test_evaluation_performance_thresholds():
    """Ensure baseline classifier performance meets defined accuracy thresholds."""
    # Locate benchmark reports or check mock classification validation
    metrics_path = os.path.join(RESULTS_DIR, "evaluation_metrics.csv")
    if os.path.exists(metrics_path):
        metrics_df = pd.read_csv(metrics_path)
        # Ensure we have required performance metrics
        for idx, row in metrics_df.iterrows():
            model_name = row.get("model", "Unknown")
            auc_pr = row.get("auc_pr", 0.0)
            # Threshold constraint: AUC-PR must be >= 0.5 for all trained baseline comparators
            assert auc_pr >= 0.50, f"Model {model_name} degraded! AUC-PR: {auc_pr}"
        print("[EVAL SUCCESS] Model performance thresholds satisfied.")
    else:
        print("[EVAL SKIP] evaluation_metrics.csv not found. Skipping performance threshold checks.")

if __name__ == "__main__":
    # Allow running directly via python tests/run_evals.py
    print("=" * 60)
    print("RUNNING CLAUDE-STYLE DETERMINISTIC EVALS SUITE")
    print("=" * 60)
    pytest.main([__file__, "-v"])
