# File: SESTRAV-Dev/tests/test_run_evals.py
#
# Renamed from tests/run_evals.py so pytest actually collects it.
# The default `python_files` globs are `test_*.py` and `*_test.py`; the old name
# matched neither, so the checks here were collected ZERO times by the normal
# suite and had never executed. Naming the file on the command line collected it,
# which is why a casual check looked fine. The file held three checks at that
# rename and holds two now: test_data_leakage_contamination_gate was retired
# on 2026-09-21; see the commit that removed it.
#
# The assertions below are unchanged from the original. What changed is that the
# branches which used to `print(...)` and pass silently now `pytest.skip(...)` with
# the concrete path or column that was missing. A gate that passes because it
# compared two empty sets is a false PASS, and a false PASS is silent.
import os

import pandas as pd
import pytest

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def test_gnn_batch_dimension_safety():
    """Assert a dense (nodes, in) x (in, hidden) projection preserves the node dimension.

    Scope note: this exercises `torch.matmul` on locally constructed tensors. It does
    NOT import or instantiate any SESTRAV graph encoder, so it cannot detect a
    regression in `src/`. It is a torch sanity check, not a SESTRAV gate. Pointing it
    at the real `GraphEncoderV2` is an owner decision, so the assertion is left as-is
    rather than quietly deleted.
    """
    import torch

    # GNN dimension invariance check
    input_dim = 128
    hidden_dim = 64
    output_dim = 1  # noqa: F841 - retained from the original for parity

    # Simulating GNN message passing dimensions
    x = torch.randn(20, input_dim)  # 20 nodes
    edge_index = torch.randint(0, 20, (2, 40), dtype=torch.long)  # noqa: F841 - never consumed

    # Verify linear projection constraints
    proj_weight = torch.randn(input_dim, hidden_dim)
    proj_out = torch.matmul(x, proj_weight)

    assert proj_out.shape == (20, hidden_dim), (
        f"GNN dimension projection failed: expected (20, {hidden_dim}), got {proj_out.shape}"
    )
    print("[EVAL SUCCESS] torch.matmul preserved the node dimension (no SESTRAV code exercised).")


def test_evaluation_performance_thresholds():
    """Ensure baseline classifier performance meets defined accuracy thresholds."""
    # Locate benchmark reports or check mock classification validation
    metrics_path = os.path.join(RESULTS_DIR, "evaluation_metrics.csv")
    if not os.path.exists(metrics_path):
        pytest.skip(
            f"{os.path.abspath(metrics_path)} does not exist, so no model metrics "
            "are available for this gate to threshold"
        )

    metrics_df = pd.read_csv(metrics_path)

    # An empty table iterates zero times and would certify nothing while reporting green.
    assert not metrics_df.empty, (
        f"{os.path.abspath(metrics_path)} has 0 rows; there are no models to threshold"
    )
    assert "auc_pr" in metrics_df.columns, (
        f"{os.path.abspath(metrics_path)} has no 'auc_pr' column "
        f"(columns present: {sorted(metrics_df.columns)}); the threshold check would "
        "compare a default of 0.0 for every row"
    )

    # Ensure we have required performance metrics
    for _idx, row in metrics_df.iterrows():
        model_name = row.get("model", "Unknown")
        auc_pr = row.get("auc_pr", 0.0)
        # Threshold constraint: AUC-PR must be >= 0.5 for all trained baseline comparators
        assert auc_pr >= 0.50, f"Model {model_name} degraded! AUC-PR: {auc_pr}"
    print(f"[EVAL SUCCESS] Model performance thresholds satisfied for {len(metrics_df)} rows.")


if __name__ == "__main__":
    # Allow running directly via python tests/test_run_evals.py
    print("=" * 60)
    print("RUNNING DETERMINISTIC EVALS SUITE")
    print("=" * 60)
    pytest.main([__file__, "-v"])
