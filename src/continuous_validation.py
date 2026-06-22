"""Continuous IEDB benchmark — scoring, baseline comparison, and regression alerting.

This logic was previously embedded in the monthly GitHub Actions job
(`.github/workflows/iedb_benchmark.yml`) as an inline heredoc, which made the
baseline-comparison and alerting behaviour impossible to unit-test. It is
extracted here as importable, testable functions per the Continuous Validation
Framework (MASTER_STRATEGIC_PLAN.md Part 11).

The monthly workflow fetches a fresh IEDB T-cell export (EBV + HPV16), scores it
with the pinned production model, and compares the AUC-PR to a stored baseline in
``data/validation_baselines.json``. If the AUC-PR degrades by more than 3%
*relative* to the baseline (Part 11 step 5), a ``REGRESSION_DETECTED`` marker file
is written, which the workflow uses to open a GitHub Issue automatically.

Importing this module is dependency-light: heavy scoring imports (scikit-learn,
joblib, the feature pipeline) are deferred into ``score_iedb_export`` so that the
pure comparison logic can be exercised without a model present.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import pathlib
from typing import Callable, Optional, Sequence

# ---------------------------------------------------------------------------
# Constants (Part 11 spec)
# ---------------------------------------------------------------------------

DEFAULT_BASELINE_PATH = "data/validation_baselines.json"
RESULTS_DIR = "results/continuous_validation"
DEFAULT_MARKER_PATH = "REGRESSION_DETECTED"

# Part 11 step 5: alert when AUC-PR degrades by more than 3% RELATIVE to baseline.
REGRESSION_REL_THRESHOLD = 0.03

# Key under which the EBV+HPV16 IEDB T-cell benchmark is stored in the baselines file.
BASELINE_KEY = "iedb_ebv_hpv16_tcell"


# ---------------------------------------------------------------------------
# Baseline I/O
# ---------------------------------------------------------------------------

def load_baseline(path: str = DEFAULT_BASELINE_PATH) -> dict:
    """Load the stored validation baselines, or ``{}`` if the file is absent."""
    if path and os.path.exists(path):
        return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    return {}


def baseline_auc_pr(baselines: dict, key: str = BASELINE_KEY) -> Optional[float]:
    """Return the stored baseline AUC-PR for ``key``, or ``None`` if unset.

    Tolerates both the nested schema (``{key: {"auc_pr": x}}``) and a flat
    ``{key: x}`` shape; a ``null``/missing value yields ``None``.
    """
    entry = baselines.get(key)
    if entry is None:
        return None
    if isinstance(entry, dict):
        val = entry.get("auc_pr")
    else:
        val = entry
    return float(val) if val is not None else None


# ---------------------------------------------------------------------------
# Regression logic (the unit-testable core)
# ---------------------------------------------------------------------------

def compute_regression(
    auc_pr: float,
    baseline: Optional[float],
    rel_threshold: float = REGRESSION_REL_THRESHOLD,
) -> dict:
    """Classify a fresh AUC-PR against a stored baseline.

    A regression is a *relative* drop greater than ``rel_threshold`` (Part 11
    specifies >3%). When no baseline exists yet (first run), this is not a
    regression — the current value seeds the baseline instead.

    Returns a dict with ``is_regression`` (bool), ``delta`` (absolute), and
    ``rel_delta`` (signed fraction of baseline), plus the inputs for logging.
    """
    if baseline is None:
        return {
            "is_regression": False,
            "delta": None,
            "rel_delta": None,
            "auc_pr": auc_pr,
            "baseline_auc_pr": None,
        }
    delta = auc_pr - baseline
    rel_delta = delta / baseline if baseline else 0.0
    return {
        "is_regression": rel_delta < -rel_threshold,
        "delta": delta,
        "rel_delta": rel_delta,
        "auc_pr": auc_pr,
        "baseline_auc_pr": baseline,
    }


def regression_message(result: dict, key: str = BASELINE_KEY) -> str:
    """Human-readable summary for the marker file / GitHub Issue body."""
    return (
        f"benchmark={key} current_auc_pr={result['auc_pr']:.4f} "
        f"baseline_auc_pr={result['baseline_auc_pr']:.4f} "
        f"delta={result['delta']:+.4f} rel_delta={result['rel_delta']:+.2%}"
    )


# ---------------------------------------------------------------------------
# Scoring (deferred heavy imports)
# ---------------------------------------------------------------------------

def load_inputs(paths: Sequence[str]):
    """Concatenate one or more IEDB export CSVs into a clean scoring frame.

    Returns a de-duplicated DataFrame with integer ``label``, or ``None`` if no
    rows were found across all inputs.
    """
    import pandas as pd

    frames = []
    for f in paths:
        if os.path.exists(f):
            df = pd.read_csv(f)
            if len(df) > 0:
                frames.append(df)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset="peptide")
    df = df.dropna(subset=["peptide", "label"])
    df["label"] = df["label"].astype(int)
    return df


def score_iedb_export(df, model_path: str, binding_matrix_path: str) -> dict:
    """Score a labeled peptide frame with the production model; return metrics."""
    from joblib import load
    from sklearn.metrics import average_precision_score

    from src.train_classifier import prepare_features_31

    clf = load(model_path)
    X = prepare_features_31(df, binding_matrix_path)
    proba = clf.predict_proba(X)[:, 1]
    auc_pr = float(average_precision_score(df["label"].values, proba))
    return {
        "auc_pr": auc_pr,
        "n_peptides": int(len(df)),
        "n_positive": int(df["label"].sum()),
    }


# ---------------------------------------------------------------------------
# Path resolution from config.yaml
# ---------------------------------------------------------------------------

def _resolve_paths(model_path: Optional[str], binding_matrix: Optional[str]) -> tuple:
    """Resolve model + binding-matrix paths, falling back to config.yaml."""
    cfg: dict = {}
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    try:
        import yaml

        if os.path.isfile(cfg_path):
            with open(cfg_path, encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
    except ImportError:
        pass

    model_path = model_path or cfg.get("model_path", "models/rf_31feature_integrated.joblib")
    binding_matrix = binding_matrix or cfg.get(
        "binding_matrix_path", "models/peptide_binding_matrix_v3.csv"
    )

    # Resolve canonical/legacy model aliases when the literal path is absent.
    try:
        from src.naming import resolve_model_path

        model_path = resolve_model_path(model_path)
    except ImportError:
        pass
    return model_path, binding_matrix


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(
    inputs: Sequence[str],
    *,
    baseline_path: str = DEFAULT_BASELINE_PATH,
    results_dir: str = RESULTS_DIR,
    model_path: Optional[str] = None,
    binding_matrix_path: Optional[str] = None,
    score_fn: Optional[Callable] = None,
    marker_path: str = DEFAULT_MARKER_PATH,
    today: Optional[str] = None,
) -> int:
    """Score inputs, persist results, and emit a regression marker if warranted.

    ``score_fn`` is injectable so tests can drive the comparison/alerting logic
    without a trained model. Returns a process exit code (0 = handled).
    """
    scorer: Callable[..., dict] = score_fn or score_iedb_export
    today = today or _dt.date.today().isoformat()

    df = load_inputs(inputs)
    if df is None or len(df) == 0:
        print("No IEDB records fetched — nothing to score. Exiting.")
        return 0

    print(f"Scoring {len(df)} unique peptides ({int(df['label'].sum())} positive)")
    metrics = scorer(df, model_path, binding_matrix_path)
    print(f"AUC-PR on fresh IEDB export: {metrics['auc_pr']:.4f}")

    baselines = load_baseline(baseline_path)
    result = compute_regression(metrics["auc_pr"], baseline_auc_pr(baselines))
    metrics["regression"] = result

    # Part 11 step 6: persist results to results/continuous_validation/.
    out_dir = pathlib.Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"date": today, **metrics}, indent=2)
    (out_dir / f"benchmark_{today}.json").write_text(payload, encoding="utf-8")
    (out_dir / "benchmark_latest.json").write_text(payload, encoding="utf-8")

    if result["baseline_auc_pr"] is None:
        print("No stored baseline - recording current value as the seed baseline.")
    else:
        print(
            f"Delta vs baseline: {result['delta']:+.4f} "
            f"({result['rel_delta']:+.2%}); threshold = -{REGRESSION_REL_THRESHOLD:.0%}"
        )
        if result["is_regression"]:
            msg = regression_message(result)
            print(f"::warning::AUC-PR regression detected - {msg}")
            pathlib.Path(marker_path).write_text(msg, encoding="utf-8")

    return 0


def main(argv: Optional[list] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="continuous_validation",
        description="Score a fresh IEDB export and alert on >3% AUC-PR regression.",
    )
    p.add_argument("--inputs", nargs="+", required=True, help="IEDB export CSV paths")
    p.add_argument("--baseline", default=DEFAULT_BASELINE_PATH, help="Baselines JSON path")
    p.add_argument("--results-dir", default=RESULTS_DIR, help="Output directory")
    p.add_argument("--model-path", default=None, help="Model path (default: config.yaml)")
    p.add_argument("--binding-matrix", default=None, help="Binding matrix (default: config.yaml)")
    p.add_argument("--marker", default=DEFAULT_MARKER_PATH, help="Regression marker file path")
    a = p.parse_args(argv)

    model_path, binding_matrix = _resolve_paths(a.model_path, a.binding_matrix)
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Skipping AUC-PR computation.")
        return 0

    return run(
        a.inputs,
        baseline_path=a.baseline,
        results_dir=a.results_dir,
        model_path=model_path,
        binding_matrix_path=binding_matrix,
        marker_path=a.marker,
    )


if __name__ == "__main__":
    import sys

    sys.exit(main())
