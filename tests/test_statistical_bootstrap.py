"""Tests for src/statistical_bootstrap.py.

Targets the uncovered paths:
  - _bootstrap_iter  (runs in joblib worker - test directly)
  - paired_bootstrap_comparison early-return (n_clean < 10)
  - None-result branch in the accumulator loop
  - main() CLI entry point
"""

import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from src.statistical_bootstrap import (
    _bootstrap_iter,
    paired_bootstrap_comparison,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _df(n_pos: int = 30, n_neg: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = np.array([1] * n_pos + [0] * n_neg)
    ref = np.concatenate(
        [
            rng.normal(0.8, 0.05, n_pos).clip(0, 1),
            rng.normal(0.2, 0.05, n_neg).clip(0, 1),
        ]
    )
    comp = np.concatenate(
        [
            rng.normal(0.6, 0.1, n_pos).clip(0, 1),
            rng.normal(0.4, 0.1, n_neg).clip(0, 1),
        ]
    )
    return pd.DataFrame({"label": labels, "ref": ref, "comp": comp})


# ---------------------------------------------------------------------------
# _bootstrap_iter - direct
# ---------------------------------------------------------------------------


class TestBootstrapIter:
    def _arrays(self, n_pos=20, n_neg=20, seed=1):
        rng = np.random.default_rng(seed)
        y = np.array([1] * n_pos + [0] * n_neg)
        ref = np.concatenate([rng.normal(0.8, 0.05, n_pos), rng.normal(0.2, 0.05, n_neg)]).clip(
            0, 1
        )
        comp = np.concatenate([rng.normal(0.6, 0.1, n_pos), rng.normal(0.4, 0.1, n_neg)]).clip(0, 1)
        return y, ref, comp

    def test_returns_tuple_on_two_class_resample(self):
        y, ref, comp = self._arrays()
        result = _bootstrap_iter(y, ref, comp, np.arange(len(y)))
        assert result is not None
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)

    def test_returns_none_on_single_class_resample(self):
        y = np.ones(20, dtype=int)
        ref = np.ones(20) * 0.9
        comp = np.ones(20) * 0.5
        result = _bootstrap_iter(y, ref, comp, np.arange(len(y)))
        assert result is None

    def test_delta_sign_matches_model_ordering(self):
        """Deltas are ref minus comp, so a strictly better ref must give positive deltas.

        The sign is the whole point of this function: paired_bootstrap_comparison
        feeds scripts/evaluate_per_virus.py, where an inverted delta inverts which
        model a published table reports as better. Bounding the delta inside
        (-1, 1) - the previous assertion - is arithmetically forced for a
        difference of two scores in [0, 1] and holds under a full sign inversion.

        Seeded indices sample repeatedly rather than trusting one draw. Each
        draw is required to be non-negative (the separation anchor below makes
        that exact), and the mean must be positive to reject all-zero deltas.
        """
        from src.statistical_bootstrap import _bootstrap_resample_indices

        y, ref, comp = self._arrays(n_pos=40, n_neg=40)
        # Precondition the sign claim rests on: ref separates the classes perfectly.
        assert ref[y == 1].min() > ref[y == 0].max()

        ap_deltas: list[float] = []
        roc_deltas: list[float] = []
        for idx in _bootstrap_resample_indices(seed=1, n_clean=len(y), n_resamples=20):
            result = _bootstrap_iter(y, ref, comp, idx)
            assert result is not None, "80 samples balanced 40/40 cannot resample single-class"
            ap_delta, roc_delta, _ = result
            assert ap_delta >= 0, f"ref outranks comp, so AP delta must not be negative: {ap_delta}"
            assert roc_delta >= 0, f"ref outranks comp, so ROC delta must not be negative: {roc_delta}"
            ap_deltas.append(ap_delta)
            roc_deltas.append(roc_delta)

        assert float(np.mean(ap_deltas)) > 0
        assert float(np.mean(roc_deltas)) > 0


# ---------------------------------------------------------------------------
# Reproducible bootstrap draws
# ---------------------------------------------------------------------------


def test_bootstrap_resample_indices_same_seed_are_identical():
    from src.statistical_bootstrap import _bootstrap_resample_indices

    a = _bootstrap_resample_indices(seed=7, n_clean=30, n_resamples=10)
    b = _bootstrap_resample_indices(seed=7, n_clean=30, n_resamples=10)
    assert len(a) == len(b) == 10
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_bootstrap_resample_indices_different_seeds_differ():
    from src.statistical_bootstrap import _bootstrap_resample_indices

    a = _bootstrap_resample_indices(seed=7, n_clean=30, n_resamples=10)
    b = _bootstrap_resample_indices(seed=8, n_clean=30, n_resamples=10)
    assert not all(np.array_equal(x, y) for x, y in zip(a, b))


def test_bootstrap_resample_indices_within_call_are_not_all_identical():
    from src.statistical_bootstrap import _bootstrap_resample_indices

    indices = _bootstrap_resample_indices(seed=7, n_clean=30, n_resamples=5)
    assert len(indices) == 5
    assert len({arr.tobytes() for arr in indices}) > 1


# ---------------------------------------------------------------------------
# paired_bootstrap_comparison - early-return path
# ---------------------------------------------------------------------------


def test_paired_bootstrap_too_few_clean_samples():
    df = pd.DataFrame(
        {
            "label": [1, 0, 1],
            "ref": [0.9, 0.1, 0.8],
            "comp": [0.7, 0.3, 0.6],
        }
    )
    result = paired_bootstrap_comparison(df, "ref", "comp", n_resamples=10)
    assert "error" in result
    assert "Too few" in result["error"]


# ---------------------------------------------------------------------------
# paired_bootstrap_comparison - None accumulator branch
# Inject a None into results_parallel via mock so the `if res is not None`
# branch is explicitly exercised in the same process as coverage.
# ---------------------------------------------------------------------------


def _inline_parallel(*args, **kwargs):
    """Replace joblib.Parallel with a synchronous in-process executor."""

    def _run(generator):
        results = []
        for fn, fn_args, fn_kwargs in generator:
            results.append(fn(*fn_args, **fn_kwargs))
        return results

    return _run


def test_paired_bootstrap_none_result_handled():
    """Force at least one _bootstrap_iter → None by using a y array that
    will produce a single-class bootstrap resample on the first draw."""
    rng = np.random.default_rng(42)
    # 10 samples: 9 positives, 1 negative - high probability of all-positive resample
    y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
    ref = np.concatenate([rng.normal(0.85, 0.05, 9), [0.2]]).clip(0, 1)
    comp = np.concatenate([rng.normal(0.6, 0.1, 9), [0.4]]).clip(0, 1)
    df = pd.DataFrame({"label": y, "ref": ref, "comp": comp})

    with patch("src.statistical_bootstrap.Parallel", _inline_parallel):
        result = paired_bootstrap_comparison(df, "ref", "comp", n_resamples=20, seed=0)

    # Result may be valid (some resamples succeeded) or have 0 valid samples
    assert isinstance(result, dict)
    if "error" not in result:
        assert "auc_pr" in result


# ---------------------------------------------------------------------------
# paired_bootstrap_comparison - happy path with inline parallel
# (ensures _bootstrap_iter body is covered in the same process)
# ---------------------------------------------------------------------------


def test_paired_bootstrap_inline_happy_path():
    df = _df(n_pos=30, n_neg=30)
    with patch("src.statistical_bootstrap.Parallel", _inline_parallel):
        result = paired_bootstrap_comparison(df, "ref", "comp", n_resamples=20, seed=42)
    assert "error" not in result
    assert result["n_samples"] == 60
    for key in ("auc_pr", "auc_roc", "issr_10"):
        assert key in result
        assert "delta_mean" in result[key]
        assert "ci_low" in result[key]
        assert "ci_high" in result[key]
        assert "p_value" in result[key]
        assert result[key]["significant_95"] in ("yes", "no")


def test_paired_bootstrap_ref_superior_to_comp():
    df = _df(n_pos=40, n_neg=40)
    with patch("src.statistical_bootstrap.Parallel", _inline_parallel):
        result = paired_bootstrap_comparison(df, "ref", "comp", n_resamples=50, seed=7)
    assert result["auc_roc"]["delta_base"] > 0


def test_paired_bootstrap_same_seed_produces_identical_output():
    df = _df(n_pos=30, n_neg=30)
    with patch("src.statistical_bootstrap.Parallel", _inline_parallel):
        first = paired_bootstrap_comparison(df, "ref", "comp", n_resamples=100, seed=7)
        second = paired_bootstrap_comparison(df, "ref", "comp", n_resamples=100, seed=7)
    assert first == second


def test_paired_bootstrap_different_seeds_produce_different_output():
    df = _df(n_pos=30, n_neg=30)
    with patch("src.statistical_bootstrap.Parallel", _inline_parallel):
        first = paired_bootstrap_comparison(df, "ref", "comp", n_resamples=100, seed=7)
        second = paired_bootstrap_comparison(df, "ref", "comp", n_resamples=100, seed=8)
    assert first != second


# ---------------------------------------------------------------------------
# paired_bootstrap_comparison - delta sign and the significance call
# ---------------------------------------------------------------------------


def _ranked_pair(n_per_class: int = 50) -> pd.DataFrame:
    """A reference that ranks every positive first, and a weaker comparator.

    No two scores tie, so each metric has one value. The comparator's ten
    highest scores hold some positives but not only positives, so its ISSR@10
    lies strictly between 0 and 1, and a sum of the two models' metrics cannot
    equal their difference.
    """
    rng = np.random.default_rng(3)
    labels = np.array([1] * n_per_class + [0] * n_per_class)
    ref = np.concatenate(
        [np.linspace(0.55, 0.99, n_per_class), np.linspace(0.01, 0.45, n_per_class)]
    )
    comp = np.concatenate([rng.uniform(0.0, 0.9, n_per_class), rng.uniform(0.1, 1.0, n_per_class)])
    return pd.DataFrame({"label": labels, "ref": ref, "comp": comp})


def _top_decile_positive_rate(labels: np.ndarray, scores: np.ndarray) -> float:
    top = np.argsort(-scores)[: len(scores) // 10]
    return float(labels[top].mean())


def test_deltas_are_reference_minus_comparator_in_either_order():
    """delta_base is the reference metric minus the comparator metric.

    Expected values come from sklearn and a direct top-decile count, not from
    the module. Swapping the two columns must negate every delta and interval,
    and both orders must be called significant: a comparison that could only
    ever report the reference as better would pass the forward half alone.
    """
    df = _ranked_pair()
    y, ref, comp = df["label"].to_numpy(), df["ref"].to_numpy(), df["comp"].to_numpy()
    assert len(np.unique(ref)) == len(np.unique(comp)) == len(df)
    assert 0 < _top_decile_positive_rate(y, comp) < 1
    expected = {
        "auc_pr": average_precision_score(y, ref) - average_precision_score(y, comp),
        "auc_roc": roc_auc_score(y, ref) - roc_auc_score(y, comp),
        "issr_10": _top_decile_positive_rate(y, ref) - _top_decile_positive_rate(y, comp),
    }

    with patch("src.statistical_bootstrap.Parallel", _inline_parallel):
        forward = paired_bootstrap_comparison(df, "ref", "comp", n_resamples=200, seed=0)
        reverse = paired_bootstrap_comparison(df, "comp", "ref", n_resamples=200, seed=0)

    for key, delta in expected.items():
        assert delta > 0
        assert forward[key]["delta_base"] == pytest.approx(delta)
        assert reverse[key]["delta_base"] == pytest.approx(-delta)
        assert reverse[key]["delta_mean"] == pytest.approx(-forward[key]["delta_mean"])
        assert reverse[key]["ci_low"] == pytest.approx(-forward[key]["ci_high"])
        assert forward[key]["ci_low"] > 0
        assert reverse[key]["ci_high"] < 0
        assert forward[key]["significant_95"] == "yes"
        assert reverse[key]["significant_95"] == "yes"


def test_identical_models_are_not_significantly_different():
    """Comparing a model with itself gives zero deltas and is not significant.

    Every bootstrap delta is exactly zero here, so both interval ends are zero.
    An interval that touches zero must not be called significant, because the
    test is ci_low > 0 or ci_high < 0, strictly.
    """
    df = _ranked_pair()
    with patch("src.statistical_bootstrap.Parallel", _inline_parallel):
        result = paired_bootstrap_comparison(df, "comp", "comp", n_resamples=200, seed=0)

    for key in ("auc_pr", "auc_roc", "issr_10"):
        assert result[key]["delta_base"] == 0.0
        assert result[key]["delta_mean"] == 0.0
        assert result[key]["ci_low"] == 0.0
        assert result[key]["ci_high"] == 0.0
        assert result[key]["significant_95"] == "no"


# ---------------------------------------------------------------------------
# main() CLI entry point
# ---------------------------------------------------------------------------


def test_main_writes_json(tmp_path):
    from src.statistical_bootstrap import main

    df = _df()
    csv_path = tmp_path / "scores.csv"
    df.rename(columns={"ref": "rf_oof_score", "comp": "gnn_score"}).to_csv(csv_path, index=False)
    out_json = tmp_path / "stats.json"

    with (
        patch(
            "sys.argv",
            [
                "statistical_bootstrap.py",
                "--merged-csv",
                str(csv_path),
                "--ref-col",
                "rf_oof_score",
                "--comp-cols",
                "gnn_score",
                "--label-col",
                "label",
                "--n-resamples",
                "20",
                "--output-json",
                str(out_json),
            ],
        ),
        patch("src.statistical_bootstrap.Parallel", _inline_parallel),
    ):
        main()

    assert out_json.exists()
    data = json.loads(out_json.read_text())
    assert "gnn_score" in data


def test_main_skips_missing_column(tmp_path):
    from src.statistical_bootstrap import main

    df = _df()
    csv_path = tmp_path / "scores.csv"
    df.rename(columns={"ref": "rf_oof_score", "comp": "gnn_score"}).to_csv(csv_path, index=False)
    out_json = tmp_path / "stats.json"

    with (
        patch(
            "sys.argv",
            [
                "statistical_bootstrap.py",
                "--merged-csv",
                str(csv_path),
                "--ref-col",
                "rf_oof_score",
                "--comp-cols",
                "nonexistent_col",
                "--output-json",
                str(out_json),
            ],
        ),
        patch("src.statistical_bootstrap.Parallel", _inline_parallel),
    ):
        main()

    data = json.loads(out_json.read_text())
    assert data == {}
