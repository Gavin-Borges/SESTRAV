"""Stage 4's ranking puts every called row above every uncalled row.

The binary ``immunogenic`` call compares the RAW ``immunogenicity_score`` with the
threshold, while the ranking sorts by ``calibrated_score``. An isotonic calibrator
maps a whole raw interval to one value, so a plateau can straddle the threshold:
rows on it tie on the calibrated score while some are called and some are not. With
the tie broken by peptide alone, uncalled rows ranked above called ones (on the
grouped v5 OOF, 1,424 uncalled rows sat above the last called row). The tie is now
broken by the raw score, so under a monotone non-decreasing calibrator the called
rows are always a prefix of the ranking. Every calibrator the repo fits is one; the
last test below pins that condition.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import functions.stage4_immunogenicity_scoring as s4
from src.features import FEATURE_COLUMNS_31


class _ReplayModel:
    """Returns caller-supplied raw scores verbatim, in row order."""

    n_features_in_ = len(FEATURE_COLUMNS_31)

    def __init__(self, raw):
        self.raw = np.asarray(raw, dtype=float)

    def predict_proba(self, X):
        assert len(X) == len(self.raw)
        return np.column_stack([1.0 - self.raw, self.raw])


class _StepCalibrator:
    """Isotonic-shaped (exposes only .predict): one plateau spans raw [0.2, 0.5)."""

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        return np.where(x < 0.2, 0.1, np.where(x < 0.5, 0.4, 0.9))


def _rank(monkeypatch, tmp_path, raw, peptides, calibrator, threshold):
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"stub")  # only needs to exist; the loader is patched
    (tmp_path / "isotonic_calibrator.joblib").write_bytes(b"stub")
    (tmp_path / "optimal_thresholds.json").write_text(json.dumps({"threshold": threshold}))
    model = _ReplayModel(raw)

    def _loader(path, required_checksum=True):
        return model if path == str(model_path) else calibrator

    monkeypatch.setattr(s4, "load_verified_joblib", _loader)
    features = pd.DataFrame({c: np.zeros(len(raw)) for c in FEATURE_COLUMNS_31})
    features["peptide"] = peptides
    ranked, _ = s4.score_immunogenicity(
        features,
        "RANKTEST",
        model_path=str(model_path),
        conformal=False,
        output_dir=str(tmp_path / "out"),
    )
    assert "calibrated_score" in ranked.columns
    return ranked


def _assert_called_rows_lead(ranked):
    called = ranked["immunogenic"].to_numpy() == 1
    n_called = int(called.sum())
    assert called[:n_called].all() and not called[n_called:].any(), (
        "called rows are not a prefix of the ranking: "
        f"{ranked[['peptide', 'immunogenicity_score', 'calibrated_score', 'immunogenic']]}"
    )
    # The calibrated score is still the primary ranking key.
    assert (np.diff(ranked["calibrated_score"].to_numpy()) <= 0).all()
    assert ranked["rank"].tolist() == list(range(1, len(ranked) + 1))


def test_a_calibrated_tie_across_the_threshold_ranks_the_called_row_first(monkeypatch, tmp_path):
    # AAA (raw 0.30, uncalled) and ZZZ (raw 0.40, called) share the 0.4 plateau, and
    # AAA sorts first by peptide. The raw score must decide that tie.
    raw = [0.30, 0.40, 0.90, 0.10]
    peptides = ["AAA", "ZZZ", "MMM", "BBB"]
    ranked = _rank(monkeypatch, tmp_path, raw, peptides, _StepCalibrator(), 0.35)
    assert ranked["calibrated_score"].tolist() == [0.9, 0.4, 0.4, 0.1]
    assert ranked["peptide"].tolist() == ["MMM", "ZZZ", "AAA", "BBB"]
    assert ranked["immunogenic"].tolist() == [1, 1, 0, 0]
    _assert_called_rows_lead(ranked)


def test_called_rows_lead_under_a_fitted_isotonic_calibrator(monkeypatch, tmp_path):
    """A real IsotonicRegression, with the threshold placed inside each of several plateaus.

    Peptides are in random order, so a peptide tie-break alone would misorder some of
    them. Each threshold is the midpoint of one plateau's raw range, so every case
    has called and uncalled rows sharing one calibrated value.
    """
    from sklearn.isotonic import IsotonicRegression

    rng = np.random.default_rng(7)
    x = rng.uniform(0, 1, size=400)
    y = (rng.uniform(0, 1, size=400) < x).astype(int)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(x, y)

    raw = rng.uniform(0, 1, size=300)
    peptides = [f"P{i:04d}" for i in rng.permutation(len(raw))]
    calibrated = iso.predict(raw)
    plateaus = [v for v in np.unique(calibrated) if np.ptp(raw[calibrated == v]) > 0]
    assert len(plateaus) >= 5
    for i in np.linspace(0, len(plateaus) - 1, 5).astype(int):
        on = raw[calibrated == plateaus[i]]
        threshold = (on.min() + on.max()) / 2
        case = tmp_path / f"plateau_{i}"
        case.mkdir()
        ranked = _rank(monkeypatch, case, raw, peptides, iso, threshold)
        called = ranked["immunogenic"].to_numpy() == 1
        shared = ranked["calibrated_score"].to_numpy()
        # Non-vacuous: a calibrated value is shared by a called and an uncalled row.
        assert set(shared[called]) & set(shared[~called])
        _assert_called_rows_lead(ranked)


def test_under_a_monotone_calibrator_the_ranking_is_the_raw_score_order(monkeypatch, tmp_path):
    """The guarantee rests on the calibrator being monotone non-decreasing.

    For such a calibrator, sorting by (calibrated desc, raw desc, peptide asc) is the
    same order as sorting by (raw desc, peptide asc), and the called rows are a raw
    prefix. Raw scores are drawn on a coarse grid so some rows tie on the raw score
    too, and the peptide tie-break is exercised.
    """
    from sklearn.isotonic import IsotonicRegression

    rng = np.random.default_rng(11)
    x = rng.uniform(0, 1, size=400)
    y = (rng.uniform(0, 1, size=400) < x).astype(int)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(x, y)

    raw = np.round(rng.uniform(0, 1, size=200), 2)
    peptides = [f"P{i:04d}" for i in rng.permutation(len(raw))]
    assert len(np.unique(raw)) < len(raw)  # non-vacuous: some raw scores tie
    ranked = _rank(monkeypatch, tmp_path, raw, peptides, iso, 0.5)
    expected = (
        pd.DataFrame({"peptide": peptides, "raw": raw})
        .sort_values(["raw", "peptide"], ascending=[False, True])["peptide"]
        .tolist()
    )
    assert ranked["peptide"].tolist() == expected
    _assert_called_rows_lead(ranked)


def test_every_calibrator_the_repo_fits_is_monotone_non_decreasing():
    """Pins the condition the ranking guarantee states.

    Every IsotonicRegression constructed under scripts/, src/ and functions/ keeps
    the default increasing=True and clips out-of-range input, so it is monotone
    non-decreasing and never returns NaN for a finite score (which _apply_calibration
    would replace with the raw score). A fitter that set increasing=False or "auto"
    could produce a calibrator under which uncalled rows outrank called ones.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    calls: list[tuple[str, ast.Call]] = []
    for top in ("scripts", "src", "functions"):
        for path in sorted((root / top).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
                if name == "IsotonicRegression":
                    calls.append((path.relative_to(root).as_posix(), node))
    assert len(calls) >= 5, calls  # non-vacuous: the fitters were found
    for where, node in calls:
        kwargs = {kw.arg: kw.value for kw in node.keywords}
        increasing = kwargs.get("increasing")
        assert increasing is None or (
            isinstance(increasing, ast.Constant) and increasing.value is True
        ), where
        bounds = kwargs.get("out_of_bounds")
        assert isinstance(bounds, ast.Constant) and bounds.value == "clip", where
