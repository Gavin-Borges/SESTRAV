"""End-to-end unit tests for ``score_immunogenicity`` and plotting.

These exercise the real decision logic on Stage 4's scoring critical path - the
prototype RandomForest fallback, the joblib/.pt model branches (with the heavy
loader mocked or a real lightweight checkpoint), calibration, freeze-mode guards,
and the plotting helper. They complement the pure-helper tests in
``test_stage4_scoring_units.py``.
"""

import os

import numpy as np
import pandas as pd
import pytest

import functions.stage4_immunogenicity_scoring as s4
from src.features import (
    FEATURE_COLUMNS,
    TRAIN_FEATURE_COLUMNS,
    FEATURE_COLUMNS_30,
    FEATURE_COLUMNS_31,
    FEATURE_COLUMNS_33,
    FEATURE_COLUMNS_35,
    FEATURE_COLUMNS_50,
)


def _feature_frame(cols, n=16, seed=0):
    """Build a DataFrame with the given feature columns plus a peptide label."""
    rng = np.random.default_rng(seed)
    data = {c: rng.normal(size=n) for c in cols}
    data["peptide"] = [f"PEP{i:03d}" for i in range(n)]
    return pd.DataFrame(data)


def _run_in_results_dir(monkeypatch, tmp_path):
    """cd into a temp dir that has a results/ folder for CSV/PNG output."""
    (tmp_path / "results").mkdir(exist_ok=True)
    monkeypatch.chdir(tmp_path)


# --- prototype RandomForest fallback -------------------------------------------------


def test_prototype_path_with_binding_score(monkeypatch, tmp_path):
    _run_in_results_dir(monkeypatch, tmp_path)
    df = _feature_frame(FEATURE_COLUMNS)  # FEATURE_COLUMNS includes binding_score
    ranked, model = s4.score_immunogenicity(df, "HPV16", model_path=None)

    assert "immunogenicity_score" in ranked.columns
    assert ranked["immunogenicity_score"].between(0, 1).all()
    # Deterministic contiguous ranks 1..N, sorted descending by score.
    assert ranked["rank"].tolist() == list(range(1, len(ranked) + 1))
    assert ranked["immunogenicity_score"].is_monotonic_decreasing
    assert model is not None
    assert os.path.isfile("results/HPV16_ranked.csv")


def test_prototype_path_with_presentation_score(monkeypatch, tmp_path):
    _run_in_results_dir(monkeypatch, tmp_path)
    cols = [c for c in FEATURE_COLUMNS if c != "binding_score"]
    df = _feature_frame(cols)
    df["presentation_score"] = np.linspace(0.1, 0.9, len(df))
    ranked, _ = s4.score_immunogenicity(df, "EBV", model_path=None)
    assert "immunogenicity_score" in ranked.columns


def test_prototype_path_no_score_columns_constant_zero(monkeypatch, tmp_path):
    """Regression: no binding_score or presentation_score → single-class guard →
    all immunogenicity scores 0.0 with no IndexError (issue #76)."""
    _run_in_results_dir(monkeypatch, tmp_path)
    # Exclude both score-derived columns so pseudo_labels is all-zero (one class).
    physico_only = [c for c in FEATURE_COLUMNS if c not in ("binding_score", "presentation_score")]
    df = _feature_frame(physico_only)
    ranked, model = s4.score_immunogenicity(df, "TEST", model_path=None)
    assert "immunogenicity_score" in ranked.columns
    assert (ranked["immunogenicity_score"] == 0.0).all()
    assert model is None  # no classifier fit in the degenerate case


def test_freeze_mode_without_model_raises(monkeypatch, tmp_path):
    _run_in_results_dir(monkeypatch, tmp_path)
    df = _feature_frame(FEATURE_COLUMNS)
    with pytest.raises(RuntimeError, match="Freeze mode requires a trained model"):
        s4.score_immunogenicity(df, "HPV16", model_path=None, freeze_mode=True)


# --- output_dir threading (regression for SESTRAV-Dev C5) ----------------------------


def test_output_dir_defaults_to_results_for_backward_compatibility(monkeypatch, tmp_path):
    """A caller that does not pass output_dir must keep writing to results/,
    exactly as before this parameter existed."""
    _run_in_results_dir(monkeypatch, tmp_path)
    df = _feature_frame(FEATURE_COLUMNS)
    s4.score_immunogenicity(df, "HPV16", model_path=None)
    assert os.path.isfile("results/HPV16_ranked.csv")


def test_output_dir_isolates_concurrent_runs_for_the_same_proteome_id(monkeypatch, tmp_path):
    """Regression for SESTRAV-Dev C5: score_immunogenicity used to hardcode
    "results/<proteome_id>_ranked.csv", so two runs for the SAME proteome_id (e.g.
    a batch sweep scoring several models against one antigen) silently overwrote
    each other's output on that one shared file. Passing distinct output_dir
    values must keep each run's file separate and correct, and the directory need
    not exist beforehand.
    """
    monkeypatch.chdir(tmp_path)  # so the old hardcoded "results/" is checkable below
    dir_a = tmp_path / "trial_a"
    dir_b = tmp_path / "trial_b"
    # Deliberately not pre-created: threading output_dir through must create it.

    df_a = _feature_frame(FEATURE_COLUMNS, seed=1)
    df_a["peptide"] = [f"A{i:03d}" for i in range(len(df_a))]
    df_b = _feature_frame(FEATURE_COLUMNS, seed=2)
    df_b["peptide"] = [f"B{i:03d}" for i in range(len(df_b))]

    s4.score_immunogenicity(df_a, "SHARED_ID", model_path=None, output_dir=str(dir_a))
    s4.score_immunogenicity(df_b, "SHARED_ID", model_path=None, output_dir=str(dir_b))

    path_a = dir_a / "SHARED_ID_ranked.csv"
    path_b = dir_b / "SHARED_ID_ranked.csv"
    assert path_a.is_file()
    assert path_b.is_file()

    on_disk_a = pd.read_csv(path_a)
    on_disk_b = pd.read_csv(path_b)
    # Each file must hold its own run's peptides, never the other run's - the
    # exact substitution the shared-file race produced (C5's measured
    # consequence was one run's output silently standing in for another's).
    assert set(on_disk_a["peptide"]) == set(df_a["peptide"])
    assert set(on_disk_b["peptide"]) == set(df_b["peptide"])
    assert set(on_disk_a["peptide"]).isdisjoint(set(on_disk_b["peptide"]))

    # Neither run touched the old shared default location.
    assert not (tmp_path / "results" / "SHARED_ID_ranked.csv").exists()


# --- joblib model branch (heavy loader mocked) ---------------------------------------


class _FakeSklearnModel:
    def __init__(self, n_features):
        self.n_features_in_ = n_features

    def predict_proba(self, X):
        # Monotonic-in-row-sum probabilities, shape (n, 2).
        p1 = 1.0 / (1.0 + np.exp(-np.asarray(X).sum(axis=1)))
        return np.column_stack([1 - p1, p1])


@pytest.mark.parametrize(
    "cols",
    [TRAIN_FEATURE_COLUMNS, FEATURE_COLUMNS_30, FEATURE_COLUMNS_31, FEATURE_COLUMNS_50],
    ids=["train21", "f30", "f31", "f50"],
)
def test_joblib_branch_scores(monkeypatch, tmp_path, cols):
    _run_in_results_dir(monkeypatch, tmp_path)
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"stub")  # only needs to exist; loader is mocked
    monkeypatch.setattr(
        s4, "load_verified_joblib", lambda p, required_checksum=True: _FakeSklearnModel(len(cols))
    )
    df = _feature_frame(cols)
    ranked, model = s4.score_immunogenicity(df, "HPV16", model_path=str(model_path))
    assert isinstance(model, _FakeSklearnModel)
    assert ranked["immunogenicity_score"].between(0, 1).all()
    assert ranked["rank"].tolist() == list(range(1, len(ranked) + 1))


class _FakeRankingModel:
    """Mimics XGBClassifier(objective='rank:pairwise'): predict_proba returns raw
    margins outside [0, 1], not probabilities. Regression test for LRF-1.

    Columns sum to 1.0 as the real ranker's do, which is what makes the output look
    probability-shaped on casual inspection; only column 1 (the one the joblib branch
    reads) is out of range. `shift` moves the whole margin vector so a single test can
    pin one side of the guard's range check at a time.
    """

    def __init__(self, n_features, shift=0.0):
        self.n_features_in_ = n_features
        self.shift = shift

    def predict_proba(self, X):
        margins = np.asarray(X).sum(axis=1) - 10.0 + self.shift
        return np.column_stack([1.0 - margins, margins])


class _FakeConstantModel:
    """Returns a caller-supplied column-1 vector verbatim, so a test can place exact
    values (0.0, 1.0, NaN) at the guard without depending on fixture arithmetic."""

    def __init__(self, n_features, values):
        self.n_features_in_ = n_features
        self.values = np.asarray(values, dtype=float)

    def predict_proba(self, X):
        col1 = np.resize(self.values, len(np.asarray(X)))
        return np.column_stack([1.0 - col1, col1])


def _patch_loader(monkeypatch, model):
    monkeypatch.setattr(
        s4, "load_verified_joblib", lambda p, required_checksum=True: model
    )


@pytest.mark.parametrize(
    "shift, side",
    [
        # The unshifted fixture is MIXED (15 margins below 0, one at ~3.01 above 1), so
        # it fires on either clause alone and pins neither. Each case below is shifted
        # entirely onto ONE side of the range, so deleting that clause fails this test.
        (-5.0, "below"),
        (100.0, "above"),
    ],
)
def test_joblib_branch_raises_on_out_of_range_score(monkeypatch, tmp_path, shift, side):
    """LRF-1: a ranking-objective model's raw margins must fail loudly, not silently
    flow through calibration/thresholding/ranking as if they were probabilities.

    Parametrized over both sides of the check so that deleting either half of
    `(x < 0.0) | (x > 1.0)` fails a test.
    """
    _run_in_results_dir(monkeypatch, tmp_path)
    model_path = tmp_path / "xgb_50feature_integrated.joblib"
    model_path.write_bytes(b"stub")
    _patch_loader(monkeypatch, _FakeRankingModel(len(FEATURE_COLUMNS_50), shift=shift))
    df = _feature_frame(FEATURE_COLUMNS_50)

    # Guard the fixture itself: if these margins ever drifted onto the other side (or
    # in-range) the test would keep passing while pinning nothing.
    margins = df[FEATURE_COLUMNS_50].to_numpy().sum(axis=1) - 10.0 + shift
    if side == "below":
        assert (margins < 0.0).all(), "below-zero fixture drifted"
    else:
        assert (margins > 1.0).all(), "above-one fixture drifted"

    with pytest.raises(RuntimeError, match=r"out of the required \[0, 1\] range"):
        s4.score_immunogenicity(df, "HPV16", model_path=str(model_path))

    # The guard's stated design point is that it fires BEFORE anything is written.
    assert not (tmp_path / "results" / "HPV16_ranked.csv").exists()


def test_joblib_branch_raises_on_nan_score(monkeypatch, tmp_path):
    """LRF-1: NaN must raise too. `(x < 0) | (x > 1)` is False for NaN, so a bare pair
    of comparisons passes it through - and a NaN score is exactly as undetectable
    downstream as the negative margin this guard exists to stop."""
    _run_in_results_dir(monkeypatch, tmp_path)
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"stub")
    _patch_loader(
        monkeypatch,
        _FakeConstantModel(len(FEATURE_COLUMNS_50), [0.4, np.nan, 0.6]),
    )
    df = _feature_frame(FEATURE_COLUMNS_50)
    with pytest.raises(RuntimeError, match=r"non-finite"):
        s4.score_immunogenicity(df, "HPV16", model_path=str(model_path))


def test_joblib_branch_accepts_exact_bounds(monkeypatch, tmp_path):
    """The bounds are inclusive: exactly 0.0 and exactly 1.0 are valid probabilities.
    Pins the strict `<`/`>` against a `<=`/`>=` mutant, which no other test catches."""
    _run_in_results_dir(monkeypatch, tmp_path)
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"stub")
    _patch_loader(
        monkeypatch, _FakeConstantModel(len(FEATURE_COLUMNS_50), [0.0, 1.0, 0.5])
    )
    df = _feature_frame(FEATURE_COLUMNS_50)
    ranked, _model = s4.score_immunogenicity(df, "HPV16", model_path=str(model_path))
    assert ranked["immunogenicity_score"].between(0.0, 1.0).all()
    assert {0.0, 1.0}.issubset(set(ranked["immunogenicity_score"]))


def test_joblib_missing_loader_warns(monkeypatch, tmp_path, capsys):
    """When joblib support is unavailable the prototype fallback still scores."""
    _run_in_results_dir(monkeypatch, tmp_path)
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"stub")
    monkeypatch.setattr(s4, "load_verified_joblib", None)
    df = _feature_frame(FEATURE_COLUMNS)
    ranked, _ = s4.score_immunogenicity(df, "HPV16", model_path=str(model_path))
    assert "immunogenicity_score" in ranked.columns
    assert "joblib not available" in capsys.readouterr().out


def test_model_path_alias_resolution(monkeypatch, tmp_path):
    _run_in_results_dir(monkeypatch, tmp_path)
    real_path = tmp_path / "resolved.joblib"
    real_path.write_bytes(b"stub")
    # A calibrator alongside the model exercises the calibrated_score assignment.
    (tmp_path / "platt_calibrator.joblib").write_bytes(b"stub")
    monkeypatch.setattr(s4, "resolve_model_path", lambda p: str(real_path))
    monkeypatch.setattr(
        s4,
        "load_verified_joblib",
        lambda p, required_checksum=True: _FakeSklearnModel(len(TRAIN_FEATURE_COLUMNS)),
    )
    df = _feature_frame(TRAIN_FEATURE_COLUMNS)
    ranked, _ = s4.score_immunogenicity(df, "HPV16", model_path="alias://model")
    assert "immunogenicity_score" in ranked.columns
    assert "calibrated_score" in ranked.columns


# --- calibration with a calibrator present -------------------------------------------


class _FakeCalibrator:
    def predict_proba(self, logits):
        p1 = 1.0 / (1.0 + np.exp(-np.asarray(logits).ravel()))
        return np.column_stack([1 - p1, p1])


def test_apply_calibration_with_calibrator(monkeypatch, tmp_path):
    (tmp_path / "platt_calibrator.joblib").write_bytes(b"stub")
    monkeypatch.setattr(
        s4, "load_verified_joblib", lambda p, required_checksum=True: _FakeCalibrator()
    )
    scores = np.array([0.2, 0.5, 0.8])
    out, applied = s4._apply_calibration(scores, str(tmp_path))
    assert applied is True
    assert out.shape == scores.shape
    assert np.all((out >= 0) & (out <= 1))


def test_apply_calibration_isotonic_branch(monkeypatch, tmp_path):
    """Isotonic calibrator (real sklearn) is detected via .predict and applied.

    Verifies the isotonic dispatch (raw-score .predict, not logit
    .predict_proba), that outputs stay in [0, 1], and that the mapping is
    monotonic non-decreasing in the input score.
    """
    from sklearn.isotonic import IsotonicRegression

    # Fit a simple monotonic calibrator on synthetic (score, label) data.
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, size=500)
    y = (rng.uniform(0, 1, size=500) < x).astype(int)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(x, y)
    assert not hasattr(iso, "predict_proba")  # isotonic exposes only .predict

    (tmp_path / "isotonic_calibrator.joblib").write_bytes(b"stub")
    monkeypatch.setattr(s4, "load_verified_joblib", lambda p, required_checksum=True: iso)

    scores = np.linspace(0.0, 1.0, 21)
    out, applied = s4._apply_calibration(scores, str(tmp_path))

    assert applied is True
    assert out.shape == scores.shape
    assert np.all((out >= 0.0) & (out <= 1.0))
    # Calibrated output is monotonic non-decreasing in the raw score.
    assert np.all(np.diff(out) >= -1e-9)


def test_apply_calibration_isotonic_preferred_over_platt(monkeypatch, tmp_path):
    """When both artifacts exist the isotonic one is resolved first."""
    (tmp_path / "isotonic_calibrator.joblib").write_bytes(b"stub")
    (tmp_path / "platt_calibrator.joblib").write_bytes(b"stub")
    resolved = s4._resolve_calibrator_path(str(tmp_path))
    assert resolved.endswith("isotonic_calibrator.joblib")


# ---------------------------------------------------------------------------
# Per-virus calibration (A1-B). No per-virus calibrator has been promoted to
# any real location - these tests exercise the resolver/dispatcher logic in
# isolation with a tmp_path standing in for a promoted per_virus_dir. The
# central property under test: an unrecognised or off-panel virus must behave
# IDENTICALLY to no virus at all, because the fallback fires from file
# ABSENCE, not from a virus allow-list.
# ---------------------------------------------------------------------------


def test_resolve_calibrator_path_per_virus_when_file_exists(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    pv_dir = tmp_path / "per_virus"
    pv_dir.mkdir()
    (pv_dir / "SARS-CoV-2.joblib").write_bytes(b"stub")

    resolved = s4._resolve_calibrator_path(
        str(model_dir), virus="SARS-CoV-2", per_virus_dir=str(pv_dir)
    )
    assert resolved == str(pv_dir / "SARS-CoV-2.joblib")


def test_resolve_calibrator_path_sanitizes_virus_for_the_filename(tmp_path):
    """HIV-1 must resolve safely; the hyphen is allowed by _sanitize_name, but
    the lookup must go through the same sanitizer as the writer or a promoted
    file could silently never be found."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    pv_dir = tmp_path / "per_virus"
    pv_dir.mkdir()
    (pv_dir / "HIV-1.joblib").write_bytes(b"stub")

    resolved = s4._resolve_calibrator_path(str(model_dir), virus="HIV-1", per_virus_dir=str(pv_dir))
    assert resolved == str(pv_dir / "HIV-1.joblib")


def test_resolve_calibrator_path_falls_back_to_global_when_per_virus_file_missing(tmp_path):
    """No file for this virus -> falls through to the global lookup, same as
    if virus had never been passed at all."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "isotonic_calibrator.joblib").write_bytes(b"stub")
    pv_dir = tmp_path / "per_virus"
    pv_dir.mkdir()  # exists, but no matching file inside it

    resolved = s4._resolve_calibrator_path(
        str(model_dir), virus="SARS-CoV-2", per_virus_dir=str(pv_dir)
    )
    assert resolved == str(model_dir / "isotonic_calibrator.joblib")


def test_resolve_calibrator_path_off_panel_virus_matches_no_virus_exactly(tmp_path):
    """An unrecognised virus name is not a special case - it is simply a name
    with no matching file, so it must resolve identically to virus=None."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "isotonic_calibrator.joblib").write_bytes(b"stub")
    pv_dir = tmp_path / "per_virus"
    pv_dir.mkdir()
    (pv_dir / "SARS-CoV-2.joblib").write_bytes(b"stub")  # a different virus is promoted

    with_unknown_virus = s4._resolve_calibrator_path(
        str(model_dir), virus="Nipah", per_virus_dir=str(pv_dir)
    )
    with_no_virus = s4._resolve_calibrator_path(str(model_dir), virus=None, per_virus_dir=str(pv_dir))
    assert with_unknown_virus == with_no_virus == str(model_dir / "isotonic_calibrator.joblib")


def test_resolve_calibrator_path_explicit_override_wins_over_per_virus(tmp_path):
    """The pre-existing explicit-config-path contract is unaffected by virus:
    it wins regardless, matching how it already wins over the global lookup."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    pv_dir = tmp_path / "per_virus"
    pv_dir.mkdir()
    (pv_dir / "SARS-CoV-2.joblib").write_bytes(b"stub")
    explicit = tmp_path / "elsewhere.joblib"
    explicit.write_bytes(b"stub")

    resolved = s4._resolve_calibrator_path(
        str(model_dir), calibration_path=str(explicit), virus="SARS-CoV-2", per_virus_dir=str(pv_dir)
    )
    assert resolved == str(explicit)


def test_apply_calibration_per_virus_and_global_give_different_scores(monkeypatch, tmp_path):
    """Functional proof the per-virus branch is actually exercised, not merely
    resolved: two differently-fitted calibrators must produce different output
    for the same raw scores."""
    from sklearn.isotonic import IsotonicRegression

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "isotonic_calibrator.joblib").write_bytes(b"stub")
    pv_dir = tmp_path / "per_virus"
    pv_dir.mkdir()
    (pv_dir / "SARS-CoV-2.joblib").write_bytes(b"stub")

    global_iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
        [0.0, 0.5, 1.0], [0.0, 0.1, 1.0]
    )
    per_virus_iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
        [0.0, 0.5, 1.0], [0.0, 0.9, 1.0]
    )

    def _fake_load(path, required_checksum=True):
        return per_virus_iso if str(pv_dir) in str(path) else global_iso

    monkeypatch.setattr(s4, "load_verified_joblib", _fake_load)

    scores = np.array([0.5])
    per_virus_out, per_virus_applied = s4._apply_calibration(
        scores, str(model_dir), virus="SARS-CoV-2", per_virus_dir=str(pv_dir)
    )
    global_out, global_applied = s4._apply_calibration(scores, str(model_dir))

    assert per_virus_applied is True and global_applied is True
    assert per_virus_out[0] != global_out[0]
    assert per_virus_out[0] == pytest.approx(0.9)
    assert global_out[0] == pytest.approx(0.1)


def test_apply_calibration_no_virus_arg_reproduces_pre_existing_behaviour(monkeypatch, tmp_path):
    """Callers that never pass virus/per_virus_dir (every call site before A1-B)
    must see byte-identical behaviour - this is the backward-compatibility
    contract the whole feature is built on top of."""
    from sklearn.isotonic import IsotonicRegression

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "isotonic_calibrator.joblib").write_bytes(b"stub")
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit([0.0, 1.0], [0.0, 1.0])
    monkeypatch.setattr(s4, "load_verified_joblib", lambda p, required_checksum=True: iso)

    scores = np.array([0.1, 0.5, 0.9])
    out_old_call_shape, applied_old = s4._apply_calibration(scores, str(model_dir))
    out_new_call_shape, applied_new = s4._apply_calibration(
        scores, str(model_dir), calibration_path=None, virus=None, per_virus_dir=None
    )
    assert applied_old == applied_new is True
    assert np.array_equal(out_old_call_shape, out_new_call_shape)


def test_apply_calibration_explicit_path(monkeypatch, tmp_path):
    """An explicit calibration_path takes precedence over model-dir search."""
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip").fit([0.0, 1.0], [0.0, 1.0])
    explicit = tmp_path / "custom_cal.joblib"
    explicit.write_bytes(b"stub")
    monkeypatch.setattr(s4, "load_verified_joblib", lambda p, required_checksum=True: iso)
    scores = np.array([0.1, 0.5, 0.9])
    out, applied = s4._apply_calibration(scores, str(tmp_path), calibration_path=str(explicit))
    assert applied is True
    assert np.all((out >= 0.0) & (out <= 1.0))


# --- PyTorch .pt branch (real lightweight checkpoint) --------------------------------


def _save_ann_checkpoint(path, n_features):
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    class _MLP(nn.Module):
        def __init__(self, input_dim, hidden=(64, 32), dropout=0.3):
            super().__init__()
            layers, prev = [], input_dim
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(-1)

    model = _MLP(n_features)
    torch.save(
        {
            "n_features": n_features,
            "scaler_mean": torch.zeros(n_features),
            "scaler_scale": torch.ones(n_features),
            "model_state_dict": model.state_dict(),
        },
        path,
    )
    # The loader enforces checksum verification (required=True); generate the
    # sidecar manifest with the project's own helper so the safe-load path runs.
    from src.artifact_integrity import default_manifest_path_for, update_checksum_manifest

    update_checksum_manifest(default_manifest_path_for(path), [path])


def test_pytorch_branch_scores(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    _run_in_results_dir(monkeypatch, tmp_path)
    pt_path = tmp_path / "ann.pt"
    _save_ann_checkpoint(str(pt_path), len(FEATURE_COLUMNS_30))
    df = _feature_frame(FEATURE_COLUMNS_30)
    ranked, model = s4.score_immunogenicity(df, "HPV16", model_path=str(pt_path))
    assert ranked["immunogenicity_score"].between(0, 1).all()
    assert model is not None


def test_pytorch_branch_mc_dropout(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    _run_in_results_dir(monkeypatch, tmp_path)
    pt_path = tmp_path / "ann.pt"
    _save_ann_checkpoint(str(pt_path), len(FEATURE_COLUMNS_30))
    df = _feature_frame(FEATURE_COLUMNS_30)
    ranked, _ = s4.score_immunogenicity(df, "HPV16", model_path=str(pt_path), mc_dropout=True)
    assert "mc_score" in ranked.columns
    assert "uncertainty_std" in ranked.columns
    assert (ranked["uncertainty_std"] >= 0).all()


def test_pytorch_freeze_mode_missing_features_raises(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    _run_in_results_dir(monkeypatch, tmp_path)
    pt_path = tmp_path / "ann.pt"
    _save_ann_checkpoint(str(pt_path), len(FEATURE_COLUMNS_30))
    # Drop a required feature so the freeze-mode guard fires.
    df = _feature_frame(FEATURE_COLUMNS_30[:-1])
    with pytest.raises(RuntimeError, match="Missing"):
        s4.score_immunogenicity(df, "HPV16", model_path=str(pt_path), freeze_mode=True)


# --- plotting ------------------------------------------------------------------------


def test_plot_immunogenicity_scores(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    ranked = pd.DataFrame(
        {
            "peptide": [f"PEP{i}" for i in range(25)],
            "immunogenicity_score": np.linspace(0.0, 1.0, 25),
        }
    )
    s4.plot_immunogenicity_scores(ranked, "HPV16", top_n=20)
    assert os.path.isfile("results/HPV16_top20_immunogenicity.png")
    assert os.path.isfile("results/HPV16_score_distribution.png")


def test_pytorch_branch_50feat(monkeypatch, tmp_path):
    """Covers lines 186-187: PyTorch checkpoint with 50-feature count."""
    pytest.importorskip("torch")
    _run_in_results_dir(monkeypatch, tmp_path)
    pt_path = tmp_path / "ann50.pt"
    _save_ann_checkpoint(str(pt_path), len(FEATURE_COLUMNS_50))
    df = _feature_frame(FEATURE_COLUMNS_50)
    ranked, model = s4.score_immunogenicity(df, "HPV16", model_path=str(pt_path))
    assert ranked["immunogenicity_score"].between(0, 1).all()
    assert model is not None


def test_pytorch_branch_train_feat(monkeypatch, tmp_path):
    """Covers lines 191-196: PyTorch checkpoint with TRAIN_FEATURE_COLUMNS count."""
    pytest.importorskip("torch")
    _run_in_results_dir(monkeypatch, tmp_path)
    pt_path = tmp_path / "ann_train.pt"
    _save_ann_checkpoint(str(pt_path), len(TRAIN_FEATURE_COLUMNS))
    df = _feature_frame(TRAIN_FEATURE_COLUMNS)
    ranked, model = s4.score_immunogenicity(df, "HPV16", model_path=str(pt_path))
    assert ranked["immunogenicity_score"].between(0, 1).all()


def test_pytorch_missing_features_warns_non_freeze(monkeypatch, tmp_path, capsys):
    """Covers line 204: missing-features WARNING path when freeze_mode=False."""
    pytest.importorskip("torch")
    _run_in_results_dir(monkeypatch, tmp_path)
    pt_path = tmp_path / "ann30.pt"
    _save_ann_checkpoint(str(pt_path), len(FEATURE_COLUMNS_30))
    # Drop one required feature so len(model_cols) < expected_n, but freeze_mode=False.
    df = _feature_frame(FEATURE_COLUMNS_30[:-1])
    # Mock _load_pytorch_model so the shape mismatch after the warning doesn't propagate.
    import torch.nn as nn

    class _TrivialModel(nn.Module):
        def forward(self, x):
            return x[:, 0]

    monkeypatch.setattr(s4, "_load_pytorch_model", lambda *_: (np.zeros(len(df)), _TrivialModel()))
    ranked, _ = s4.score_immunogenicity(df, "HPV16", model_path=str(pt_path), freeze_mode=False)
    assert "immunogenicity_score" in ranked.columns
    assert "WARNING" in capsys.readouterr().out


def test_score_immunogenicity_no_calibrate(monkeypatch, tmp_path):
    """Covers branch 282->289: calibrate=False skips _apply_calibration."""
    _run_in_results_dir(monkeypatch, tmp_path)
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"stub")
    monkeypatch.setattr(
        s4,
        "load_verified_joblib",
        lambda p, required_checksum=True: _FakeSklearnModel(len(FEATURE_COLUMNS_30)),
    )
    df = _feature_frame(FEATURE_COLUMNS_30)
    ranked, _ = s4.score_immunogenicity(df, "HPV16", model_path=str(model_path), calibrate=False)
    assert "immunogenicity_score" in ranked.columns
    assert "calibrated_score" not in ranked.columns


def test_pytorch_branch_else_legacy(monkeypatch, tmp_path):
    """A PyTorch checkpoint of unrecognised width uses the legacy fallback layout.

    ``_resolve_feature_layout`` returns FEATURE_COLUMNS for any width absent from
    ``_FEATURE_LAYOUTS``, and the ANN branch warns rather than refuses.
    """
    pytest.importorskip("torch")
    _run_in_results_dir(monkeypatch, tmp_path)
    pt_path = tmp_path / "ann_legacy.pt"
    # 15 matches no shipped layout, so the legacy 22-column set is the fallback.
    _save_ann_checkpoint(str(pt_path), 15)
    df = _feature_frame(FEATURE_COLUMNS[:15])
    ranked, _ = s4.score_immunogenicity(df, "HPV16", model_path=str(pt_path))
    assert "immunogenicity_score" in ranked.columns


def test_joblib_branch_legacy_else_fallback(monkeypatch, tmp_path):
    """An unrecognised width whose column count happens to match still scores.

    ``_resolve_feature_layout`` falls back to the legacy 22-column set for any width
    not in ``_FEATURE_LAYOUTS``; ``_select_model_columns`` then refuses only when the
    selection size disagrees with the model. Here 15 of the 22 are present against a
    15-feature model, so the counts agree and scoring proceeds. This is deliberately
    preserved behaviour - the refusal added for modes 33/35 must not widen into it.
    """
    _run_in_results_dir(monkeypatch, tmp_path)
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"stub")

    # Use a feature count that matches none of the 50/30/21 sets (e.g. 15)
    _LEGACY_COLS = FEATURE_COLUMNS[:15]
    monkeypatch.setattr(
        s4,
        "load_verified_joblib",
        lambda p, required_checksum=True: _FakeSklearnModel(15),
    )
    df = _feature_frame(_LEGACY_COLS)
    ranked, _ = s4.score_immunogenicity(df, "HPV16", model_path=str(model_path))
    assert "immunogenicity_score" in ranked.columns


# --- feature-layout selection and refusal (regression: 2026-08-17 stress test) --------
#
# Stage 4 used to pick model columns with an if/elif ladder written out twice - once in
# the ANN branch, once in the joblib branch - and both copies handled only widths 50, 30,
# 22 and 21. A 31-feature model (config.yaml's default) fell through to the 22-column
# legacy set, so the canonical production mode could not be scored at all: the mismatch
# surfaced only as a third-party feature-name error from inside predict_proba. These tests
# pin the width->layout mapping and the refusal behaviour that replaced the silent
# substitution.


def test_layout_table_covers_every_shipped_feature_mode():
    """Each shipped mode maps to a layout of exactly its own width."""
    for cols in (
        TRAIN_FEATURE_COLUMNS,
        FEATURE_COLUMNS,
        FEATURE_COLUMNS_30,
        FEATURE_COLUMNS_31,
        FEATURE_COLUMNS_33,
        FEATURE_COLUMNS_35,
        FEATURE_COLUMNS_50,
    ):
        resolved, _label = s4._resolve_feature_layout(len(cols))
        assert len(resolved) == len(cols)
        assert resolved == cols


@pytest.mark.parametrize(
    "cols, missing_names",
    [
        (FEATURE_COLUMNS_33, ["netchop_score", "tap_score"]),
        (
            FEATURE_COLUMNS_35,
            ["netchop_score", "tap_score", "self_similarity_max_identity"],
        ),
    ],
    ids=["f33", "f35"],
)
def test_modes_the_pipeline_cannot_feed_are_refused_by_name(cols, missing_names):
    """Modes 33/35 need columns Stage 3 never computes, so they must refuse clearly.

    Deliberately NOT 'fixed' by supplying the cached values: per D18 the antigen-processing
    scores are mock and not reproducible, so making these modes run would put unreproducible
    features into user-facing output.
    """
    df = _feature_frame(FEATURE_COLUMNS_31)  # what the pipeline actually produces
    with pytest.raises(RuntimeError, match="Cannot score a") as exc:
        s4._select_model_columns(len(cols), df)
    for name in missing_names:
        assert name in str(exc.value)


def test_partial_allele_panel_is_refused_not_silently_narrowed():
    """A restricted --alleles panel drops bind_* columns a mode-30 model was fit on."""
    reduced = [c for c in FEATURE_COLUMNS_30 if not c.startswith("bind_") or c == "bind_A0201"]
    df = _feature_frame(reduced)
    with pytest.raises(RuntimeError, match="Cannot score a 30-feature model"):
        s4._select_model_columns(len(FEATURE_COLUMNS_30), df)


def test_named_but_missing_model_refuses_instead_of_fabricating(monkeypatch, tmp_path):
    """A caller who names a model must get it or an error - never the prototype.

    The prototype trains on pseudo-labels derived from binding_score and its output is then
    calibrated and thresholded, producing a ranked CSV indistinguishable from a real run.
    """
    _run_in_results_dir(monkeypatch, tmp_path)
    df = _feature_frame(FEATURE_COLUMNS)
    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        s4.score_immunogenicity(df, "TEST", model_path=str(tmp_path / "absent.joblib"))


def test_prototype_is_still_reachable_when_no_model_is_named(monkeypatch, tmp_path):
    """model_path=None remains the supported way to ask for the prototype on purpose."""
    _run_in_results_dir(monkeypatch, tmp_path)
    df = _feature_frame(FEATURE_COLUMNS)
    ranked, _model = s4.score_immunogenicity(df, "TEST", model_path=None, calibrate=False)
    assert "immunogenicity_score" in ranked.columns
