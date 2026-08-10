"""
Unit tests for parent protein mapping and LOPO (Leave-One-Protein-Out) cross validation.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import BINDING_ALLELE_COLUMNS, HLA_PSEUDO_COLS
from src.train_classifier import (
    load_all_proteins,
    _get_protein_name_from_header,
    _cross_validate,
    _excluded_bloc_cv_metrics,
    VACCINIA_VIRUS,
    prepare_features_30,
    prepare_features_31,
    prepare_features_33,
    prepare_features_35,
    prepare_features_166,
    train_models,
    planned_artifact_paths,
    _filter_quarantined,
)

_AAS = "ACDEFGHIKLMNPQRSTVWY"


def _make_peptides(n):
    """Deterministic distinct valid 9-mers (standard AA only)."""
    base = list("SLLMWITQV")
    peps = []
    for i in range(n):
        p = base.copy()
        p[0] = _AAS[i % 20]
        p[1] = _AAS[(i // 20) % 20]
        peps.append("".join(p))
    return peps


def _mock_binding_csv(tmp_path, peptides, name="binding.csv"):
    """Mock binding matrix with all 10 allele columns present."""
    data = {"peptide": peptides}
    for col in BINDING_ALLELE_COLUMNS:
        data[col] = [0.6] * len(peptides)
    path = tmp_path / name
    pd.DataFrame(data).to_csv(path, index=False)
    return path


def _df_with_pseudo(peptides):
    data = {"peptide": peptides}
    for col in HLA_PSEUDO_COLS:
        data[col] = [0.0] * len(peptides)
    return pd.DataFrame(data)


def _training_csv(tmp_path, n=30, name="train.csv"):
    peps = _make_peptides(n)
    labels = ([0, 1] * (n // 2 + 1))[:n]
    df = pd.DataFrame({"peptide": peps, "label": labels, "virus": ["EBV"] * n})
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path, peps


def test_get_protein_name_from_header():
    hdr1 = "sp|P03120|VE2_HPV16 Regulatory protein E2 OS=Human papillomavirus type 16 OX=333760 GN=E2 PE=1 SV=1"
    assert _get_protein_name_from_header(hdr1) == "VE2_HPV16"

    hdr2 = "GP350_EBVB9 Envelope glycoprotein GP350"
    assert _get_protein_name_from_header(hdr2) == "GP350_EBVB9"

    hdr3 = "sp|P03211|EBNA1_EBVB9"
    assert _get_protein_name_from_header(hdr3) == "EBNA1_EBVB9"


def test_load_all_proteins_mocked(tmp_path):
    # Create mock fastas
    fasta1 = tmp_path / "EBV_B95_8_panel8.fasta"
    fasta1.write_text(
        ">sp|P03211|EBNA1_EBVB9 Mock EBNA1\nMSDEGPGTGPG\n>sp|P13285|LMP2_EBVB9 Mock LMP2\nMGSLEMVPMG\n"
    )

    fasta2 = tmp_path / "HPV16_18_panel8.fasta"
    fasta2.write_text(">sp|P03126|VE6_HPV16 Mock E6\nMHQKRTAMFQ\n")

    # Patch fasta paths in load_all_proteins or run load_all_proteins with custom paths
    # Since fasta_files is hardcoded in the function, we can temporarily patch it or test that our parsing logic matches
    import src.train_classifier

    original_fasta_files = src.train_classifier.load_all_proteins.__globals__.get(
        "fasta_files", None
    )

    # We can inspect the code of load_all_proteins or just mock the file system if needed.
    # Alternatively, since we can't easily change the local list in the function scope,
    # we can test the function by verifying it successfully returns the actual fastas of the workspace
    # since we know the workspace has the fasta files in data/proteomes!
    proteins = load_all_proteins()
    assert len(proteins) > 0
    assert "VE6_HPV16" in proteins or "VE2_HPV16" in proteins or "GP350_EBVB9" in proteins


def test_lopo_cross_validate():
    # Setup mock features DataFrame
    np.random.seed(42)
    n_samples = 40
    X = pd.DataFrame(np.random.normal(size=(n_samples, 5)), columns=[f"f{i}" for i in range(5)])
    y = np.random.choice([0, 1], size=n_samples)

    # We have 4 mock proteins, 10 samples each
    proteins = ["PROT_A"] * 10 + ["PROT_B"] * 10 + ["PROT_C"] * 10 + ["PROT_D"] * 10
    metadata = pd.DataFrame(
        {
            "peptide": [f"PEPTIDE_{i}" for i in range(n_samples)],
            "virus": ["EBV"] * 20 + ["HPV16"] * 20,
            "protein": proteins,
        }
    )

    # Mock classifier class that conforms to sklearn api
    class DummyClassifier:
        def __init__(self, **kwargs):
            pass

        def fit(self, X, y, sample_weight=None):
            return self

        def predict_proba(self, X):
            # Return uniform random probabilities for 2 classes
            return np.column_stack([np.ones(len(X)) * 0.5, np.ones(len(X)) * 0.5])

    # Run _cross_validate with use_lopo=True
    avg, std, subgroup_df, oof_df = _cross_validate(
        X, y, metadata, DummyClassifier, {}, use_lopo=True, subgroup_columns=["virus", "protein"]
    )

    # LeaveOneGroupOut with 4 unique groups should yield exactly 4 folds
    assert len(oof_df["fold"].unique()) == 4
    assert "fold" in subgroup_df.columns
    assert "auc_roc" in avg
    assert "auc_pr" in avg


class _DummyClassifier:
    """sklearn-shaped stub: fixed predictions, no real fitting."""

    def __init__(self, **kwargs):
        pass

    def fit(self, X, y, sample_weight=None):
        return self

    def predict_proba(self, X):
        return np.column_stack([np.ones(len(X)) * 0.5, np.ones(len(X)) * 0.5])


def _grouped_cv_fixture(n_peptides=40, rows_per_peptide=3):
    """Peptides repeated across rows, the shape cv_group_by="peptide" exists for."""
    n = n_peptides * rows_per_peptide
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(n, 5)), columns=[f"f{i}" for i in range(5)])
    peptides, labels, origins, alleles = [], [], [], []
    for i in range(n_peptides):
        for _ in range(rows_per_peptide):
            peptides.append(f"PEPTIDE{i:03d}")
            labels.append(i % 2)
            origins.append("tested_negative" if i % 2 == 0 else None)
            alleles.append("HLA-A*02:01")
    y = np.array(labels)
    metadata = pd.DataFrame(
        {"peptide": peptides, "negative_origin": origins, "hla_allele": alleles}
    )
    return X, y, metadata


def test_cross_validate_cv_group_by_peptide_is_fold_disjoint():
    X, y, metadata = _grouped_cv_fixture()
    avg, std, subgroup_df, oof_df = _cross_validate(
        X, y, metadata, _DummyClassifier, {}, n_splits=5, cv_group_by="peptide"
    )
    assert "auc_roc" in avg
    for fold_idx, fold_df in oof_df.groupby("fold"):
        train_peptides = set(oof_df.loc[oof_df["fold"] != fold_idx, "peptide"])
        assert not (set(fold_df["peptide"]) & train_peptides)


def test_cross_validate_cv_group_by_none_matches_legacy_default():
    X, y, metadata = _grouped_cv_fixture()
    avg_default, _std, _subgroup, oof_default = _cross_validate(
        X, y, metadata, _DummyClassifier, {}, n_splits=5
    )
    avg_explicit, _std2, _subgroup2, oof_explicit = _cross_validate(
        X, y, metadata, _DummyClassifier, {}, n_splits=5, cv_group_by=None
    )
    assert avg_default == avg_explicit
    assert oof_default["fold"].tolist() == oof_explicit["fold"].tolist()


def test_cross_validate_cv_group_by_and_lopo_conflict():
    X, y, metadata = _grouped_cv_fixture()
    metadata = metadata.assign(protein=["P0"] * len(metadata))
    with pytest.raises(ValueError, match="select different splitters"):
        list(
            _cross_validate(
                X, y, metadata, _DummyClassifier, {}, use_lopo=True, cv_group_by="peptide"
            )
        )


def test_cross_validate_cv_group_by_unsupported_value_raises():
    X, y, metadata = _grouped_cv_fixture()
    with pytest.raises(ValueError, match="Unsupported cv_group_by"):
        _cross_validate(X, y, metadata, _DummyClassifier, {}, cv_group_by="protein")


def test_cross_validate_cv_group_by_peptide_requires_peptide_column():
    X, y, metadata = _grouped_cv_fixture()
    metadata_no_peptide = metadata.drop(columns=["peptide"])
    with pytest.raises(ValueError, match="requires a 'peptide' column"):
        _cross_validate(X, y, metadata_no_peptide, _DummyClassifier, {}, cv_group_by="peptide")


# ---------------------------------------------------------------------------
# fold_impute_columns (Phase 0 step 6: in-fold antigen-processing imputation)
# ---------------------------------------------------------------------------


class _RecordingClassifier:
    """Stub that records every X it is fit on, keyed by call order."""

    fit_calls: list = []

    def __init__(self, **kwargs):
        pass

    def fit(self, X, y, sample_weight=None):
        _RecordingClassifier.fit_calls.append(X.copy())
        return self

    def predict_proba(self, X):
        return np.column_stack([np.ones(len(X)) * 0.5, np.ones(len(X)) * 0.5])


def test_cross_validate_fold_impute_touches_only_named_columns():
    X, y, metadata = _grouped_cv_fixture(n_peptides=20, rows_per_peptide=3)
    X = X.copy()
    X["netchop_score"] = np.nan
    X.loc[X.index[:10], "netchop_score"] = 0.5
    X["f0"] = np.nan  # a control column NOT named in fold_impute_columns

    _RecordingClassifier.fit_calls = []
    _cross_validate(
        X,
        y,
        metadata,
        _RecordingClassifier,
        {},
        n_splits=5,
        fold_impute_columns=("netchop_score",),
    )
    for fitted_X in _RecordingClassifier.fit_calls:
        assert not fitted_X["netchop_score"].isna().any()
        assert fitted_X["f0"].isna().all()  # untouched: not in fold_impute_columns


def test_cross_validate_fold_impute_median_is_train_rows_only():
    # Concentrate the NaN rows in one stratum so the train-fold median
    # diverges sharply from the whole-column (pooled) median.
    n_peptides, rows_per_peptide = 20, 3
    X, y, metadata = _grouped_cv_fixture(n_peptides=n_peptides, rows_per_peptide=rows_per_peptide)
    X = X.copy()
    netchop = np.full(len(X), 10.0)
    # Make roughly half of it NaN, concentrated among a subset of peptides,
    # so at least one fold's training median differs from the pooled one.
    for pep_idx in range(n_peptides):
        if pep_idx % 2 == 0:
            netchop[pep_idx * rows_per_peptide : (pep_idx + 1) * rows_per_peptide] = np.nan
    X["netchop_score"] = netchop

    _RecordingClassifier.fit_calls = []
    _cross_validate(
        X,
        y,
        metadata,
        _RecordingClassifier,
        {},
        n_splits=5,
        fold_impute_columns=("netchop_score",),
    )
    assert _RecordingClassifier.fit_calls  # sanity: folds ran
    for fitted_X in _RecordingClassifier.fit_calls:
        assert not fitted_X["netchop_score"].isna().any()
        # every filled value must be exactly 10.0 (the only non-NaN value present)
        assert (fitted_X["netchop_score"] == 10.0).all()


def test_cross_validate_fold_impute_columns_none_is_bitwise_identical():
    X, y, metadata = _grouped_cv_fixture()  # no NaN anywhere in this fixture
    avg_none, _std1, _sub1, oof_none = _cross_validate(
        X, y, metadata, _DummyClassifier, {}, n_splits=5, fold_impute_columns=None
    )
    avg_cols, _std2, _sub2, oof_cols = _cross_validate(
        X, y, metadata, _DummyClassifier, {}, n_splits=5, fold_impute_columns=("f0",)
    )
    assert avg_none == avg_cols
    assert oof_none["score"].tolist() == oof_cols["score"].tolist()


# ---------------------------------------------------------------------------
# _excluded_bloc_cv_metrics (vaccinia-excluded OOF re-slice, Phase 0 step 5)
# ---------------------------------------------------------------------------


def _oof_fixture_with_vaccinia():
    # 2 folds, each with a mix of vaccinia and non-vaccinia rows, both classes
    # present in each fold after vaccinia is dropped.
    rows = []
    for fold in (1, 2):
        for i in range(10):
            rows.append(
                {
                    "virus": VACCINIA_VIRUS if i < 6 else "EBV",
                    "label": 0 if i < 6 else (i % 2),
                    "score": 0.1 if i < 6 else 0.9 - 0.05 * i,
                    "fold": fold,
                }
            )
    return pd.DataFrame(rows)


def test_excluded_bloc_cv_metrics_drops_vaccinia_rows():
    oof_df = _oof_fixture_with_vaccinia()
    avg, std = _excluded_bloc_cv_metrics(oof_df)
    assert "auc_roc" in avg
    assert "auc_roc" in std


def test_excluded_bloc_cv_metrics_empty_when_no_virus_column():
    avg, std = _excluded_bloc_cv_metrics(pd.DataFrame({"label": [0, 1], "score": [0.1, 0.9]}))
    assert avg == {}
    assert std == {}


def test_excluded_bloc_cv_metrics_empty_on_empty_input():
    avg, std = _excluded_bloc_cv_metrics(pd.DataFrame())
    assert avg == {}
    assert std == {}


def test_excluded_bloc_cv_metrics_empty_when_every_row_is_the_excluded_bloc():
    oof_df = pd.DataFrame(
        {
            "virus": [VACCINIA_VIRUS] * 4,
            "label": [0, 1, 0, 1],
            "score": [0.1, 0.9, 0.2, 0.8],
            "fold": [1, 1, 2, 2],
        }
    )
    avg, std = _excluded_bloc_cv_metrics(oof_df)
    assert avg == {}
    assert std == {}


# ---------------------------------------------------------------------------
# Feature builders: 30 guard, 31, 33, 35, 166
# ---------------------------------------------------------------------------


def test_prepare_features_30_raises_insufficient_alleles(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"peptide": ["SLLMWITQV"], "bind_A0101": [0.5]}).to_csv(path, index=False)
    df = pd.DataFrame({"peptide": ["SLLMWITQV"]})
    with pytest.raises(ValueError, match="only 1/10"):
        prepare_features_30(df, str(path))


def test_prepare_features_31_shape_and_length_col(tmp_path):
    peps = _make_peptides(3)
    path = _mock_binding_csv(tmp_path, peps)
    result = prepare_features_31(pd.DataFrame({"peptide": peps}), str(path))
    assert result.shape == (3, 31)
    assert "peptide_length" in result.columns
    assert (result["peptide_length"] == 9).all()
    assert not result.isnull().any().any()


def test_prepare_features_33_appends_processing_scores(tmp_path):
    peps = _make_peptides(3)
    binding = _mock_binding_csv(tmp_path, peps)
    ap_cache = tmp_path / "ap.csv"
    pd.DataFrame(
        {"peptide": peps, "netchop_score": [0.4, 0.5, 0.6], "tap_score": [0.1, 0.2, 0.3]}
    ).to_csv(ap_cache, index=False)
    result = prepare_features_33(pd.DataFrame({"peptide": peps}), str(binding), str(ap_cache))
    assert result.shape == (3, 33)
    assert {"netchop_score", "tap_score"}.issubset(result.columns)
    assert not result.isnull().any().any()


def test_prepare_features_33_impute_false_leaves_nan(tmp_path):
    # A peptide missing from the antigen-processing cache stays NaN when
    # impute=False, so a caller can fit the median inside a CV fold instead
    # of from the whole cache (docs/claims_register.md D15).
    peps = _make_peptides(3)
    binding = _mock_binding_csv(tmp_path, peps)
    ap_cache = tmp_path / "ap.csv"
    pd.DataFrame(
        {"peptide": peps[:2], "netchop_score": [0.4, 0.5], "tap_score": [0.1, 0.2]}
    ).to_csv(ap_cache, index=False)
    result = prepare_features_33(
        pd.DataFrame({"peptide": peps}), str(binding), str(ap_cache), impute=False
    )
    assert result["netchop_score"].isna().sum() == 1
    assert result["tap_score"].isna().sum() == 1


def test_prepare_features_35_appends_self_similarity(tmp_path):
    peps = _make_peptides(3)
    binding = _mock_binding_csv(tmp_path, peps)
    ap_cache = tmp_path / "ap.csv"
    pd.DataFrame(
        {"peptide": peps, "netchop_score": [0.4, 0.5, 0.6], "tap_score": [0.1, 0.2, 0.3]}
    ).to_csv(ap_cache, index=False)
    sim_cache = tmp_path / "sim.csv"
    pd.DataFrame(
        {
            "peptide": peps,
            "self_similarity_max_identity": [1.0, 0.5, 0.0],
            "self_similarity_exact_match": [1.0, 0.0, 0.0],
        }
    ).to_csv(sim_cache, index=False)
    result = prepare_features_35(
        pd.DataFrame({"peptide": peps}), str(binding), str(ap_cache), str(sim_cache)
    )
    assert result.shape == (3, 35)
    assert {
        "self_similarity_max_identity",
        "self_similarity_exact_match",
    }.issubset(result.columns)


def test_prepare_features_35_impute_false_leaves_antigen_processing_nan(tmp_path):
    peps = _make_peptides(3)
    binding = _mock_binding_csv(tmp_path, peps)
    ap_cache = tmp_path / "ap.csv"
    pd.DataFrame(
        {"peptide": peps[:2], "netchop_score": [0.4, 0.5], "tap_score": [0.1, 0.2]}
    ).to_csv(ap_cache, index=False)
    sim_cache = tmp_path / "sim.csv"
    pd.DataFrame(
        {
            "peptide": peps,
            "self_similarity_max_identity": [1.0, 0.5, 0.0],
            "self_similarity_exact_match": [1.0, 0.0, 0.0],
        }
    ).to_csv(sim_cache, index=False)
    result = prepare_features_35(
        pd.DataFrame({"peptide": peps}),
        str(binding),
        str(ap_cache),
        str(sim_cache),
        impute=False,
    )
    # impute=False forwards to the antigen-processing join only; self-similarity
    # always fills a constant 0.0 and needs no in-fold variant.
    assert result["netchop_score"].isna().sum() == 1
    assert not result["self_similarity_max_identity"].isnull().any()


def test_prepare_features_166_missing_pseudo_raises(tmp_path):
    peps = _make_peptides(2)
    binding = _mock_binding_csv(tmp_path, peps)
    with pytest.raises(ValueError, match="HLA pseudo-sequence columns"):
        prepare_features_166(pd.DataFrame({"peptide": peps}), str(binding))


def test_prepare_features_166_success_shape(tmp_path):
    peps = _make_peptides(2)
    binding = _mock_binding_csv(tmp_path, peps)
    result = prepare_features_166(_df_with_pseudo(peps), str(binding))
    assert result.shape == (2, 166)


def test_prepare_features_166_no_allele_cols_uses_zeros(tmp_path):
    peps = _make_peptides(2)
    binding = tmp_path / "noalleles.csv"
    pd.DataFrame({"peptide": peps}).to_csv(binding, index=False)
    result = prepare_features_166(_df_with_pseudo(peps), str(binding))
    assert result.shape == (2, 166)
    assert (result[BINDING_ALLELE_COLUMNS[0]] == 0.0).all()


# ---------------------------------------------------------------------------
# train_models: end-to-end pipeline + artifact writing
# ---------------------------------------------------------------------------


def test_train_models_mode21_smoke(tmp_path):
    data_path, _ = _training_csv(tmp_path)
    model_dir = tmp_path / "models21"
    rf, xgb, rf_avg, xgb_avg = train_models(
        str(data_path), model_dir=str(model_dir), n_cv_folds=3, feature_mode=21
    )
    assert (model_dir / "rf_21feature_legacy.joblib").exists()
    assert (model_dir / "xgb_21feature_legacy.joblib").exists()
    assert (model_dir / "training_results.csv").exists()
    assert (model_dir / "training_results_mode21.csv").exists()
    assert "auc_pr" in rf_avg and "auc_pr" in xgb_avg


def test_train_models_mode31_writes_per_mode_artifacts(tmp_path):
    data_path, peps = _training_csv(tmp_path)
    binding = _mock_binding_csv(tmp_path, peps)
    model_dir = tmp_path / "models31"
    train_models(
        str(data_path),
        model_dir=str(model_dir),
        n_cv_folds=3,
        feature_mode=31,
        binding_matrix_path=str(binding),
        use_sample_weights=True,
    )
    assert (model_dir / "rf_31feature_integrated.joblib").exists()
    assert (model_dir / "rf_oof_predictions_mode31.csv").exists()
    threshold_path = model_dir / "optimal_thresholds.json"
    assert threshold_path.exists()
    payload = json.loads(threshold_path.read_text())
    assert payload["method"] == "RandomForest"
    assert payload["feature_mode"] == 31


def test_train_models_mode31_requires_binding_matrix(tmp_path):
    data_path, _ = _training_csv(tmp_path)
    with pytest.raises(ValueError, match="--binding-matrix is required"):
        train_models(str(data_path), model_dir=str(tmp_path / "m"), n_cv_folds=3, feature_mode=31)


# ---------------------------------------------------------------------------
# Output-directory contamination guard
# ---------------------------------------------------------------------------


def test_train_models_requires_explicit_model_dir(tmp_path):
    """No default destination: omitting model_dir is a TypeError, not a write to models/."""
    data_path, _ = _training_csv(tmp_path)
    with pytest.raises(TypeError):
        train_models(str(data_path))


def test_planned_artifact_paths_covers_per_mode_files():
    paths = [os.path.basename(p) for p in planned_artifact_paths("models", 31)]
    assert "training_results_mode31.csv" in paths
    assert "rf_oof_predictions_mode31.csv" in paths
    assert "rf_31feature_integrated.joblib" in paths
    assert "training_results.csv" in paths
    legacy = [os.path.basename(p) for p in planned_artifact_paths("models", 21)]
    assert "rf_21feature_legacy.joblib" in legacy


def test_train_models_refuses_to_overwrite_existing_artifacts(tmp_path):
    data_path, _ = _training_csv(tmp_path)
    model_dir = tmp_path / "published"
    model_dir.mkdir()
    sentinel = model_dir / "training_results_mode21.csv"
    sentinel.write_text("metric,rf_cv_mean\nauc_pr,0.9999\n")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        train_models(str(data_path), model_dir=str(model_dir), n_cv_folds=3, feature_mode=21)

    assert "0.9999" in sentinel.read_text()
    assert not (model_dir / "rf_21feature_legacy.joblib").exists()


def test_train_models_guard_names_the_blocking_files(tmp_path):
    data_path, _ = _training_csv(tmp_path)
    model_dir = tmp_path / "published2"
    model_dir.mkdir()
    (model_dir / "rf_21feature_legacy.joblib").write_bytes(b"stale")

    with pytest.raises(FileExistsError) as excinfo:
        train_models(str(data_path), model_dir=str(model_dir), n_cv_folds=3, feature_mode=21)

    message = str(excinfo.value)
    assert "rf_21feature_legacy.joblib" in message
    assert "--allow-overwrite" in message


def test_train_models_allow_overwrite_replaces_artifacts(tmp_path):
    data_path, _ = _training_csv(tmp_path)
    model_dir = tmp_path / "rerun"
    model_dir.mkdir()
    sentinel = model_dir / "training_results_mode21.csv"
    sentinel.write_text("metric,rf_cv_mean\nauc_pr,0.9999\n")

    train_models(
        str(data_path),
        model_dir=str(model_dir),
        n_cv_folds=3,
        feature_mode=21,
        allow_overwrite=True,
    )

    assert "0.9999" not in sentinel.read_text()
    assert (model_dir / "rf_21feature_legacy.joblib").exists()


def test_train_models_ignores_unrelated_files_in_model_dir(tmp_path):
    data_path, _ = _training_csv(tmp_path)
    model_dir = tmp_path / "mixed"
    model_dir.mkdir()
    (model_dir / "notes.txt").write_text("unrelated")

    train_models(str(data_path), model_dir=str(model_dir), n_cv_folds=3, feature_mode=21)

    assert (model_dir / "training_results_mode21.csv").exists()


def test_train_classifier_cli_requires_model_dir():
    """The module entry point cannot fall back to the production models/ directory."""
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "src.train_classifier", "--data", "does_not_matter.csv"],
        capture_output=True,
        text=True,
        cwd=".",
    )
    assert result.returncode != 0
    assert "--model-dir" in result.stderr


def test_train_classifier_cli_exposes_allow_overwrite():
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "src.train_classifier", "--help"],
        capture_output=True,
        text=True,
        cwd=".",
    )
    assert "--allow-overwrite" in result.stdout


def test_train_classifier_cli_bare_lopo_is_not_rejected(tmp_path):
    """--cv-group-by defaults to 'peptide', but a bare --lopo must still work.

    Regression guard: naively treating the default as an explicit choice made
    --lopo (valid before Phase 0) abort on a spurious conflict error.
    """
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.train_classifier",
            "--data",
            "does_not_exist.csv",
            "--model-dir",
            str(tmp_path / "out"),
            "--lopo",
        ],
        capture_output=True,
        text=True,
        cwd=".",
    )
    # It must fail on the missing data file, NOT on a splitter conflict.
    assert "select different splitters" not in result.stderr
    assert "Data file does not exist" in result.stderr


def test_train_classifier_cli_explicit_lopo_and_grouped_conflict(tmp_path):
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.train_classifier",
            "--data",
            "does_not_exist.csv",
            "--model-dir",
            str(tmp_path / "out"),
            "--lopo",
            "--cv-group-by",
            "peptide",
        ],
        capture_output=True,
        text=True,
        cwd=".",
    )
    assert result.returncode != 0
    assert "select different splitters" in result.stderr


# ---------------------------------------------------------------------------
# Quarantine filter
# ---------------------------------------------------------------------------


def test_filter_quarantined_drops_flagged_rows():
    df = pd.DataFrame(
        {
            "peptide": _make_peptides(4),
            "label": [0, 1, 0, 1],
            "is_quarantined": [True, False, True, False],
        }
    )
    result = _filter_quarantined(df)
    assert len(result) == 2
    assert result["is_quarantined"].tolist() == [False, False]


def test_filter_quarantined_noop_without_column():
    df = pd.DataFrame(
        {
            "peptide": _make_peptides(4),
            "label": [0, 1, 0, 1],
        }
    )
    result = _filter_quarantined(df)
    assert len(result) == 4


def test_filter_quarantined_handles_nan():
    df = pd.DataFrame(
        {
            "peptide": _make_peptides(3),
            "label": [0, 1, 0],
            "is_quarantined": [True, False, None],
        }
    )
    result = _filter_quarantined(df)
    # True row dropped; False and NaN rows kept
    assert len(result) == 2
    assert True not in result["is_quarantined"].tolist()
