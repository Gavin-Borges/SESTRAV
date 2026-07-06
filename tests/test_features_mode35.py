"""Tests for FEATURE_COLUMNS_35 and prepare_features_35 in train_classifier."""

import numpy as np
import pandas as pd

from src.features import (
    FEATURE_COLUMNS_33,
    FEATURE_COLUMNS_35,
    load_self_similarity_cache,
)


# ---------------------------------------------------------------------------
# Column-list contract
# ---------------------------------------------------------------------------


def test_feature_columns_35_length():
    assert len(FEATURE_COLUMNS_35) == 35


def test_feature_columns_35_is_superset_of_33():
    assert set(FEATURE_COLUMNS_33).issubset(set(FEATURE_COLUMNS_35))


def test_feature_columns_35_adds_similarity_columns():
    extra = [c for c in FEATURE_COLUMNS_35 if c not in FEATURE_COLUMNS_33]
    assert sorted(extra) == ["self_similarity_exact_match", "self_similarity_max_identity"]


def test_feature_columns_35_no_duplicates():
    assert len(FEATURE_COLUMNS_35) == len(set(FEATURE_COLUMNS_35))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_minimal_df():
    peptides = ["AAAAAAAA", "ACDEFGHIK", "LMVKQSRTY", "NPQRSTVWY"]
    return pd.DataFrame(
        {
            "peptide": peptides,
            "label": [1, 0, 1, 0],
            "virus": ["EBV", "HPV", "EBV", "HPV"],
        }
    )


def _write_binding_matrix(peptides, path):
    allele_cols = [
        "bind_A0101",
        "bind_A0201",
        "bind_A0301",
        "bind_A1101",
        "bind_A2402",
        "bind_B0702",
        "bind_B0801",
        "bind_B2705",
        "bind_B3501",
        "bind_B4402",
    ]
    pd.DataFrame([{"peptide": p, **{c: 0.5 for c in allele_cols}} for p in peptides]).to_csv(
        path, index=False
    )


def _write_ap_cache(peptides, path, netchop=0.4, tap=0.6):
    pd.DataFrame(
        [{"peptide": p, "netchop_score": netchop, "tap_score": tap} for p in peptides]
    ).to_csv(path, index=False)


def _write_sim_cache(peptides, path, identity=0.0, exact=False):
    pd.DataFrame(
        [
            {
                "peptide": p,
                "self_similarity_max_identity": identity,
                "self_similarity_exact_match": exact,
            }
            for p in peptides
        ]
    ).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# load_self_similarity_cache
# ---------------------------------------------------------------------------


class TestLoadSelfSimilarityCache:
    def test_returns_dataframe_with_new_columns(self, tmp_path):
        df = _make_minimal_df()
        cache = tmp_path / "sim.csv"
        _write_sim_cache(df["peptide"].tolist(), cache)
        result = load_self_similarity_cache(str(cache), df)
        assert "self_similarity_max_identity" in result.columns
        assert "self_similarity_exact_match" in result.columns

    def test_identity_values_match_cache(self, tmp_path):
        df = _make_minimal_df()
        cache = tmp_path / "sim.csv"
        _write_sim_cache(df["peptide"].tolist(), cache, identity=1.0, exact=True)
        result = load_self_similarity_cache(str(cache), df)
        np.testing.assert_allclose(result["self_similarity_max_identity"].values, 1.0)

    def test_exact_match_stored_as_float(self, tmp_path):
        df = _make_minimal_df()
        cache = tmp_path / "sim.csv"
        _write_sim_cache(df["peptide"].tolist(), cache, exact=True)
        result = load_self_similarity_cache(str(cache), df)
        assert result["self_similarity_exact_match"].dtype == float

    def test_missing_peptide_defaults_to_zero(self, tmp_path):
        df = _make_minimal_df()
        cache = tmp_path / "sim.csv"
        # Only cache the first two peptides
        _write_sim_cache(df["peptide"].tolist()[:2], cache, identity=1.0)
        result = load_self_similarity_cache(str(cache), df)
        # Last two peptides should default to 0.0
        assert result.iloc[2]["self_similarity_max_identity"] == 0.0
        assert result.iloc[3]["self_similarity_max_identity"] == 0.0

    def test_no_nans_in_output(self, tmp_path):
        df = _make_minimal_df()
        cache = tmp_path / "sim.csv"
        _write_sim_cache(df["peptide"].tolist(), cache)
        result = load_self_similarity_cache(str(cache), df)
        assert not result["self_similarity_max_identity"].isna().any()
        assert not result["self_similarity_exact_match"].isna().any()

    def test_identity_bounded_0_to_1(self, tmp_path):
        df = _make_minimal_df()
        cache = tmp_path / "sim.csv"
        _write_sim_cache(df["peptide"].tolist(), cache, identity=0.75)
        result = load_self_similarity_cache(str(cache), df)
        assert (result["self_similarity_max_identity"] >= 0.0).all()
        assert (result["self_similarity_max_identity"] <= 1.0).all()


# ---------------------------------------------------------------------------
# prepare_features_35
# ---------------------------------------------------------------------------


class TestPrepareFeatures35:
    def test_shape(self, tmp_path):
        from src.train_classifier import prepare_features_35

        df = _make_minimal_df()
        bm = tmp_path / "bm.csv"
        _write_binding_matrix(df["peptide"].tolist(), bm)
        ap = tmp_path / "ap.csv"
        _write_ap_cache(df["peptide"].tolist(), ap)
        sim = tmp_path / "sim.csv"
        _write_sim_cache(df["peptide"].tolist(), sim)

        X = prepare_features_35(df, str(bm), str(ap), str(sim))
        assert X.shape == (len(df), 35)

    def test_column_names_match_feature_columns_35(self, tmp_path):
        from src.train_classifier import prepare_features_35

        df = _make_minimal_df()
        bm = tmp_path / "bm.csv"
        _write_binding_matrix(df["peptide"].tolist(), bm)
        ap = tmp_path / "ap.csv"
        _write_ap_cache(df["peptide"].tolist(), ap)
        sim = tmp_path / "sim.csv"
        _write_sim_cache(df["peptide"].tolist(), sim)

        X = prepare_features_35(df, str(bm), str(ap), str(sim))
        assert list(X.columns) == FEATURE_COLUMNS_35

    def test_similarity_columns_reflect_cache(self, tmp_path):
        from src.train_classifier import prepare_features_35

        df = _make_minimal_df()
        bm = tmp_path / "bm.csv"
        _write_binding_matrix(df["peptide"].tolist(), bm)
        ap = tmp_path / "ap.csv"
        _write_ap_cache(df["peptide"].tolist(), ap)
        sim = tmp_path / "sim.csv"
        _write_sim_cache(df["peptide"].tolist(), sim, identity=0.8, exact=False)

        X = prepare_features_35(df, str(bm), str(ap), str(sim))
        np.testing.assert_allclose(X["self_similarity_max_identity"].values, 0.8, atol=1e-6)
        np.testing.assert_allclose(X["self_similarity_exact_match"].values, 0.0, atol=1e-6)

    def test_no_nans_when_all_caches_complete(self, tmp_path):
        from src.train_classifier import prepare_features_35

        df = _make_minimal_df()
        bm = tmp_path / "bm.csv"
        _write_binding_matrix(df["peptide"].tolist(), bm)
        ap = tmp_path / "ap.csv"
        _write_ap_cache(df["peptide"].tolist(), ap)
        sim = tmp_path / "sim.csv"
        _write_sim_cache(df["peptide"].tolist(), sim)

        X = prepare_features_35(df, str(bm), str(ap), str(sim))
        assert not X.isnull().any().any()

    def test_missing_sim_cache_peptide_defaults_to_zero(self, tmp_path):
        from src.train_classifier import prepare_features_35

        df = _make_minimal_df()
        bm = tmp_path / "bm.csv"
        _write_binding_matrix(df["peptide"].tolist(), bm)
        ap = tmp_path / "ap.csv"
        _write_ap_cache(df["peptide"].tolist(), ap)
        sim = tmp_path / "sim.csv"
        # Empty cache - all peptides missing
        pd.DataFrame(
            columns=["peptide", "self_similarity_max_identity", "self_similarity_exact_match"]
        ).to_csv(sim, index=False)

        X = prepare_features_35(df, str(bm), str(ap), str(sim))
        assert (X["self_similarity_max_identity"] == 0.0).all()

    def test_is_superset_of_features_33(self, tmp_path):
        from src.train_classifier import prepare_features_33, prepare_features_35

        df = _make_minimal_df()
        bm = tmp_path / "bm.csv"
        _write_binding_matrix(df["peptide"].tolist(), bm)
        ap = tmp_path / "ap.csv"
        _write_ap_cache(df["peptide"].tolist(), ap)
        sim = tmp_path / "sim.csv"
        _write_sim_cache(df["peptide"].tolist(), sim)

        X33 = prepare_features_33(df, str(bm), str(ap))
        X35 = prepare_features_35(df, str(bm), str(ap), str(sim))

        # First 33 columns of mode-35 must equal mode-33 exactly
        pd.testing.assert_frame_equal(
            X35.iloc[:, :33].reset_index(drop=True),
            X33.reset_index(drop=True),
            check_names=True,
        )


def test_train_classifier_argparse_accepts_mode_35():
    """train_classifier.py argparse does not reject --feature-mode 35."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "src.train_classifier", "--help"],
        capture_output=True,
        text=True,
        cwd=".",
    )
    assert "35" in result.stdout
