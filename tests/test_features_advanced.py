"""Tests for uncovered src/features.py paths.

Targets:
  - compute_sample_weights  (virus + length bias correction)
  - compute_features_for_dataset  (vectorised batch extraction)
  - compute_weisfeiler_lehman_features  (NetworkX WL kernel)
"""
import numpy as np
import pandas as pd
import pytest

from src.features import (
    compute_features,
    compute_features_for_dataset,
    compute_sample_weights,
    compute_weisfeiler_lehman_features,
    FEATURE_COLUMNS,
)


# ---------------------------------------------------------------------------
# compute_sample_weights
# ---------------------------------------------------------------------------

class TestComputeSampleWeights:
    def _df(self, viruses, peptides=None):
        if peptides is None:
            peptides = ["CLGGLLTMV"] * len(viruses)
        return pd.DataFrame({"virus": viruses, "peptide": peptides})

    def test_single_virus_no_correction(self):
        df = self._df(["EBV"] * 10)
        w = compute_sample_weights(df)
        assert w.shape == (10,)
        np.testing.assert_allclose(w, 1.0, atol=1e-9)

    def test_mean_weight_is_one(self):
        viruses = ["EBV"] * 6 + ["HPV16"] * 2 + ["CMV"] * 2
        df = self._df(viruses)
        w = compute_sample_weights(df)
        np.testing.assert_allclose(w.mean(), 1.0, atol=1e-9)

    def test_minority_virus_upweighted(self):
        viruses = ["EBV"] * 8 + ["HPV16"] * 2
        df = self._df(viruses)
        w = compute_sample_weights(df)
        ebv_mean = w[:8].mean()
        hpv_mean = w[8:].mean()
        assert hpv_mean > ebv_mean

    def test_length_correction_upweights_non_9mers(self):
        viruses = ["EBV"] * 10
        peptides = ["CLGGLLTMV"] * 7 + ["TIHDIILECV"] * 3  # 7 9-mers, 3 10-mers
        df = self._df(viruses, peptides)
        w = compute_sample_weights(df, virus_weight=0.0, length_weight=1.0)
        w_9 = w[:7].mean()
        w_10 = w[7:].mean()
        assert w_10 > w_9

    def test_all_9mers_no_length_correction(self):
        df = self._df(["EBV"] * 4, ["CLGGLLTMV"] * 4)
        w = compute_sample_weights(df, virus_weight=0.0, length_weight=1.0)
        np.testing.assert_allclose(w.mean(), 1.0, atol=1e-9)

    def test_missing_virus_column_returns_length_weights(self):
        df = pd.DataFrame({"peptide": ["CLGGLLTMV"] * 3 + ["TIHDIILECV"] * 3})
        w = compute_sample_weights(df, virus_col="virus")
        assert w.shape == (6,)
        np.testing.assert_allclose(w.mean(), 1.0, atol=1e-9)

    def test_custom_length_col(self):
        df = pd.DataFrame({
            "virus": ["EBV"] * 4,
            "seq": ["CLGGLLTMV"] * 2 + ["TIHDIILECV"] * 2,
        })
        w = compute_sample_weights(df, length_col="seq")
        assert w.shape == (4,)
        np.testing.assert_allclose(w.mean(), 1.0, atol=1e-9)

    def test_output_is_ndarray(self):
        df = self._df(["EBV", "HPV16"])
        w = compute_sample_weights(df)
        assert isinstance(w, np.ndarray)


# ---------------------------------------------------------------------------
# compute_features_for_dataset
# ---------------------------------------------------------------------------

class TestComputeFeaturesForDataset:
    def _df(self, peptides, scores=None):
        rows = {"peptide": peptides}
        if scores is not None:
            rows["presentation_score"] = scores
        return pd.DataFrame(rows)

    def test_single_9mer_matches_scalar(self):
        pep = "CLGGLLTMV"
        df_out = compute_features_for_dataset(self._df([pep]))
        scalar = compute_features(pep)
        for col in ("p4_hydrophobicity", "p5_hydrophobicity", "p7_hydrophobicity",
                    "p4_charge", "p7_charge", "peptide_length"):
            assert df_out[col].iloc[0] == pytest.approx(scalar[col], abs=1e-9)

    def test_8mer_zero_imputes_p7_p8(self):
        df_out = compute_features_for_dataset(self._df(["RAKFKQLL"]))
        assert df_out["p7_hydrophobicity"].iloc[0] == 0.0
        assert df_out["p8_hydrophobicity"].iloc[0] == 0.0

    def test_binding_score_passthrough(self):
        df_out = compute_features_for_dataset(
            self._df(["CLGGLLTMV"], scores=[0.75])
        )
        assert df_out["binding_score"].iloc[0] == pytest.approx(0.75)

    def test_missing_binding_col_defaults_zero(self):
        df_out = compute_features_for_dataset(self._df(["CLGGLLTMV"]))
        assert df_out["binding_score"].iloc[0] == 0.0

    def test_nan_binding_score_imputed_zero(self):
        df_out = compute_features_for_dataset(
            self._df(["CLGGLLTMV"], scores=[float("nan")])
        )
        assert df_out["binding_score"].iloc[0] == 0.0

    def test_mixed_length_peptides(self):
        peptides = ["RAKFKQLL", "CLGGLLTMV", "TIHDIILECV", "HPVGEADYFEY"]
        df_out = compute_features_for_dataset(self._df(peptides))
        assert list(df_out["peptide_length"].values) == [8, 9, 10, 11]

    def test_output_contains_all_feature_columns(self):
        df_out = compute_features_for_dataset(self._df(["CLGGLLTMV"]))
        for col in FEATURE_COLUMNS:
            assert col in df_out.columns, f"Missing column: {col}"

    def test_original_columns_preserved(self):
        df = pd.DataFrame({"peptide": ["CLGGLLTMV"], "label": [1], "hla": ["A*02:01"]})
        df_out = compute_features_for_dataset(df)
        assert "label" in df_out.columns
        assert "hla" in df_out.columns

    def test_batch_length_matches_input(self):
        peptides = ["CLGGLLTMV"] * 50
        df_out = compute_features_for_dataset(self._df(peptides))
        assert len(df_out) == 50

    def test_10mer_p7_p8_populated(self):
        df_out = compute_features_for_dataset(self._df(["TIHDIILECV"]))
        scalar = compute_features("TIHDIILECV")
        assert df_out["p7_hydrophobicity"].iloc[0] == pytest.approx(
            scalar["p7_hydrophobicity"]
        )
        assert df_out["p8_hydrophobicity"].iloc[0] == pytest.approx(
            scalar["p8_hydrophobicity"]
        )


# ---------------------------------------------------------------------------
# compute_weisfeiler_lehman_features
# ---------------------------------------------------------------------------

class TestComputeWeisfeilerLehmanFeatures:
    def _graph(self, n_nodes=4):
        import networkx as nx
        G = nx.path_graph(n_nodes)
        for node in G.nodes():
            G.nodes[node]["x"] = str(node % 3)
        return G

    def test_output_shape(self):
        G = self._graph(5)
        wl = compute_weisfeiler_lehman_features(G)
        assert wl.shape == (32,)

    def test_output_nonnegative(self):
        G = self._graph(6)
        wl = compute_weisfeiler_lehman_features(G)
        assert (wl >= 0).all()

    def test_single_node_graph(self):
        import networkx as nx
        G = nx.Graph()
        G.add_node(0)
        G.nodes[0]["x"] = "A"
        wl = compute_weisfeiler_lehman_features(G)
        assert wl.shape == (32,)

    def test_deterministic(self):
        G = self._graph(7)
        wl1 = compute_weisfeiler_lehman_features(G)
        wl2 = compute_weisfeiler_lehman_features(G)
        np.testing.assert_array_equal(wl1, wl2)

    def test_different_graphs_differ(self):
        import networkx as nx
        G1 = nx.path_graph(5)
        G2 = nx.complete_graph(5)
        for G in (G1, G2):
            for n in G.nodes():
                G.nodes[n]["x"] = "0"
        wl1 = compute_weisfeiler_lehman_features(G1)
        wl2 = compute_weisfeiler_lehman_features(G2)
        assert not np.array_equal(wl1, wl2)
