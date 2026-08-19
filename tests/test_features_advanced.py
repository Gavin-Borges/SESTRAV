"""Tests for uncovered src/features.py paths.

Targets:
  - compute_sample_weights  (virus + length bias correction)
  - compute_features_for_dataset  (vectorised batch extraction)
  - compute_weisfeiler_lehman_features  (NetworkX WL kernel)
  - get_esm_cls_token         (ESM-2 success path via mocked transformers)
"""

import hashlib
import sys
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from src.features import (
    compute_features,
    compute_features_for_dataset,
    compute_sample_weights,
    compute_weisfeiler_lehman_features,
    get_esm_cls_token,
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
        df = pd.DataFrame(
            {
                "virus": ["EBV"] * 4,
                "seq": ["CLGGLLTMV"] * 2 + ["TIHDIILECV"] * 2,
            }
        )
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
        for col in (
            "p4_hydrophobicity",
            "p5_hydrophobicity",
            "p7_hydrophobicity",
            "p4_charge",
            "p7_charge",
            "peptide_length",
        ):
            assert df_out[col].iloc[0] == pytest.approx(scalar[col], abs=1e-9)

    def test_8mer_zero_imputes_p7_p8(self):
        df_out = compute_features_for_dataset(self._df(["RAKFKQLL"]))
        assert df_out["p7_hydrophobicity"].iloc[0] == 0.0
        assert df_out["p8_hydrophobicity"].iloc[0] == 0.0

    def test_binding_score_passthrough(self):
        df_out = compute_features_for_dataset(self._df(["CLGGLLTMV"], scores=[0.75]))
        assert df_out["binding_score"].iloc[0] == pytest.approx(0.75)

    def test_missing_binding_col_defaults_zero(self):
        df_out = compute_features_for_dataset(self._df(["CLGGLLTMV"]))
        assert df_out["binding_score"].iloc[0] == 0.0

    def test_nan_binding_score_imputed_zero(self):
        df_out = compute_features_for_dataset(self._df(["CLGGLLTMV"], scores=[float("nan")]))
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
        assert df_out["p7_hydrophobicity"].iloc[0] == pytest.approx(scalar["p7_hydrophobicity"])
        assert df_out["p8_hydrophobicity"].iloc[0] == pytest.approx(scalar["p8_hydrophobicity"])


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
        """In-process determinism only. See test_wl_features_are_stable_across_processes.

        This assertion held even while WL features were re-randomised on every
        interpreter launch, because CPython's hash salt is fixed within a process.
        It is kept as a cheap smoke check, not as the determinism gate.
        """
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


# ---------------------------------------------------------------------------
# WL cross-process determinism
# ---------------------------------------------------------------------------
# The node-colour hash must not be CPython's builtin hash(), which is salted per
# interpreter via PYTHONHASHSEED. While it was, a model trained in one process and
# scored in another saw different graph_wl_* features, and the in-process
# test_deterministic above passed throughout.
#
# The seeds below are deliberately DIFFERENT from each other. Pinning PYTHONHASHSEED
# to one value here (or in pytest.ini / ci.yml) would make a reintroduced hash() look
# stable and hide exactly the defect this test exists to catch.

_WL_PROBE = """
import sys
sys.path.insert(0, sys.argv[1])
import networkx as nx
from src.features import (
    compute_wl_features,
    compute_weisfeiler_lehman_features,
    get_cb_cb_edges,
)

peptide = "GLFYTRTGL"
production = compute_wl_features(peptide, get_cb_cb_edges(len(peptide)))

G = nx.path_graph(6)
for node in G.nodes():
    G.nodes[node]["x"] = str(node % 3)
kernel = compute_weisfeiler_lehman_features(G)

print(",".join(str(int(v)) for v in list(production) + list(kernel)))
"""


@pytest.mark.parametrize("seeds", [("1", "2"), ("0", "12345")], ids=["1v2", "0v12345"])
def test_wl_features_are_stable_across_processes(seeds):
    """Both WL paths must yield identical vectors under different hash seeds."""
    import os
    import subprocess
    from pathlib import Path

    repo_root = str(Path(__file__).resolve().parents[1])

    outputs = []
    for seed in seeds:
        env = dict(os.environ, PYTHONHASHSEED=seed, CUDA_VISIBLE_DEVICES="")
        proc = subprocess.run(
            [sys.executable, "-c", _WL_PROBE, repo_root],
            capture_output=True,
            text=True,
            env=env,
            cwd=repo_root,
            timeout=300,
        )
        assert proc.returncode == 0, f"probe failed (seed {seed}): {proc.stderr}"
        outputs.append(proc.stdout.strip().splitlines()[-1])

    assert outputs[0] == outputs[1], (
        f"WL features changed across processes: PYTHONHASHSEED={seeds[0]} gave "
        f"{outputs[0]}, PYTHONHASHSEED={seeds[1]} gave {outputs[1]}"
    )
    assert outputs[0].count(",") == 63  # 32 production + 32 kernel values


# ---------------------------------------------------------------------------
# get_esm_cls_token - ESM-2 success path via mocked transformers
# ---------------------------------------------------------------------------
# Every assertion below must be able to tell the ESM-2 path from the except-branch
# fallback. `assert result.shape == (320,)` cannot: the fallback returns (320,) too,
# so that assertion survived a mutation that made EsmModel.from_pretrained raise.
# Each test therefore pins the exact vector it expects - the mock's sentinel on the
# success paths, the sha256-seeded stream on the fallback path.


class TestGetEsmClsToken:
    def _sentinel(self):
        """The CLS vector the mocked ESM-2 model hands back.

        A ramp rather than zeros or ones: it is float32 (the fallback is float64),
        it is unmistakably not a Gaussian, and every element differs, so a wrong
        tensor slice or a broadcast placeholder shows up as a mismatch too.
        """
        return np.arange(320, dtype=np.float32) * 0.01

    def _mock_transformers(self, cls_vector=None):
        """Build a sys.modules['transformers'] mock that makes the ESM-2
        success path (lines 524-535) run without a network call.

        ``cls_vector`` defaults to the shared sentinel; pass one explicitly when a
        test needs the freshly loaded model to be distinguishable from a cached one.
        """
        if cls_vector is None:
            cls_vector = self._sentinel()

        mock_cls_repr = MagicMock()
        mock_cls_repr.numpy.return_value = cls_vector

        mock_last_hidden_state = MagicMock()
        mock_last_hidden_state.__getitem__.return_value = mock_cls_repr

        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = mock_last_hidden_state

        mock_model_instance = MagicMock()
        mock_model_instance.return_value = mock_outputs

        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.return_value = {}

        MockAutoTokenizer = MagicMock()
        MockAutoTokenizer.from_pretrained.return_value = mock_tokenizer_instance

        MockEsmModel = MagicMock()
        MockEsmModel.from_pretrained.return_value = mock_model_instance

        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer = MockAutoTokenizer
        mock_transformers.EsmModel = MockEsmModel
        return mock_transformers

    def test_esm_cls_token_success_path(self, monkeypatch, capsys):
        """Covers lines 524-535: ESM-2 transformers path runs when model not cached."""
        import src.features as f

        # Reset the module-level cache so the load path runs.
        monkeypatch.setattr(f, "_esm_model", None)
        monkeypatch.setattr(f, "_esm_tokenizer", None)
        monkeypatch.setitem(sys.modules, "transformers", self._mock_transformers())

        result = get_esm_cls_token("CLGGLLTMV")
        assert result.shape == (320,)
        # This is the assertion that fails if the ESM-2 path never ran: the
        # fallback returns float64 Gaussians, not the mock's ramp.
        np.testing.assert_array_equal(result, self._sentinel())
        assert result.dtype == np.float32  # the fallback returns float64
        assert "Falling back" not in capsys.readouterr().out

    def test_esm_cls_token_reuses_cached_model(self, monkeypatch, capsys):
        """When the model is already cached, the load block (524-527) is skipped."""
        import src.features as f

        mock_transformers = self._mock_transformers()

        cls_vector = np.ones(320, dtype=np.float32)
        mock_cls = MagicMock()
        mock_cls.numpy.return_value = cls_vector
        mock_lhs = MagicMock()
        mock_lhs.__getitem__.return_value = mock_cls
        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = mock_lhs
        mock_model_cached = MagicMock()
        mock_model_cached.return_value = mock_outputs
        mock_tok_cached = MagicMock()
        mock_tok_cached.return_value = {}

        monkeypatch.setattr(f, "_esm_model", mock_model_cached)
        monkeypatch.setattr(f, "_esm_tokenizer", mock_tok_cached)
        monkeypatch.setitem(sys.modules, "transformers", mock_transformers)

        result = get_esm_cls_token("CLGGLLTMV")
        assert result.shape == (320,)
        # Three outcomes are now distinguishable: ones means the cached model was
        # used, the _mock_transformers ramp means it was reloaded anyway, float64
        # Gaussians mean the fallback fired.
        np.testing.assert_array_equal(result, cls_vector)
        assert result.dtype == np.float32
        mock_model_cached.assert_called_once()
        # What the docstring claims, finally checked.
        mock_transformers.EsmModel.from_pretrained.assert_not_called()
        mock_transformers.AutoTokenizer.from_pretrained.assert_not_called()
        assert "Falling back" not in capsys.readouterr().out

    def test_esm_cls_token_fallback_on_error(self, monkeypatch, capsys):
        """When the model raises, the except path returns a deterministic mock vector."""
        import src.features as f

        monkeypatch.setattr(f, "_esm_model", None)
        monkeypatch.setattr(f, "_esm_tokenizer", None)
        # Inject a broken transformers module that raises on import.
        bad_transformers = MagicMock()
        bad_transformers.AutoTokenizer.from_pretrained.side_effect = RuntimeError("net")
        monkeypatch.setitem(sys.modules, "transformers", bad_transformers)

        result = get_esm_cls_token("CLGGLLTMV")
        assert result.shape == (320,)
        assert "Falling back" in capsys.readouterr().out
        # Pin the stream. The seed is sha256-derived and NOT the builtin hash():
        # hash() is salted per interpreter (see _wl_color), so a revert to it would
        # re-randomise these 320 features on every process launch and shape alone
        # would not notice. assert_allclose, not assert_array_equal, so a last-bit
        # change in a future numpy PCG64 is not a spurious failure.
        seed = int.from_bytes(hashlib.sha256(b"CLGGLLTMV").digest()[:4], "big")
        expected = np.random.default_rng(seed).normal(0, 1, 320)
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-12)
        # Measured literal, independent of the line above, so the pin cannot be
        # satisfied by a source-and-test change that merely agrees with itself.
        assert float(result.sum()) == pytest.approx(30.199397801339, abs=1e-9)
