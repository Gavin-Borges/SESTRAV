"""train_gnn (v1) must refuse a feature mode it does not build.

v1 builds two feature sets: the 21 sequence-only physico columns and mode 50's
expanded multi-allele set. Any other mode fell through to the 21-column branch
while the feature cache stayed keyed by the REQUESTED mode, so
`--architecture v1 --feature-mode 31 --binding-matrix M` wrote 21 columns under
the name `_feature_cache_name` gives mode 31 with that corpus and matrix, which
is the name train_gnn_v2 computes for the same arguments, in the same store. A
later v2 mode-31 run loaded those 21 columns on a cache hit, because
`_validate_cached_features` checks the row count only, and recorded
feature_mode=31. That breaks the invariant tests/test_train_gnn_feature_cache_key.py
states - a cached matrix must contain exactly the features its filename claims -
by a route that file does not cover.

The help half ties the CLI text to the code rather than to a literal: it reads
the mode set each trainer accepts from its own guard, and requires the
--feature-mode help to name every one and the --binding-matrix help to name
exactly the modes that need the matrix.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="GNN modules require torch")

import src.core.feature_store as feature_store  # noqa: E402
import src.train_gnn as tg  # noqa: E402

TRAIN_GNN = Path(tg.__file__)
TREE = ast.parse(TRAIN_GNN.read_text(encoding="utf-8"))
_AAS = "ACDEFGHIKLMNPQRSTVWY"


@pytest.fixture(autouse=True)
def _isolated_anomaly_flag():
    """train_gnn enables anomaly detection process-globally; do not leak it from here."""
    previous = torch.is_anomaly_enabled()
    try:
        yield
    finally:
        torch.set_anomaly_enabled(previous)


class _StopAtFeatures(Exception):
    """Raised by the spy so an accepted run ends before any training happens."""


class _RecordingStore:
    """Stands in for FeatureStore and records every cache access."""

    def __init__(self, *args, **kwargs):
        self.loaded: list[str] = []
        self.saved: list[str] = []

    def load_cached_features(self, cache_name):
        self.loaded.append(cache_name)
        return None

    def save_cached_features(self, df, cache_name):
        self.saved.append(cache_name)


def _corpus(tmp_path):
    peptides = []
    for i in range(4):
        p = list("SLLMWITQV")
        p[0] = _AAS[i % 20]
        p[1] = _AAS[(i // 20) % 20]
        peptides.append("".join(p))
    path = tmp_path / "corpus.csv"
    pd.DataFrame({"peptide": peptides, "label": [0, 1, 0, 1], "virus": ["HIV-1"] * 4}).to_csv(
        path, index=False
    )
    return str(path)


def _matrix(tmp_path):
    path = tmp_path / "binding.csv"
    pd.DataFrame({"peptide": ["SLLMWITQV"], "bind_A0201": [0.5]}).to_csv(path, index=False)
    return str(path)


def _run_v1(tmp_path, monkeypatch, mode):
    store = _RecordingStore()
    monkeypatch.setattr(feature_store, "FeatureStore", lambda *a, **k: store)

    def _stop(*args, **kwargs):
        raise _StopAtFeatures

    monkeypatch.setattr(tg, "prepare_features", _stop)
    monkeypatch.setattr(tg, "prepare_features_50", _stop)
    model_dir = tmp_path / "gnn"
    run = lambda: tg.train_gnn(  # noqa: E731
        _corpus(tmp_path),
        model_dir=str(model_dir),
        feature_mode=mode,
        binding_matrix_path=_matrix(tmp_path),
    )
    return store, model_dir, run


@pytest.mark.parametrize("mode", [31, 30, 33, 35, 10, 51, 166])
def test_v1_refuses_a_mode_it_does_not_build_before_touching_anything(tmp_path, monkeypatch, mode):
    store, model_dir, run = _run_v1(tmp_path, monkeypatch, mode)

    with pytest.raises(ValueError, match=r"supports feature_mode 21 or 50"):
        run()

    # Neither the cache nor the output directory was touched.
    assert store.loaded == [] and store.saved == []
    assert not model_dir.exists()


@pytest.mark.parametrize("mode", [21, 50])
def test_v1_still_reaches_feature_extraction_for_the_modes_it_builds(tmp_path, monkeypatch, mode):
    """Control: the guard admits 21 and 50, and both get as far as the cache lookup."""
    store, _, run = _run_v1(tmp_path, monkeypatch, mode)

    with pytest.raises(_StopAtFeatures):
        run()

    assert len(store.loaded) == 1
    assert store.loaded[0].startswith(f"physico_features_mode{mode}_")


def _accepted_modes(function_name: str) -> set[int]:
    """The tuple in the function's own `if feature_mode not in (...)` guard."""
    function = next(
        node
        for node in ast.walk(TREE)
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Name)
            and node.left.id == "feature_mode"
            and len(node.ops) == 1
            and isinstance(node.ops[0], ast.NotIn)
            and isinstance(node.comparators[0], ast.Tuple)
        ):
            return {element.value for element in node.comparators[0].elts}
    raise AssertionError(f"no `feature_mode not in (...)` guard in {function_name}")


def _help_of(flag: str) -> str:
    for node in ast.walk(TREE):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == flag
        ):
            keywords = {keyword.arg: keyword.value for keyword in node.keywords}
            return ast.literal_eval(keywords["help"])
    raise AssertionError(f"no add_argument call for {flag}")


def test_each_trainer_accepts_the_modes_it_builds():
    assert _accepted_modes("train_gnn") == {21, 50}
    assert _accepted_modes("train_gnn_v2") == {21, 31}


def test_the_feature_mode_help_names_every_mode_and_which_architecture_builds_it():
    help_text = _help_of("--feature-mode")
    named = {int(n) for n in re.findall(r"\b(\d+) \(", help_text)}
    assert named == _accepted_modes("train_gnn") | _accepted_modes("train_gnn_v2")
    assert "31 (physico + per-allele binding; --architecture v2 only)" in help_text
    assert "50 (expanded multi-allele; --architecture v1 only)" in help_text


def test_the_binding_matrix_help_names_exactly_the_modes_that_need_it():
    help_text = _help_of("--binding-matrix")
    match = re.search(r"required for --feature-mode ([^.]*)\.", help_text)
    assert match, f"--binding-matrix help no longer says which modes need it: {help_text!r}"
    named = {int(n) for n in re.findall(r"\d+", match.group(1))}
    # Mode 21 is the one mode either trainer builds without a matrix.
    assert named == (_accepted_modes("train_gnn") | _accepted_modes("train_gnn_v2")) - {21}
