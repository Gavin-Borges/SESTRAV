"""Quarantine-filter wiring tests for the three ANN training entry points.

tests/test_train_classifier.py already covers `_filter_quarantined` in
isolation. Those three tests stay green if any call site is deleted, so they
cannot pin the wiring. These go through the real entry points instead, the way
tests/test_ann_benchmark_guard.py pins its guard call.

There are THREE such entry points, not two: `src/train_ann.py`'s `train_ann`
(what pipeline.smk runs), `src/ann_benchmark.py`'s different `train_ann`
(imported by `src/bias_skew_finalization.py`), and `src/ablation_study.py`'s
`run_ablation`, which trains the same ANN through `src.model.run_cv`. An
earlier version of this file covered only the first two.

Every case stops at feature extraction: the spy records the frame it is handed
and raises, so nothing here trains a network or touches the feature machinery.
"""

from __future__ import annotations

import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="ANN training requires torch")

import src.ablation_study as ablation_study  # noqa: E402
import src.ann_benchmark as ann_benchmark  # noqa: E402
import src.train_ann as train_ann_module  # noqa: E402

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


class _StopAtFeatures(Exception):
    """Raised by the spy so a run ends before any training happens."""


def _corpus(tmp_path, quarantined):
    """Write a four-row corpus. `quarantined` is the is_quarantined column, or None for v4 shape."""
    peptides = _make_peptides(4)
    data = {
        "peptide": peptides,
        "label": [0, 1, 0, 1],
        "virus": ["HIV-1"] * 4,
    }
    if quarantined is not None:
        data["is_quarantined"] = quarantined
    path = tmp_path / "corpus.csv"
    pd.DataFrame(data).to_csv(path, index=False)
    return str(path), peptides


def _spy(seen):
    def capture(df, *args, **kwargs):
        seen.append(df["peptide"].tolist())
        raise _StopAtFeatures

    return capture


def _out_dir(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    return str(out)


def test_train_ann_drops_quarantined_rows_before_feature_extraction(tmp_path, monkeypatch):
    path, peptides = _corpus(tmp_path, [True, False, True, False])
    seen = []
    monkeypatch.setattr(train_ann_module, "prepare_features", _spy(seen))

    with pytest.raises(_StopAtFeatures):
        train_ann_module.train_ann(path, model_dir=_out_dir(tmp_path), feature_mode=21)

    assert seen == [[peptides[1], peptides[3]]]


def test_train_ann_keeps_every_row_when_the_column_is_absent(tmp_path, monkeypatch):
    """The v4 corpus carries no is_quarantined column, so the filter must be a no-op there."""
    path, peptides = _corpus(tmp_path, None)
    seen = []
    monkeypatch.setattr(train_ann_module, "prepare_features", _spy(seen))

    with pytest.raises(_StopAtFeatures):
        train_ann_module.train_ann(path, model_dir=_out_dir(tmp_path), feature_mode=21)

    assert seen == [peptides]


def test_ann_benchmark_drops_quarantined_rows_before_feature_extraction(tmp_path, monkeypatch):
    path, peptides = _corpus(tmp_path, [True, False, True, False])
    seen = []
    monkeypatch.setattr(ann_benchmark, "_prepare_features_21", _spy(seen))

    with pytest.raises(_StopAtFeatures):
        ann_benchmark.train_ann(path, model_dir=_out_dir(tmp_path), feature_mode=21)

    assert seen == [[peptides[1], peptides[3]]]


def test_ann_benchmark_keeps_every_row_when_the_column_is_absent(tmp_path, monkeypatch):
    path, peptides = _corpus(tmp_path, None)
    seen = []
    monkeypatch.setattr(ann_benchmark, "_prepare_features_21", _spy(seen))

    with pytest.raises(_StopAtFeatures):
        ann_benchmark.train_ann(path, model_dir=_out_dir(tmp_path), feature_mode=21)

    assert seen == [peptides]


def test_ablation_study_drops_quarantined_rows_before_feature_extraction(tmp_path, monkeypatch):
    """The third ANN path. run_ablation feeds src.model.run_cv, the same ANN CV trainer."""
    path, peptides = _corpus(tmp_path, [True, False, True, False])
    seen = []
    monkeypatch.setattr(ablation_study, "prepare_features_30", _spy(seen))

    with pytest.raises(_StopAtFeatures):
        ablation_study.run_ablation(
            path, binding_matrix_path="unused.csv", output_dir=_out_dir(tmp_path)
        )

    assert seen == [[peptides[1], peptides[3]]]


def test_ablation_study_keeps_every_row_when_the_column_is_absent(tmp_path, monkeypatch):
    path, peptides = _corpus(tmp_path, None)
    seen = []
    monkeypatch.setattr(ablation_study, "prepare_features_30", _spy(seen))

    with pytest.raises(_StopAtFeatures):
        ablation_study.run_ablation(
            path, binding_matrix_path="unused.csv", output_dir=_out_dir(tmp_path)
        )

    assert seen == [peptides]
