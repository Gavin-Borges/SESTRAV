"""The v1 GNN trainer must honour is_quarantined, the way train_gnn_v2 already does.

src/train_gnn.py has TWO corpus readers. train_gnn_v2 filters; train_gnn (v1)
did not, and v1 is reachable: --architecture is registered with choices
["v1", "v2"] and the CLI dispatches anything that is not "v2" to train_gnn. So
`python -m src.train_gnn --architecture v1` against the v5 corpus trained on
every quarantined row.

tests/test_train_classifier.py covers _filter_quarantined in isolation and stays
green if a call site is deleted, so it cannot pin this. These go through the
real entry point and stop at feature extraction: the spy records the frame it is
handed and raises, so nothing here trains a network.
"""

from __future__ import annotations

import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="GNN modules require torch")

import src.train_gnn as tg  # noqa: E402

_AAS = "ACDEFGHIKLMNPQRSTVWY"


@pytest.fixture(autouse=True)
def _isolated_anomaly_flag():
    """train_gnn enables anomaly detection process-globally; do not leak it from here."""
    previous = torch.is_anomaly_enabled()
    try:
        yield
    finally:
        torch.set_anomaly_enabled(previous)


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


def _run_v1(tmp_path, monkeypatch, quarantined):
    path, peptides = _corpus(tmp_path, quarantined)
    seen = []
    monkeypatch.setattr(tg, "prepare_features", _spy(seen))
    model_dir = tmp_path / "gnn"
    model_dir.mkdir()

    with pytest.raises(_StopAtFeatures):
        tg.train_gnn(path, model_dir=str(model_dir))

    return seen, peptides


def test_v1_drops_quarantined_rows_before_feature_extraction(tmp_path, monkeypatch):
    seen, peptides = _run_v1(tmp_path, monkeypatch, [True, False, True, False])

    assert seen == [[peptides[1], peptides[3]]]


def test_v1_keeps_every_row_when_the_column_is_absent(tmp_path, monkeypatch):
    """The v4 corpus carries no is_quarantined column, so the filter must be a no-op there."""
    seen, peptides = _run_v1(tmp_path, monkeypatch, None)

    assert seen == [peptides]
