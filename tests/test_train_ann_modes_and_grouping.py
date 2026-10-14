"""src/train_ann.py trains only what it builds, and its CV keeps a peptide in one fold.

pipeline.smk's train_ann rule passed config's feature_mode (31) and the binding matrix,
but train_ann builds mode 21 or 50 only, so it trained the 21 sequence-only columns under
a mode-31 request and read no matrix. Its CV was a label-only StratifiedKFold, while the
v5 corpus repeats peptides across alleles and every feature is a function of the peptide
string, so a peptide could sit on both sides of a fold (docs/claims_register.md D15).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("torch", reason="train_ann requires torch")

import src.train_ann as ta  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
_AAS = "ACDEFGHIKLMNPQRSTVWY"


class _Stop(Exception):
    """Raised by a spy so a run ends before any network is trained."""


def _peptides(n):
    out = []
    for i in range(n):
        p = list("SLLMWITQV")
        p[0] = _AAS[i % 20]
        p[1] = _AAS[(i // 20) % 20]
        out.append("".join(p))
    return out


def _corpus(tmp_path):
    """40 peptides, each on two alleles: every peptide appears twice."""
    peptides = _peptides(40)
    rows = []
    for i, pep in enumerate(peptides):
        for allele in ("HLA-A*02:01", "HLA-B*07:02"):
            rows.append({"peptide": pep, "label": i % 2, "virus": "HIV-1", "hla_allele": allele})
    path = tmp_path / "corpus.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


@pytest.mark.parametrize("mode", [31, 30, 33, 35, 10])
def test_train_ann_refuses_a_mode_it_does_not_build(tmp_path, mode):
    model_dir = tmp_path / "ann"
    with pytest.raises(ValueError, match=r"supports feature_mode 21 or 50"):
        ta.train_ann(_corpus(tmp_path), model_dir=str(model_dir), feature_mode=mode)
    assert not model_dir.exists()


def test_train_ann_folds_never_split_a_peptide(tmp_path, monkeypatch):
    recorded = []
    real = ta.PeptideGroupedKFold

    class Recording(real):
        def split(self, *args, **kwargs):
            folds = list(super().split(*args, **kwargs))
            recorded.extend(folds)
            return iter(folds)

    def stop(*args, **kwargs):
        raise _Stop

    monkeypatch.setattr(ta, "PeptideGroupedKFold", Recording)
    monkeypatch.setattr(ta, "StandardScaler", stop)
    path = _corpus(tmp_path)

    with pytest.raises(_Stop):
        ta.train_ann(path, model_dir=str(tmp_path / "ann"), feature_mode=21)

    peptides = pd.read_csv(path)["peptide"].to_numpy()
    assert len(recorded) == 5
    for train_idx, val_idx in recorded:
        assert not set(peptides[train_idx]) & set(peptides[val_idx])


def _built_modes() -> set[int]:
    tree = ast.parse((REPO_ROOT / "src" / "train_ann.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Name)
            and node.left.id == "feature_mode"
            and isinstance(node.ops[0], ast.NotIn)
            and isinstance(node.comparators[0], ast.Tuple)
        ):
            return {element.value for element in node.comparators[0].elts}
    raise AssertionError("no `feature_mode not in (...)` guard in src/train_ann.py")


def test_the_pipeline_rule_asks_train_ann_for_a_mode_it_builds():
    smk = (REPO_ROOT / "pipeline.smk").read_text(encoding="utf-8")
    rule = re.search(r"rule train_ann:\n(.*?)\n\n", smk, re.S)
    assert rule, "rule train_ann not found"
    body = rule.group(1)
    requested = re.search(r"--feature-mode (\S+)", body)
    assert requested, "the train_ann rule names no feature mode"
    assert requested.group(1).isdigit(), f"not a literal mode: {requested.group(1)!r}"
    assert int(requested.group(1)) in _built_modes()
    # Mode 21 reads no binding matrix, so the rule must not feed one in.
    if int(requested.group(1)) == 21:
        assert "binding" not in body
