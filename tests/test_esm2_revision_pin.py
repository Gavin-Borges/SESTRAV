"""The ESM-2 weights load from one pinned Hugging Face Hub commit, defined once.

Without ``revision=``, ``from_pretrained`` follows the model repository's default
branch, so a moved or compromised branch changes the weights with no error. The pin
lives in ``src/features.py``; ``scripts/precompute_esm2_embeddings.py`` imports it
rather than repeating the hash, so the two cannot drift.

A revision names a commit of ONE model repository, so the pin covers
``src.features._ESM_MODEL_NAME`` and no other model. The script's other registered
models therefore need an explicit full commit SHA, and are refused without one.

``transformers`` is replaced in ``sys.modules`` by a fake that records every
``from_pretrained`` call, so nothing here downloads a model or touches the network.
"""

from __future__ import annotations

import ast
import re
import runpy
import sys
import types
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import src.features as features
import scripts.precompute_esm2_embeddings as precompute

SCRIPT = _ROOT / "scripts" / "precompute_esm2_embeddings.py"
FEATURES = _ROOT / "src" / "features.py"
T6 = "facebook/esm2_t6_8M_UR50D"
T12 = "facebook/esm2_t12_35M_UR50D"
FULL_SHA = re.compile(r"[0-9a-f]{40}")
# Built rather than written out, so no 40-hex literal sits in this file for the
# cited-commit gate to look up as a commit of this repository.
OTHER_SHA = "ab" * 20


def _fake_transformers(calls: list, dim: int) -> types.ModuleType:
    """A transformers stand-in whose loaders record (kind, name, kwargs)."""

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            calls.append(("tokenizer", name, kwargs))
            return cls()

        def __call__(self, seqs, return_tensors="pt", padding=False, truncation=False):
            seqs = [seqs] if isinstance(seqs, str) else seqs
            width = max(len(s) for s in seqs) + 2  # [CLS] residues [EOS]
            return {"input_ids": torch.zeros((len(seqs), width), dtype=torch.long)}

    class FakeModel:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            calls.append(("model", name, kwargs))
            return cls()

        def eval(self):
            return self

        def __call__(self, input_ids):
            shape = (input_ids.shape[0], input_ids.shape[1], dim)
            return types.SimpleNamespace(last_hidden_state=torch.ones(shape))

    module = types.ModuleType("transformers")
    module.EsmTokenizer = FakeTokenizer
    module.EsmModel = FakeModel
    return module


def _dataset(tmp_path: Path) -> Path:
    path = tmp_path / "dataset.csv"
    path.write_text("peptide\nSIINFEKL\nGILGFVFTL\n", encoding="utf-8")
    return path


def _string_constants(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


# --- The one definition ------------------------------------------------------


def test_the_pin_is_a_full_commit_sha():
    # A branch or tag name can move; only a full commit SHA names fixed content.
    assert FULL_SHA.fullmatch(features._ESM_MODEL_REVISION)


def test_features_defines_the_pin_once():
    shas = [s for s in _string_constants(FEATURES) if FULL_SHA.fullmatch(s)]
    assert shas == [features._ESM_MODEL_REVISION]


def test_script_imports_the_pin_and_holds_no_hash_of_its_own():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "src.features"
        for alias in node.names
    }
    assert {"_ESM_MODEL_NAME", "_ESM_MODEL_REVISION"} <= imported
    assert [s for s in _string_constants(SCRIPT) if FULL_SHA.fullmatch(s)] == []


def test_script_pins_exactly_the_model_the_revision_belongs_to():
    assert precompute.PINNED_REVISIONS == {features._ESM_MODEL_NAME: features._ESM_MODEL_REVISION}
    assert features._ESM_MODEL_NAME in precompute.ESM_MODEL_DIMS


# --- src/features.py loads at the pin ------------------------------------------


def test_features_loads_tokenizer_and_model_at_the_pin(monkeypatch):
    calls: list = []
    fake = _fake_transformers(calls, 320)
    fake.AutoTokenizer = fake.EsmTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake)
    monkeypatch.setattr(features, "_esm_model", None)
    monkeypatch.setattr(features, "_esm_tokenizer", None)
    monkeypatch.delenv("SESTRAV_ALLOW_ESM_FALLBACK", raising=False)
    # The load path switches torch's process-wide determinism flag on; put it back.
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        features.get_esm_cls_token("SIINFEKL")
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)

    pin = {"revision": features._ESM_MODEL_REVISION}
    assert calls == [("tokenizer", T6, pin), ("model", T6, pin)]


# --- The precompute script loads at the pin ------------------------------------


def test_default_model_loads_at_the_shared_pin(tmp_path, monkeypatch):
    calls: list = []
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers(calls, 320))
    out = tmp_path / "out.pt"

    precompute.precompute_esm2(str(_dataset(tmp_path)), str(out), model_name=T6)

    pin = {"revision": features._ESM_MODEL_REVISION}
    assert calls == [("tokenizer", T6, pin), ("model", T6, pin)]
    saved = torch.load(out, map_location="cpu", weights_only=True)
    assert sorted(saved) == ["GILGFVFTL", "SIINFEKL"]
    assert saved["SIINFEKL"].shape == (precompute.MAX_LEN, 320)


def test_unpinned_model_without_a_revision_is_refused_before_loading(tmp_path, monkeypatch):
    calls: list = []
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers(calls, 480))
    out = tmp_path / "out.pt"

    with pytest.raises(ValueError, match="No pinned Hugging Face Hub revision"):
        precompute.precompute_esm2(str(_dataset(tmp_path)), str(out), model_name=T12)

    assert calls == []
    assert not out.exists()


def test_unpinned_model_with_an_explicit_commit_loads_at_that_commit(tmp_path, monkeypatch):
    calls: list = []
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers(calls, 480))

    precompute.precompute_esm2(
        str(_dataset(tmp_path)), str(tmp_path / "out.pt"), model_name=T12, revision=OTHER_SHA
    )

    explicit = {"revision": OTHER_SHA}
    assert calls == [("tokenizer", T12, explicit), ("model", T12, explicit)]


@pytest.mark.parametrize("revision", ["main", "v1.0", "ab" * 4, "AB" * 20, "ab" * 21])
def test_a_revision_that_is_not_a_full_commit_sha_is_refused(tmp_path, monkeypatch, revision):
    calls: list = []
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers(calls, 320))

    with pytest.raises(ValueError, match="full 40-character"):
        precompute.precompute_esm2(
            str(_dataset(tmp_path)), str(tmp_path / "out.pt"), model_name=T6, revision=revision
        )

    assert calls == []


def test_cli_revision_flag_reaches_both_loaders(tmp_path, monkeypatch):
    calls: list = []
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers(calls, 480))
    argv = [str(SCRIPT), "--data", str(_dataset(tmp_path)), "--model", T12]
    argv += ["--output", str(tmp_path / "out.pt"), "--revision", OTHER_SHA]
    monkeypatch.setattr(sys, "argv", argv)

    runpy.run_path(str(SCRIPT), run_name="__main__")

    explicit = {"revision": OTHER_SHA}
    assert calls == [("tokenizer", T12, explicit), ("model", T12, explicit)]


def test_cli_refuses_an_unpinned_model_without_a_revision(tmp_path, monkeypatch):
    # The function-level refusal alone would not catch the CLI defaulting --revision
    # to the t6 pin, which would load a t6 commit hash against the t12 repository.
    calls: list = []
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers(calls, 480))
    out = tmp_path / "out.pt"
    argv = [str(SCRIPT), "--data", str(_dataset(tmp_path)), "--model", T12, "--output", str(out)]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(ValueError, match="No pinned Hugging Face Hub revision"):
        runpy.run_path(str(SCRIPT), run_name="__main__")

    assert calls == []
    assert not out.exists()


@pytest.mark.parametrize(
    "model, revision, message",
    [(T12, None, "No pinned Hugging Face Hub revision"), (T6, "main", "full 40-character")],
)
def test_refusal_happens_before_transformers_is_imported(
    tmp_path, monkeypatch, model, revision, message
):
    # A None entry in sys.modules makes any `import transformers` raise ImportError, so
    # getting the ValueError proves the check ran first.
    monkeypatch.setitem(sys.modules, "transformers", None)

    with pytest.raises(ValueError, match=message):
        precompute.precompute_esm2(
            str(_dataset(tmp_path)), str(tmp_path / "out.pt"), model_name=model, revision=revision
        )
