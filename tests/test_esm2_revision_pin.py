"""The ESM-2 weights load from a pinned Hugging Face Hub commit.

Without ``revision=``, ``from_pretrained`` follows the model repository's default
branch, so a moved or compromised branch changes the weights with no error. The t6
default's pin lives in ``src/features.py``; ``scripts/precompute_esm2_embeddings.py``
imports it rather than repeating the hash, so the two cannot drift.

A revision names a commit of ONE model repository, so each pinned model needs its
own. Two are pinned: ``src.features._ESM_MODEL_NAME`` (the t6 default, whose pin lives
in ``src/features.py`` because that module loads it) and the t12 variant the script's
own usage notes document, whose pin has no other home and so lives in the script. The
remaining registered models are unpinned and are refused without an explicit full
commit SHA.

What the anti-drift tests assert, and no more: ``src/features.py`` holds exactly one
40-hex string constant, the t6 pin; the script imports that pin and never holds its
value; and every 40-hex string constant the script does hold is the pin of a model
other than the t6 default. Nothing here scans the rest of the tree, where the t6 value
also appears, twice, as a fixture in ``tests/test_check_doc_commit_refs.py``.

``transformers`` is replaced in ``sys.modules`` by a fake that records every
``from_pretrained`` call, so nothing here downloads a model or touches the network.
"""

from __future__ import annotations

import ast
import json
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
T30 = "facebook/esm2_t30_150M_UR50D"
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


def test_script_imports_the_t6_pin_and_never_repeats_it():
    """The script takes the t6 pin from src/features.py and never holds its value. A
    hash for another model may live in the script, since src/features.py does not load
    that model, but the shared one may not be copied: two copies of it is two chances
    to drift."""
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "src.features"
        for alias in node.names
    }
    assert {"_ESM_MODEL_NAME", "_ESM_MODEL_REVISION"} <= imported
    hashes = [s for s in _string_constants(SCRIPT) if FULL_SHA.fullmatch(s)]
    assert features._ESM_MODEL_REVISION not in hashes
    # Every hash the script does hold is a pin it owns, for a model features.py does
    # not load, so none of them is a stray or a second copy of the shared one.
    owned = {
        revision
        for model, revision in precompute.PINNED_REVISIONS.items()
        if model != features._ESM_MODEL_NAME
    }
    assert set(hashes) == owned


def test_every_pin_names_a_registered_model_and_a_full_commit_sha():
    assert precompute.PINNED_REVISIONS
    for model, revision in precompute.PINNED_REVISIONS.items():
        assert model in precompute.ESM_MODEL_DIMS, model
        assert FULL_SHA.fullmatch(revision), (model, revision)


def test_the_t6_pin_is_the_one_from_features():
    assert precompute.PINNED_REVISIONS[features._ESM_MODEL_NAME] == (features._ESM_MODEL_REVISION)
    assert features._ESM_MODEL_NAME in precompute.ESM_MODEL_DIMS


def test_the_t12_model_is_pinned():
    """t12 is the variant the script's own usage notes document, so leaving its revision
    to be typed by hand every run is a pin nobody applies."""
    assert T12 in precompute.PINNED_REVISIONS
    assert FULL_SHA.fullmatch(precompute.PINNED_REVISIONS[T12])


def test_the_t12_pin_is_not_the_t6_pin():
    """A commit of one model repository does not exist in another, so reusing the t6
    hash for t12 would be the inverted pin this file already warns about."""
    assert precompute.PINNED_REVISIONS[T12] != features._ESM_MODEL_REVISION


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
        precompute.precompute_esm2(str(_dataset(tmp_path)), str(out), model_name=T30)

    assert calls == []
    assert not out.exists()


def test_t12_without_a_revision_loads_at_its_own_pin(tmp_path, monkeypatch):
    """Before the t12 pin existed this call was refused, so every t12 run had to be
    handed a hash by hand. An unpinned model is still refused; T30 covers that."""
    calls: list = []
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers(calls, 480))

    precompute.precompute_esm2(str(_dataset(tmp_path)), str(tmp_path / "out.pt"), model_name=T12)

    pin = {"revision": precompute.PINNED_REVISIONS[T12]}
    assert calls == [("tokenizer", T12, pin), ("model", T12, pin)]


def test_unpinned_model_with_an_explicit_commit_loads_at_that_commit(tmp_path, monkeypatch):
    """T30 has no pin, so the explicit commit is the only revision it can load at.
    This used T12 until T12 gained a pin, after which it no longer exercised an
    unpinned model at all."""
    assert T30 not in precompute.PINNED_REVISIONS
    calls: list = []
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers(calls, 640))

    precompute.precompute_esm2(
        str(_dataset(tmp_path)), str(tmp_path / "out.pt"), model_name=T30, revision=OTHER_SHA
    )

    explicit = {"revision": OTHER_SHA}
    assert calls == [("tokenizer", T30, explicit), ("model", T30, explicit)]


def test_a_pinned_model_with_an_explicit_commit_loads_at_that_commit(tmp_path, monkeypatch):
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
    # to the t6 pin, which would load a t6 commit hash against another repository.
    calls: list = []
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers(calls, 480))
    out = tmp_path / "out.pt"
    argv = [str(SCRIPT), "--data", str(_dataset(tmp_path)), "--model", T30, "--output", str(out)]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(ValueError, match="No pinned Hugging Face Hub revision"):
        runpy.run_path(str(SCRIPT), run_name="__main__")

    assert calls == []
    assert not out.exists()


@pytest.mark.parametrize(
    "model, revision, message",
    [(T30, None, "No pinned Hugging Face Hub revision"), (T6, "main", "full 40-character")],
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


# --- The v2.3 notebook loads t12 at the script's pin ---------------------------

NOTEBOOK = _ROOT / "notebooks" / "SESTRAV_GNN_v23_canonical_retrain.ipynb"


def _notebook_code_cells(path: Path) -> list[str]:
    """Each code cell's source, with each IPython magic line replaced by ``pass``."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        lines = "".join(cell["source"]).splitlines()
        cells.append(
            "\n".join(
                line[: len(line) - len(line.lstrip())] + "pass"
                if line.lstrip().startswith(("%", "!"))
                else line
                for line in lines
            )
        )
    return cells


def _notebook_hub_loads() -> list[tuple[ast.Call, ast.Module]]:
    loads = []
    for source in _notebook_code_cells(NOTEBOOK):
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "from_pretrained"
            ):
                loads.append((node, tree))
    return loads


def test_the_notebook_loads_the_tokenizer_and_model_at_the_scripts_pin():
    """The notebook rebuilds the t12 cache itself, and its two from_pretrained calls
    once passed no revision, so they followed the model repository's main branch.
    Each must now pass a revision computed by the script's own resolve_revision."""
    loads = _notebook_hub_loads()
    assert len(loads) == 2, "expected the tokenizer load and the model load"
    for call, tree in loads:
        imported = {
            (node.module, alias.name)
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        assert ("scripts.precompute_esm2_embeddings", "resolve_revision") in imported
        resolved_names = {
            target.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "resolve_revision"
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        revision = [keyword.value for keyword in call.keywords if keyword.arg == "revision"]
        assert revision, ast.unparse(call)
        assert isinstance(revision[0], ast.Name), ast.unparse(call)
        assert revision[0].id in resolved_names, ast.unparse(call)


def test_the_notebooks_model_is_t12_and_the_script_resolves_it_to_its_pin():
    _, tree = _notebook_hub_loads()[0]
    models = [
        node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "MODEL" for t in node.targets)
        and isinstance(node.value, ast.Constant)
    ]
    assert models == [T12]
    assert precompute.resolve_revision(T12, None) == precompute.PINNED_REVISIONS[T12]


def test_the_notebook_resolves_the_pin_for_the_model_it_loads():
    """The revision is ``resolve_revision(MODEL, None)``, and both loads take MODEL.

    The test above checks only that ``revision=`` names a variable assigned from
    some resolve_revision call. Its arguments matter as much: another model's name
    resolves that model's pin, a literal hash in place of None bypasses the pin, and
    a different first argument to from_pretrained loads a model the revision was not
    resolved for."""
    loads = _notebook_hub_loads()
    trees = {id(tree) for _, tree in loads}
    assert len(trees) == 1, "the two loads and the resolution must share one cell"
    tree = loads[0][1]
    resolutions = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "resolve_revision"
    ]
    assert len(resolutions) == 1
    assert [ast.unparse(arg) for arg in resolutions[0].args] == ["MODEL", "None"]
    assert resolutions[0].keywords == []
    for call, _ in loads:
        assert [ast.unparse(arg) for arg in call.args] == ["MODEL"], ast.unparse(call)
        assert [k.arg for k in call.keywords] == ["revision"], ast.unparse(call)
