"""Unit tests for scripts/precompute_esm2_embeddings.py.

Covers the ESM_MODEL_DIMS registry, auto-output-path naming, and the
unknown-model validation guard - without running actual ESM-2 inference.
"""

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.precompute_esm2_embeddings import ESM_MODEL_DIMS


# ---------------------------------------------------------------------------
# Registry completeness
# ---------------------------------------------------------------------------


def test_registry_contains_t6():
    assert "facebook/esm2_t6_8M_UR50D" in ESM_MODEL_DIMS


def test_registry_contains_t12():
    assert "facebook/esm2_t12_35M_UR50D" in ESM_MODEL_DIMS


def test_registry_contains_t30():
    assert "facebook/esm2_t30_150M_UR50D" in ESM_MODEL_DIMS


def test_t6_dim_is_320():
    assert ESM_MODEL_DIMS["facebook/esm2_t6_8M_UR50D"] == 320


def test_t12_dim_is_480():
    assert ESM_MODEL_DIMS["facebook/esm2_t12_35M_UR50D"] == 480


def test_t30_dim_is_640():
    assert ESM_MODEL_DIMS["facebook/esm2_t30_150M_UR50D"] == 640


def test_all_dims_are_positive_ints():
    for name, dim in ESM_MODEL_DIMS.items():
        assert isinstance(dim, int) and dim > 0, f"{name} has invalid dim {dim}"


# ---------------------------------------------------------------------------
# Auto output-path naming (simulated via the same logic as __main__)
# ---------------------------------------------------------------------------


def _auto_output_path(model_name: str) -> str:
    variant = model_name.split("/")[-1].lower().replace("_ur50d", "")
    return f"data/esm2_embeddings_{variant}.pt"


def test_auto_path_t6():
    path = _auto_output_path("facebook/esm2_t6_8M_UR50D")
    assert path == "data/esm2_embeddings_esm2_t6_8m.pt"


def test_auto_path_t12():
    path = _auto_output_path("facebook/esm2_t12_35M_UR50D")
    assert path == "data/esm2_embeddings_esm2_t12_35m.pt"


def test_auto_path_t30():
    path = _auto_output_path("facebook/esm2_t30_150M_UR50D")
    assert path == "data/esm2_embeddings_esm2_t30_150m.pt"


# ---------------------------------------------------------------------------
# Unknown model guard
# ---------------------------------------------------------------------------


def test_unknown_model_raises_value_error(tmp_path):
    from scripts.precompute_esm2_embeddings import precompute_esm2

    with pytest.raises(ValueError, match="Unknown model"):
        precompute_esm2(
            "data/immunogenicity_dataset_v4.csv",
            str(tmp_path / "out.pt"),
            model_name="facebook/esm2_t99_FAKE",
        )
