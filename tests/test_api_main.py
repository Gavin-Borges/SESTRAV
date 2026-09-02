"""First test coverage for api/main.py (D31, docs/claims_register.md).

Before this file, api/main.py had zero test coverage - no unit test, no
TestClient test, no CI smoke request - which is how an inert scorer (the
served bind_* feature block silently stayed all-zero regardless of MHCflurry
availability or the requested allele) survived from 2026-06-13 to 2026-08-19
across 13 commits that edited the file (`git log --since=2026-06-13
--until=2026-08-19 -- api/main.py`).

These call the route handlers directly rather than going through
fastapi.testclient.TestClient. That is deliberate: TestClient pulls in
starlette.testclient, which on current starlette hard-requires the httpx2
package, and CI installs from hash-pinned lockfiles (`pip install
--require-hashes`), so satisfying it would mean adding a dependency plus
SBOM/pip-audit churn purely to reach a test helper. The defect this file
exists for lives in _score_peptide's feature-vector assembly and the
handler's error mapping, not in HTTP serialisation, so a direct call covers
it without that cost. Pydantic still validates PeptideInput on construction,
and HTTPException carries the same status_code the HTTP layer would emit.

Tests avoid loading the real 128MB rf_31feature_integrated.joblib
(gitignored, absent from a clean CI checkout): the fixture pre-seeds
`_manager` with a fake model and a small binding matrix, and sets
`_loaded = True` so nothing tries to load the real artifact.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from fastapi import HTTPException

import api.main as api_main
from src.features import BINDING_ALLELE_COLUMNS, FEATURE_COLUMNS_31

PANEL_PEPTIDE = "GILGFVFTL"
PANEL_ROW = {col: 0.5 + 0.01 * i for i, col in enumerate(BINDING_ALLELE_COLUMNS)}
OUT_OF_PANEL_PEPTIDE = "KKKKKKKKK"


class _FakeRFModel:
    """Records the feature vector it was scored on, for assertion."""

    def __init__(self) -> None:
        self.last_feature_vector: np.ndarray | None = None

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.last_feature_vector = X
        return np.array([[0.2, 0.8]])


@pytest.fixture
def fake_manager() -> Iterator[_FakeRFModel]:
    """Seeds the ModelManager singleton so no real artifact is loaded."""
    fake_model = _FakeRFModel()
    api_main._manager._loaded = True
    api_main._manager.rf_model = fake_model
    api_main._manager.binding_matrix = {PANEL_PEPTIDE: dict(PANEL_ROW)}
    yield fake_model
    api_main._manager._loaded = False


def _score(sequence: str, allele: str) -> api_main.ScoreResponse:
    return api_main.score_peptide(api_main.PeptideInput(sequence=sequence, allele=allele))


def _binding_block(vector: np.ndarray) -> list[float]:
    return [vector[FEATURE_COLUMNS_31.index(c)] for c in BINDING_ALLELE_COLUMNS]


def test_panel_present_peptide_gets_a_nonzero_binding_block(
    fake_manager: _FakeRFModel,
) -> None:
    """The regression this file exists for: bind_* columns must not be all-zero
    for a peptide that has a real panel row, regardless of MHCflurry."""
    resp = _score(PANEL_PEPTIDE, "HLA-A*02:01")
    assert resp.immunogenicity_score == pytest.approx(0.8)

    assert fake_manager.last_feature_vector is not None
    bind_values = _binding_block(fake_manager.last_feature_vector[0])
    assert any(v != 0.0 for v in bind_values), "binding block was all-zero for a panel peptide"
    assert bind_values == [PANEL_ROW[c] for c in BINDING_ALLELE_COLUMNS]


def test_binding_block_is_identical_across_different_alleles(
    fake_manager: _FakeRFModel,
) -> None:
    """D31: the panel is fixed and keyed by peptide, not by the caller's allele.

    This pins the documented, by-design behaviour rather than hiding it: two
    callers requesting different alleles for the same peptide must receive an
    identical binding block, so a future test that expects the allele to move
    the model's score is testing for the wrong thing.
    """
    _score(PANEL_PEPTIDE, "HLA-A*02:01")
    vec_a = fake_manager.last_feature_vector[0].copy()
    _score(PANEL_PEPTIDE, "HLA-B*07:02")
    vec_b = fake_manager.last_feature_vector[0].copy()

    assert np.array_equal(vec_a, vec_b)


def test_out_of_panel_peptide_returns_422_not_silent_zeros(
    fake_manager: _FakeRFModel,
) -> None:
    """Explicit failure, not a silently zero-filled score."""
    with pytest.raises(HTTPException) as excinfo:
        _score(OUT_OF_PANEL_PEPTIDE, "HLA-A*02:01")
    assert excinfo.value.status_code == 422
    assert "binding panel" in excinfo.value.detail


def test_score_peptide_raises_503_when_model_not_loaded() -> None:
    """No fake_manager fixture here: the singleton is deliberately unloaded."""
    api_main._manager._loaded = False
    with pytest.raises(HTTPException) as excinfo:
        _score(PANEL_PEPTIDE, "HLA-A*02:01")
    assert excinfo.value.status_code == 503


def test_health_check_reports_loaded_state(fake_manager: _FakeRFModel) -> None:
    assert api_main.health_check()["model_loaded"] is True


def test_provenance_points_at_tracked_v5_corpus_not_missing_root_file() -> None:
    """Cruft item 2: /provenance used a root CSV that is not in the tree."""
    expected = api_main._PROJECT_ROOT / "data" / "immunogenicity_dataset_v5.csv"
    assert api_main._TRAINING_DATASET_PATH == expected
    historical_root = api_main._PROJECT_ROOT / "immunogenicity_dataset.csv"
    assert api_main._TRAINING_DATASET_PATH != historical_root
    # verify_tier_a_provenance.py git-shows the historical name at 69e0e5c
    # on purpose; this route must not be "fixed" onto that path.
    assert api_main._TRAINING_DATASET_PATH.name != "immunogenicity_dataset.csv"
