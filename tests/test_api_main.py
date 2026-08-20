"""First test coverage for api/main.py (D31, docs/claims_register.md).

Before this file, api/main.py had zero test coverage - no unit test, no
TestClient test, no CI smoke request - which is how an inert scorer (the
served bind_* feature block silently stayed all-zero regardless of MHCflurry
availability or the requested allele) survived from 2026-06-13 to 2026-08-19
across 13 commits that edited the file (`git log --since=2026-06-13
--until=2026-08-19 -- api/main.py`).

Tests here avoid loading the real 128MB rf_31feature_integrated.joblib
(gitignored, not present in a clean CI checkout): ModelManager.load() returns
immediately when `_loaded` is already True, so the fixtures below pre-seed
`_manager` with a fake model and a small binding matrix before that check
ever runs.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from fastapi.testclient import TestClient

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
    """Seeds the ModelManager singleton so lifespan.load() is a no-op."""
    fake_model = _FakeRFModel()
    api_main._manager._loaded = True
    api_main._manager.rf_model = fake_model
    api_main._manager.binding_matrix = {PANEL_PEPTIDE: dict(PANEL_ROW)}
    yield fake_model
    api_main._manager._loaded = False


@pytest.fixture
def client(fake_manager: _FakeRFModel) -> Iterator[TestClient]:
    with TestClient(api_main.app) as c:
        yield c


def test_panel_present_peptide_gets_a_nonzero_binding_block(
    client: TestClient, fake_manager: _FakeRFModel
) -> None:
    """The regression this file exists for: bind_* columns must not be all-zero
    for a peptide that has a real panel row, regardless of MHCflurry."""
    resp = client.post(
        "/score", json={"sequence": PANEL_PEPTIDE, "allele": "HLA-A*02:01"}
    )
    assert resp.status_code == 200, resp.text

    assert fake_manager.last_feature_vector is not None
    vec = fake_manager.last_feature_vector[0]
    bind_indices = [FEATURE_COLUMNS_31.index(c) for c in BINDING_ALLELE_COLUMNS]
    bind_values = [vec[i] for i in bind_indices]
    assert any(v != 0.0 for v in bind_values), "binding block was all-zero for a panel peptide"
    assert bind_values == [PANEL_ROW[c] for c in BINDING_ALLELE_COLUMNS]


def test_binding_block_is_identical_across_different_alleles(
    client: TestClient, fake_manager: _FakeRFModel
) -> None:
    """D31: the panel is fixed and keyed by peptide, not by the caller's allele.

    This pins the (documented, not-yet-fixable) behaviour rather than hiding
    it: two callers requesting different alleles for the same peptide must
    receive the identical binding block, so a test that ever expects the
    allele to change the model's score is testing for the wrong thing.
    """
    resp_a = client.post("/score", json={"sequence": PANEL_PEPTIDE, "allele": "HLA-A*02:01"})
    vec_a = fake_manager.last_feature_vector[0].copy()
    resp_b = client.post("/score", json={"sequence": PANEL_PEPTIDE, "allele": "HLA-B*07:02"})
    vec_b = fake_manager.last_feature_vector[0].copy()

    assert resp_a.status_code == 200
    assert resp_b.status_code == 200
    assert np.array_equal(vec_a, vec_b)


def test_out_of_panel_peptide_returns_422_not_silent_zeros(client: TestClient) -> None:
    """Explicit failure, not a silently zero-filled score (T4 recommendation)."""
    resp = client.post(
        "/score", json={"sequence": OUT_OF_PANEL_PEPTIDE, "allele": "HLA-A*02:01"}
    )
    assert resp.status_code == 422
    assert "binding panel" in resp.json()["detail"]


def test_health_check_does_not_require_the_real_model(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is True
