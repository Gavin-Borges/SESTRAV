"""/health must report a cached MHCflurry presentation-predictor load failure.

api/main.py loads the MHCflurry presentation predictor lazily, on the first /score
that reaches it, and caches a failed import or load: every later call re-raises it as
PresentationPredictorUnavailable, which _score_peptide catches, so each such /score
still answers 200, with binding_score null, until the server restarts. /health keyed
only on the RF model, so it said healthy throughout, and only an INFO log line on each
request said why binding_score was null.

/health now reports status degraded, with model_loaded true and a reason naming the
exception type, once that load has failed. Before any load has been attempted the
predictor's state is unknown, not failed, so /health stays healthy then and does not
load the predictor itself. An error raised by predict() is not cached and does not
degrade /health either. /ready keeps answering 200 in this state: whether it should
answer 503 changes the readiness contract docker-compose.yml gates the demo on, which
is an owner decision, so the last test pins the current behaviour.

As in tests/test_api_presentation_predictor.py, the `mhcflurry` module is replaced by a
stand-in through sys.modules and the ModelManager singleton is seeded with a fake RF
model and a one-row binding panel, so nothing reads model files. GETs go through
api_main.app as raw ASGI calls, as in tests/test_api_readiness.py, so the HTTP status a
client would see is what is asserted.
"""

from __future__ import annotations

import asyncio
import json
import sys
import types
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
import pytest

import api.main as api_main
from src.features import BINDING_ALLELE_COLUMNS

PANEL_PEPTIDE = "GILGFVFTL"
PANEL_ROW = {col: 0.5 + 0.01 * i for i, col in enumerate(BINDING_ALLELE_COLUMNS)}
PRESENTATION_SCORE = 0.8977
# mhcflurry 2.2.1's message for missing model data quotes the downloads path, so the
# reason /health returns must not carry the exception's message.
DOWNLOADS_PATH = "/data/mhcflurry/4/2.2.0/models_class1_presentation"
MISSING_DATA_MESSAGE = f"Missing MHCflurry downloadable file: {DOWNLOADS_PATH}."
# Matches _ALLELE_PATTERN, so it passes request validation; the stand-in's predict()
# raises for it, as mhcflurry 2.2.1's does for an allele it has no sequence for.
UNSUPPORTED_ALLELE = "HLA-A*99:99"
HEALTHY = {"status": "healthy", "model_loaded": True, "reason": None}


class _FakeRFModel:
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.array([[0.2, 0.8]])


class _FakePresentationPredictor:
    def predict(
        self, peptides: list[str], alleles: list[str], include_affinity_percentile: bool
    ) -> pd.DataFrame:
        if UNSUPPORTED_ALLELE in alleles:
            raise ValueError(f"No sequences for allele(s): {UNSUPPORTED_ALLELE}.")
        return pd.DataFrame({"presentation_score": [PRESENTATION_SCORE] * len(peptides)})


def _install_mhcflurry(
    monkeypatch: pytest.MonkeyPatch, load_error: Exception | None = None
) -> list[int]:
    """Installs a stand-in `mhcflurry`; returns a list that grows by one per load()."""
    load_calls: list[int] = []

    class Class1PresentationPredictor:
        @classmethod
        def load(cls) -> _FakePresentationPredictor:
            load_calls.append(1)
            if load_error is not None:
                raise load_error
            return _FakePresentationPredictor()

    module = types.ModuleType("mhcflurry")
    module.Class1PresentationPredictor = Class1PresentationPredictor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mhcflurry", module)
    return load_calls


@pytest.fixture
def seeded_manager(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A loaded RF model and an untried presentation predictor; monkeypatch restores both.

    The cache is cleared because tests/test_api_main.py scores through the real MHCflurry,
    which caches a load failure wherever its model data is absent.
    """
    manager = api_main._manager
    monkeypatch.setattr(manager, "_loaded", True)
    monkeypatch.setattr(manager, "_load_error", None)
    monkeypatch.setattr(manager, "rf_model", _FakeRFModel(), raising=False)
    monkeypatch.setattr(manager, "binding_matrix", {PANEL_PEPTIDE: dict(PANEL_ROW)}, raising=False)
    monkeypatch.setattr(manager, "_presentation_predictor", None, raising=False)
    monkeypatch.setattr(manager, "_presentation_load_error", None, raising=False)
    monkeypatch.setattr(manager, "_presentation_load_error_type", None, raising=False)
    yield


def _score(allele: str = "HLA-A*02:01") -> api_main.ScoreResponse:
    return api_main.score_peptide(api_main.PeptideInput(sequence=PANEL_PEPTIDE, allele=allele))


def _asgi_get(path: str) -> tuple[int, Any]:
    """Sends one GET through the real ASGI app; returns (status, parsed JSON body)."""
    messages: list[dict[str, Any]] = []

    async def _receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def _send(message: dict[str, Any]) -> None:
        messages.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"localhost:8000")],
        "client": ("127.0.0.1", 50000),
        "server": ("localhost", 8000),
    }
    asyncio.run(api_main.app(scope, _receive, _send))
    starts = [m for m in messages if m["type"] == "http.response.start"]
    assert len(starts) == 1, f"expected one response start for GET {path}, got {messages!r}"
    body = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
    return starts[0]["status"], json.loads(body)


@pytest.mark.parametrize(
    ("not_installed", "expected_type"),
    [(False, "RuntimeError"), (True, "ModuleNotFoundError")],
    ids=["model-data-missing", "not-installed"],
)
def test_health_reports_a_cached_presentation_load_failure(
    seeded_manager: None,
    monkeypatch: pytest.MonkeyPatch,
    not_installed: bool,
    expected_type: str,
) -> None:
    load_calls = _install_mhcflurry(monkeypatch, RuntimeError(MISSING_DATA_MESSAGE))
    if not_installed:
        # A None entry makes `from mhcflurry import ...` raise ModuleNotFoundError.
        monkeypatch.setitem(sys.modules, "mhcflurry", None)

    for _ in range(2):
        assert _score().binding_score is None
    assert len(load_calls) == (0 if not_installed else 1)

    status, body = _asgi_get("/health")
    assert status == 200, f"/health is a liveness probe and must answer 200: {body!r}"
    assert body["status"] == "degraded", f"/health hid a cached MHCflurry failure: {body!r}"
    assert body["model_loaded"] is True
    assert expected_type in body["reason"]
    assert "MHCflurry" in body["reason"]
    assert DOWNLOADS_PATH not in body["reason"]


def test_an_empty_failure_message_is_still_reported(
    seeded_manager: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cached failure is recognised by being set, not by its text."""
    _install_mhcflurry(monkeypatch, RuntimeError(""))
    assert _score().binding_score is None

    _, body = _asgi_get("/health")
    assert body["status"] == "degraded", body
    assert "RuntimeError" in body["reason"]


def test_health_is_unchanged_before_any_presentation_load_is_attempted(
    seeded_manager: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Untried is unknown, not failed: /health neither reports a failure nor loads."""
    load_calls = _install_mhcflurry(monkeypatch, RuntimeError(MISSING_DATA_MESSAGE))

    status, body = _asgi_get("/health")
    assert (status, body) == (200, HEALTHY)
    assert load_calls == [], "/health must not load the presentation predictor"


def test_health_stays_healthy_after_a_successful_load(
    seeded_manager: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_mhcflurry(monkeypatch)
    assert _score().binding_score == pytest.approx(PRESENTATION_SCORE)

    assert _asgi_get("/health") == (200, HEALTHY)


def test_a_predict_error_does_not_degrade_health(
    seeded_manager: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """predict() errors are not cached: they cost one request its binding_score only."""
    _install_mhcflurry(monkeypatch)
    assert _score(UNSUPPORTED_ALLELE).binding_score is None
    assert _score().binding_score == pytest.approx(PRESENTATION_SCORE)

    assert _asgi_get("/health") == (200, HEALTHY)


def test_ready_keeps_answering_200_after_a_presentation_load_failure(
    seeded_manager: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pins the current readiness contract; changing it is an owner decision.

    /ready answers 503 only while the RF model is not loaded, so with a cached MHCflurry
    failure it answers 200 and carries /health's degraded body.
    """
    _install_mhcflurry(monkeypatch, RuntimeError(MISSING_DATA_MESSAGE))
    assert _score().binding_score is None

    ready_status, ready_body = _asgi_get("/ready")
    _, health_body = _asgi_get("/health")
    assert ready_status == 200
    assert ready_body == health_body
    assert ready_body["status"] == "degraded"
