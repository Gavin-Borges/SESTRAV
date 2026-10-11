"""/score must not reload the MHCflurry presentation predictor on every request.

api/main.py's module docstring promised "Models loaded once at startup via
ModelManager singleton (no per-request I/O)", while _score_peptide called
Class1PresentationPredictor.load() on every /score request that passed the
readiness check and the binding-panel lookup. load() deserializes the predictor
from MHCflurry's downloaded models directory, so every such request read model
files from disk before answering.

These tests replace the `mhcflurry` module with a stand-in through sys.modules,
so they never import the real package or read its model data, and they count
load() calls directly. Like tests/test_api_main.py they call the route handlers
directly rather than through fastapi.testclient (see that module's docstring for
why), and they seed the ModelManager singleton with a fake RF model and a
one-row binding panel so nothing loads the real RF artifact.

The failure-path test pins what an unavailable MHCflurry (not installed, or
installed without its model data, as in Dockerfile.api) returned before this
change: the request still succeeds, binding_score is None, and each such
request logs the same INFO line. Only a failed import or load is cached; an
error raised by predict() costs its own request the binding_score and nothing
more.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
import types
from collections.abc import Callable, Iterator

import numpy as np
import pandas as pd
import pytest

import api.main as api_main
from src.features import BINDING_ALLELE_COLUMNS

PANEL_PEPTIDE = "GILGFVFTL"
PANEL_ROW = {col: 0.5 + 0.01 * i for i, col in enumerate(BINDING_ALLELE_COLUMNS)}
PRESENTATION_SCORE = 0.8977
MISSING_DATA_MESSAGE = "Missing MHCflurry downloadable file: models_class1_presentation"
# Matches _ALLELE_PATTERN, so it passes request validation, but mhcflurry 2.2.1's
# predict() raises ValueError("No sequences for allele(s): ...") for it.
UNSUPPORTED_ALLELE = "HLA-A*99:99"
UNAVAILABLE_PREFIX = "MHCflurry unavailable for the informational binding_score field: "


class _FakeRFModel:
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.array([[0.2, 0.8]])


class _FakePresentationPredictor:
    """Answers predict() like Class1PresentationPredictor and records concurrency."""

    def __init__(self, predict_seconds: float, fail_alleles: frozenset[str]) -> None:
        self.predict_seconds = predict_seconds
        self.fail_alleles = fail_alleles
        self.alleles_seen: list[str] = []
        self.active = 0
        self.max_active = 0
        self._counter_lock = threading.Lock()

    def predict(
        self, peptides: list[str], alleles: list[str], include_affinity_percentile: bool
    ) -> pd.DataFrame:
        with self._counter_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.predict_seconds:
                time.sleep(self.predict_seconds)
            self.alleles_seen.extend(alleles)
            unsupported = [a for a in alleles if a in self.fail_alleles]
            if unsupported:
                raise ValueError(f"No sequences for allele(s): {', '.join(unsupported)}.")
            return pd.DataFrame({"presentation_score": [PRESENTATION_SCORE] * len(peptides)})
        finally:
            with self._counter_lock:
                self.active -= 1


class _FakeMhcflurry:
    """Builds a stand-in `mhcflurry` module whose Class1PresentationPredictor.load() counts."""

    def __init__(
        self,
        load_error: Exception | None = None,
        load_seconds: float = 0.0,
        predict_seconds: float = 0.0,
        fail_alleles: frozenset[str] = frozenset(),
    ) -> None:
        self.load_calls = 0
        self.predictor = _FakePresentationPredictor(predict_seconds, fail_alleles)
        owner = self

        class Class1PresentationPredictor:
            @classmethod
            def load(cls) -> _FakePresentationPredictor:
                owner.load_calls += 1
                if load_seconds:
                    time.sleep(load_seconds)
                if load_error is not None:
                    raise load_error
                return owner.predictor

        self.module = types.ModuleType("mhcflurry")
        self.module.Class1PresentationPredictor = Class1PresentationPredictor  # type: ignore[attr-defined]


@pytest.fixture
def seeded_manager(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Seeds the singleton with fakes and an empty presentation-predictor cache.

    monkeypatch restores whatever another test left cached, so neither direction
    leaks: a predictor or load failure cached by tests/test_api_main.py's real
    MHCflurry calls never answers here, and the stand-ins here never answer there.
    """
    manager = api_main._manager
    monkeypatch.setattr(manager, "_loaded", True)
    monkeypatch.setattr(manager, "rf_model", _FakeRFModel(), raising=False)
    monkeypatch.setattr(manager, "binding_matrix", {PANEL_PEPTIDE: dict(PANEL_ROW)}, raising=False)
    monkeypatch.setattr(manager, "_presentation_predictor", None, raising=False)
    monkeypatch.setattr(manager, "_presentation_load_error", None, raising=False)
    yield


def _install(monkeypatch: pytest.MonkeyPatch, fake: _FakeMhcflurry) -> None:
    monkeypatch.setitem(sys.modules, "mhcflurry", fake.module)


def _score(allele: str = "HLA-A*02:01") -> api_main.ScoreResponse:
    return api_main.score_peptide(api_main.PeptideInput(sequence=PANEL_PEPTIDE, allele=allele))


def test_presentation_predictor_is_loaded_once_across_requests(
    seeded_manager: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeMhcflurry()
    _install(monkeypatch, fake)

    first = _score("HLA-A*02:01")
    second = _score("HLA-B*07:02")

    assert fake.load_calls == 1, f"load() ran {fake.load_calls} times for 2 requests"
    assert first.binding_score == round(PRESENTATION_SCORE, 4)
    assert second.binding_score == round(PRESENTATION_SCORE, 4)
    # The reused predictor still answers for each request's own allele.
    assert fake.predictor.alleles_seen == ["HLA-A*02:01", "HLA-B*07:02"]


def _not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    # A None entry makes `from mhcflurry import ...` raise ImportError, as an absent package does.
    monkeypatch.setitem(sys.modules, "mhcflurry", None)


def _no_model_data(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _FakeMhcflurry(load_error=RuntimeError(MISSING_DATA_MESSAGE)))


@pytest.mark.parametrize("make_unavailable", [_not_installed, _no_model_data])
def test_unavailable_mhcflurry_still_answers_with_a_null_binding_score(
    seeded_manager: None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    make_unavailable: Callable[[pytest.MonkeyPatch], None],
) -> None:
    """The failure behaviour from before the fix, request by request, is unchanged."""
    make_unavailable(monkeypatch)
    caplog.set_level(logging.INFO, logger=api_main.logger.name)

    responses = [_score(), _score()]

    for resp in responses:
        assert resp.binding_score is None
        assert resp.immunogenicity_score == pytest.approx(0.8)
        assert resp.rank == "HIGH"
    messages = [
        r.getMessage()
        for r in caplog.records
        if r.name == api_main.logger.name and r.getMessage().startswith(UNAVAILABLE_PREFIX)
    ]
    assert len(messages) == 2, f"expected one INFO line per request, got {messages}"
    assert messages[0] == messages[1]
    assert all(
        r.levelno == logging.INFO
        for r in caplog.records
        if r.getMessage().startswith(UNAVAILABLE_PREFIX)
    )
    if make_unavailable is _no_model_data:
        assert messages[0] == UNAVAILABLE_PREFIX + MISSING_DATA_MESSAGE


def test_failed_load_is_not_retried_on_every_request(
    seeded_manager: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dockerfile.api never fetches the weights, so this is the shipped container's path."""
    fake = _FakeMhcflurry(load_error=RuntimeError(MISSING_DATA_MESSAGE))
    _install(monkeypatch, fake)

    for _ in range(3):
        assert _score().binding_score is None

    assert fake.load_calls == 1, f"a failed load() was retried: {fake.load_calls} calls"


def test_a_failed_load_with_an_empty_message_is_still_cached(
    seeded_manager: None, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """str(RuntimeError()) is "", so the cached failure must be tested against None, not
    for truthiness: an empty message is still a failed load."""
    fake = _FakeMhcflurry(load_error=RuntimeError())
    _install(monkeypatch, fake)
    caplog.set_level(logging.INFO, logger=api_main.logger.name)

    for _ in range(3):
        assert _score().binding_score is None

    assert fake.load_calls == 1, f"a failed load() was retried: {fake.load_calls} calls"
    messages = [
        r.getMessage()
        for r in caplog.records
        if r.name == api_main.logger.name and r.getMessage().startswith(UNAVAILABLE_PREFIX)
    ]
    assert messages == [UNAVAILABLE_PREFIX] * 3, messages
    with pytest.raises(api_main.PresentationPredictorUnavailable):
        api_main._manager.presentation_score(PANEL_PEPTIDE, "HLA-A*02:01")
    assert fake.load_calls == 1


def test_a_predict_error_is_not_cached_as_a_load_failure(
    seeded_manager: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unsupported allele must cost only its own request the binding_score."""
    fake = _FakeMhcflurry(fail_alleles=frozenset({UNSUPPORTED_ALLELE}))
    _install(monkeypatch, fake)

    # The failing request comes first, so its predict() raises in the same call
    # that loaded the predictor.
    scores = [
        _score(allele).binding_score
        for allele in (UNSUPPORTED_ALLELE, "HLA-A*02:01", UNSUPPORTED_ALLELE, "HLA-B*07:02")
    ]

    expected = round(PRESENTATION_SCORE, 4)
    assert scores == [None, expected, None, expected]
    assert fake.load_calls == 1, f"load() ran {fake.load_calls} times for 4 requests"


def test_concurrent_requests_load_once_and_predict_one_at_a_time(
    seeded_manager: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sync routes run on a thread pool. predict() mutates the shared predictor (it
    builds networks lazily and rewrites their allele embeddings), so calls on the
    one shared instance must not overlap."""
    fake = _FakeMhcflurry(load_seconds=0.05, predict_seconds=0.02)
    _install(monkeypatch, fake)
    n_threads = 8
    barrier = threading.Barrier(n_threads)
    results: list[tuple[float, float | None]] = []
    errors: list[BaseException] = []

    def _request() -> None:
        try:
            barrier.wait(timeout=10)
            results.append(api_main._score_peptide(PANEL_PEPTIDE, "HLA-A*02:01"))
        except BaseException as exc:  # pragma: no cover - surfaced by the asserts below
            errors.append(exc)

    # Daemon threads, so a deadlock regression cannot hang the worker at interpreter
    # exit: the joins share one 30 s deadline, so they return within about 30 s in
    # total, and the results-count assertion fails. (A deadlock that leaves the
    # singleton's lock held could still block later tests in the same worker.)
    threads = [threading.Thread(target=_request, daemon=True) for _ in range(n_threads)]
    for t in threads:
        t.start()
    deadline = time.monotonic() + 30
    for t in threads:
        t.join(max(0.0, deadline - time.monotonic()))

    assert not errors, errors
    assert len(results) == n_threads
    assert all(bind == PRESENTATION_SCORE for _, bind in results)
    assert fake.load_calls == 1, (
        f"{n_threads} concurrent requests ran load() {fake.load_calls} times"
    )
    assert fake.predictor.max_active == 1, (
        f"{fake.predictor.max_active} predict() calls overlapped on the shared predictor"
    )
