"""app/demo.py loads the MHCflurry presentation predictor once, not per score.

_get_binding_score imported mhcflurry and called Class1PresentationPredictor.load()
on every call, so every scored peptide re-read the predictor's model files from
MHCflurry's downloaded models directory. It now goes through one holder, kept by
st.cache_resource, that loads on first use and serializes every predict().

Why the lock is tested: Streamlit runs each session's script on its own thread
and st.cache_resource hands every session the same object, while MHCflurry's
predict() mutates the predictor it runs on (mhcflurry 2.2.1 rewrites the
network's allele embedding on each call). The fake predictor below models that:
it stores the call's allele on the instance, yields, then scores from the stored
value, so two overlapping calls on one instance can return the wrong score.

streamlit is stubbed, as in tests/test_demo_feature_vector.py, but this stub's
cache_resource caches: one object per function, shared by every caller and
every re-execution of the script, and nothing stored when the function raises.
Those are the properties of streamlit 1.64.0's st.cache_resource the fix relies
on. Each test executes a fresh copy of app/demo.py, so no cache leaks between
tests or into the app.demo module other test files import.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

DEMO_SOURCE = Path(__file__).resolve().parents[1] / "app" / "demo.py"

PEPTIDE = "GILGFVFTL"
SCORES = {
    "HLA-A*01:01": 0.11,
    "HLA-A*02:01": 0.42,
    "HLA-B*07:02": 0.73,
    "HLA-B*44:02": 0.94,
}
MISSING_DATA = "Missing MHCflurry downloadable file: models_class1_presentation"


def _caching_streamlit() -> types.ModuleType:
    """A streamlit stand-in whose cache_resource memoizes like the real one.

    Entries are keyed by module and qualified name. Streamlit's key also hashes
    the function's source, which a rerun leaves unchanged, so in both a
    re-executed script finds the entry its previous run stored.
    """
    st = types.ModuleType("streamlit")
    entries: dict[str, tuple[dict[tuple[Any, ...], Any], threading.Lock]] = {}

    def cache_resource(*dargs: Any, **dkwargs: Any) -> Any:
        def _wrap(fn: Any) -> Any:
            key = f"{fn.__module__}.{fn.__qualname__}"
            cache, lock = entries.setdefault(key, ({}, threading.Lock()))

            def cached(*args: Any) -> Any:
                with lock:
                    if args not in cache:
                        cache[args] = fn(*args)  # a raise stores nothing
                    return cache[args]

            return cached

        if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
            return _wrap(dargs[0])
        return _wrap

    st.cache_resource = cache_resource  # type: ignore[attr-defined]
    st.set_page_config = lambda *a, **k: None  # type: ignore[attr-defined]
    return st


def _exec_demo(st: types.ModuleType, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Execute app/demo.py afresh against `st`, as one Streamlit (re)run does."""
    monkeypatch.setitem(sys.modules, "streamlit", st)
    spec = importlib.util.spec_from_file_location("_demo_under_test", DEMO_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Recorder:
    def __init__(self) -> None:
        self.guard = threading.Lock()
        self.loads = 0
        self.predicts = 0
        self.max_overlap = 0  # most predict() calls running at once on ONE instance


def _fake_mhcflurry(
    rec: _Recorder,
    *,
    load_seconds: float = 0.0,
    predict_seconds: float = 0.0,
    load_error: Exception | None = None,
    failing_predicts: int = 0,
) -> types.ModuleType:
    class Class1PresentationPredictor:
        def __init__(self) -> None:
            self.allele: str | None = None
            self.running = 0

        @classmethod
        def load(cls) -> Class1PresentationPredictor:
            with rec.guard:
                rec.loads += 1
            time.sleep(load_seconds)
            if load_error is not None:
                raise load_error
            return cls()

        def predict(
            self, peptides: list[str], alleles: list[str], include_affinity_percentile: bool
        ) -> pd.DataFrame:
            with rec.guard:
                rec.predicts += 1
                call = rec.predicts
                self.running += 1
                rec.max_overlap = max(rec.max_overlap, self.running)
            try:
                self.allele = alleles[0]  # per-instance state, like the allele embedding
                time.sleep(predict_seconds)
                if call <= failing_predicts:
                    raise RuntimeError(f"predict failure {call}")
                return pd.DataFrame({"presentation_score": [SCORES[self.allele]]})
            finally:
                with rec.guard:
                    self.running -= 1

    module = types.ModuleType("mhcflurry")
    module.Class1PresentationPredictor = Class1PresentationPredictor  # type: ignore[attr-defined]
    return module


@pytest.fixture
def st_stub() -> types.ModuleType:
    return _caching_streamlit()


@pytest.fixture
def demo(st_stub: types.ModuleType, monkeypatch: pytest.MonkeyPatch) -> Any:
    return _exec_demo(st_stub, monkeypatch)


def _assert_silent(capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture) -> None:
    """A failed binding score is reported only by the None return, as before."""
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == "", captured
    assert caplog.records == [], [r.getMessage() for r in caplog.records]


def test_repeated_scores_load_the_predictor_once(
    demo: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    rec = _Recorder()
    monkeypatch.setitem(sys.modules, "mhcflurry", _fake_mhcflurry(rec))

    alleles = ["HLA-A*02:01", "HLA-B*07:02", "HLA-A*02:01"]
    scores = [demo._get_binding_score(PEPTIDE, allele) for allele in alleles]

    assert scores == [SCORES[a] for a in alleles]
    assert rec.loads == 1, f"Class1PresentationPredictor.load() ran {rec.loads} times for 3 scores"
    assert rec.predicts == 3


def test_the_loaded_predictor_survives_a_script_rerun(
    st_stub: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Streamlit re-executes the script in a fresh module on every rerun, so a
    module-level holder would be rebuilt on every rerun. The holder kept by
    st.cache_resource is not."""
    rec = _Recorder()
    monkeypatch.setitem(sys.modules, "mhcflurry", _fake_mhcflurry(rec))

    first_run = _exec_demo(st_stub, monkeypatch)
    assert first_run._get_binding_score(PEPTIDE, "HLA-A*02:01") == SCORES["HLA-A*02:01"]
    rerun = _exec_demo(st_stub, monkeypatch)
    assert rerun is not first_run
    assert rerun._get_binding_score(PEPTIDE, "HLA-B*44:02") == SCORES["HLA-B*44:02"]

    assert rec.loads == 1, f"load() ran {rec.loads} times across one rerun"


def test_concurrent_sessions_load_once_and_never_overlap_predict(
    demo: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Eight session threads, released together, five scores each, against a slow
    load: one load, one predict() at a time, and every score its own allele's."""
    rec = _Recorder()
    monkeypatch.setitem(
        sys.modules,
        "mhcflurry",
        _fake_mhcflurry(rec, load_seconds=0.05, predict_seconds=0.005),
    )
    threads_n, calls_n = 8, 5
    alleles = list(SCORES)
    start = threading.Barrier(threads_n)
    results: list[tuple[str, float | None]] = []
    results_lock = threading.Lock()

    def session(i: int) -> None:
        allele = alleles[i % len(alleles)]
        start.wait(timeout=10)
        for _ in range(calls_n):
            score = demo._get_binding_score(PEPTIDE, allele)
            with results_lock:
                results.append((allele, score))

    # Daemon threads and one shared 60 s deadline for all the joins, so a deadlock
    # regression fails the assertion below within about 60 s instead of hanging
    # pytest at exit.
    threads = [threading.Thread(target=session, args=(i,), daemon=True) for i in range(threads_n)]
    for t in threads:
        t.start()
    deadline = time.monotonic() + 60
    for t in threads:
        t.join(max(0.0, deadline - time.monotonic()))
    assert not any(t.is_alive() for t in threads), "a session thread hung"

    assert rec.loads == 1, f"load() ran {rec.loads} times under {threads_n} concurrent sessions"
    assert rec.max_overlap == 1, f"{rec.max_overlap} predict() calls overlapped on one predictor"
    wrong = [(a, s) for a, s in results if s != SCORES[a]]
    assert wrong == [], f"{len(wrong)} of {len(results)} scores wrong: {wrong[:5]}"
    assert len(results) == rec.predicts == threads_n * calls_n


def test_missing_model_data_is_loaded_once_and_still_returns_none(
    demo: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """mhcflurry imports but load() raises, as MHCflurry 2.2.1's downloads.get_path
    does when the model data was never fetched. Each call still returns None, the
    disk is not tried again, and the failure keeps its original message."""
    caplog.set_level(logging.DEBUG)
    rec = _Recorder()
    monkeypatch.setitem(
        sys.modules, "mhcflurry", _fake_mhcflurry(rec, load_error=RuntimeError(MISSING_DATA))
    )

    assert demo._get_binding_score(PEPTIDE, "HLA-A*02:01") is None
    assert demo._get_binding_score(PEPTIDE, "HLA-A*02:01") is None
    assert rec.loads == 1, f"a failed load() was retried: {rec.loads} attempts"
    _assert_silent(capsys, caplog)

    with pytest.raises(RuntimeError, match=MISSING_DATA):
        demo._presentation_predictor().score(PEPTIDE, "HLA-A*02:01")
    assert rec.loads == 1


def test_a_failed_load_with_an_empty_message_is_still_cached(
    demo: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """str(RuntimeError()) is "", so the cached failure must be tested against None,
    not for truthiness: an empty message is still a failed load."""
    caplog.set_level(logging.DEBUG)
    rec = _Recorder()
    monkeypatch.setitem(sys.modules, "mhcflurry", _fake_mhcflurry(rec, load_error=RuntimeError()))

    assert demo._get_binding_score(PEPTIDE, "HLA-A*02:01") is None
    assert demo._get_binding_score(PEPTIDE, "HLA-A*02:01") is None
    assert rec.loads == 1, f"a failed load() was retried: {rec.loads} attempts"
    _assert_silent(capsys, caplog)

    with pytest.raises(demo._PresentationPredictorUnavailable):
        demo._presentation_predictor().score(PEPTIDE, "HLA-A*02:01")
    assert rec.loads == 1


def test_missing_mhcflurry_is_not_reimported_and_still_returns_none(
    demo: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The import failing is cached like a load failure: installing mhcflurry
    afterwards changes nothing until the cached holder is discarded."""
    caplog.set_level(logging.DEBUG)
    monkeypatch.setitem(sys.modules, "mhcflurry", None)  # the import raises ImportError
    assert demo._get_binding_score(PEPTIDE, "HLA-A*02:01") is None

    rec = _Recorder()
    monkeypatch.setitem(sys.modules, "mhcflurry", _fake_mhcflurry(rec))
    assert demo._get_binding_score(PEPTIDE, "HLA-A*02:01") is None
    assert rec.loads == 0, "the failed import was retried"
    _assert_silent(capsys, caplog)

    with pytest.raises(RuntimeError, match="import of mhcflurry halted"):
        demo._presentation_predictor().score(PEPTIDE, "HLA-A*02:01")


def test_a_predict_error_fails_that_call_only(
    demo: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    rec = _Recorder()
    monkeypatch.setitem(sys.modules, "mhcflurry", _fake_mhcflurry(rec, failing_predicts=1))

    assert demo._get_binding_score(PEPTIDE, "HLA-A*02:01") is None
    _assert_silent(capsys, caplog)
    assert demo._get_binding_score(PEPTIDE, "HLA-A*02:01") == SCORES["HLA-A*02:01"]
    assert rec.loads == 1, f"a predict() error forced a reload: {rec.loads} loads"
    assert rec.predicts == 2
