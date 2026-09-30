"""Readiness vs liveness for api/main.py, and the compose gate that consumes it.

docker-compose.yml starts the demo only once the api service reports healthy
(`depends_on: api: condition: service_healthy`), and that health is decided by
a `curl -f` against one api URL. curl -f fails only on an HTTP status of 400 or
above. /health is a LIVENESS probe: it answers 200 in degraded mode too, with
`model_loaded: false`, while /score answers 503. So a compose healthcheck
pointed at /health reported the API healthy while /score answered 503, and the
demo's start is gated on that signal. (The demo is not an API client: it loads
its own model and makes no HTTP call, so the defect is the signal, not a demo
request.) /ready is the readiness probe: same body as /health, but 503 until
the model is loaded.

Requests are driven through api_main.app itself, as a raw ASGI call, rather
than by calling the route functions: the property under test is the HTTP
status a real client sees at a real path, which a direct call cannot show (it
would not prove the route is registered, or that the status reaches the wire).
fastapi.testclient is avoided for the reason tests/test_api_main.py records -
it needs an httpx package CI's hash-pinned lockfiles do not carry.

No real model is loaded: the tests only flip the ModelManager singleton's
`_loaded` flag, which is all /health and /ready read.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest
import yaml

import api.main as api_main

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILE = REPO_ROOT / "docker-compose.yml"


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


@pytest.fixture
def model_state() -> Iterator[None]:
    """Restores the singleton's loaded flag and load error after each test."""
    saved = (api_main._manager._loaded, api_main._manager._load_error)
    yield
    api_main._manager._loaded, api_main._manager._load_error = saved


def _set_loaded(loaded: bool) -> None:
    api_main._manager._loaded = loaded
    api_main._manager._load_error = None


def test_ready_answers_503_while_no_model_is_loaded(model_state: None) -> None:
    _set_loaded(False)
    status, body = _asgi_get("/ready")
    assert status == 503, f"/ready answered {status} with no model loaded: {body!r}"
    assert body["model_loaded"] is False
    assert body["status"] == "degraded"
    assert body["reason"]


def test_ready_answers_200_once_the_model_is_loaded(model_state: None) -> None:
    _set_loaded(True)
    status, body = _asgi_get("/ready")
    assert status == 200, f"/ready answered {status} with the model loaded: {body!r}"
    assert body == {"status": "healthy", "model_loaded": True, "reason": None}


def test_health_stays_a_liveness_probe_in_degraded_mode(model_state: None) -> None:
    """/health's contract is unchanged: 200 whether or not a model is loaded."""
    _set_loaded(False)
    status, body = _asgi_get("/health")
    assert status == 200
    assert body == {
        "status": "degraded",
        "model_loaded": False,
        "reason": "Model has not been loaded.",
    }

    _set_loaded(True)
    status, body = _asgi_get("/health")
    assert status == 200
    assert body == {"status": "healthy", "model_loaded": True, "reason": None}


@pytest.mark.parametrize("loaded", [False, True])
def test_ready_carries_the_same_body_as_health(model_state: None, loaded: bool) -> None:
    _set_loaded(loaded)
    _, health_body = _asgi_get("/health")
    _, ready_body = _asgi_get("/ready")
    assert ready_body == health_body


def _compose() -> dict[str, Any]:
    with COMPOSE_FILE.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_compose_api_healthcheck_fails_while_no_model_is_loaded(model_state: None) -> None:
    """The defect: compose reported a non-scoring API healthy, gating the demo on it.

    The demo must still gate on the api's health, the check must still be a
    failing curl (-f/--fail), and the URL it probes must answer >= 400 while no
    model is loaded. The loaded case must answer 200, so a probe pointed at a
    path that does not exist (a 404, which also fails curl -f, and would keep
    the demo from ever starting) cannot satisfy this test.
    """
    services = _compose()["services"]
    assert services["demo"]["depends_on"]["api"]["condition"] == "service_healthy"

    check = services["api"]["healthcheck"]["test"]
    assert check[0] == "CMD" and check[1] == "curl", f"unexpected healthcheck form: {check!r}"
    assert "-f" in check or "--fail" in check, f"healthcheck curl lacks -f/--fail: {check!r}"
    urls = [arg for arg in check if arg.startswith("http")]
    assert len(urls) == 1, f"expected exactly one probe URL in {check!r}"
    path = urlsplit(urls[0]).path

    _set_loaded(False)
    status, body = _asgi_get(path)
    assert status >= 400, (
        f"compose healthcheck target {path} answered {status} with no model loaded "
        f"({body!r}), so curl -f passes and compose reports an API that cannot score healthy"
    )

    _set_loaded(True)
    status, body = _asgi_get(path)
    assert status == 200, (
        f"compose healthcheck target {path} answered {status} when ready: {body!r}"
    )
