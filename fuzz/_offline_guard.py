"""Offline guard for the atheris harnesses in ``fuzz/``.

Every harness calls :func:`install_and_verify` as its FIRST action, before atheris
is imported and therefore before ``atheris.instrument_imports()`` runs. Three of the
targets live in modules that talk to third-party web services
(``src/external_predictors.py`` posts to DTU and UCM;
``src/verify/iedb_multi_virus_extractor.py`` queries IEDB and VDJdb), so a harness
that reached one of those code paths by accident would send fuzzed payloads to a
third-party server. The guard makes that impossible rather than unlikely:

* ``socket.socket``, ``socket.create_connection``, ``socket.getaddrinfo`` and
  ``ssl.SSLContext.wrap_socket`` are replaced by callables that raise
  :class:`OfflineGuardViolation`;
* ``no_proxy`` is set to ``*`` and ``http_proxy`` and ``https_proxy`` (both cases)
  point at ``http://127.0.0.1:1``, a port nothing listens on;
* a POSITIVE CONTROL then attempts one outbound connection through each patched
  entry point (and through ``requests`` when it is installed) and aborts the
  process with exit status 97 unless every attempt raised
  :class:`OfflineGuardViolation` specifically. Any other exception type does not
  count as a pass, because an unpatched ``create_connection`` to an unroutable
  address also raises (a timeout), which would make a weaker control vacuous.

Standard library only, so it can be imported before atheris.
"""

from __future__ import annotations

import os
import socket
import ssl
import sys
from typing import Callable, NoReturn

PROXY_SINK = "http://127.0.0.1:1"
# TEST-NET-1 (RFC 5737): documentation-only, never routed on the public internet.
CONTROL_ADDRESS = ("192.0.2.1", 80)
# https, not http: the guard refuses the socket before any byte is sent, so the scheme
# changes nothing the control proves, and an http literal in a requests call is a
# semgrep p/python finding (request-with-http) that CI would publish as an alert.
CONTROL_URL = "https://192.0.2.1/"
ABORT_STATUS = 97


class OfflineGuardViolation(RuntimeError):
    """Raised by every network entry point the guard replaces."""


def _refuse(*_args: object, **_kwargs: object) -> NoReturn:
    raise OfflineGuardViolation("network access is disabled inside fuzz harnesses")


class _RefusingSocket(socket.socket):
    """A ``socket.socket`` subclass that can never be constructed.

    A subclass rather than a bare function so that ``isinstance(x, socket.socket)``
    in already-imported code keeps working.
    """

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        _refuse()


_installed = False


def install() -> None:
    """Patch the network entry points and point the http(s) proxy variables at a dead port."""
    global _installed
    os.environ["no_proxy"] = "*"
    os.environ["NO_PROXY"] = "*"
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ[key] = PROXY_SINK
    socket.socket = _RefusingSocket  # type: ignore[misc]
    socket.create_connection = _refuse  # type: ignore[assignment]
    socket.getaddrinfo = _refuse  # type: ignore[assignment]
    ssl.SSLContext.wrap_socket = _refuse  # type: ignore[assignment,method-assign]
    _installed = True


def _control_tls_context() -> ssl.SSLContext:
    """The context the wrap_socket control uses, with its TLS 1.2 floor stated explicitly.

    ``create_default_context`` already sets that floor (its ``minimum_version`` is TLSv1_2,
    measured on Python 3.11 and 3.14), but CodeQL's insecure-protocol query does not model
    the default and reports TLSv1 and TLSv1_1 as allowed. Setting ``minimum_version`` states
    the same policy where the analyzer can see it.
    """
    context = ssl.create_default_context()
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    return context


def _control_attempts() -> list[tuple[str, Callable[[], object]]]:
    attempts: list[tuple[str, Callable[[], object]]] = [
        (
            "socket.create_connection",
            lambda: socket.create_connection(CONTROL_ADDRESS, timeout=1),
        ),
        ("socket.socket", lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM)),
        ("socket.getaddrinfo", lambda: socket.getaddrinfo("example.org", 443)),
        (
            "ssl.SSLContext.wrap_socket",
            lambda: _control_tls_context().wrap_socket(
                None,  # type: ignore[arg-type]
                server_hostname="example.org",
            ),
        ),
    ]
    try:
        import requests
    except ImportError:
        return attempts
    attempts.append(("requests.get", lambda: requests.get(CONTROL_URL, timeout=1)))
    return attempts


def positive_control() -> list[str]:
    """Attempt one outbound connection per entry point; return one report line each.

    Raises SystemExit(ABORT_STATUS) unless every attempt raised OfflineGuardViolation.
    """
    lines = []
    failures = 0
    for label, attempt in _control_attempts():
        try:
            attempt()
        except OfflineGuardViolation:
            lines.append(f"OFFLINE_GUARD control {label}: refused (OfflineGuardViolation)")
            continue
        except Exception as exc:  # any other outcome means the patch did not take
            failures += 1
            lines.append(f"OFFLINE_GUARD control {label}: FAIL raised {type(exc).__name__}")
            continue
        failures += 1
        lines.append(f"OFFLINE_GUARD control {label}: FAIL did not raise")
    verdict = "PASS" if failures == 0 else "FAIL"
    lines.append(
        f"OFFLINE_GUARD control verdict: {verdict} "
        f"({len(lines) - failures}/{len(lines)} entry points refused)"
    )
    if failures:
        sys.stderr.write("\n".join(lines) + "\n")
        raise SystemExit(ABORT_STATUS)
    return lines


def install_and_verify(quiet: bool = False) -> list[str]:
    """Install the guard, run the positive control, and report it on stderr."""
    install()
    lines = [
        "OFFLINE_GUARD installed: no_proxy={} http_proxy={} https_proxy={}".format(
            os.environ["no_proxy"], os.environ["http_proxy"], os.environ["https_proxy"]
        )
    ]
    lines.extend(positive_control())
    if not quiet:
        sys.stderr.write("\n".join(lines) + "\n")
    return lines
