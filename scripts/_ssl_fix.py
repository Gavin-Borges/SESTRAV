"""Monkey-patch ssl.create_default_context to bypass broken Windows cert store.

Python 3.13 on Windows calls ssl._load_windows_store_certs() which fails with
``ssl.SSLError: [ASN1: NOT_ENOUGH_DATA]`` when a malformed certificate exists
in the Windows Trusted Root CA store.

Import this module *before* any network calls to use certifi's CA bundle instead:

    import _ssl_fix  # noqa: F401  - patches ssl.create_default_context
    import urllib.request
    ...

This is a no-op on non-Windows platforms or if the Windows store works fine.
"""

from __future__ import annotations

import ssl
import sys


def _patched_create_default_context(
    purpose: ssl.Purpose = ssl.Purpose.SERVER_AUTH,
    *,
    cafile: str | None = None,
    capath: str | None = None,
    cadata: str | None = None,
) -> ssl.SSLContext:
    """Create an SSLContext using certifi, skipping the Windows cert store."""
    import certifi

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.verify_mode = ssl.CERT_REQUIRED
    ctx.check_hostname = True

    if cafile or capath or cadata:
        ctx.load_verify_locations(cafile=cafile, capath=capath, cadata=cadata)
    else:
        ctx.load_verify_locations(cafile=certifi.where())

    return ctx


def _apply() -> None:
    """Apply the patch only on Windows where the store is broken."""
    if sys.platform != "win32":
        return
    # Quick probe: does the default context creation work?
    try:
        ssl.create_default_context()
    except ssl.SSLError:
        # Broken - apply patch
        ssl.create_default_context = _patched_create_default_context  # type: ignore[assignment]
        ssl._create_default_https_context = _patched_create_default_context  # type: ignore[attr-defined]


_apply()
