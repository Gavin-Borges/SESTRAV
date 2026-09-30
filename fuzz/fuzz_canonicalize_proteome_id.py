#!/usr/bin/env python3
"""Atheris harness: ``src.naming.canonicalize_proteome_id``.

Untrusted input: proteome identifiers from config files and CLI arguments.

Contract, from the docstring ("Return canonical proteome_id if a legacy alias is
supplied"): returns a ``str``; an alias maps to its canonical id and anything else
is returned unchanged. Any exception, a result that is neither the input nor a
canonical id, or more than 1 s of CPU is a finding.

Input: the whole fuzz input, decoded as UTF-8 with replacement.
"""

from __future__ import annotations

import sys

import _offline_guard

_offline_guard.install_and_verify()

import _harness_common as common  # noqa: E402

common.prepare_import_path()

import atheris  # noqa: E402

atheris.enabled_hooks.add("str")
with atheris.instrument_imports(include=["src"]):
    from src import naming

CANONICAL = frozenset(naming.PROTEOME_ID_ALIASES.values())
RUNNER = common.Runner("canonicalize_proteome_id")
RUNNER.announce(naming.canonicalize_proteome_id)


def body(data: bytes) -> None:
    proteome_id = common.decode_text(data)
    out = common.call_with_cpu_limit(naming.canonicalize_proteome_id, proteome_id)
    if out != proteome_id and out not in CANONICAL:
        raise common.ContractViolation(f"{proteome_id!r} mapped to non-canonical {out!r}")


def TestOneInput(data: bytes) -> None:
    RUNNER.run(body, data)


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
