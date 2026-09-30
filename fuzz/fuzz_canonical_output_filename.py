#!/usr/bin/env python3
"""Atheris harness: ``src.naming.canonical_output_filename``.

Untrusted input: base names, dataset modes and dataset versions taken from config
and CLI arguments.

Contract, from the docstring: returns ``"<base>__<mode>__<version>.csv"``. Any
exception, any other result, or more than 1 s of CPU is a finding.

Input: the fuzz input split on NUL bytes into base name, mode and version (missing
fields are empty), each decoded as UTF-8 with replacement.
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
    from src.naming import canonical_output_filename

RUNNER = common.Runner("canonical_output_filename")
RUNNER.announce(canonical_output_filename)


def body(data: bytes) -> None:
    fields = [common.decode_text(part) for part in data.split(b"\x00", 2)]
    base, mode, version = (fields + ["", "", ""])[:3]
    out = common.call_with_cpu_limit(canonical_output_filename, base, mode, version)
    if out != f"{base}__{mode}__{version}.csv":
        raise common.ContractViolation(f"unexpected filename {out!r}")


def TestOneInput(data: bytes) -> None:
    RUNNER.run(body, data)


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
