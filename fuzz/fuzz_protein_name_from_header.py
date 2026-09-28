#!/usr/bin/env python3
"""Atheris harness: ``src.build_predig_recombinant_input._protein_name_from_header``.

Untrusted input: FASTA header lines from downloaded proteomes. ``parse_fasta``
strips each line and passes everything after a leading ``>`` to this function.

Contract, from the docstring ("Use UniProt-style ID token when present, else first
whitespace token"): returns a ``str`` and does not raise for any ``str`` header.
Any exception, a non-str result, or more than 1 s of CPU is a finding.

Input: the whole fuzz input, decoded as UTF-8 with replacement.
"""

from __future__ import annotations

import sys

import _offline_guard

_offline_guard.install_and_verify()

import _harness_common as common  # noqa: E402

common.prepare_import_path()

import atheris  # noqa: E402
import pandas  # noqa: E402,F401  imported early so instrument_imports leaves it alone

atheris.enabled_hooks.add("str")
with atheris.instrument_imports(include=["src"]):
    from src.build_predig_recombinant_input import _protein_name_from_header

RUNNER = common.Runner("_protein_name_from_header")
RUNNER.announce(_protein_name_from_header)


def body(data: bytes) -> None:
    header = common.decode_text(data)
    name = common.call_with_cpu_limit(_protein_name_from_header, header)
    if not isinstance(name, str):
        raise common.ContractViolation(f"non-str result {name!r}")


def TestOneInput(data: bytes) -> None:
    RUNNER.run(body, data)


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
