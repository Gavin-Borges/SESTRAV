#!/usr/bin/env python3
"""Atheris harness: ``src.prepare_external_validation_inputs.hla_to_prime_compact``.

Untrusted input: allele strings read from ``config.yaml`` and external tables.

Contract, from the docstring ("HLA-A*02:01 -> A0201; HLA-B*08:01 -> B0801"): returns
a ``str`` and does not raise for any ``str``; an input of the documented shape
``HLA-<gene>*<field>:<field>`` becomes ``<gene><field><field>``. Any exception, a
non-str result, a documented-shape input mapped to anything else, or more than 1 s
of CPU is a finding.

The docstring promises nothing about colons outside the allele fields. An earlier
version of this harness asserted that EVERY colon is removed; the fuzzer refuted
that within seconds (a colon in the gene field survives, by the code's design), and
the assertion was the harness over-reading the docstring, not a target bug.

Input: the whole fuzz input, decoded as UTF-8 with replacement.
"""

from __future__ import annotations

import re
import sys

import _offline_guard

_offline_guard.install_and_verify()

import _harness_common as common  # noqa: E402

common.prepare_import_path()

import atheris  # noqa: E402
import openpyxl  # noqa: E402,F401  imported early so instrument_imports leaves it alone
import pandas  # noqa: E402,F401  same reason
import yaml  # noqa: E402,F401  same reason

atheris.enabled_hooks.add("str")
with atheris.instrument_imports(include=["src"]):
    from src.prepare_external_validation_inputs import hla_to_prime_compact

DOCUMENTED_SHAPE = re.compile(r"HLA-([A-Z]+)\*([0-9]{2,3}):([0-9]{2,3})")
RUNNER = common.Runner("hla_to_prime_compact")
RUNNER.announce(hla_to_prime_compact)


def body(data: bytes) -> None:
    hla = common.decode_text(data)
    out = common.call_with_cpu_limit(hla_to_prime_compact, hla)
    if not isinstance(out, str):
        raise common.ContractViolation(f"non-str result {out!r}")
    shaped = DOCUMENTED_SHAPE.fullmatch(hla)
    if shaped and out != "".join(shaped.groups()):
        raise common.ContractViolation(f"{hla!r} mapped to {out!r}")


def TestOneInput(data: bytes) -> None:
    RUNNER.run(body, data)


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
