#!/usr/bin/env python3
"""Atheris harness: ``src.external_predictors.parse_tapreg_html``.

Untrusted input: the body the TAPreg CGI service at UCM returns, which
``query_tapreg`` passes straight to this parser.

Contract, from the function's code and docstring:

* Documented behaviour: returns a dict mapping some subset of the query peptides
  to a float transport score. A peptide with no nearby decimal is simply absent.
* No exception is documented, and the only caller (``query_tapreg``) catches
  ``requests.exceptions.RequestException`` alone, so any exception raised here
  escapes the network client and its mock fallback. Every exception is therefore
  a finding to triage.
* More than 1 s of CPU on one input is also a finding. The text pattern puts a
  whitespace run and a lazy wildcard between the peptide and its score, so run time
  is worth watching.

Input layout: see ``_harness_common.split_parser_input``.
"""

from __future__ import annotations

import sys

import _offline_guard

_offline_guard.install_and_verify()

import _harness_common as common  # noqa: E402

common.prepare_import_path()

import atheris  # noqa: E402
import pandas  # noqa: E402,F401  imported early so instrument_imports leaves it alone
import requests  # noqa: E402,F401  same reason

atheris.enabled_hooks.add("RegEx")
atheris.enabled_hooks.add("str")
with atheris.instrument_imports(include=["src"]):
    from src.external_predictors import parse_tapreg_html

RUNNER = common.Runner("parse_tapreg_html")
RUNNER.announce(parse_tapreg_html)


def check(peptides: list[str], html: str) -> None:
    result = common.call_with_cpu_limit(parse_tapreg_html, html, peptides)
    if not isinstance(result, dict):
        raise common.ContractViolation("result is not a dict")
    extra = set(result) - set(peptides)
    if extra:
        raise common.ContractViolation(f"keys not in the peptide list: {sorted(extra)!r}")
    if not all(isinstance(v, float) for v in result.values()):
        raise common.ContractViolation("non-float transport score")


def body(data: bytes) -> None:
    peptides, html = common.split_parser_input(data)
    check(peptides, html)


def TestOneInput(data: bytes) -> None:
    RUNNER.run(body, data)


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
