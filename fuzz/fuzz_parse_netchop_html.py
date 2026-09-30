#!/usr/bin/env python3
"""Atheris harness: ``src.external_predictors.parse_netchop_html``.

Untrusted input: the body the NetChop 3.1 web service at DTU returns, which
``query_netchop`` passes straight to this parser.

Contract, from the function's code and docstring:

* Documented behaviour: returns a dict whose keys are exactly the distinct peptides
  passed in, each mapping to ``{"scores": [float, ...], "cleavages": [str, ...]}``
  with one entry per parsed residue row, so the two lists always have equal
  length. Rows that do not match the table pattern, or whose ``pep_N`` index is
  out of range, are skipped.
* No exception is documented, and the only caller (``query_netchop``) catches
  ``requests.exceptions.RequestException`` alone, so any exception raised here
  escapes the network client and its mock fallback. Every exception is therefore
  outside the contract: a finding to triage, not an expected outcome.
* More than 1 s of CPU on one input is also a finding.

Input layout: see ``_harness_common.split_parser_input``.

Run from the repository root, for example::

    python fuzz/build_seed_corpora.py --out /path/to/seeds
    python fuzz/fuzz_parse_netchop_html.py corpus/ /path/to/seeds/fuzz_parse_netchop_html \
        -max_total_time=300 -max_len=16384
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
    from src.external_predictors import parse_netchop_html

RUNNER = common.Runner("parse_netchop_html")
RUNNER.announce(parse_netchop_html)


def check(peptides: list[str], html: str) -> None:
    result = common.call_with_cpu_limit(parse_netchop_html, html, peptides)
    if not isinstance(result, dict) or set(result) != set(peptides):
        raise common.ContractViolation("result keys differ from the peptide list")
    for pep, entry in result.items():
        scores = entry.get("scores")
        cleavages = entry.get("cleavages")
        if not isinstance(scores, list) or not isinstance(cleavages, list):
            raise common.ContractViolation(f"{pep!r}: scores/cleavages are not lists")
        if len(scores) != len(cleavages):
            raise common.ContractViolation(f"{pep!r}: {len(scores)} scores, {len(cleavages)} calls")
        if not all(isinstance(s, float) for s in scores):
            raise common.ContractViolation(f"{pep!r}: non-float score")
        if not all(isinstance(c, str) and len(c) == 1 for c in cleavages):
            raise common.ContractViolation(f"{pep!r}: cleavage call is not one character")


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
