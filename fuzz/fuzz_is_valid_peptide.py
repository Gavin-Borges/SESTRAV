#!/usr/bin/env python3
"""Atheris harness: ``src.verify.iedb_multi_virus_extractor.is_valid_peptide``.

Untrusted input: ``linear_sequence`` values from the IEDB REST API, which
``clean_and_pool_epitopes`` filters through this predicate before upper-casing
them into the verification cohort table (results/verify/<target>_verify.csv).

Contract, from the function's code (it has no docstring) and the module's own
``STANDARD_AA`` set:

* Documented behaviour: returns a bool; never raises for a ``str`` argument.
* The predicate's evident meaning, used as the oracle here: with the default
  window, the result is True exactly when the whitespace-stripped input is 8 to 11
  characters long and every character is one of the 20 standard amino-acid
  letters in either case. The function normalises case and surrounding
  whitespace before checking, and tests/test_iedb_data_loader.py pins that
  normalisation for the sibling loader, so both are part of the contract.
* Any exception, any non-bool result, a disagreement with the oracle, or more than
  1 s of CPU is a finding.

Input: the whole fuzz input, decoded as UTF-8 with replacement.
"""

from __future__ import annotations

import sys

import _offline_guard

_offline_guard.install_and_verify()

import _harness_common as common  # noqa: E402

common.prepare_import_path()

import atheris  # noqa: E402
import numpy  # noqa: E402,F401  imported early so instrument_imports leaves it alone
import pandas  # noqa: E402,F401  same reason
import requests  # noqa: E402,F401  same reason

atheris.enabled_hooks.add("str")
with atheris.instrument_imports(include=["src"]):
    from src.verify.iedb_multi_virus_extractor import is_valid_peptide

MIN_LEN = 8
MAX_LEN = 11
RUNNER = common.Runner("is_valid_peptide")
RUNNER.announce(is_valid_peptide)


def expected(seq: str) -> bool:
    stripped = seq.strip()
    return MIN_LEN <= len(stripped) <= MAX_LEN and all(
        ch in common.STANDARD_AA_EITHER_CASE for ch in stripped
    )


def body(data: bytes) -> None:
    seq = common.decode_text(data)
    got = common.call_with_cpu_limit(is_valid_peptide, seq)
    if not isinstance(got, bool):
        raise common.ContractViolation(f"non-bool result {got!r}")
    want = expected(seq)
    if got != want:
        raise common.ContractViolation(f"is_valid_peptide({seq!r}) returned {got}, oracle {want}")


def TestOneInput(data: bytes) -> None:
    RUNNER.run(body, data)


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
