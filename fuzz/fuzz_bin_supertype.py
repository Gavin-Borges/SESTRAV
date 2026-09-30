#!/usr/bin/env python3
"""Atheris harness: ``src.ml_utils._bin_supertype``.

Untrusted input: the HLA allele column of a training or cohort table, binned into a
stratification component.

Contract, from the code: returns a ``str`` and never raises. It converts
``get_hla_supertype``'s ``ValueError`` into the unknown bin, so a result is either
a supertype label from ``src.hla_supertypes`` or ``_SUPERTYPE_UNKNOWN``. Any
exception, any other result, or more than 1 s of CPU is a finding.

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
import sklearn.model_selection  # noqa: E402,F401  same reason

atheris.enabled_hooks.add("RegEx")
atheris.enabled_hooks.add("str")
with atheris.instrument_imports(include=["src"]):
    from src import hla_supertypes, ml_utils

ALLOWED = (
    frozenset(hla_supertypes.HLA_SUPERTYPE_MAP.values())
    | frozenset(hla_supertypes.HLA_FAMILY_SUPERTYPE_MAP.values())
    | {ml_utils._SUPERTYPE_UNKNOWN}
)
RUNNER = common.Runner("_bin_supertype")
RUNNER.announce(ml_utils._bin_supertype)


def body(data: bytes) -> None:
    allele = common.decode_text(data)
    out = common.call_with_cpu_limit(ml_utils._bin_supertype, allele)
    if out not in ALLOWED:
        raise common.ContractViolation(f"unexpected bin {out!r}")


def TestOneInput(data: bytes) -> None:
    RUNNER.run(body, data)


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
