#!/usr/bin/env python3
"""Atheris harness: ``src.hla_supertypes.get_hla_supertype``.

Untrusted input: allele strings from IEDB exports and user-supplied tables.

Contract, from the function's code and docstring:

* Documented behaviour: returns one of the supertype labels that appear as values
  in ``HLA_SUPERTYPE_MAP`` or ``HLA_FAMILY_SUPERTYPE_MAP``, or raises ``ValueError``
  for an unknown or unmapped allele (the function's only raise, and the one
  ``src.ml_utils._bin_supertype`` catches).
* Any other exception is a finding. ``UnicodeError`` subclasses ``ValueError`` and is
  re-raised on purpose: an encoding failure is a bug, not the documented
  unmapped-allele outcome.
* A return value outside the label set, or more than 1 s of CPU, is a finding.

Input: the whole fuzz input, decoded as UTF-8 with replacement.
"""

from __future__ import annotations

import sys

import _offline_guard

_offline_guard.install_and_verify()

import _harness_common as common  # noqa: E402

common.prepare_import_path()

import atheris  # noqa: E402

atheris.enabled_hooks.add("RegEx")
atheris.enabled_hooks.add("str")
with atheris.instrument_imports(include=["src"]):
    from src import hla_supertypes
    from src.hla_supertypes import get_hla_supertype

LABELS = frozenset(hla_supertypes.HLA_SUPERTYPE_MAP.values()) | frozenset(
    hla_supertypes.HLA_FAMILY_SUPERTYPE_MAP.values()
)
RUNNER = common.Runner("get_hla_supertype")
RUNNER.announce(get_hla_supertype)


def body(data: bytes) -> None:
    allele = common.decode_text(data)
    try:
        label = common.call_with_cpu_limit(get_hla_supertype, allele)
    except ValueError as exc:
        if isinstance(exc, UnicodeError):
            raise
        return
    if label not in LABELS:
        raise common.ContractViolation(f"unexpected supertype label {label!r}")


def TestOneInput(data: bytes) -> None:
    RUNNER.run(body, data)


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
