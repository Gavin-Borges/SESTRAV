"""Shared helpers for the NetMHCpan 4.1 leave-one-out scoring harness.

This module holds the two pieces of logic that both the input-preparation
script and the output-parsing script depend on:

1. Allele-name conversion from the test-set format ("HLA-B*07:02") to the
   NetMHCpan CLI format ("HLA-B07:02") - the asterisk is stripped and the
   colon is kept. The transform is loci-agnostic (A/B/C) and idempotent.
2. Header-driven column location for the whitespace-delimited NetMHCpan 4.1
   output, so a change in column order does not silently break parsing.

Keeping this minimal and dependency-light on purpose: only the standard
library is used here.
"""

from __future__ import annotations

import re

# Presentation (eluted-ligand) column name candidates emitted by NetMHCpan 4.1.
# The raw EL score is preferred; the percentile rank is a fallback. Both are
# matched case-insensitively against the located header tokens.
EL_SCORE_NAMES = ("EL_Score", "EL-score", "EL_score", "Score_EL", "ELScore")
EL_RANK_NAMES = ("%Rank_EL", "Rank_EL", "%Rank_el", "EL_Rank", "%Rank")
PEPTIDE_NAMES = ("Peptide", "peptide")
ALLELE_NAMES = ("MHC", "HLA", "Allele", "allele")


def convert_allele_to_cli(allele: str) -> str:
    """Convert a test-set allele label to the NetMHCpan CLI form.

    The only transform NetMHCpan requires here is removal of the asterisk
    separator while the two-field colon notation is preserved::

        "HLA-B*07:02" -> "HLA-B07:02"
        "HLA-A*02:01" -> "HLA-A02:01"
        "HLA-C*07:01" -> "HLA-C07:01"

    Alleles that already lack an asterisk (for example "HLA-B53") are returned
    unchanged, which makes the function idempotent: applying it twice yields the
    same result as applying it once. Surrounding whitespace is trimmed.

    Args:
        allele: An HLA allele string in test-set notation.

    Returns:
        The allele string with any asterisk removed and whitespace trimmed.
    """
    return allele.strip().replace("*", "")


def parse_header_columns(header_line: str) -> dict[str, int]:
    """Map lowercased NetMHCpan header tokens to their column index.

    NetMHCpan 4.1 output is whitespace-delimited. The header row (the line that
    contains the "Peptide" token) is split on runs of whitespace and each token
    is recorded with its positional index. Callers then resolve the specific
    columns they need by name via :func:`find_column`.

    Args:
        header_line: A single header line from a NetMHCpan output file.

    Returns:
        A dict mapping each lowercased header token to its 0-based column index.
        When a token repeats, the first occurrence wins.
    """
    tokens = header_line.split()
    columns: dict[str, int] = {}
    for idx, token in enumerate(tokens):
        key = token.strip().lower()
        if key and key not in columns:
            columns[key] = idx
    return columns


def find_column(columns: dict[str, int], candidates: tuple[str, ...]) -> int | None:
    """Return the column index for the first matching candidate name.

    Matching is case-insensitive. Candidates are tried in order so a caller can
    express a preference (for example, raw score before percentile rank).

    Args:
        columns: Header-token -> index map from :func:`parse_header_columns`.
        candidates: Candidate header names, most-preferred first.

    Returns:
        The 0-based column index of the first candidate present, else ``None``.
    """
    for name in candidates:
        idx = columns.get(name.strip().lower())
        if idx is not None:
            return idx
    return None


def is_header_line(line: str) -> bool:
    """Return True if a line is the NetMHCpan column header row.

    The header is the whitespace-delimited row that names the columns; it is
    identified by the presence of a "Peptide" token. Comment and rule lines
    (those beginning with '#' or '-') are never treated as headers.

    Args:
        line: A raw line from a NetMHCpan output file.

    Returns:
        True if the line looks like the column header row.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or stripped.startswith("-"):
        return False
    return bool(re.search(r"\bpeptide\b", stripped, flags=re.IGNORECASE))
