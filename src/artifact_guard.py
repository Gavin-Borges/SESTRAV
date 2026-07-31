"""Shared overwrite guard for entry points that write tracked release artifacts.

Several entry points in this repository write git-tracked artifacts under
`models/` or `results/`. Each one had defaulted its output directory at the
argparse layer, so a bare invocation silently rewrote published files in place.
The repair pattern that closed those instances is always the same three pieces:

1. the output-directory flag is required, with no default;
2. a `planned_*_paths()` function enumerating *every* path the run writes;
3. a guard that raises `FileExistsError` listing the collisions before any work
   starts, unless `--allow-overwrite` / `allow_overwrite=True` is passed.

Piece 3 was copy-pasted ten times before this module existed. The duplication
itself was cheap, but the per-module tests written against each copy were not
equivalent to each other, and that mattered: during the `--results-dir` repair a
dedicated 16-test file passed in full against a live regression it existed to
catch, because its assertions searched all of stderr for a flag name that the
guard's own error message also contains. A single shared implementation is what
lets one generic, parametrized test run against the modules that use it, so a
module cannot quietly be held to a weaker standard than its siblings.

Six of the twelve modules that carry this pattern now delegate here - the
`results/` family (`scripts/run_analysis.py`, `src/final_validation_report.py`,
`src/bias_skew_finalization.py`, `src/h2_tier_a_evaluation.py`) plus
`src/data_bias_audit.py` and `src/gold_standard_sensitivity.py`. The other six
still carry their own copy: `src/train_classifier.py`, `src/train_gnn.py`,
`src/ann_benchmark.py` and `src/gnn_benchmark.py` share this exact pattern, while
`src/ablation_study.py` and `scripts/compute_ann_baseline_summary.py` are
single-path variants whose message omits the count and the listing. Folding those
six in means first growing the template: two of them add a
"(for example models/scratch/<run-name>)" clause this message cannot express, and
the single-path variants would have their user-facing text changed, which would
stop the move being behaviour-preserving. Until then, treat the contract test as
covering the delegating modules only.

`data_bias_audit.py` and `gold_standard_sensitivity.py` needed the template to
grow first too, for a different reason: neither has a single output directory.
Both take independent file-path flags (`--provenance-csv`, `--audit-csv`,
`--audit-md` / `--output-csv`, `--output-md`), so the original message's
`"under '{output_dir}'"` and `"Point {flag} at a fresh directory"` clauses would
have been actively wrong advice - there is no one directory to point at, and the
flags name files, not directories. `guard_planned_paths` gained optional `scope`
and `remedy` keyword parameters to let those two callers substitute accurate
clauses while leaving every other caller's message byte-identical (both default
to reconstructing the original clause exactly when omitted).

This module deliberately contains no policy about *which* files are planned.
Enumeration stays with each entry point, because it is the part that requires
reading that specific module's writes (including derived filenames and delegate
writes) and is the part most likely to drift as a module changes.
"""

from __future__ import annotations

import os
from typing import Iterable, Sequence


def planned_paths_under(output_dir: str, names: Iterable[str]) -> list[str]:
    """Join every artifact name onto output_dir.

    The shared tail of all four `planned_*_paths()` implementations. Kept here
    so the join convention cannot diverge between a guard and the writer it
    protects.
    """
    return [os.path.join(output_dir, name) for name in names]


def existing_planned_paths(planned_paths: Sequence[str]) -> list[str]:
    """The subset of planned_paths already present on disk, sorted."""
    return sorted(p for p in planned_paths if os.path.isfile(p))


def guard_planned_paths(
    output_dir: str,
    planned_paths: Sequence[str],
    allow_overwrite: bool,
    *,
    flag: str,
    api_hint: str,
    detail: str = "",
    scope: str | None = None,
    remedy: str | None = None,
) -> None:
    """Refuse to clobber planned artifacts already on disk.

    Args:
        output_dir: The directory the run was pointed at. Named in the error
            unless `scope` overrides that clause (see below). Still required
            even when `scope` is given, so every caller states plainly what
            location its planned paths are rooted under.
        planned_paths: Every path the run intends to write. Callers build this
            from their own `planned_*_paths()` enumeration.
        allow_overwrite: When True the guard is disarmed and returns immediately.
        flag: The CLI flag to redirect, e.g. "--results-dir" or "--output-dir".
            Entry points in this family do not all use the same name.
        api_hint: How to disarm the guard from Python rather than the CLI, e.g.
            "run_h2_tier_a(..., allow_overwrite=True)".
        detail: Optional clause appended to "These may be published results",
            for modules where naming the specific artifact at risk is worth the
            extra sentence. Must read as a continuation, e.g. a leading ": ".
        scope: Optional replacement for the "under '{output_dir}'" clause. Set
            this for entry points with no single output directory - e.g. a
            module whose planned paths straddle independent `--flag`-supplied
            file paths rather than one directory - so the message does not
            claim a directory-shaped destination that does not exist. Defaults
            to `under '{output_dir}'` when omitted, reproducing the original
            message exactly.
        remedy: Optional replacement for the "Point {flag} at a fresh
            directory, " clause. Pairs with `scope`: "point the flag at a
            fresh directory" is wrong advice when the flag names a file path,
            not a directory. Defaults to `Point {flag} at a fresh directory, `
            when omitted, reproducing the original message exactly.

    Raises:
        FileExistsError: If any planned path exists and allow_overwrite is False.
            The message lists every collision, not just the first, so one run
            surfaces the full extent of what a retry would replace.
    """
    if allow_overwrite:
        return
    existing = existing_planned_paths(planned_paths)
    if not existing:
        return
    listing = "\n  ".join(existing)
    scope_clause = scope if scope is not None else f"under '{output_dir}'"
    remedy_clause = remedy if remedy is not None else f"Point {flag} at a fresh directory, "
    raise FileExistsError(
        f"Refusing to overwrite {len(existing)} existing artifact(s) {scope_clause}:\n  "
        f"{listing}\n"
        f"These may be published results{detail}. {remedy_clause}"
        f"or pass --allow-overwrite ({api_hint}) to replace them deliberately."
    )
