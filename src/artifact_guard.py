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

All nineteen modules that carry this pattern now delegate here. The `results/`
family (`scripts/run_analysis.py`, `src/final_validation_report.py`,
`src/bias_skew_finalization.py`, `src/h2_tier_a_evaluation.py`) and several more
(`src/data_bias_audit.py`, `src/gold_standard_sensitivity.py`,
`src/shap_analysis.py`, `src/prepare_external_validation_inputs.py`,
`src/external_benchmark_comparison.py`, `src/external_validation_cross_virus.py`,
`src/calibration_analysis.py`, `src/baseline_comparison.py`,
`scripts/compute_population_coverage.py`) migrated first, followed by
`src/train_classifier.py` and `src/ann_benchmark.py`, which matched the shape
exactly once `remedy` could carry a "(for example models/scratch/<run-name>)"
clause. The last four - `src/train_gnn.py`, `src/gnn_benchmark.py`,
`src/ablation_study.py`, `scripts/compute_ann_baseline_summary.py` - needed the
template to grow further, each for a different reason:

- `src/train_gnn.py` omits the `under '{output_dir}'` scope clause entirely and
  appends a trailing sentence about OOF predictions the original message had no
  slot for.
- `src/gnn_benchmark.py` also omits the scope clause, calls its writes
  "result CSV(s)" instead of "artifact(s)", and omits the Python-API-hint
  parenthetical altogether.
- `src/ablation_study.py` and `scripts/compute_ann_baseline_summary.py` guard a
  single named path rather than a list: "Refusing to overwrite existing artifact
  at '{path}'." with no collision count and no indented listing, "This may be a
  published result" (singular) rather than "These may be published results",
  and no API-hint parenthetical either.

`guard_planned_paths` gained `noun`, `trailing` and `single_path` keyword
parameters, and `scope`/`api_hint` became omittable (an empty string suppresses
the clause instead of replacing it), to let these four substitute an accurate
shape while leaving every other caller's message byte-identical - each default
reconstructs the original clause exactly when omitted, verified caller-by-caller
before folding this group in.

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
    noun: str = "artifact(s)",
    trailing: str = "",
    single_path: bool = False,
) -> None:
    """Refuse to clobber planned artifacts already on disk.

    Args:
        output_dir: The directory the run was pointed at. Named in the error
            unless `scope` overrides that clause (see below). Still required
            even when `scope` is given, so every caller states plainly what
            location its planned paths are rooted under.
        planned_paths: Every path the run intends to write. Callers build this
            from their own `planned_*_paths()` enumeration. When `single_path`
            is True this must resolve to exactly one path; the message names
            it directly rather than reporting a count and an indented listing.
        allow_overwrite: When True the guard is disarmed and returns immediately.
        flag: The CLI flag to redirect, e.g. "--results-dir" or "--output-dir".
            Entry points in this family do not all use the same name.
        api_hint: How to disarm the guard from Python rather than the CLI, e.g.
            "run_h2_tier_a(..., allow_overwrite=True)". Pass "" to omit the
            parenthetical entirely, for modules with no Python-level entry
            point worth naming.
        detail: Optional clause appended to "These may be published results"
            ("This may be a published result" when `single_path` is True), for
            modules where naming the specific artifact at risk is worth the
            extra sentence. Must read as a continuation, e.g. a leading ": ".
        scope: Optional replacement for the "under '{output_dir}'" clause. Set
            this for entry points with no single output directory - e.g. a
            module whose planned paths straddle independent `--flag`-supplied
            file paths rather than one directory - so the message does not
            claim a directory-shaped destination that does not exist. Pass ""
            to drop the clause entirely rather than replace it. Defaults to
            `under '{output_dir}'` when omitted, reproducing the original
            message exactly. Unused when `single_path` is True, since that
            shape names the one colliding path inline instead.
        remedy: Optional replacement for the "Point {flag} at a fresh
            directory, " clause. Pairs with `scope`: "point the flag at a
            fresh directory" is wrong advice when the flag names a file path,
            not a directory. Defaults to `Point {flag} at a fresh directory, `
            when omitted, reproducing the original message exactly.
        noun: What to call a colliding path. Defaults to "artifact(s)".
            Ignored when `single_path` is True, which always says "artifact".
        trailing: Optional free text appended after the final sentence, for a
            module-specific caveat the rest of the message has no slot for
            (e.g. a note that a derived file lives outside `output_dir`).
            Must include its own leading space to read as a continuation.
            Defaults to "".
        single_path: When True, switches to the single-path message shape:
            "Refusing to overwrite existing artifact at '{path}'." with no
            collision count or listing, "This may be a published result"
            (singular) rather than "These may be published results", and
            "replace it deliberately" rather than "replace them deliberately".
            Defaults to False, reproducing the original multi-path shape.

    Raises:
        ValueError: If `single_path` is True and `planned_paths` does not
            resolve to exactly one path - a caller-contract violation, checked
            before `allow_overwrite` so it cannot be silently bypassed.
        FileExistsError: If any planned path exists and allow_overwrite is False.
            The message lists every collision, not just the first (unless
            `single_path` is True, where there is only ever one to name), so
            one run surfaces the full extent of what a retry would replace.
    """
    if single_path and len(planned_paths) > 1:
        raise ValueError(
            f"single_path=True requires exactly one planned path, got "
            f"{len(planned_paths)}: {list(planned_paths)}"
        )
    if allow_overwrite:
        return
    existing = existing_planned_paths(planned_paths)
    if not existing:
        return

    hint_clause = f" ({api_hint})" if api_hint else ""
    remedy_clause = remedy if remedy is not None else f"Point {flag} at a fresh directory, "

    if single_path:
        raise FileExistsError(
            f"Refusing to overwrite existing artifact at '{existing[0]}'.\n"
            f"This may be a published result{detail}. {remedy_clause}"
            f"or pass --allow-overwrite{hint_clause} to replace it deliberately."
            f"{trailing}"
        )

    listing = "\n  ".join(existing)
    scope_clause = scope if scope is not None else f"under '{output_dir}'"
    scope_segment = f" {scope_clause}" if scope_clause else ""
    raise FileExistsError(
        f"Refusing to overwrite {len(existing)} existing {noun}{scope_segment}:\n  "
        f"{listing}\n"
        f"These may be published results{detail}. {remedy_clause}"
        f"or pass --allow-overwrite{hint_clause} to replace them deliberately."
        f"{trailing}"
    )
