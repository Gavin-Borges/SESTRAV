"""The two population-coverage implementations agree.

Why this test exists
--------------------
`src.optimizer.compute_population_coverage` (the cocktail optimizer's
objective) and `scripts/compute_population_coverage.py` (which backs the
tracked `results/population_coverage_v5.json`) each implement the same
Hardy-Weinberg coverage, 1 - (prod_i(1 - h_i))^2, with their own allele table.
Nothing tied them together, so either could drift and the optimizer's
objective and the published coverage table would quietly disagree. Pinned:

- the two built-in frequency tables and population lists are identical;
- on any shared table the per-population values agree exactly once the
  optimizer's rounding to 4 decimals is applied, and so does the global
  mean (the optimizer averages its ROUNDED values, then rounds again);
- the two deliberate differences in the optimizer's contract: an allele
  listed twice counts once, and an allele missing from the table counts as
  frequency 0 rather than raising.

Measured 2026-10-15 before writing: 0.0 maximum difference over 2,000 seeded
random tables, and the two built-in tables compare equal.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src import optimizer

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "compute_population_coverage.py"

_spec = importlib.util.spec_from_file_location("compute_population_coverage_under_test", SCRIPT)
assert _spec is not None and _spec.loader is not None
script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(script)

POPULATIONS = list(optimizer.WHO_SUPER_POPULATIONS)


def _script_values(freqs: dict[str, dict[str, float]]) -> dict[str, float]:
    """The script's panel coverage per population plus the optimizer-rounded global mean."""
    _, panel, _ = script.compute_all_coverage(freqs, POPULATIONS)
    values = {pop: round(panel[pop], 4) for pop in POPULATIONS}
    values["global_mean"] = round(sum(values[pop] for pop in POPULATIONS) / len(POPULATIONS), 4)
    return values


def test_the_built_in_tables_and_populations_are_identical() -> None:
    assert script.ALLELE_HAPLOTYPE_FREQUENCIES == optimizer.DEFAULT_ALLELE_FREQUENCIES
    assert list(script.POPULATIONS) == POPULATIONS


def test_the_shipped_table_gives_the_same_coverage() -> None:
    freqs = optimizer.DEFAULT_ALLELE_FREQUENCIES
    assert optimizer.compute_population_coverage(list(freqs), freqs, POPULATIONS) == _script_values(freqs)


@pytest.mark.parametrize("seed", range(25))
def test_random_shared_tables_agree(seed: int) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(20):
        alleles = [f"HLA-X*{i:02d}:01" for i in range(int(rng.integers(1, 13)))]
        freqs = {a: {p: float(rng.random()) * 0.4 for p in POPULATIONS} for a in alleles}
        assert optimizer.compute_population_coverage(alleles, freqs, POPULATIONS) == _script_values(freqs)


def test_the_optimizer_counts_a_repeated_allele_once() -> None:
    freqs = optimizer.DEFAULT_ALLELE_FREQUENCIES
    allele = next(iter(freqs))
    once = optimizer.compute_population_coverage([allele], freqs, POPULATIONS)
    twice = optimizer.compute_population_coverage([allele, allele], freqs, POPULATIONS)
    assert once == twice


def test_the_optimizer_scores_an_unknown_allele_as_zero_frequency() -> None:
    freqs = optimizer.DEFAULT_ALLELE_FREQUENCIES
    allele = next(iter(freqs))
    known = optimizer.compute_population_coverage([allele], freqs, POPULATIONS)
    padded = optimizer.compute_population_coverage([allele, "HLA-Z*99:99"], freqs, POPULATIONS)
    assert known == padded
    assert optimizer.compute_population_coverage(["HLA-Z*99:99"], freqs, POPULATIONS) == {
        **{pop: 0.0 for pop in POPULATIONS}, "global_mean": 0.0
    }
