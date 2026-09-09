"""
Unit and property tests for SESTRAV Vaccine Cocktail ILP Optimizer (src/optimizer.py).
"""

import numpy as np
import pandas as pd
import pytest

from src.optimizer import (
    BIND_COL_TO_HLA,
    PANEL_CEILINGS,
    WHO_SUPER_POPULATIONS,
    CocktailResult,
    compute_population_coverage,
    load_afnd_frequencies,
    optimize_vaccine_cocktail,
)


def _synthetic_candidates(n_peptides=20, seed=42):
    """Generate reproducible synthetic peptide candidates with binding profiles and conformal bounds."""
    rng = np.random.default_rng(seed)
    data = {
        "peptide": [f"PEP{i:03d}" for i in range(n_peptides)],
        "immunogenicity_score": rng.uniform(0.3, 0.95, size=n_peptides),
        "lower_bound": rng.uniform(0.1, 0.85, size=n_peptides),
        "upper_bound": rng.uniform(0.5, 0.99, size=n_peptides),
    }
    # Ensure lower_bound <= upper_bound
    data["lower_bound"] = np.minimum(data["lower_bound"], data["upper_bound"] - 0.05)
    data["interval_width"] = data["upper_bound"] - data["lower_bound"]

    # Generate binding matrix (10 alleles)
    for col in BIND_COL_TO_HLA:
        # Sparsity: ~30% binding rate
        data[col] = (rng.uniform(0.0, 1.0, size=n_peptides) > 0.65).astype(float)

    return pd.DataFrame(data)


def test_load_afnd_frequencies():
    """Verify versioned AFND frequency file contains 10 alleles across 5 WHO populations."""
    freqs = load_afnd_frequencies()
    assert len(freqs) == 10
    for allele, pop_dict in freqs.items():
        assert allele.startswith("HLA-")
        for pop in WHO_SUPER_POPULATIONS:
            assert pop in pop_dict
            assert 0.0 <= pop_dict[pop] <= 1.0


def test_compute_population_coverage_single_allele():
    """Verify HWE coverage formula for a single allele matches analytical expectation."""
    # HLA-A*02:01 in EUR has h = 0.270
    # Expected coverage = 1 - (1 - 0.27)^2 = 1 - 0.5329 = 0.4671
    cov = compute_population_coverage(["HLA-A*02:01"])
    assert cov["EUR"] == pytest.approx(0.4671, abs=1e-4)


def test_compute_population_coverage_panel_ceilings():
    """Verify coverage over the entire 10-allele panel matches documented ceilings."""
    all_alleles = list(BIND_COL_TO_HLA.values())
    cov = compute_population_coverage(all_alleles)
    assert cov["EUR"] == pytest.approx(PANEL_CEILINGS["EUR"], abs=0.01)
    assert cov["AFR"] == pytest.approx(PANEL_CEILINGS["AFR"], abs=0.01)
    assert cov["global_mean"] == pytest.approx(PANEL_CEILINGS["global_mean"], abs=0.01)


def test_optimize_vaccine_cocktail_budget_constraint():
    """Selected cocktail size must not exceed max_peptides budget K."""
    df = _synthetic_candidates(n_peptides=25, seed=123)
    for k in (1, 3, 5, 8):
        res = optimize_vaccine_cocktail(df, max_peptides=k)
        assert isinstance(res, CocktailResult)
        assert res.status == "Optimal"
        assert len(res.selected_peptides) <= k


def test_optimize_vaccine_cocktail_coverage_monotonicity():
    """Increasing cocktail size budget K must produce non-decreasing global coverage."""
    df = _synthetic_candidates(n_peptides=30, seed=456)
    coverages = []
    for k in (2, 5, 10):
        res = optimize_vaccine_cocktail(df, max_peptides=k, immunogenicity_weight=0.0)
        coverages.append(res.population_coverage["global_mean"])

    assert coverages[0] <= coverages[1] <= coverages[2]


def test_optimize_vaccine_cocktail_min_lower_bound_filter():
    """Peptides with conformal lower bound below threshold must not be selected."""
    df = _synthetic_candidates(n_peptides=20, seed=789)
    threshold = 0.50
    res = optimize_vaccine_cocktail(df, max_peptides=5, min_lower_bound=threshold)
    assert res.status == "Optimal"
    assert (res.selected_peptides["lower_bound"] >= threshold).all()


def test_optimize_vaccine_cocktail_required_alleles():
    """Required alleles must be covered when feasible."""
    df = _synthetic_candidates(n_peptides=25, seed=999)
    req_allele = "HLA-A*02:01"
    res = optimize_vaccine_cocktail(df, max_peptides=5, required_alleles=[req_allele])
    assert res.status == "Optimal"
    assert res.allele_coverage[req_allele] >= 1


def test_optimize_vaccine_cocktail_empty_df_raises():
    """Empty candidate DataFrame raises ValueError."""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="is empty"):
        optimize_vaccine_cocktail(empty_df)


def test_optimize_vaccine_cocktail_to_dict_serialization():
    """CocktailResult serialization to dictionary produces expected structure."""
    df = _synthetic_candidates(n_peptides=15, seed=101)
    res = optimize_vaccine_cocktail(df, max_peptides=4)
    summary = res.to_dict()
    assert summary["status"] == "Optimal"
    assert summary["cocktail_size"] <= 4
    assert "population_coverage" in summary
    assert "allele_coverage" in summary
    assert "panel_ceilings" in summary
