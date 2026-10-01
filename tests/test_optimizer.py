"""
Unit and property tests for SESTRAV Vaccine Cocktail ILP Optimizer (src/optimizer.py).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.optimizer import (
    BIND_COL_TO_HLA,
    DEFAULT_ALLELE_FREQUENCIES,
    PANEL_CEILINGS,
    WHO_SUPER_POPULATIONS,
    CocktailResult,
    compute_population_coverage,
    load_afnd_frequencies,
    optimize_vaccine_cocktail,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKED_FREQUENCY_TABLE = REPO_ROOT / "data" / "population" / "afnd_frequencies.json"


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


def test_load_afnd_frequencies_default_reads_the_tracked_table(monkeypatch):
    """With no path, the loader reads the tracked table verbatim.

    Pins the documented default: same table, same values. The canonical path is
    relative, so the working directory is set to the repository root to make the
    resolution deterministic rather than dependent on where pytest was launched.
    """
    monkeypatch.chdir(REPO_ROOT)
    expected = json.loads(TRACKED_FREQUENCY_TABLE.read_text(encoding="utf-8"))["alleles"]
    assert load_afnd_frequencies() == expected
    assert load_afnd_frequencies(None) == expected


def test_load_afnd_frequencies_default_still_falls_back_silently(monkeypatch, tmp_path):
    """With no path and no tracked table on disk, the built-in defaults are returned.

    This fallback is deliberate and must stay silent: only an explicitly supplied
    path refuses to be substituted.
    """
    monkeypatch.chdir(tmp_path)
    assert load_afnd_frequencies() == DEFAULT_ALLELE_FREQUENCIES


def test_load_afnd_frequencies_explicit_path_to_the_tracked_table_loads():
    """An explicit path that can be honoured is still honoured."""
    expected = json.loads(TRACKED_FREQUENCY_TABLE.read_text(encoding="utf-8"))["alleles"]
    assert load_afnd_frequencies(str(TRACKED_FREQUENCY_TABLE)) == expected


def test_load_afnd_frequencies_missing_explicit_path_raises(tmp_path, monkeypatch):
    """A missing explicit path raises instead of substituting another table.

    The working directory is the repository root, so the pre-fix fallback chain
    had a tracked table to substitute: the failure this pins is substitution, not
    an incidental absence of the default table.
    """
    monkeypatch.chdir(REPO_ROOT)
    missing = tmp_path / "frequencies_absent.json"
    with pytest.raises(FileNotFoundError) as excinfo:
        load_afnd_frequencies(str(missing))
    assert "frequencies_absent.json" in str(excinfo.value)


def test_load_afnd_frequencies_explicit_path_without_alleles_key_raises(tmp_path, monkeypatch):
    """A JSON object lacking 'alleles' raises and names the file."""
    monkeypatch.chdir(REPO_ROOT)
    malformed = tmp_path / "frequencies_no_key.json"
    malformed.write_text(json.dumps({"frequencies": {"HLA-A*02:01": {}}}), encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        load_afnd_frequencies(str(malformed))
    message = str(excinfo.value)
    assert "frequencies_no_key.json" in message
    assert "alleles" in message


def test_load_afnd_frequencies_explicit_path_that_is_not_an_object_raises(tmp_path, monkeypatch):
    """Valid JSON that is not an object raises rather than falling through."""
    monkeypatch.chdir(REPO_ROOT)
    not_an_object = tmp_path / "frequencies_list.json"
    not_an_object.write_text(json.dumps([{"HLA-A*02:01": {}}]), encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        load_afnd_frequencies(str(not_an_object))
    assert "frequencies_list.json" in str(excinfo.value)


def test_load_afnd_frequencies_explicit_unparseable_json_still_propagates(tmp_path):
    """Unparseable JSON raises json.JSONDecodeError, which is a ValueError.

    Pre-existing behaviour, pinned because the docstring now states it.
    """
    broken = tmp_path / "frequencies_broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        load_afnd_frequencies(str(broken))


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


def test_min_lower_bound_refuses_a_frame_without_conformal_bounds():
    """tau_conf on a frame lacking 'lower_bound' raises instead of thresholding raw scores.

    Stage 4 emits lower_bound/upper_bound/interval_width only when a conformal
    calibrator resolves, and models/v5/conformal_calibrator.joblib is gitignored, so a
    clone routinely produces ranked frames without them. Applying min_lower_bound to
    immunogenicity_score in that case silently changes what the threshold means.
    """
    df = _synthetic_candidates(n_peptides=15, seed=7).drop(
        columns=["lower_bound", "upper_bound", "interval_width"]
    )
    assert "immunogenicity_score" in df.columns
    with pytest.raises(ValueError, match="no 'lower_bound' column"):
        optimize_vaccine_cocktail(df, max_peptides=4, min_lower_bound=0.5)


def test_min_lower_bound_refuses_when_only_a_flat_default_is_available():
    """The 0.5-constant fallback is refused too, not just the score substitution."""
    df = _synthetic_candidates(n_peptides=15, seed=8).drop(
        columns=["lower_bound", "upper_bound", "interval_width", "immunogenicity_score"]
    )
    with pytest.raises(ValueError, match="default_0.5"):
        optimize_vaccine_cocktail(df, max_peptides=4, min_lower_bound=0.4)


def test_substitution_without_a_threshold_is_recorded_not_silent():
    """min_lower_bound=0.0 still optimizes, but the result says the bounds were substituted."""
    df = _synthetic_candidates(n_peptides=15, seed=9).drop(
        columns=["lower_bound", "upper_bound", "interval_width"]
    )
    res = optimize_vaccine_cocktail(df, max_peptides=4, min_lower_bound=0.0)
    assert res.lower_bound_source == "immunogenicity_score"
    assert res.to_dict()["lower_bound_source"] == "immunogenicity_score"


def test_genuine_conformal_bounds_are_reported_as_such():
    """A frame that really carries conformal bounds is labelled 'lower_bound'."""
    df = _synthetic_candidates(n_peptides=15, seed=10)
    res = optimize_vaccine_cocktail(df, max_peptides=4, min_lower_bound=0.2)
    assert res.lower_bound_source == "lower_bound"
    assert res.to_dict()["lower_bound_source"] == "lower_bound"
