"""
SESTRAV Vaccine Cocktail Optimization via Integer Linear Programming (ILP)

Selects a minimal or bounded-size cocktail of candidate T-cell epitopes to
maximize population HLA coverage across global populations, prioritizing
peptides with high lower-bound predicted immunogenicity from conformal
uncertainty quantification (Cross Venn-Abers intervals).

Population coverage model:
    Hardy-Weinberg equilibrium (HWE) independence across Class I loci
    (Lundegaard et al. 2010; IEDB Population Coverage Tool, Bui et al. 2006):

        phenotype_freq_a = 1 - (1 - haplotype_freq_a)^2
        pop_coverage     = 1 - product_{a in covered}(1 - phenotype_freq_a)
                         = 1 - (product_{a in covered}(1 - haplotype_freq_a))^2

    Under this model, maximizing population coverage is equivalent to minimizing
    product_{a in covered}(1 - h_a)^2, which linearizes exactly in the log domain:

        max sum_{a in A} (-2 * ln(1 - h_{p, a})) * z_a

    where z_a in {0, 1} indicates whether allele a is covered by at least one
    selected peptide in the cocktail.

Literature citations:
    - Allele frequencies: Gonzalez-Galarza FF et al. (2020) Nucleic Acids Res 48(D1):D783-D788.
    - Coverage formulation: Bui HH et al. (2006) BMC Bioinformatics 7:153.
    - Conformal bounds: Vovk V, Shen G, Manokhin V, Xie Z (2019) Cross-conformal predictive distributions.

Disclosed limitations (standing project constraints):
    1. HWE assumes locus independence and ignores linkage disequilibrium.
    2. The 10-allele Class I panel imposes an intrinsic population coverage ceiling
       (AFR 0.621, AMR 0.814, EAS 0.704, EUR 0.919, SAS 0.840; global mean 0.789).
       No cocktail can exceed the panel ceiling.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pulp

#: Default fallback frequencies if data/population/afnd_frequencies.json is absent
DEFAULT_ALLELE_FREQUENCIES: dict[str, dict[str, float]] = {
    "HLA-A*01:01": {"AFR": 0.065, "AMR": 0.082, "EAS": 0.015, "EUR": 0.165, "SAS": 0.105},
    "HLA-A*02:01": {"AFR": 0.090, "AMR": 0.180, "EAS": 0.040, "EUR": 0.270, "SAS": 0.150},
    "HLA-A*03:01": {"AFR": 0.070, "AMR": 0.075, "EAS": 0.008, "EUR": 0.135, "SAS": 0.080},
    "HLA-A*11:01": {"AFR": 0.040, "AMR": 0.060, "EAS": 0.238, "EUR": 0.055, "SAS": 0.180},
    "HLA-A*24:02": {"AFR": 0.035, "AMR": 0.120, "EAS": 0.210, "EUR": 0.085, "SAS": 0.090},
    "HLA-B*07:02": {"AFR": 0.055, "AMR": 0.078, "EAS": 0.035, "EUR": 0.120, "SAS": 0.078},
    "HLA-B*08:01": {"AFR": 0.030, "AMR": 0.050, "EAS": 0.005, "EUR": 0.108, "SAS": 0.062},
    "HLA-B*27:05": {"AFR": 0.011, "AMR": 0.026, "EAS": 0.012, "EUR": 0.052, "SAS": 0.039},
    "HLA-B*35:01": {"AFR": 0.050, "AMR": 0.076, "EAS": 0.038, "EUR": 0.075, "SAS": 0.057},
    "HLA-B*44:02": {"AFR": 0.025, "AMR": 0.048, "EAS": 0.015, "EUR": 0.095, "SAS": 0.045},
}

#: Mapping of feature matrix binding columns to standard HLA allele notation
BIND_COL_TO_HLA: dict[str, str] = {
    "bind_A0101": "HLA-A*01:01",
    "bind_A0201": "HLA-A*02:01",
    "bind_A0301": "HLA-A*03:01",
    "bind_A1101": "HLA-A*11:01",
    "bind_A2402": "HLA-A*24:02",
    "bind_B0702": "HLA-B*07:02",
    "bind_B0801": "HLA-B*08:01",
    "bind_B2705": "HLA-B*27:05",
    "bind_B3501": "HLA-B*35:01",
    "bind_B4402": "HLA-B*44:02",
}

HLA_TO_BIND_COL: dict[str, str] = {v: k for k, v in BIND_COL_TO_HLA.items()}

WHO_SUPER_POPULATIONS: tuple[str, ...] = ("AFR", "AMR", "EAS", "EUR", "SAS")

PANEL_CEILINGS: dict[str, float] = {
    "AFR": 0.621,
    "AMR": 0.814,
    "EAS": 0.704,
    "EUR": 0.919,
    "SAS": 0.840,
    "global_mean": 0.789,
}


@dataclass(frozen=True)
class CocktailResult:
    """Immutable result of a vaccine cocktail optimization run."""

    selected_peptides: pd.DataFrame
    population_coverage: dict[str, float]
    allele_coverage: dict[str, int]
    mean_conformal_lower_bound: float
    mean_score: float
    objective_value: float
    status: str
    panel_ceilings: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Convert result summary to a serializable dictionary."""
        return {
            "status": self.status,
            "cocktail_size": len(self.selected_peptides),
            "peptides": self.selected_peptides["peptide"].tolist()
            if "peptide" in self.selected_peptides.columns
            else [],
            "population_coverage": self.population_coverage,
            "allele_coverage": self.allele_coverage,
            "mean_conformal_lower_bound": round(self.mean_conformal_lower_bound, 4),
            "mean_score": round(self.mean_score, 4),
            "objective_value": round(self.objective_value, 4),
            "panel_ceilings": self.panel_ceilings,
        }


def load_afnd_frequencies(path: str | None = None) -> dict[str, dict[str, float]]:
    """Load versioned AFND haplotype frequencies from JSON or default table."""
    if path is not None and os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if "alleles" in data:
            return data["alleles"]
    canonical_path = os.path.join("data", "population", "afnd_frequencies.json")
    if os.path.isfile(canonical_path):
        with open(canonical_path, encoding="utf-8") as f:
            data = json.load(f)
        if "alleles" in data:
            return data["alleles"]
    return DEFAULT_ALLELE_FREQUENCIES


def compute_population_coverage(
    covered_alleles: Sequence[str],
    allele_freqs: dict[str, dict[str, float]] | None = None,
    populations: Sequence[str] = WHO_SUPER_POPULATIONS,
) -> dict[str, float]:
    """Compute exact Hardy-Weinberg population coverage for a set of covered alleles.

    Formula:
        coverage_p = 1.0 - (product_{a in covered} (1.0 - h_{p, a}))^2
    """
    freqs = allele_freqs or DEFAULT_ALLELE_FREQUENCIES
    coverage: dict[str, float] = {}

    for pop in populations:
        complement_prod = 1.0
        for allele in set(covered_alleles):
            h = freqs.get(allele, {}).get(pop, 0.0)
            complement_prod *= 1.0 - h
        cov = 1.0 - (complement_prod**2)
        coverage[pop] = round(float(cov), 4)

    if populations:
        mean_cov = float(np.mean([coverage[p] for p in populations]))
        coverage["global_mean"] = round(mean_cov, 4)

    return coverage


def optimize_vaccine_cocktail(
    candidates_df: pd.DataFrame,
    max_peptides: int = 10,
    min_lower_bound: float = 0.0,
    immunogenicity_weight: float = 0.1,
    min_binding_threshold: float = 0.5,
    min_alleles_per_peptide: int = 1,
    required_alleles: Sequence[str] | None = None,
    population_weights: dict[str, float] | None = None,
    allele_freqs: dict[str, dict[str, float]] | None = None,
    solver: pulp.LpSolver | None = None,
) -> CocktailResult:
    """Formulate and solve the Vaccine Cocktail Optimization ILP.

    Parameters
    ----------
    candidates_df : pd.DataFrame
        Ranked epitope candidates (e.g. from Stage 4 ranked output). Must contain
        'peptide' and either 'lower_bound' (conformal lower bound) or
        'immunogenicity_score'. If 'lower_bound' is missing, 'immunogenicity_score'
        is used.
    max_peptides : int
        Maximum number of epitopes in the selected cocktail (K). Default 10.
    min_lower_bound : float
        Hard minimum conformal lower bound threshold (tau_conf). Candidates
        with lower_bound < min_lower_bound are pruned prior to optimization.
    immunogenicity_weight : float
        Trade-off weight (beta) balancing population coverage against
        conformal lower bound confidence. Default 0.1.
    min_binding_threshold : float
        Threshold above which a peptide is considered to present an allele
        (if binding values in bind_* columns are affinities/probabilities).
        Default 0.5.
    min_alleles_per_peptide : int
        Pruning threshold: ignore candidates that present fewer than this
        many panel alleles. Default 1.
    required_alleles : Sequence[str], optional
        Alleles that MUST be covered by at least one selected epitope.
    population_weights : dict[str, float], optional
        Weights alpha_p across super-populations. Default equal weighting.
    allele_freqs : dict[str, dict[str, float]], optional
        Per-allele haplotype frequencies. Defaults to AFND table.
    solver : pulp.LpSolver, optional
        Custom PuLP solver. Defaults to PULP_CBC_CMD(msg=False).

    Returns
    -------
    CocktailResult
        Data structure with selected peptides, population coverage, allele coverage,
        and diagnostic statistics.
    """
    if candidates_df.empty:
        raise ValueError("[CocktailOptimizer] Input candidate DataFrame is empty.")

    freqs = allele_freqs or load_afnd_frequencies()
    all_alleles = sorted(freqs.keys())
    pops = list(population_weights.keys()) if population_weights else list(WHO_SUPER_POPULATIONS)

    if population_weights is None:
        weights = {p: 1.0 / len(pops) for p in pops}
    else:
        total_w = sum(population_weights.values())
        weights = {p: w / total_w for p, w in population_weights.items()}

    # Determine score and lower bound columns
    df = candidates_df.copy().reset_index(drop=True)
    if "lower_bound" not in df.columns:
        if "immunogenicity_score" in df.columns:
            df["lower_bound"] = df["immunogenicity_score"]
        else:
            df["lower_bound"] = 0.5

    if "immunogenicity_score" not in df.columns:
        df["immunogenicity_score"] = df["lower_bound"]

    # Filter by minimum lower bound
    if min_lower_bound > 0.0:
        df = df[df["lower_bound"] >= min_lower_bound].reset_index(drop=True)
        if df.empty:
            raise ValueError(
                f"[CocktailOptimizer] No candidate peptides meet min_lower_bound >= {min_lower_bound:.3f}."
            )

    # Detect allele binding matrix columns in candidates_df
    allele_bind_cols: dict[str, str] = {}
    for col in df.columns:
        if col in BIND_COL_TO_HLA:
            allele_bind_cols[BIND_COL_TO_HLA[col]] = col
        elif col in freqs:
            allele_bind_cols[col] = col

    # Precompute peptide-allele incidence matrix: M[i, a] in {0, 1}
    n_peptides = len(df)
    peptide_indices = list(range(n_peptides))
    binds: dict[tuple[int, str], int] = {}

    for i, row in df.iterrows():
        for allele in all_alleles:
            is_bound = 0
            if allele in allele_bind_cols:
                col_name = allele_bind_cols[allele]
                val = float(row[col_name])
                # Value may be normalized affinity [0, 1] or raw binding score
                if val >= min_binding_threshold:
                    is_bound = 1
            else:
                # Fallback: if 'alleles' column exists as list/str
                if "alleles" in row and isinstance(row["alleles"], (str, list)):
                    if allele in row["alleles"]:
                        is_bound = 1
                elif "allele" in row and allele == str(row["allele"]):
                    is_bound = 1
            binds[(i, allele)] = is_bound

    # Prune candidates presenting 0 alleles if min_alleles_per_peptide > 0
    if min_alleles_per_peptide > 0:
        active_indices = [
            i
            for i in peptide_indices
            if sum(binds[(i, a)] for a in all_alleles) >= min_alleles_per_peptide
        ]
        if not active_indices:
            # Fall back to all peptides if none meet the allele threshold
            active_indices = peptide_indices
    else:
        active_indices = peptide_indices

    # Calculate linearized HWE coverage weights:
    # W_a = sum_p alpha_p * (-2 * ln(1 - h_{p, a}))
    allele_linear_weights: dict[str, float] = {}
    for allele in all_alleles:
        w_a = 0.0
        for pop in pops:
            h = freqs.get(allele, {}).get(pop, 0.0)
            # Clip h to avoid ln(0)
            h_clipped = min(max(h, 0.0), 0.999)
            coeff = -2.0 * math.log(1.0 - h_clipped)
            w_a += weights[pop] * coeff
        allele_linear_weights[allele] = w_a

    # Build Integer Linear Program (ILP)
    prob = pulp.LpProblem("Vaccine_Cocktail_Optimization", pulp.LpMaximize)

    # Decision variables:
    # x_i in {0, 1}: 1 if peptide i is selected in the cocktail
    x = {i: pulp.LpVariable(f"x_{i}", cat=pulp.LpBinary) for i in active_indices}

    # z_a in {0, 1}: 1 if allele a is covered by at least one selected peptide
    z = {a: pulp.LpVariable(f"z_{a}", cat=pulp.LpBinary) for a in all_alleles}

    # Objective function:
    # Maximize sum_a W_a * z_a + beta * sum_i L_i * x_i
    coverage_term = pulp.lpSum([allele_linear_weights[a] * z[a] for a in all_alleles])
    immunogenicity_term = pulp.lpSum(
        [float(df.loc[i, "lower_bound"]) * x[i] for i in active_indices]
    )

    prob += coverage_term + (immunogenicity_weight * immunogenicity_term), "Objective"

    # Constraint 1: Cocktail budget sum_i x_i <= K
    prob += pulp.lpSum([x[i] for i in active_indices]) <= max_peptides, "Cocktail_Budget"

    # Constraint 2: Allele coverage linkage
    # z_a <= sum_{i: binds(i, a)} x_i
    for a in all_alleles:
        presenting_peptides = [x[i] for i in active_indices if binds[(i, a)] == 1]
        if presenting_peptides:
            prob += z[a] <= pulp.lpSum(presenting_peptides), f"Linkage_{a}"
        else:
            prob += z[a] == 0, f"Linkage_Zero_{a}"

    # Constraint 3: Required alleles
    if required_alleles:
        for a in required_alleles:
            if a in z:
                prob += z[a] == 1, f"Required_{a}"

    # Solve
    solver_cmd = solver or pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver_cmd)

    status = pulp.LpStatus[prob.status]

    # Extract solution
    selected_idx = [i for i in active_indices if pulp.value(x[i]) and pulp.value(x[i]) > 0.5]
    selected_df = df.loc[selected_idx].copy().reset_index(drop=True)

    # Determine covered alleles and counts
    allele_counts: dict[str, int] = {}
    for a in all_alleles:
        cnt = sum(binds[(i, a)] for i in selected_idx)
        allele_counts[a] = cnt

    covered_alleles = [a for a, cnt in allele_counts.items() if cnt > 0]
    pop_cov = compute_population_coverage(covered_alleles, freqs, pops)

    mean_lb = float(selected_df["lower_bound"].mean()) if not selected_df.empty else 0.0
    mean_sc = (
        float(selected_df["immunogenicity_score"].mean()) if not selected_df.empty else 0.0
    )
    obj_val = float(pulp.value(prob.objective)) if prob.objective is not None else 0.0

    return CocktailResult(
        selected_peptides=selected_df,
        population_coverage=pop_cov,
        allele_coverage=allele_counts,
        mean_conformal_lower_bound=mean_lb,
        mean_score=mean_sc,
        objective_value=obj_val,
        status=status,
        panel_ceilings=PANEL_CEILINGS,
    )
