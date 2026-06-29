"""
compute_population_coverage.py - HLA population coverage for the SESTRAV v5 panel.

Computes the fraction of each WHO super-population predicted to present at least
one peptide via the 10-allele MHCflurry Class I panel, using the
Hardy-Weinberg equilibrium independence model.

Coverage method (Lundegaard et al. 2010; also implemented in the IEDB
Population Coverage Tool, Bui et al. 2006 BMC Bioinformatics 7:153):

    phenotype_freq_i = 1 - (1 - haplotype_freq_i)^2
    panel_coverage   = 1 - product_i(1 - phenotype_freq_i)

This is the "at least one allele recognised" probability under HWE and
allele-independence assumptions.  Because (1 - phenotype_freq_i) = (1-h_i)^2,
the panel coverage simplifies to:

    coverage = 1 - (product_i(1 - haplotype_freq_i))^2

Frequency data source:
    Gonzalez-Galarza FF, McCabe A, Santos EJMD, Jones J, Takeshita L,
    Ortega-Rivera ND, Cid-Pavon GMD, Ramsbottom K, Ghattaoraya G,
    Alfirevic A, Middleton D, Jones AR.
    Allele frequency net database (AFND) 2020 update: gold-standard data
    classification, open access genotype data and new query tools.
    Nucleic Acids Res. 2020;48(D1):D783-D788.
    https://www.allelefrequencies.net

    Per-allele haplotype frequencies are aggregated medians from AFND
    gold-standard (>=100 individuals, >4-digit resolution) studies for each
    WHO super-population.  Values are approximate (+/-0.01-0.02) due to
    within-group heterogeneity; they are adequate for order-of-magnitude
    coverage estimation and consistent with published vaccine design analyses
    (e.g. Cao et al. 2001 Tissue Antigens 58:339; Mack et al. 2007
    Eur J Immunogenet).

Panel alleles (feature column -> HLA notation):
    bind_A0101 -> HLA-A*01:01
    bind_A0201 -> HLA-A*02:01
    bind_A0301 -> HLA-A*03:01
    bind_A1101 -> HLA-A*11:01
    bind_A2402 -> HLA-A*24:02
    bind_B0702 -> HLA-B*07:02
    bind_B0801 -> HLA-B*08:01
    bind_B2705 -> HLA-B*27:05
    bind_B3501 -> HLA-B*35:01
    bind_B4402 -> HLA-B*44:02

WHO super-populations:
    AFR - African
    AMR - American (admixed Americas)
    EAS - East Asian
    EUR - European
    SAS - South Asian

Usage:
    python scripts/compute_population_coverage.py
    python scripts/compute_population_coverage.py --output results/population_coverage_v5.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POPULATIONS: list[str] = ["AFR", "AMR", "EAS", "EUR", "SAS"]

POPULATION_LABELS: dict[str, str] = {
    "AFR": "African",
    "AMR": "American",
    "EAS": "East Asian",
    "EUR": "European",
    "SAS": "South Asian",
}

# HLA haplotype (= allele) frequencies per WHO super-population.
#
# Source: AFND 2020 (Gonzalez-Galarza et al., Nucleic Acids Res 2020).
#   Values are aggregated medians from gold-standard AFND studies per
#   super-population region.  Within-group heterogeneity is +/-0.01-0.02
#   for most alleles; EAS A*11:01 and A*24:02 have tighter agreement.
#
# Notes on specific alleles:
#   A*02:01 (EAS=0.040): In East Asia the dominant A2 subtypes are A*02:07,
#     A*02:06, and A*02:03; A*02:01 itself is uncommon (~4%).
#   A*11:01 (EAS=0.238, SAS=0.180): Exceptionally common in East and South
#     Asia; this is the main driver of EAS panel coverage.
#   A*24:02 (EAS=0.210): Very common in Japan and Taiwan; second-highest
#     EAS driver.
#   B*08:01 (EAS=0.005): Essentially absent from East Asian populations.
#   B*27:05 (AFR=0.011): Rare in Sub-Saharan Africa; most B27 in Africa
#     is B*27:03 or B*27:04.
#
# Reference haplotype frequencies used:
#
#   Allele         AFR    AMR    EAS    EUR    SAS
#   HLA-A*01:01   0.065  0.082  0.015  0.165  0.105
#   HLA-A*02:01   0.090  0.180  0.040  0.270  0.150
#   HLA-A*03:01   0.070  0.075  0.008  0.135  0.080
#   HLA-A*11:01   0.040  0.060  0.238  0.055  0.180
#   HLA-A*24:02   0.035  0.120  0.210  0.085  0.090
#   HLA-B*07:02   0.055  0.078  0.035  0.120  0.078
#   HLA-B*08:01   0.030  0.050  0.005  0.108  0.062
#   HLA-B*27:05   0.011  0.026  0.012  0.052  0.039
#   HLA-B*35:01   0.050  0.076  0.038  0.075  0.057
#   HLA-B*44:02   0.025  0.048  0.015  0.095  0.045

ALLELE_HAPLOTYPE_FREQUENCIES: dict[str, dict[str, float]] = {
    "HLA-A*01:01": {
        "AFR": 0.065,
        "AMR": 0.082,
        "EAS": 0.015,
        "EUR": 0.165,
        "SAS": 0.105,
    },
    "HLA-A*02:01": {
        "AFR": 0.090,
        "AMR": 0.180,
        "EAS": 0.040,
        "EUR": 0.270,
        "SAS": 0.150,
    },
    "HLA-A*03:01": {
        "AFR": 0.070,
        "AMR": 0.075,
        "EAS": 0.008,
        "EUR": 0.135,
        "SAS": 0.080,
    },
    "HLA-A*11:01": {
        "AFR": 0.040,
        "AMR": 0.060,
        "EAS": 0.238,
        "EUR": 0.055,
        "SAS": 0.180,
    },
    "HLA-A*24:02": {
        "AFR": 0.035,
        "AMR": 0.120,
        "EAS": 0.210,
        "EUR": 0.085,
        "SAS": 0.090,
    },
    "HLA-B*07:02": {
        "AFR": 0.055,
        "AMR": 0.078,
        "EAS": 0.035,
        "EUR": 0.120,
        "SAS": 0.078,
    },
    "HLA-B*08:01": {
        "AFR": 0.030,
        "AMR": 0.050,
        "EAS": 0.005,
        "EUR": 0.108,
        "SAS": 0.062,
    },
    "HLA-B*27:05": {
        "AFR": 0.011,
        "AMR": 0.026,
        "EAS": 0.012,
        "EUR": 0.052,
        "SAS": 0.039,
    },
    "HLA-B*35:01": {
        "AFR": 0.050,
        "AMR": 0.076,
        "EAS": 0.038,
        "EUR": 0.075,
        "SAS": 0.057,
    },
    "HLA-B*44:02": {
        "AFR": 0.025,
        "AMR": 0.048,
        "EAS": 0.015,
        "EUR": 0.095,
        "SAS": 0.045,
    },
}

CITATION_SOURCE: str = (
    "Gonzalez-Galarza FF et al. Allele frequency net database (AFND) 2020 update. "
    "Nucleic Acids Res. 2020;48(D1):D783-D788. https://www.allelefrequencies.net"
)

CITATION_METHOD: str = (
    "Lundegaard et al. 2010; IEDB Population Coverage Tool (Bui et al. 2006 "
    "BMC Bioinformatics 7:153). "
    "Formula: phenotype_freq = 1-(1-h)^2; "
    "panel_coverage = 1-product_i(1-phenotype_freq_i)."
)

# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def phenotype_frequency(haplotype_freq: float) -> float:
    """Return the phenotype frequency (probability of carrying >= 1 copy).

    Under Hardy-Weinberg equilibrium:
        p(phenotype) = 1 - (1 - h)^2
    where h is the haplotype (allele) frequency.
    """
    return 1.0 - (1.0 - haplotype_freq) ** 2


def panel_coverage(haplotype_freqs: list[float]) -> float:
    """Return the population coverage for a set of alleles.

    Assumes HWE and allele independence across loci:
        coverage = 1 - product_i((1 - h_i)^2)
                 = 1 - (product_i(1 - h_i))^2

    Parameters
    ----------
    haplotype_freqs:
        List of per-allele haplotype frequencies for a single population.
    """
    complement_product = math.prod(1.0 - h for h in haplotype_freqs)
    return 1.0 - complement_product ** 2


def compute_all_coverage(
    freq_table: dict[str, dict[str, float]],
    populations: list[str],
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
    dict[str, float],
]:
    """Compute per-allele and panel coverage for every population.

    Returns
    -------
    per_allele_cov : dict[allele, dict[pop, phenotype_freq]]
    panel_cov      : dict[pop, coverage]
    global_mean    : dict with single key "global_mean"
    """
    alleles = list(freq_table.keys())

    per_allele_cov: dict[str, dict[str, float]] = {}
    for allele in alleles:
        per_allele_cov[allele] = {}
        for pop in populations:
            h = freq_table[allele][pop]
            per_allele_cov[allele][pop] = phenotype_frequency(h)

    panel_cov: dict[str, float] = {}
    for pop in populations:
        h_list = [freq_table[allele][pop] for allele in alleles]
        panel_cov[pop] = panel_coverage(h_list)

    global_mean = sum(panel_cov[p] for p in populations) / len(populations)

    return per_allele_cov, panel_cov, global_mean


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def _best_allele(
    pop: str,
    per_allele_cov: dict[str, dict[str, float]],
) -> tuple[str, float]:
    """Return (allele_name, phenotype_freq) for the highest-coverage allele."""
    best_allele = max(per_allele_cov, key=lambda a: per_allele_cov[a][pop])
    return best_allele, per_allele_cov[best_allele][pop]


def _n_above_threshold(
    pop: str,
    per_allele_cov: dict[str, dict[str, float]],
    threshold: float = 0.05,
) -> int:
    """Count alleles with phenotype_freq > threshold in a given population."""
    return sum(
        1 for pf in per_allele_cov.values() if pf[pop] > threshold
    )


def print_markdown_table(
    populations: list[str],
    panel_cov: dict[str, float],
    per_allele_cov: dict[str, dict[str, float]],
    global_mean: float,
) -> None:
    """Print a markdown summary table suitable for inclusion in the paper."""
    header = (
        "| Population | Individuals covered (%) | "
        "Alleles covering >5% of pop. | Best single allele |"
    )
    sep = "|---|---|---|---|"
    print()
    print("## SESTRAV v5 Panel - HLA Population Coverage")
    print()
    print(header)
    print(sep)

    for pop in populations:
        label = POPULATION_LABELS[pop]
        cov_pct = panel_cov[pop] * 100.0
        n_above = _n_above_threshold(pop, per_allele_cov, threshold=0.05)
        best_a, best_pf = _best_allele(pop, per_allele_cov)
        best_pct = best_pf * 100.0
        print(
            f"| {label} ({pop}) "
            f"| {cov_pct:.1f} "
            f"| {n_above}/10 "
            f"| {best_a} ({best_pct:.1f}%) |"
        )

    global_pct = global_mean * 100.0
    print(f"| **Global mean** | **{global_pct:.1f}** | - | - |")
    print()
    print(
        "Coverage method: "
        "Lundegaard et al. 2010 / IEDB Population Coverage Tool. "
        "Frequencies: AFND 2020 (Gonzalez-Galarza et al.)."
    )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_output(
    per_allele_cov: dict[str, dict[str, float]],
    panel_cov: dict[str, float],
    global_mean: float,
) -> dict:
    """Assemble the JSON output structure."""
    panel_cov_out = dict(panel_cov)
    panel_cov_out["global_mean"] = global_mean

    return {
        "meta": {
            "source": CITATION_SOURCE,
            "method": CITATION_METHOD,
            "date": "2026-06-28",
            "panel_size": len(ALLELE_HAPLOTYPE_FREQUENCIES),
            "populations": POPULATIONS,
        },
        "allele_haplotype_frequencies": ALLELE_HAPLOTYPE_FREQUENCIES,
        "per_allele_coverage": per_allele_cov,
        "panel_coverage": panel_cov_out,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute HLA population coverage for the SESTRAV v5 panel"
    )
    parser.add_argument(
        "--output",
        default="results/population_coverage_v5.json",
        help="Output JSON path (default: results/population_coverage_v5.json)",
    )
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Suppress markdown table output to stdout",
    )
    args = parser.parse_args(argv)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / args.output

    per_allele_cov, panel_cov, global_mean = compute_all_coverage(
        ALLELE_HAPLOTYPE_FREQUENCIES, POPULATIONS
    )

    result = build_output(per_allele_cov, panel_cov, global_mean)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    print(f"Coverage JSON written: {output_path}")

    # Per-population summary to stdout
    print()
    for pop in POPULATIONS:
        label = POPULATION_LABELS[pop]
        cov_pct = panel_cov[pop] * 100.0
        print(f"  {label:15s} ({pop}): {cov_pct:.1f}%")
    print(f"  {'Global mean':20s}: {global_mean * 100.0:.1f}%")

    if not args.no_markdown:
        print_markdown_table(POPULATIONS, panel_cov, per_allele_cov, global_mean)

    return 0


if __name__ == "__main__":
    sys.exit(main())
