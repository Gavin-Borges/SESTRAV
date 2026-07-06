"""
Download authoritative UniProt FASTA sequences for SESTRAV viral proteome panels.

Writes to data/proteomes/ following the same naming convention used by the
existing HPV and EBV panels.  Run once before using a new panel config:

    python scripts/fetch_viral_proteomes.py [--output-dir data/proteomes]
    python scripts/fetch_viral_proteomes.py --panels HIV1_HXB2_panel4 SARSCOV2_wuhan1_panel4

UniProt accessions are pinned to reviewed (Swiss-Prot) entries and logged to
a provenance JSON alongside each FASTA so reruns can confirm the source.

Supported panels
----------------
HBV_ayw_panel4        Hepatitis B virus genotype D (ayw) - 4 proteins
HCV_1a_panel4         Hepatitis C virus genotype 1a (H77) - 4 proteins
HIV1_HXB2_panel4      HIV-1 clade B reference strain HXB2 - 4 proteins
SARSCOV2_wuhan1_panel4  SARS-CoV-2 Wuhan-1 (Hu-1) - 4 proteins
IAV_PR8_panel4        Influenza A virus A/Puerto Rico/8/1934 (H1N1) - 4 proteins
CMV_AD169_panel4      Human cytomegalovirus strain AD169 - 4 proteins
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _ssl_fix  # noqa: F401, E402 - patch SSL before any network calls

UNIPROT_FASTA_URL = "https://rest.uniprot.org/uniprotkb/{accession}.fasta"

# ---------------------------------------------------------------------------
# Panel definitions - each entry is (uniprot_accession, protein_short_name)
# ---------------------------------------------------------------------------

HBV_AWY_PANEL4 = [
    ("P03147", "HBcAg"),  # Core antigen (capsid), genotype A ayw
    ("P03165", "HBx"),  # X protein (transcriptional transactivator), genotype D subtype ayw
    ("P03138", "HBsAg_S"),  # Small surface antigen (a-determinant bearer)
    ("P03157", "HBpol"),  # Polymerase / reverse transcriptase
]

HCV_1A_PANEL4 = [
    ("P26664", "Core"),  # Genome polyprotein, genotype 1a (H77); Core/NS regions are contiguous
    ("O92972", "NS3"),  # Genome polyprotein, genotype 1b (HC-J4); best-reviewed NS3 source
    ("O92975", "NS5A"),  # Genome polyprotein fragment, Hepacivirus hominis; NS5A region
    ("O92976", "NS5B"),  # Genome polyprotein fragment, Hepacivirus hominis; NS5B/polymerase region
]


# HIV-1 clade B reference strain HXB2 (GenBank K03455).
# Gag and Env are the dominant CD8+ T-cell targets; Pol and Nef are well-studied.
# Clade B is predominant in North America/Western Europe. Clade C (Sub-Saharan Africa,
# South Asia) shows ~10 % amino-acid divergence - predictions carry additional uncertainty
# when applied to non-clade-B populations. See antigen_accessions.md §5 for full caveats.
HIV1_HXB2_PANEL4 = [
    ("P04591", "Gag"),  # Gag polyprotein (MA, CA, p24, NC) - dominant CD8 target
    ("P04585", "Pol"),  # Pol polyprotein (PR, RT, RNase H, IN)
    ("P04601", "Nef"),  # Negative factor - accessory protein, strong CD8 target
    ("P04578", "Env"),  # Envelope glycoprotein gp160 (gp120 + gp41)
]

# SARS-CoV-2 Wuhan-1 reference (Hu-1; GenBank MN908947).
# Spike/N/M/ORF3a are the primary CD8+ T-cell targets documented in IEDB.
# Omicron BA.2+ has >30 Spike mutations that may reduce cross-reactivity with
# Wuhan-1-derived predictions. See antigen_accessions.md §6.
SARSCOV2_WUHAN1_PANEL4 = [
    ("P0DTC2", "Spike"),  # Surface glycoprotein (S / ORF2) - 1273 AA
    ("P0DTC9", "N"),  # Nucleocapsid phosphoprotein (N / ORF9) - highly conserved
    ("P0DTC5", "M"),  # Membrane glycoprotein (M / ORF5) - conserved CD8 target
    ("P0DTC3", "ORF3a"),  # Accessory protein 3a - pore-forming, immunogenic
]

# Influenza A virus A/Puerto Rico/8/1934 (PR8; H1N1).
# NP and M1 are cross-strain conserved - the basis for universal flu vaccine proposals.
# HA is strain-specific and will not generalize to H3N2 or avian strains without retraining.
# PB1-F2 is 90 AA (PR8); produces few peptides but is a well-documented CD8 target
# associated with viral pathogenicity. See antigen_accessions.md §7.
IAV_PR8_PANEL4 = [
    ("P03466", "NP"),  # Nucleoprotein - dominant, cross-strain conserved CD8 target
    ("P03485", "M1"),  # Matrix protein 1 - highly conserved CD8 target
    ("P03437", "HA"),  # Hemagglutinin - strain-specific (H1N1 subtype only)
    ("P0C0U1", "PB1F2"),  # PB1-F2 - pathogenicity factor; 90 AA in PR8 strain
]

# Human cytomegalovirus (HCMV) strain AD169.
# pp65 (UL83) is immunodominant in CMV-seropositive individuals; it drives the
# CMV-specific CD8+ T-cell pool that can reach 10-20 % of total CD8+ T-cells.
# IE1 (UL123) is the dominant target early in infection (lytic phase).
# See antigen_accessions.md §8.
CMV_AD169_PANEL4 = [
    ("P06725", "pp65"),  # UL83 lower matrix phosphoprotein - immunodominant CD8 target
    ("P13202", "IE1"),  # UL123 immediate-early antigen 1 - dominant lytic target
    ("P16785", "pp50"),  # UL44 DNA polymerase processivity factor
    ("P06473", "gB"),  # UL55 envelope glycoprotein B - CD8 target in primary infection
]

# Dengue virus serotype 2 New Guinea C (DENV-2 NGC) reference strain.
# NGC is the most-sequenced DENV-2 strain in the T-cell literature; polyprotein
# P29991 encodes all structural (C/prM/E) and non-structural (NS1-NS5) proteins.
# DENV-2 and DENV-3 share ~70% amino-acid identity; NS3/NS5 are most conserved
# across serotypes. Per-serotype divergence (~30%) means predictions carry
# additional uncertainty when applied to DENV-1/3/4 epitopes.
DENV2_NGC_PANEL1 = [
    (
        "P29991",
        "Polyprotein",
    ),  # Full DENV-2 NGC polyprotein (C/prM/E/NS1/NS2A/NS2B/NS3/NS4A/NS4B/NS5)
]

PANELS = {
    "HBV_ayw_panel4": HBV_AWY_PANEL4,
    "HCV_1a_panel4": HCV_1A_PANEL4,
    "HIV1_HXB2_panel4": HIV1_HXB2_PANEL4,
    "SARSCOV2_wuhan1_panel4": SARSCOV2_WUHAN1_PANEL4,
    "IAV_PR8_panel4": IAV_PR8_PANEL4,
    "CMV_AD169_panel4": CMV_AD169_PANEL4,
    "DENV2_NGC_panel1": DENV2_NGC_PANEL1,
}


def fetch_fasta(accession: str, retries: int = 3, delay: float = 1.0) -> str:
    url = UNIPROT_FASTA_URL.format(accession=accession)
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:  # nosec B310
                return resp.read().decode("utf-8")
        except Exception as exc:
            print(f"  Attempt {attempt}/{retries} failed for {accession}: {exc}")
            if attempt < retries:
                time.sleep(delay * attempt)
    raise RuntimeError(f"Could not fetch {accession} after {retries} attempts.")


def build_panel_fasta(panel: list, panel_name: str, output_dir: str) -> str:
    """Download all proteins for a panel, concatenate into one FASTA file."""
    sequences = []
    sources = []
    for accession, short_name in panel:
        print(f"  Fetching {accession} ({short_name})...")
        fasta = fetch_fasta(accession)
        # Ensure a blank line between entries
        sequences.append(fasta.rstrip("\n"))
        sources.append({"accession": accession, "name": short_name})

    out_path = os.path.join(output_dir, f"{panel_name}.fasta")
    with open(out_path, "w", newline="\n") as fh:
        fh.write("\n".join(sequences) + "\n")
    print(f"  Written: {out_path}")

    prov_path = os.path.join(output_dir, f"{panel_name}_provenance.json")
    provenance = {
        "panel": panel_name,
        "sources": sources,
        "fetched_utc": datetime.now(timezone.utc).isoformat(),
        "uniprot_base_url": UNIPROT_FASTA_URL,
    }
    with open(prov_path, "w") as fh:
        json.dump(provenance, fh, indent=2)
    print(f"  Provenance: {prov_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="data/proteomes",
        help="Directory to write FASTA and provenance files (default: data/proteomes)",
    )
    parser.add_argument(
        "--panels",
        nargs="+",
        choices=list(PANELS.keys()),
        default=list(PANELS.keys()),
        help="Which panels to fetch (default: all)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for panel_name in args.panels:
        print(f"\nFetching panel: {panel_name}")
        try:
            build_panel_fasta(PANELS[panel_name], panel_name, args.output_dir)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)

    print("\nAll panels fetched successfully.")


if __name__ == "__main__":
    main()
