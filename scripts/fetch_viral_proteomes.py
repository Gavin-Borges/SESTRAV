"""
Download authoritative UniProt FASTA sequences for HBV and HCV proteome panels.

Writes to data/proteomes/ following the same naming convention used by the
existing HPV and EBV panels.  Run once before using HBV/HCV configs:

    python scripts/fetch_viral_proteomes.py [--output-dir data/proteomes]

UniProt accessions are pinned to reviewed (Swiss-Prot) entries and logged to
a provenance JSON alongside each FASTA so reruns can confirm the source.
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone

UNIPROT_FASTA_URL = "https://rest.uniprot.org/uniprotkb/{accession}.fasta"

# ---------------------------------------------------------------------------
# Panel definitions — each entry is (uniprot_accession, protein_short_name)
# ---------------------------------------------------------------------------

HBV_AWY_PANEL4 = [
    ("P03147", "HBcAg"),      # Core antigen (capsid), genotype A ayw
    ("P03165", "HBx"),        # X protein (transcriptional transactivator), genotype D subtype ayw
    ("P03138", "HBsAg_S"),    # Small surface antigen (a-determinant bearer)
    ("P03157", "HBpol"),      # Polymerase / reverse transcriptase
]

HCV_1A_PANEL4 = [
    ("P26664", "Core"),       # Genome polyprotein, genotype 1a (H77); Core/NS regions are contiguous
    ("O92972", "NS3"),        # Genome polyprotein, genotype 1b (HC-J4); best-reviewed NS3 source
    ("O92975", "NS5A"),       # Genome polyprotein fragment, Hepacivirus hominis; NS5A region
    ("O92976", "NS5B"),       # Genome polyprotein fragment, Hepacivirus hominis; NS5B/polymerase region
]

PANELS = {
    "HBV_ayw_panel4": HBV_AWY_PANEL4,
    "HCV_1a_panel4": HCV_1A_PANEL4,
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
