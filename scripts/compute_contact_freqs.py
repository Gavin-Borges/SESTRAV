"""Compute per-allele TCR contact frequencies from MHC-I + TCR complex PDB files.

Reads PDB structures of pMHC-I:TCR complexes (from ATLAS tar.gz or STCRDab),
identifies the peptide chain by length, collects Cb coordinates from all non-MHC
non-peptide chains (assumed TCR), and computes the fraction of structures in which
each peptide position p4-p8 places its Cb atom within --cutoff Angstroms of any
TCR Cb atom.

Output replaces the literature-informed priors in ALLELE_CONTACT_WEIGHTS
(src/features.py) with structure-derived values.

Input requirements:
  --pdb-dir    : directory of .pdb / .ent files
  --allele-map : optional CSV (pdb_id,hla_allele) for structures without
                 REMARK 950 allele annotation; STCRDab exports include REMARK 950
  --cutoff     : Cb-Cb distance threshold in Angstroms (default 8.0 A;
                 standard in TCR-pMHC contact literature)
  --output     : output CSV path

Output CSV columns: allele, position, contact_freq, n_contacts, n_structures

Usage (after downloading ATLAS or STCRDab PDB corpus):
  python scripts/compute_contact_freqs.py \\
      --pdb-dir data/stcrdab_structures/ \\
      --output data/contact_freqs/per_allele_contact_freq.csv

Update ALLELE_CONTACT_WEIGHTS in src/features.py with the computed values
for each allele in the canonical-10 panel.
"""

import argparse
import csv
import pathlib
import sys
from collections import defaultdict

try:
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import is_aa
except ImportError:
    sys.exit("BioPython is required: pip install biopython>=1.87")


PEPTIDE_MIN_LEN = 8
PEPTIDE_MAX_LEN = 11
DEFAULT_CUTOFF = 8.0
POSITIONS = ['p4', 'p5', 'p6', 'p7', 'p8']


def _cb_coord(residue):
    """Return Cb coordinate; falls back to Ca for Gly (no Cb)."""
    if residue.get_resname() == 'GLY':
        atom = residue.get('CA')
    else:
        atom = residue.get('CB')
    return atom.get_vector() if atom is not None else None


def _tcr_position_index(pep_len: int, pos: str) -> int:
    """Map p4/p5/p6/p7/p8 to 0-based peptide index (mirrors src/features.py)."""
    mapping = {'p4': 3, 'p5': 4, 'p6': 5, 'p7': pep_len - 3, 'p8': pep_len - 2}
    return mapping.get(pos, -1)


def _peptide_chain(model):
    """Return the chain whose AA length falls in [PEPTIDE_MIN_LEN, PEPTIDE_MAX_LEN].

    If multiple candidates exist, prefer the shortest (most likely the peptide).
    Returns None when no peptide-length chain is found.
    """
    candidates = [
        (len([r for r in ch if is_aa(r, standard=True)]), ch)
        for ch in model
        if PEPTIDE_MIN_LEN <= len([r for r in ch if is_aa(r, standard=True)]) <= PEPTIDE_MAX_LEN
    ]
    return min(candidates, key=lambda x: x[0])[1] if candidates else None


def _tcr_cb_coords(model, pep_chain_id: str) -> list:
    """Collect all Cb coordinates from chains that are not the peptide chain.

    Excludes the peptide chain only; MHC and TCR chains are all included.
    False-positive contacts from MHC side-chains are rare at the 8 A cutoff
    because MHC groove residues face inward, away from the exposed peptide face.
    For precision, a future iteration can restrict to CDR loop IMGT ranges.
    """
    coords = []
    for chain in model:
        if chain.id == pep_chain_id:
            continue
        for res in chain:
            if is_aa(res, standard=True):
                coord = _cb_coord(res)
                if coord is not None:
                    coords.append(coord)
    return coords


def _contacts_for_pdb(pdb_path: pathlib.Path, cutoff: float) -> dict | None:
    """Return {pos_name: bool} contact map for a single PDB structure.

    Returns None when the structure cannot be parsed or lacks a peptide chain.
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    except Exception:
        return None

    model = structure[0]
    pep_chain = _peptide_chain(model)
    if pep_chain is None:
        return None

    pep_residues = [r for r in pep_chain if is_aa(r, standard=True)]
    pep_len = len(pep_residues)

    tcr_coords = _tcr_cb_coords(model, pep_chain.id)
    if not tcr_coords:
        return None

    contacts = {}
    for pos in POSITIONS:
        idx = _tcr_position_index(pep_len, pos)
        if idx < 0 or idx >= pep_len:
            contacts[pos] = False
            continue
        pep_coord = _cb_coord(pep_residues[idx])
        if pep_coord is None:
            contacts[pos] = False
            continue
        contacts[pos] = any((pep_coord - tc).norm() <= cutoff for tc in tcr_coords)

    return contacts


def _allele_from_remark(pdb_path: pathlib.Path) -> str | None:
    """Extract HLA allele from REMARK 950 (STCRDab format) or return None."""
    try:
        with open(pdb_path, encoding='ascii', errors='ignore') as fh:
            for line in fh:
                if line.startswith('REMARK 950') and 'HLA' in line:
                    for token in line.split():
                        if token.startswith('HLA-'):
                            return token.strip()
    except OSError:
        pass
    return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--pdb-dir', required=True,
                    help='Directory of .pdb or .ent structure files')
    ap.add_argument('--allele-map',
                    help='CSV with columns pdb_id,hla_allele for explicit allele assignment')
    ap.add_argument('--cutoff', type=float, default=DEFAULT_CUTOFF,
                    help=f'Cb-Cb contact cutoff in Angstroms (default {DEFAULT_CUTOFF})')
    ap.add_argument('--output', default='data/contact_freqs/per_allele_contact_freq.csv',
                    help='Output CSV path')
    args = ap.parse_args()

    pdb_dir = pathlib.Path(args.pdb_dir)
    if not pdb_dir.is_dir():
        sys.exit(f"PDB directory not found: {pdb_dir}")

    pdb_files = sorted(pdb_dir.glob('*.pdb')) + sorted(pdb_dir.glob('*.ent'))
    if not pdb_files:
        sys.exit(f"No .pdb or .ent files found in {pdb_dir}")

    allele_map: dict[str, str] = {}
    if args.allele_map:
        with open(args.allele_map, newline='') as fh:
            for row in csv.DictReader(fh):
                allele_map[row['pdb_id'].lower()] = row['hla_allele']

    # contact_counts[allele][pos] = [n_contacts, n_total]
    contact_counts: dict[str, dict[str, list]] = defaultdict(
        lambda: {pos: [0, 0] for pos in POSITIONS}
    )
    n_parsed = n_skipped = 0

    for pdb_path in pdb_files:
        allele = allele_map.get(pdb_path.stem.lower()) or _allele_from_remark(pdb_path)
        if allele is None:
            n_skipped += 1
            continue

        contacts = _contacts_for_pdb(pdb_path, args.cutoff)
        if contacts is None:
            n_skipped += 1
            continue

        for pos, in_contact in contacts.items():
            contact_counts[allele][pos][1] += 1
            if in_contact:
                contact_counts[allele][pos][0] += 1
        n_parsed += 1

    print(f"Parsed {n_parsed} structures; {n_skipped} skipped (no allele or no peptide chain)")
    if n_parsed == 0:
        sys.exit("No structures parsed. Verify --pdb-dir contents and --allele-map.")

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for allele in sorted(contact_counts):
        for pos in POSITIONS:
            n_con, n_tot = contact_counts[allele][pos]
            rows.append({
                'allele': allele,
                'position': pos,
                'contact_freq': round(n_con / n_tot, 4) if n_tot else 0.0,
                'n_contacts': n_con,
                'n_structures': n_tot,
            })

    fieldnames = ['allele', 'position', 'contact_freq', 'n_contacts', 'n_structures']
    with open(out_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")
    print(f"Alleles: {sorted(contact_counts)}")


if __name__ == '__main__':
    main()
