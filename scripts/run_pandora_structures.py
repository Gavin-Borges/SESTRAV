"""Run PANDORA pHLA-I structure modelling for a set of peptide+allele pairs.

Generates Cb-Cb distance tensors (.pt files) for use as structural graph edges
in GNN v2.5 (build_pyg_structural_graph) and as SASA/torsion feature inputs
for RF modes 37/39 (M11).

SETUP REQUIRED BEFORE RUNNING:
  1. Obtain a MODELLER license (free for academic use):
       https://salilab.org/modeller/registration.html
  2. Install MODELLER via conda:
       conda install -c salilab modeller
  3. Install PANDORA and dependencies:
       pip install pandora pdb2sql mhctools
  4. Download MHC template database used by PANDORA:
       python -c "import pandora; pandora.fetch_templates()"

Usage:
  python scripts/run_pandora_structures.py \\
      --input  data/immunogenicity_dataset_v5.csv \\
      --alleles HLA-A*02:01 HLA-B*07:02 \\
      --n-peptides 50 \\
      --output-dir data/structural_cache/ \\
      --seed 42

  Full v5 run (25,386 unique peptides):
      Omit --n-peptides; set --alleles to all 10 canonical alleles.
      Expected wall-time per peptide: document during --n-peptides 50 test run.

Output:
  data/structural_cache/{peptide}_{allele_key}_dist.pt  (L x L float32 tensor)
  data/structural_cache/run_summary.json                (timing + success counts)
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch

try:
    from pandora import Pandora
    from pandora.peptide import Peptide
except ImportError:
    sys.exit(
        "PANDORA is not installed. Follow the SETUP instructions in this script's "
        "docstring to install MODELLER (license required), PANDORA, pdb2sql, and mhctools."
    )

CANONICAL_10_ALLELES = [
    'HLA-A*01:01', 'HLA-A*02:01', 'HLA-A*03:01', 'HLA-A*11:01', 'HLA-A*24:02',
    'HLA-B*07:02', 'HLA-B*08:01', 'HLA-B*27:05', 'HLA-B*35:01', 'HLA-B*44:02',
]


def _allele_key(allele: str) -> str:
    """Convert 'HLA-A*02:01' -> 'A0201' for use in filenames."""
    return allele.replace('HLA-', '').replace('*', '').replace(':', '')


def _cb_cb_distance_matrix(structure_path: pathlib.Path) -> torch.Tensor | None:
    """Extract Cb-Cb (Gly: Ca-Ca) distance matrix for the peptide chain.

    Returns an (L, L) float32 tensor or None on failure.
    """
    try:
        from Bio.PDB import PDBParser
        from Bio.PDB.Polypeptide import is_aa
    except ImportError:
        sys.exit("BioPython is required: pip install biopython>=1.87")

    parser = PDBParser(QUIET=True)
    try:
        struct = parser.get_structure(structure_path.stem, str(structure_path))
    except Exception:
        return None

    model = struct[0]
    # PANDORA names the peptide chain 'C' by convention
    pep_chain = model.get('C')
    if pep_chain is None:
        # Fallback: shortest chain of AA length 8-11
        candidates = [
            (len([r for r in ch if is_aa(r, standard=True)]), ch)
            for ch in model
            if 8 <= len([r for r in ch if is_aa(r, standard=True)]) <= 11
        ]
        if not candidates:
            return None
        pep_chain = min(candidates, key=lambda x: x[0])[1]

    residues = [r for r in pep_chain if is_aa(r, standard=True)]
    if not residues:
        return None

    coord_list: list[np.ndarray] = []
    for res in residues:
        if res.get_resname() == 'GLY':
            atom = res.get('CA')
        else:
            atom = res.get('CB')
        if atom is None:
            atom = res.get('CA')
        coord_list.append(atom.get_vector().get_array() if atom else np.zeros(3))

    coords: np.ndarray = np.array(coord_list, dtype=np.float32)  # (L, 3)
    diff: np.ndarray = coords[:, None, :] - coords[None, :, :]   # (L, L, 3)
    dist: np.ndarray = np.sqrt((diff ** 2).sum(axis=-1))         # (L, L)
    return torch.from_numpy(dist)


def run_pandora_for_pair(peptide_seq: str, allele: str, out_dir: pathlib.Path) -> bool:
    """Model one peptide+allele pair and write the distance tensor.

    Returns True on success.
    """
    allele_key = _allele_key(allele)
    out_path = out_dir / f"{peptide_seq}_{allele_key}_dist.pt"
    if out_path.exists():
        return True  # already computed

    try:
        p = Peptide(peptide_seq, mhc_allele=allele)
        pand = Pandora(p)
        pand.run()
        # PANDORA writes the top model as {peptide}.B99990001.pdb in the working dir
        pdb_candidates = list(pathlib.Path('.').glob(f'*{peptide_seq}*.pdb'))
        if not pdb_candidates:
            return False
        pdb_path = pdb_candidates[0]
        dist = _cb_cb_distance_matrix(pdb_path)
        if dist is None:
            return False
        torch.save(dist, out_path)  # nosec B614 - saving our own computed ndarray, not loading untrusted data
        pdb_path.unlink(missing_ok=True)  # clean up PANDORA output
        return True
    except Exception:
        return False


def select_representative_peptides(
    dataset_path: pathlib.Path, n: int, seed: int
) -> pd.DataFrame:
    """Sample n peptides from v5 dataset, stratified by length (8/9/10/11-mer)."""
    df = pd.read_csv(dataset_path, low_memory=False)
    if 'is_quarantined' in df.columns:
        df = df[df['is_quarantined'] == False]  # noqa: E712
    df['pep_len'] = df['peptide'].str.len()
    df = df[df['pep_len'].between(8, 11)].drop_duplicates(subset='peptide')

    rng = np.random.default_rng(seed)
    per_len = max(1, n // 4)
    rows = []
    for length in [8, 9, 10, 11]:
        pool = df[df['pep_len'] == length]
        k = min(per_len, len(pool))
        rows.append(pool.sample(n=k, random_state=int(rng.integers(10**6))))
    return pd.concat(rows).head(n)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--input', default='data/immunogenicity_dataset_v5.csv',
                    help='v5 dataset CSV (default: data/immunogenicity_dataset_v5.csv)')
    ap.add_argument('--alleles', nargs='+', default=['HLA-A*02:01', 'HLA-B*07:02'],
                    help='Alleles to model (default: HLA-A*02:01 HLA-B*07:02)')
    ap.add_argument('--n-peptides', type=int, default=None,
                    help='Limit to N peptides per allele (omit for full v5 run)')
    ap.add_argument('--output-dir', default='data/structural_cache/',
                    help='Output directory for .pt distance tensors')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = pathlib.Path(args.input)
    if not dataset_path.exists():
        sys.exit(f"Dataset not found: {dataset_path}")

    if args.n_peptides is not None:
        peptide_df = select_representative_peptides(dataset_path, args.n_peptides, args.seed)
        peptides = peptide_df['peptide'].tolist()
        print(f"Selected {len(peptides)} representative peptides "
              f"(lengths: {peptide_df['pep_len'].value_counts().to_dict()})")
    else:
        df = pd.read_csv(dataset_path, low_memory=False)
        if 'is_quarantined' in df.columns:
            df = df[df['is_quarantined'] == False]  # noqa: E712
        peptides = df['peptide'].drop_duplicates().tolist()
        print(f"Full v5 run: {len(peptides)} unique peptides")

    summary: dict = {'n_peptides': len(peptides), 'alleles': args.alleles,
                     'cutoff_ang': 8.0, 'pairs': {}}
    total_wall = 0.0

    for allele in args.alleles:
        allele_key = _allele_key(allele)
        n_ok = n_fail = 0
        allele_times = []

        for pep in peptides:
            t0 = time.perf_counter()
            ok = run_pandora_for_pair(pep, allele, out_dir)
            elapsed = time.perf_counter() - t0
            if ok:
                n_ok += 1
                allele_times.append(elapsed)
            else:
                n_fail += 1

        mean_t = float(np.mean(allele_times)) if allele_times else 0.0
        total_wall += mean_t * len(peptides)
        summary['pairs'][allele_key] = {
            'n_ok': n_ok, 'n_fail': n_fail,
            'mean_sec_per_peptide': round(mean_t, 2),
        }
        print(f"  {allele}: {n_ok} ok / {n_fail} fail, "
              f"mean {mean_t:.1f}s/peptide")

    full_v5_peptides = 25386
    extrap_total_sec = mean_t * full_v5_peptides * len(args.alleles) if allele_times else None
    if extrap_total_sec:
        extrap_days = extrap_total_sec / 86400
        extrap_cpu_h = extrap_total_sec / 3600
        summary['extrapolation'] = {
            'full_v5_peptides': full_v5_peptides,
            'n_alleles': len(args.alleles),
            'estimated_total_days_single_cpu': round(extrap_days, 1),
            'estimated_cpu_hours': round(extrap_cpu_h, 1),
            'feasible': extrap_days <= 500 and extrap_cpu_h <= 2100,
        }
        print(f"\nExtrapolation to full v5 ({full_v5_peptides} peptides x {len(args.alleles)} alleles):")
        print(f"  ~{extrap_days:.0f} calendar days single-CPU")
        print(f"  ~{extrap_cpu_h:.0f} CPU-hours")
        if not summary['extrapolation']['feasible']:
            print("  STATUS: COMPUTE INFEASIBLE pre-paper (>500 days or >2100 CPU-hours)")
            print("  -> M11/M12 move to future work in paper Section 5")
        else:
            print("  STATUS: feasible - proceed with full v5 run")

    run_summary_path = out_dir / 'run_summary.json'
    with open(run_summary_path, 'w') as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSummary written to {run_summary_path}")


if __name__ == '__main__':
    main()
