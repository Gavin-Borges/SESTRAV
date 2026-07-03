"""Update ALLELE_CONTACT_WEIGHTS in src/features.py from structure-derived contact frequencies.

Reads the output CSV produced by scripts/compute_contact_freqs.py and replaces the
literature-informed prior values in ALLELE_CONTACT_WEIGHTS with per-structure empirical
contact frequencies from TCR-pMHC-I crystal structures (TCR3d 2.0 corpus).

Alleles with fewer than --min-structures confirmed structures retain their existing
prior (printed as a warning). The update is idempotent: running twice with the same
CSV yields identical output.

Usage:
  python scripts/update_contact_weights.py \\
      --contact-csv data/contact_freqs/per_allele_contact_freq.csv \\
      --features-py src/features.py

Dry-run (print proposed changes without writing):
  python scripts/update_contact_weights.py \\
      --contact-csv data/contact_freqs/per_allele_contact_freq.csv \\
      --dry-run
"""

import argparse
import csv
import pathlib
import re
import sys


POSITIONS = ['p4', 'p5', 'p6', 'p7', 'p8']

CANONICAL_ALLELES = [
    'HLA-A*01:01',
    'HLA-A*02:01',
    'HLA-A*03:01',
    'HLA-A*11:01',
    'HLA-A*24:02',
    'HLA-B*07:02',
    'HLA-B*08:01',
    'HLA-B*27:05',
    'HLA-B*35:01',
    'HLA-B*44:02',
]

FEATURES_PY_DEFAULT = pathlib.Path('src/features.py')
CONTACT_CSV_DEFAULT = pathlib.Path('data/contact_freqs/per_allele_contact_freq.csv')


def _load_contact_csv(csv_path: pathlib.Path) -> dict[str, dict[str, tuple[float, int]]]:
    """Return {allele: {position: (contact_freq, n_structures)}} from the CSV."""
    data: dict[str, dict[str, tuple[float, int]]] = {}
    with open(csv_path, newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            allele = row['allele'].strip()
            pos = row['position'].strip()
            try:
                freq = float(row['contact_freq'])
                n_struct = int(row['n_structures'])
            except (ValueError, KeyError) as exc:
                print(f"WARNING: skipping malformed row {row!r}: {exc}", file=sys.stderr)
                continue
            if allele not in data:
                data[allele] = {}
            data[allele][pos] = (freq, n_struct)
    return data


def _build_weight_list(
    allele: str,
    pos_data: dict[str, tuple[float, int]],
    min_structures: int,
) -> list[float] | None:
    """Return [p4, p5, p6, p7, p8] weight list, or None if data is insufficient.

    Returns None when any position has n_structures < min_structures.
    """
    weights = []
    for pos in POSITIONS:
        if pos not in pos_data:
            print(
                f"WARNING: {allele} missing position {pos} in CSV - keeping prior",
                file=sys.stderr,
            )
            return None
        freq, n_struct = pos_data[pos]
        if n_struct < min_structures:
            print(
                f"WARNING: {allele} pos {pos}: only {n_struct} structures "
                f"(< {min_structures} minimum) - keeping prior",
                file=sys.stderr,
            )
            return None
        weights.append(round(freq, 4))
    return weights


def _population_avg(allele_weights: dict[str, list[float]]) -> list[float]:
    """Return mean contact frequency across all updated alleles, per position."""
    if not allele_weights:
        return []
    n = len(allele_weights)
    avg = []
    for i in range(len(POSITIONS)):
        avg.append(round(sum(v[i] for v in allele_weights.values()) / n, 4))
    return avg


def _format_weights_block(
    allele_weights: dict[str, list[float]],
    unchanged_alleles: list[str],
    existing_priors: dict[str, list[float]],
    pop_avg: list[float],
) -> str:
    """Return the Python source text for the updated ALLELE_CONTACT_WEIGHTS block."""
    lines = [
        'ALLELE_CONTACT_WEIGHTS: dict[str, list[float]] = {',
    ]
    for allele in CANONICAL_ALLELES:
        if allele in allele_weights:
            weights = allele_weights[allele]
            tag = '  # structure-derived (TCR3d)'
        elif allele in existing_priors:
            weights = existing_priors[allele]
            tag = '  # literature prior (insufficient structures)'
        else:
            continue
        weights_repr = '[' + ', '.join(f'{w:.4f}' for w in weights) + ']'
        lines.append(f"    '{allele}': {weights_repr},{tag}")
    lines.append('}')
    return '\n'.join(lines)


def _parse_existing_weights(source: str) -> dict[str, list[float]]:
    """Extract current ALLELE_CONTACT_WEIGHTS values from features.py source."""
    pattern = re.compile(
        r"'(HLA-[AB]\*\d{2}:\d{2})':\s*\[([0-9., ]+)\]"
    )
    result = {}
    for m in pattern.finditer(source):
        allele = m.group(1)
        weights = [float(x.strip()) for x in m.group(2).split(',') if x.strip()]
        result[allele] = weights
    return result


def _parse_existing_pop_avg(source: str) -> list[float]:
    """Extract current POPULATION_AVG_CONTACT_WEIGHTS from features.py source."""
    m = re.search(
        r'POPULATION_AVG_CONTACT_WEIGHTS:\s*list\[float\]\s*=\s*\[([0-9., ]+)\]',
        source,
    )
    if m:
        return [float(x.strip()) for x in m.group(1).split(',') if x.strip()]
    return []


def _replace_weights_in_source(
    source: str,
    new_allele_block: str,
    new_pop_avg: list[float],
) -> str:
    """Splice updated weight values into features.py source text."""
    # Replace ALLELE_CONTACT_WEIGHTS dict block
    allele_pattern = re.compile(
        r"ALLELE_CONTACT_WEIGHTS:\s*dict\[str,\s*list\[float\]\]\s*=\s*\{[^}]+\}",
        re.DOTALL,
    )
    if not allele_pattern.search(source):
        raise ValueError("Could not locate ALLELE_CONTACT_WEIGHTS block in features.py")
    source = allele_pattern.sub(new_allele_block, source)

    # Replace POPULATION_AVG_CONTACT_WEIGHTS list
    pop_repr = '[' + ', '.join(f'{w:.4f}' for w in new_pop_avg) + ']'
    pop_pattern = re.compile(
        r"POPULATION_AVG_CONTACT_WEIGHTS:\s*list\[float\]\s*=\s*\[[0-9., ]+\]"
    )
    pop_replacement = f'POPULATION_AVG_CONTACT_WEIGHTS: list[float] = {pop_repr}'
    if not pop_pattern.search(source):
        print("WARNING: POPULATION_AVG_CONTACT_WEIGHTS not found; skipping update", file=sys.stderr)
    else:
        source = pop_pattern.sub(pop_replacement, source)

    return source


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        '--contact-csv',
        type=pathlib.Path,
        default=CONTACT_CSV_DEFAULT,
        help=f'Contact frequency CSV from compute_contact_freqs.py (default: {CONTACT_CSV_DEFAULT})',
    )
    ap.add_argument(
        '--features-py',
        type=pathlib.Path,
        default=FEATURES_PY_DEFAULT,
        help=f'Path to src/features.py (default: {FEATURES_PY_DEFAULT})',
    )
    ap.add_argument(
        '--min-structures',
        type=int,
        default=3,
        help='Minimum number of structures required to use empirical freq (default: 3)',
    )
    ap.add_argument(
        '--dry-run',
        action='store_true',
        help='Print proposed changes without modifying features.py',
    )
    args = ap.parse_args()

    if not args.contact_csv.exists():
        sys.exit(f"Contact CSV not found: {args.contact_csv}")
    if not args.features_py.exists():
        sys.exit(f"features.py not found: {args.features_py}")

    # Load contact frequency data
    csv_data = _load_contact_csv(args.contact_csv)
    print(f"Loaded contact data for {len(csv_data)} alleles: {sorted(csv_data)}")

    # Load existing source to parse current priors
    source = args.features_py.read_text(encoding='utf-8')
    existing_priors = _parse_existing_weights(source)
    print(f"Found {len(existing_priors)} existing allele priors in features.py")

    # Build updated weights for each canonical allele
    allele_weights: dict[str, list[float]] = {}
    unchanged: list[str] = []

    for allele in CANONICAL_ALLELES:
        if allele not in csv_data:
            print(f"INFO: {allele} - no structures in CSV; keeping prior")
            unchanged.append(allele)
            continue
        weights = _build_weight_list(allele, csv_data[allele], args.min_structures)
        if weights is None:
            unchanged.append(allele)
        else:
            allele_weights[allele] = weights
            print(f"UPDATE: {allele} -> {weights}")

    if not allele_weights:
        print("No alleles have sufficient structure data. No changes made.")
        return

    # Compute new population average from updated alleles + unchanged priors
    combined = {**{a: existing_priors[a] for a in unchanged if a in existing_priors},
                **allele_weights}
    pop_avg = _population_avg(combined)
    print(f"Population average (all alleles): {pop_avg}")

    # Generate updated source block
    new_block = _format_weights_block(allele_weights, unchanged, existing_priors, pop_avg)

    if args.dry_run:
        print("\n--- Proposed ALLELE_CONTACT_WEIGHTS replacement ---")
        print(new_block)
        print("\n--- Proposed POPULATION_AVG_CONTACT_WEIGHTS ---")
        print(f"POPULATION_AVG_CONTACT_WEIGHTS: list[float] = {pop_avg}")
        print("\nDry run - no files modified.")
        return

    # Apply changes to features.py
    new_source = _replace_weights_in_source(source, new_block, pop_avg)
    args.features_py.write_text(new_source, encoding='utf-8')
    print(
        f"\nUpdated {args.features_py}: "
        f"{len(allele_weights)} alleles from structures, "
        f"{len(unchanged)} alleles kept as priors."
    )
    print("Run: python -m ruff check src/features.py && pytest tests/test_features.py -v")


if __name__ == '__main__':
    main()
