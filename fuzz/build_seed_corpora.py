#!/usr/bin/env python3
"""Build the seed corpora for the atheris harnesses from TRACKED repository content.

Nothing here is downloaded and nothing is committed: the seeds are regenerated from
files ``git ls-files`` reports, so a clone and CI build the same set. Output layout
is ``<out>/<harness stem>/seed_<NNNN>``, one directory per ``fuzz_*.py`` harness.

Sources, per harness family:

* NetChop / TAPreg parsers: the response literals in
  tests/test_external_predictors.py, plus pages synthesised in the row formats the
  two parsers' docstrings and comments document, filled with PEPTIDE_POOL entries.
  No real service response is tracked in this repository, so none is used.
* Allele harnesses: the ``allele`` column of
  results/external_predig_peptide_allele_pairs.csv and of the allele-aware
  holdout table, the keys of ``HLA_SUPERTYPE_MAP`` in src/hla_supertypes.py (the
  family map's keys are regexes, not alleles), and the tokens of
  results/external_prime_alleles_compact.txt.
* is_valid_peptide: the ``peptide`` column of the allele-aware holdout table and
  PEPTIDE_POOL.
* FASTA header harness: every header line of every tracked ``*.fasta`` file.
* Naming harnesses: the keys and values of ``PROTEOME_ID_ALIASES`` (read with ast,
  not imported), the example in ``canonical_output_filename``'s docstring, and up
  to 40 stems of tracked ``results/*.csv``.

Usage::

    python fuzz/build_seed_corpora.py --out <directory>
"""

from __future__ import annotations

import argparse
import ast
import csv
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import _harness_common as common  # noqa: E402

HOLDOUTS = "data/allele_aware/IEDB-20260704-MULTI_VIRUS_ALLELE_AWARE-v2_holdouts.csv"
PAIRS = "results/external_predig_peptide_allele_pairs.csv"
PRIME_ALLELES = "results/external_prime_alleles_compact.txt"
PREDICTOR_TESTS = "tests/test_external_predictors.py"
SUPERTYPES = "src/hla_supertypes.py"
NAMING = "src/naming.py"
MAX_SEEDS_PER_HARNESS = 400


def tracked(pattern: str) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", pattern],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    ).stdout
    return sorted(p for p in out.decode("utf-8").split("\0") if p)


def require_tracked(path: str) -> Path:
    if path not in tracked(path):
        raise SystemExit(f"seed source is not a tracked file: {path}")
    return REPO_ROOT / path


def csv_column(path: str, column: str) -> list[str]:
    with require_tracked(path).open(newline="", encoding="utf-8") as fh:
        return sorted({row[column] for row in csv.DictReader(fh) if row.get(column)})


def string_literals(path: str) -> list[str]:
    tree = ast.parse(require_tracked(path).read_text(encoding="utf-8"))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def dict_literal(path: str, name: str) -> dict:
    tree = ast.parse(require_tracked(path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        else:
            continue
        if isinstance(target, ast.Name) and target.id == name:
            return ast.literal_eval(value)
    raise SystemExit(f"{name} not found in {path}")


def netchop_page(indices: list[int]) -> str:
    rows = []
    for ident, idx in enumerate(indices):
        pep = common.PEPTIDE_POOL[idx]
        for pos, aa in enumerate(pep, start=1):
            score = ((ord(aa) * 37 + pos * 11) % 1000) / 1000.0
            call = "S" if score >= 0.5 else "."
            rows.append(f"{pos:>4} {aa:>3} {call:>3} {score:>9.5f}  pep_{ident}")
    rule = "-" * 38
    body = "\n".join([rule, " pos  AA  C  score    Ident", rule, *rows, rule])
    return f"<html>\n<body>\n<pre>\n{body}\n</pre>\n</body>\n</html>\n"


def tapreg_text(indices: list[int]) -> str:
    lines = []
    for ident, idx in enumerate(indices):
        score = 0.5 + (idx % 7) * 0.125
        lines.append(f"pep_{ident}  {common.PEPTIDE_POOL[idx]}  {score:.4f}")
    return "\n".join(lines)


def tapreg_table(indices: list[int]) -> str:
    rows = [
        f"  <tr><td>{common.PEPTIDE_POOL[idx]}</td><td>{-1.25 + idx * 0.25:.4f}</td></tr>"
        for idx in indices
    ]
    return "<table>\n" + "\n".join(rows) + "\n</table>\n"


def parser_seeds(kind: str) -> list[bytes]:
    pool = list(common.PEPTIDE_POOL)
    literals = string_literals(PREDICTOR_TESTS)
    seeds: list[bytes] = []
    for text in literals:
        if kind == "netchop" and "pep_" in text:
            seeds.append(common.encode_parser_input([2, 3], text))
            seeds.append(common.encode_parser_input([0, 1], text))
        if kind == "tapreg" and any(p in text for p in pool) and "." in text:
            seeds.append(common.encode_parser_input([0, 1], text))
    groups = [[i] for i in range(len(pool))]
    groups += [[i, (i + 5) % len(pool), (i + 9) % len(pool)] for i in range(len(pool))]
    for group in groups:
        if kind == "netchop":
            seeds.append(common.encode_parser_input(group, netchop_page(group)))
        else:
            seeds.append(common.encode_parser_input(group, tapreg_text(group)))
            seeds.append(common.encode_parser_input(group, tapreg_table(group)))
    return seeds


def allele_seeds() -> list[str]:
    alleles = set(csv_column(PAIRS, "allele")) | set(csv_column(HOLDOUTS, "allele"))
    alleles |= set(dict_literal(SUPERTYPES, "HLA_SUPERTYPE_MAP"))
    prime = require_tracked(PRIME_ALLELES).read_text(encoding="utf-8")
    alleles |= {tok.strip() for tok in prime.split(",") if tok.strip()}
    return sorted(alleles)


def peptide_seeds() -> list[str]:
    return sorted(set(csv_column(HOLDOUTS, "peptide")) | set(common.PEPTIDE_POOL))


def header_seeds() -> list[str]:
    headers = set()
    for path in tracked("*.fasta"):
        with (REPO_ROOT / path).open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith(">"):
                    headers.add(line[1:])
    return sorted(headers)


def proteome_id_seeds() -> list[str]:
    aliases = dict_literal(NAMING, "PROTEOME_ID_ALIASES")
    return sorted(set(aliases) | set(aliases.values()))


def filename_seeds() -> list[bytes]:
    mode, version = "modeA_baseline", "IEDB-20260424-EBV_HPV16_BASELINE-v1"
    stems = ["h2_tier_a_summary"] + [Path(p).stem for p in tracked("results/*.csv")][:40]
    return [b"\x00".join(s.encode("utf-8") for s in (stem, mode, version)) for stem in stems]


def build(out: Path) -> dict[str, int]:
    plan: dict[str, list[bytes]] = {
        "fuzz_parse_netchop_html": parser_seeds("netchop"),
        "fuzz_parse_tapreg_html": parser_seeds("tapreg"),
        "fuzz_get_hla_supertype": [s.encode("utf-8") for s in allele_seeds()],
        "fuzz_bin_supertype": [s.encode("utf-8") for s in allele_seeds()],
        "fuzz_hla_to_prime_compact": [s.encode("utf-8") for s in allele_seeds()],
        "fuzz_is_valid_peptide": [s.encode("utf-8") for s in peptide_seeds()],
        "fuzz_protein_name_from_header": [s.encode("utf-8") for s in header_seeds()],
        "fuzz_canonicalize_proteome_id": [s.encode("utf-8") for s in proteome_id_seeds()],
        "fuzz_canonical_output_filename": filename_seeds(),
    }
    counts = {}
    for harness, seeds in plan.items():
        target = out / harness
        target.mkdir(parents=True, exist_ok=True)
        unique = sorted(set(seeds))[:MAX_SEEDS_PER_HARNESS]
        for n, seed in enumerate(unique):
            (target / f"seed_{n:04d}").write_bytes(seed)
        counts[harness] = len(unique)
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path, help="output directory")
    args = parser.parse_args(argv)
    counts = build(args.out)
    for harness, n in counts.items():
        print(f"{harness}\t{n}")
    return 0 if all(counts.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
