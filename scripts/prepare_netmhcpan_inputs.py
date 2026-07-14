"""Prepare NetMHCpan 4.1 peptide-list inputs from the LOO held-out test sets.

NetMHCpan is an academic-license-gated binary that is not installed in this
environment. This script does all of the work that does NOT need the binary:
it reads the per-virus held-out TSVs, groups peptides by (virus, allele),
writes one deduplicated peptide-list file per group in the format NetMHCpan's
``-p`` mode expects (one peptide per line), and emits a single ready-to-run
shell script so a human only has to install the binary and run one command.

Outputs (all under the gitignored ``_local/`` workspace):

    _local/ext_scores/inputs/<virus>__<allele_cli>.pep   peptide lists
    _local/ext_scores/inputs/manifest.csv                group manifest
    _local/ext_scores/run_netmhcpan.sh                   generated runner
    _local/ext_scores/raw/                                (created; runner target)

The manifest columns are:
    virus, allele_cli, allele_original, peptide_list_path, n_peptides

The process is deterministic and idempotent: groups, peptide order and file
names are sorted, so re-running with unchanged inputs reproduces byte-identical
outputs.

Usage:
    python scripts/prepare_netmhcpan_inputs.py
    python scripts/prepare_netmhcpan_inputs.py --test-set-dir results/loo_test_sets
    python scripts/prepare_netmhcpan_inputs.py --out-dir _local/ext_scores
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

# Allow "python scripts/prepare_netmhcpan_inputs.py" from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _netmhcpan_common import convert_allele_to_cli  # noqa: E402

HELD_OUT_SUFFIX = "_held_out.tsv"
DEFAULT_TEST_SET_DIR = "results/loo_test_sets"
DEFAULT_OUT_DIR = "_local/ext_scores"

MANIFEST_COLUMNS = [
    "virus",
    "allele_cli",
    "allele_original",
    "peptide_list_path",
    "n_peptides",
]

# NetMHCpan's -p mode accepts only the 20 standard amino acids. A group whose
# CLI allele is malformed cannot be scored, so it is skipped for the runner but
# still recorded in the manifest so the counts stay auditable. "Scorable" means
# a fully well-formed CLI allele: locus letter plus digits, optionally a
# two-field colon suffix, and NOTHING else (this rejects bare loci like "HLA-A"
# and free-text mutant annotations like "HLA-A02:01 K66A mutant").
_VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
_SCORABLE_ALLELE = re.compile(r"^HLA-[ABC]\d+(:\d+)?$")


def virus_from_filename(path: Path) -> str:
    """Return the virus label encoded in a held-out TSV filename.

    Args:
        path: Path to a ``<virus>_held_out.tsv`` file.

    Returns:
        The virus label (the filename with the held-out suffix removed).
    """
    return path.name[: -len(HELD_OUT_SUFFIX)]


def read_groups(tsv_path: Path) -> dict[tuple[str, str], list[str]]:
    """Group deduplicated peptides by (allele_cli, allele_original) for one file.

    Only clean peptides (composed solely of the 20 standard amino acids) are
    kept; anything else would be rejected by NetMHCpan and would corrupt the
    join on the parse side. Peptides are deduplicated within each group while
    preserving determinism via sorting downstream.

    Args:
        tsv_path: Path to one ``<virus>_held_out.tsv`` file.

    Returns:
        Mapping of (allele_cli, allele_original) -> sorted unique peptide list.
    """
    groups: dict[tuple[str, str], set[str]] = {}
    with tsv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            peptide = (row.get("peptide") or "").strip().upper()
            allele_original = (row.get("hla_allele") or "").strip()
            if not peptide or not allele_original:
                continue
            if any(aa not in _VALID_AA for aa in peptide):
                continue
            allele_cli = convert_allele_to_cli(allele_original)
            key = (allele_cli, allele_original)
            groups.setdefault(key, set()).add(peptide)
    return {key: sorted(peps) for key, peps in groups.items()}


def _safe_filename(virus: str, allele_cli: str) -> str:
    """Return a filesystem-safe stem for a (virus, allele) peptide list.

    Args:
        virus: Virus label.
        allele_cli: CLI-form allele string.

    Returns:
        A sanitized ``<virus>__<allele>`` stem with non-word characters mapped
        to underscores.
    """
    safe_virus = re.sub(r"[^\w.-]+", "_", virus)
    safe_allele = re.sub(r"[^\w.-]+", "_", allele_cli)
    return f"{safe_virus}__{safe_allele}"


def prepare_inputs(test_set_dir: Path, out_dir: Path) -> list[dict[str, object]]:
    """Write peptide lists and a manifest for every (virus, allele) group.

    Args:
        test_set_dir: Directory containing ``*_held_out.tsv`` files.
        out_dir: Root output directory (the ``_local/ext_scores`` workspace).

    Returns:
        A list of manifest records (one dict per group), sorted deterministically.
    """
    inputs_dir = out_dir / "inputs"
    raw_dir = out_dir / "raw"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    tsv_files = sorted(test_set_dir.glob(f"*{HELD_OUT_SUFFIX}"))
    if not tsv_files:
        print(f"[error] no *{HELD_OUT_SUFFIX} files under {test_set_dir}", file=sys.stderr)
        return []

    records: list[dict[str, object]] = []
    for tsv_path in tsv_files:
        virus = virus_from_filename(tsv_path)
        groups = read_groups(tsv_path)
        for (allele_cli, allele_original), peptides in sorted(groups.items()):
            stem = _safe_filename(virus, allele_cli)
            pep_path = inputs_dir / f"{stem}.pep"
            pep_path.write_text("\n".join(peptides) + "\n", encoding="utf-8")
            records.append(
                {
                    "virus": virus,
                    "allele_cli": allele_cli,
                    "allele_original": allele_original,
                    "peptide_list_path": pep_path.as_posix(),
                    "n_peptides": len(peptides),
                }
            )

    records.sort(key=lambda r: (str(r["virus"]), str(r["allele_cli"])))
    _write_manifest(inputs_dir / "manifest.csv", records)
    _write_runner(out_dir, raw_dir, records)
    return records


def _write_manifest(manifest_path: Path, records: list[dict[str, object]]) -> None:
    """Write the group manifest CSV.

    Args:
        manifest_path: Destination CSV path.
        records: Manifest records produced by :func:`prepare_inputs`.
    """
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def _write_runner(out_dir: Path, raw_dir: Path, records: list[dict[str, object]]) -> None:
    """Write the ready-to-run shell script that invokes NetMHCpan per group.

    Only groups whose CLI allele is well-formed (locus + digit) get a command;
    unscorable groups (for example a bare "HLA-A") are emitted as comments so a
    reviewer can see they were deliberately skipped.

    Args:
        out_dir: Root output directory; the script is written here.
        raw_dir: Directory the NetMHCpan outputs should land in.
        records: Manifest records produced by :func:`prepare_inputs`.
    """
    runner_path = out_dir / "run_netmhcpan.sh"
    raw_rel = raw_dir.relative_to(out_dir).as_posix()

    lines: list[str] = [
        "#!/bin/bash",
        "# Generated by scripts/prepare_netmhcpan_inputs.py - do not edit by hand.",
        "#",
        "# Install NetMHCpan 4.1 (academic license) so 'netMHCpan' is on PATH,",
        "# then run this script from the _local/ext_scores/ directory:",
        "#     cd _local/ext_scores && bash run_netmhcpan.sh",
        "# Outputs land in raw/ as one .xls and one .txt per (virus, allele) group.",
        "set -euo pipefail",
        "",
        'HERE="$(cd "$(dirname "$0")" && pwd)"',
        'cd "$HERE"',
        f'mkdir -p "{raw_rel}"',
        "",
    ]

    scorable = 0
    for record in records:
        virus = str(record["virus"])
        allele_cli = str(record["allele_cli"])
        pep_posix = str(record["peptide_list_path"])
        pep_name = Path(pep_posix).name
        stem = Path(pep_name).stem
        pep_arg = f"inputs/{pep_name}"
        out_xls = f"{raw_rel}/{stem}.xls"
        out_txt = f"{raw_rel}/{stem}.txt"
        if not _SCORABLE_ALLELE.match(allele_cli):
            lines.append(
                f"# SKIPPED (unscorable allele) virus={virus} allele={allele_cli}"
            )
            continue
        scorable += 1
        lines.append(f"# {virus} / {allele_cli}")
        lines.append(
            f'netMHCpan -p "{pep_arg}" -a "{allele_cli}" -BA -xls '
            f'-xlsfile "{out_xls}" > "{out_txt}"'
        )
    lines.append("")
    lines.append(f'echo "Done: {scorable} NetMHCpan runs completed."')
    lines.append("")

    runner_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    """Command-line entry point.

    Returns:
        Process exit code (0 on success, 1 when no inputs were found).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-set-dir",
        default=DEFAULT_TEST_SET_DIR,
        help="Directory with *_held_out.tsv files (default: %(default)s).",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help="Output root under _local/ (default: %(default)s).",
    )
    args = parser.parse_args()

    test_set_dir = Path(args.test_set_dir)
    out_dir = Path(args.out_dir)

    records = prepare_inputs(test_set_dir, out_dir)
    if not records:
        return 1

    n_groups = len(records)
    n_peptides = sum(int(str(r["n_peptides"])) for r in records)
    viruses = sorted({str(r["virus"]) for r in records})
    print(f"[prepare] viruses:        {len(viruses)} ({', '.join(viruses)})")
    print(f"[prepare] (virus,allele): {n_groups} groups")
    print(f"[prepare] peptides:       {n_peptides} total (deduplicated per group)")
    print(f"[prepare] manifest:       {(out_dir / 'inputs' / 'manifest.csv').as_posix()}")
    print(f"[prepare] runner:         {(out_dir / 'run_netmhcpan.sh').as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
