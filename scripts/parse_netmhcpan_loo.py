"""Parse NetMHCpan 4.1 output and score the LOO benchmark by AUC-ROC.

Once a human has installed NetMHCpan 4.1 and run
``_local/ext_scores/run_netmhcpan.sh``, this script consumes the resulting raw
output files, joins the presentation scores back to the original held-out test
rows, and computes per-virus AUC-ROC of label vs predictor.

Predictor orientation (IMPORTANT)
---------------------------------
NetMHCpan predicts PRESENTATION / BINDING, not immunogenicity. We use the
eluted-ligand (EL) presentation score as the predictor, oriented so that
HIGHER means more likely positive:

* Preferred: the raw ``EL_Score`` column (higher = stronger presentation).
* Fallback: if only ``%Rank_EL`` is available, the predictor is
  ``1 - (%Rank_EL / 100)`` so that a lower (stronger) rank maps to a higher
  score.

The column actually used for each file is logged so the choice is auditable.
This means a "good" NetMHCpan AUC here reflects how well presentation ranks the
labelled immunogenic peptides - it is a binding-based baseline, not a direct
immunogenicity predictor. Reviewers should keep that framing in mind.

Output
------
A NEW file ``results/loo_benchmark_comparison_with_netmhcpan.csv`` is written
with the ``netmhcpan_auc`` column filled. The existing verified CSV is never
overwritten. Per-virus n_pos / n_neg are checked against the reference CSV and
mismatches are warned about loudly.

Usage:
    python scripts/parse_netmhcpan_loo.py
    python scripts/parse_netmhcpan_loo.py --raw-dir _local/ext_scores/raw
    python scripts/parse_netmhcpan_loo.py --test-set-dir results/loo_test_sets
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from sklearn.metrics import roc_auc_score

# Allow "python scripts/parse_netmhcpan_loo.py" from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _netmhcpan_common import (  # noqa: E402
    ALLELE_NAMES,
    EL_RANK_NAMES,
    EL_SCORE_NAMES,
    PEPTIDE_NAMES,
    convert_allele_to_cli,
    find_column,
    is_header_line,
    parse_header_columns,
)

HELD_OUT_SUFFIX = "_held_out.tsv"
DEFAULT_TEST_SET_DIR = "results/loo_test_sets"
DEFAULT_RAW_DIR = "_local/ext_scores/raw"
DEFAULT_REFERENCE_CSV = "results/loo_benchmark_comparison.csv"
DEFAULT_OUTPUT_CSV = "results/loo_benchmark_comparison_with_netmhcpan.csv"

# Score keyed by (peptide, allele_cli); allele_cli is the CLI-form allele so the
# same conversion is applied on both sides of the join.
ScoreKey = tuple[str, str]


def parse_netmhcpan_file(path: Path) -> dict[ScoreKey, float]:
    """Parse one NetMHCpan output file into (peptide, allele_cli) -> predictor.

    Columns are located by header NAME, never by fixed position. The raw EL
    score is preferred; a percentile rank is converted with ``1 - rank/100``.
    Comment ('#') and rule ('---') lines are ignored. The predictor orientation
    for the file is logged once.

    Args:
        path: Path to a NetMHCpan ``-p`` output ``.txt`` (or ``.xls``) file.

    Returns:
        Mapping of (peptide, allele_cli) -> predictor value (higher = positive).
    """
    scores: dict[ScoreKey, float] = {}
    columns: dict[str, int] | None = None
    pep_idx = allele_idx = value_idx = None
    use_rank = False
    logged = False

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if columns is None:
                if is_header_line(line):
                    columns = parse_header_columns(line)
                    pep_idx = find_column(columns, PEPTIDE_NAMES)
                    allele_idx = find_column(columns, ALLELE_NAMES)
                    score_idx = find_column(columns, EL_SCORE_NAMES)
                    rank_idx = find_column(columns, EL_RANK_NAMES)
                    if score_idx is not None:
                        value_idx, use_rank = score_idx, False
                    else:
                        value_idx, use_rank = rank_idx, True
                    if pep_idx is None or allele_idx is None or value_idx is None:
                        print(
                            f"[warn] {path.name}: could not locate required columns "
                            f"(peptide={pep_idx}, allele={allele_idx}, value={value_idx})",
                            file=sys.stderr,
                        )
                        return {}
                continue

            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("-"):
                continue
            fields = stripped.split()
            need = max(pep_idx or 0, allele_idx or 0, value_idx or 0)
            if len(fields) <= need:
                continue
            try:
                raw_value = float(fields[value_idx])  # type: ignore[index]
            except (TypeError, ValueError):
                continue
            peptide = fields[pep_idx].strip().upper()  # type: ignore[index]
            allele_cli = convert_allele_to_cli(fields[allele_idx].strip())  # type: ignore[index]
            predictor = (1.0 - raw_value / 100.0) if use_rank else raw_value
            if not logged:
                col_name = "%Rank_EL (converted 1 - rank/100)" if use_rank else "EL_Score"
                print(f"[parse] {path.name}: predictor column = {col_name}")
                logged = True
            scores[(peptide, allele_cli)] = predictor

    return scores


def load_all_scores(raw_dir: Path) -> dict[ScoreKey, float]:
    """Parse every NetMHCpan output file under a directory into one score map.

    ``.txt`` files are preferred; ``.xls`` files are parsed only when no ``.txt``
    sibling exists (both are whitespace-delimited in NetMHCpan 4.1 -p mode).

    Args:
        raw_dir: Directory containing NetMHCpan output files.

    Returns:
        Merged (peptide, allele_cli) -> predictor map across all files.
    """
    scores: dict[ScoreKey, float] = {}
    txt_stems = {p.stem for p in raw_dir.glob("*.txt")}
    candidates = sorted(raw_dir.glob("*.txt")) + [
        p for p in sorted(raw_dir.glob("*.xls")) if p.stem not in txt_stems
    ]
    for path in candidates:
        scores.update(parse_netmhcpan_file(path))
    return scores


def virus_from_filename(path: Path) -> str:
    """Return the virus label encoded in a held-out TSV filename.

    Args:
        path: Path to a ``<virus>_held_out.tsv`` file.

    Returns:
        The virus label with the held-out suffix removed.
    """
    return path.name[: -len(HELD_OUT_SUFFIX)]


def score_virus(
    tsv_path: Path, scores: dict[ScoreKey, float]
) -> tuple[list[int], list[float], int, int]:
    """Join scores onto one virus test set and collect labels/predictors.

    Args:
        tsv_path: Path to one ``<virus>_held_out.tsv`` file.
        scores: Merged (peptide, allele_cli) -> predictor map.

    Returns:
        (labels, predictors, n_pos, n_neg) where labels/predictors are aligned
        lists over rows that were successfully joined.
    """
    labels: list[int] = []
    predictors: list[float] = []
    n_pos = n_neg = 0
    with tsv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            peptide = (row.get("peptide") or "").strip().upper()
            allele_original = (row.get("hla_allele") or "").strip()
            label_raw = (row.get("label") or "").strip()
            if not peptide or not allele_original or label_raw not in {"0", "1"}:
                continue
            label = int(label_raw)
            if label == 1:
                n_pos += 1
            else:
                n_neg += 1
            key = (peptide, convert_allele_to_cli(allele_original))
            predictor = scores.get(key)
            if predictor is None:
                continue
            labels.append(label)
            predictors.append(predictor)
    return labels, predictors, n_pos, n_neg


def load_reference(reference_csv: Path) -> dict[str, tuple[int, int]]:
    """Load expected per-virus (n_pos, n_neg) from the reference benchmark CSV.

    Args:
        reference_csv: Path to ``loo_benchmark_comparison.csv``.

    Returns:
        Mapping of virus -> (expected_n_pos, expected_n_neg).
    """
    expected: dict[str, tuple[int, int]] = {}
    with reference_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                expected[row["virus"]] = (int(row["n_pos"]), int(row["n_neg"]))
            except (KeyError, ValueError):
                continue
    return expected


def write_output(
    reference_csv: Path, output_csv: Path, auc_by_virus: dict[str, float | None]
) -> None:
    """Write a copy of the reference CSV with netmhcpan_auc filled in.

    The reference file is never modified. Rows for which no AUC could be computed
    keep an empty ``netmhcpan_auc`` cell.

    Args:
        reference_csv: Path to the verified reference CSV (read-only).
        output_csv: Destination for the augmented CSV.
        auc_by_virus: Mapping of virus -> AUC (or ``None`` when unavailable).
    """
    with reference_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if "netmhcpan_auc" not in fieldnames:
        fieldnames.append("netmhcpan_auc")
    for row in rows:
        auc = auc_by_virus.get(row.get("virus", ""))
        row["netmhcpan_auc"] = "" if auc is None else f"{auc:.4f}"
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(
    test_set_dir: Path, raw_dir: Path, reference_csv: Path, output_csv: Path
) -> dict[str, float | None]:
    """Parse outputs, compute per-virus AUC, verify counts, and write the CSV.

    Args:
        test_set_dir: Directory with ``*_held_out.tsv`` files.
        raw_dir: Directory with NetMHCpan output files.
        reference_csv: Verified reference benchmark CSV (read-only).
        output_csv: Destination for the augmented CSV.

    Returns:
        Mapping of virus -> AUC (or ``None`` when it could not be computed).
    """
    scores = load_all_scores(raw_dir)
    print(f"[parse] loaded {len(scores)} (peptide, allele) scores from {raw_dir.as_posix()}")
    expected = load_reference(reference_csv)

    auc_by_virus: dict[str, float | None] = {}
    table: list[tuple[str, int, int, str]] = []

    for tsv_path in sorted(test_set_dir.glob(f"*{HELD_OUT_SUFFIX}")):
        virus = virus_from_filename(tsv_path)
        labels, predictors, n_pos, n_neg = score_virus(tsv_path, scores)

        if virus in expected:
            exp_pos, exp_neg = expected[virus]
            if (n_pos, n_neg) != (exp_pos, exp_neg):
                print(
                    f"[WARN] {virus}: count mismatch vs reference - "
                    f"got pos={n_pos} neg={n_neg}, expected pos={exp_pos} neg={exp_neg}",
                    file=sys.stderr,
                )

        auc: float | None = None
        if len(set(labels)) < 2:
            print(
                f"[warn] {virus}: cannot compute AUC "
                f"(joined {len(labels)} rows, classes={sorted(set(labels))})",
                file=sys.stderr,
            )
        else:
            auc = float(roc_auc_score(labels, predictors))
        auc_by_virus[virus] = auc
        auc_str = "n/a" if auc is None else f"{auc:.4f}"
        table.append((virus, n_pos, n_neg, auc_str))

    write_output(reference_csv, output_csv, auc_by_virus)

    print("\n=== NetMHCpan LOO benchmark (presentation EL score) ===")
    print(f"{'virus':<12}{'n_pos':>8}{'n_neg':>8}{'netmhcpan_auc':>16}")
    for virus, n_pos, n_neg, auc_str in table:
        print(f"{virus:<12}{n_pos:>8}{n_neg:>8}{auc_str:>16}")
    print(f"\n[parse] wrote {output_csv.as_posix()}")
    return auc_by_virus


def main() -> int:
    """Command-line entry point.

    Returns:
        Process exit code (0 on success, 2 when inputs are missing).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-set-dir", default=DEFAULT_TEST_SET_DIR)
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--reference-csv", default=DEFAULT_REFERENCE_CSV)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    args = parser.parse_args()

    test_set_dir = Path(args.test_set_dir)
    raw_dir = Path(args.raw_dir)
    reference_csv = Path(args.reference_csv)
    output_csv = Path(args.output_csv)

    if not test_set_dir.is_dir():
        print(f"[error] test-set dir not found: {test_set_dir}", file=sys.stderr)
        return 2
    if not reference_csv.is_file():
        print(f"[error] reference CSV not found: {reference_csv}", file=sys.stderr)
        return 2
    if not raw_dir.is_dir():
        print(
            f"[error] raw NetMHCpan output dir not found: {raw_dir}\n"
            f"        Install NetMHCpan 4.1 and run "
            f"_local/ext_scores/run_netmhcpan.sh first.",
            file=sys.stderr,
        )
        return 2

    run(test_set_dir, raw_dir, reference_csv, output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
