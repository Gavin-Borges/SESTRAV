#!/usr/bin/env python
"""Fail if the corpus provenance sidecar disagrees with the corpus or with itself.

`scripts/data_qc_gate.py` governs the shipped corpus with a class RATIO, and
`docs/data_qc_criteria.md` records why that instrument is structurally blind to a
whole family of build failures: a ratio cannot see a change that moves numerator
and denominator together, and it cannot see anything at all in a stream that is
entirely quarantined. Measured for the v5 corpus, every perturbation of either
decoy stream up to 2x moves the full-file ratio by at most 13%, landing inside
the derived bound; and the 5,000 self-proteome decoys are quarantined in full, so
the active and in-panel ratios are literally invariant to them.

This is the companion instrument the derivation names. It asserts per-stream row
counts by HARD EQUALITY against `data/immunogenicity_dataset_v5_provenance.json`,
which `scripts/build_dataset_v5.py` writes and which nothing previously read.

Four independent checks, deliberately separate so a failure names its own cause:

1. **Internal arithmetic.** The parts the build actually concatenates, minus the
   rows deduplication drops, must equal the merged total. Note `v4_other_negatives`
   is EXCLUDED on purpose: `scripts/build_dataset_v5.py` counts it into provenance
   but never appends it to `parts`, so including it here would make a correct
   sidecar fail. That asymmetry is the reason this check is written by hand rather
   than summing every field.
2. **The sidecar describes the shipped file.** Its `merged_total` must equal the
   corpus row count, and its `output_checksum_sha256` must equal the corpus digest.
3. **The sidecar agrees with config.yaml.** Its checksum must equal
   `dataset_governance.provenance.checksum`, the value `freeze_mode` enforces.
4. **Per-stream pins.** Each count in `dataset_governance.provenance.source_counts`
   must match the sidecar exactly. This is what catches a decoy stream that
   silently halved, a dedup key regression, and selection of the wrong IEDB input
   file - `data/` carries both `iedb_negatives_v5.csv` (32,506 rows) and
   `iedb_negatives_v5_merged.csv` (36,689), and substituting one for the other is
   invisible to the class-ratio gate.

The digest is computed over LF-normalised bytes, matching the portable git-blob
convention `.gitattributes` pins for this file. A CRLF working-tree digest is what
made `config.yaml`'s pin wrong for two months before 2026-09-13; computing the
other convention here would reintroduce exactly that class of error on Windows.

Usage:
    python tools/check_dataset_provenance.py            # human-readable report
    python tools/check_dataset_provenance.py --check    # same, exit 1 on any gap
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

SIDECAR = "data/immunogenicity_dataset_v5_provenance.json"
CONFIG = "config.yaml"

# The parts scripts/build_dataset_v5.py actually appends to `parts`. See check 1
# for why v4_other_negatives is absent.
MERGED_PARTS = ("v4_positives", "published_panels", "v4_hard_decoys", "iedb_negatives")


def _sha256_lf(path: pathlib.Path) -> str:
    """Digest of the file with CRLF normalised to LF, the portable git-blob form."""
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _count_rows(path: pathlib.Path) -> int:
    """Data rows in a CSV, excluding the header. Streamed: the corpus is ~8 MB."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _load_config_provenance(config_path: pathlib.Path) -> dict:
    import yaml

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return (cfg.get("dataset_governance") or {}).get("provenance") or {}


def check(repo_root: pathlib.Path) -> list[str]:
    problems: list[str] = []

    sidecar_path = repo_root / SIDECAR
    config_path = repo_root / CONFIG
    for path in (sidecar_path, config_path):
        if not path.is_file():
            return [f"{path.relative_to(repo_root).as_posix()}: missing (fail closed)"]

    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    counts = sidecar.get("source_counts") or {}
    provenance = _load_config_provenance(config_path)

    # --- 1. internal arithmetic -------------------------------------------------
    missing = [key for key in (*MERGED_PARTS, "dedup_dropped", "merged_total") if key not in counts]
    if missing:
        problems.append(f"{SIDECAR}: source_counts is missing {', '.join(missing)}")
    else:
        parts_total = sum(int(counts[key]) for key in MERGED_PARTS)
        expected = parts_total - int(counts["dedup_dropped"])
        if expected != int(counts["merged_total"]):
            problems.append(
                f"{SIDECAR}: {' + '.join(MERGED_PARTS)} = {parts_total}, minus "
                f"dedup_dropped {counts['dedup_dropped']} = {expected}, but merged_total "
                f"is {counts['merged_total']} (the sidecar contradicts itself)"
            )

    # --- 2. the sidecar describes the shipped file ------------------------------
    # output_file is written with the build host's separator; normalise it.
    rel = str(sidecar.get("output_file", "")).replace("\\", "/")
    corpus_path = repo_root / rel if rel else None
    if corpus_path is None or not corpus_path.is_file():
        problems.append(
            f"{SIDECAR}: output_file {rel!r} does not resolve to a file in this checkout"
        )
    else:
        actual_rows = _count_rows(corpus_path)
        if "merged_total" in counts and int(counts["merged_total"]) != actual_rows:
            problems.append(
                f"{rel}: has {actual_rows} data rows but the sidecar records "
                f"merged_total {counts['merged_total']}"
            )
        actual_digest = _sha256_lf(corpus_path)
        recorded = sidecar.get("output_checksum_sha256")
        if recorded != actual_digest:
            problems.append(
                f"{rel}: LF-normalised sha256 is {actual_digest} but the sidecar records "
                f"{recorded} (stale sidecar, or the corpus changed without a rebuild)"
            )

        # --- 3. the sidecar agrees with config.yaml -----------------------------
        pinned = provenance.get("checksum")
        if pinned and pinned != actual_digest:
            problems.append(
                f"{CONFIG}: dataset_governance.provenance.checksum is {pinned} but the "
                f"corpus digest is {actual_digest} (freeze_mode would reject this corpus)"
            )

    # --- 4. per-stream pins -----------------------------------------------------
    pinned_counts = provenance.get("source_counts") or {}
    if not pinned_counts:
        problems.append(
            f"{CONFIG}: dataset_governance.provenance.source_counts is absent, so no "
            "per-stream count is pinned and this gate cannot see a stream failure"
        )
    for key, want in sorted(pinned_counts.items()):
        if key not in counts:
            problems.append(f"{SIDECAR}: source_counts has no {key}, but {CONFIG} pins it to {want}")
        elif int(counts[key]) != int(want):
            problems.append(
                f"{SIDECAR}: source_counts.{key} is {counts[key]} but {CONFIG} pins {want} "
                "(a build input changed; this is the failure class the class-ratio gate cannot see)"
            )

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="exit 1 on any disagreement (CI mode)"
    )
    parser.add_argument(
        "--repo-root", default=str(REPO_ROOT), help="repository root (default: this file's repo)"
    )
    args = parser.parse_args()

    repo_root = pathlib.Path(args.repo_root).resolve()
    problems = check(repo_root)

    if problems:
        print(f"Dataset provenance: found {len(problems)} problem(s):", file=sys.stderr)
        for problem in problems:
            print(f"  ERROR: {problem}", file=sys.stderr)
        print(
            "\nThe sidecar is written by scripts/build_dataset_v5.py. Do not hand-edit it "
            "to clear this gate: it is the record of what the build actually produced, and "
            "editing it to match a changed corpus destroys the only evidence of the change.",
            file=sys.stderr,
        )
        return 1 if args.check else 0

    print("Dataset provenance: sidecar, corpus and config.yaml agree on every pinned count.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
