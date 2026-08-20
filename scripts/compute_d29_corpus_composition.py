"""Reproduce the corpus-composition counts behind claims_register D29.

D29 corrected the manuscript's "pan-allele" self-description: no production
feature mode carries an allele-identity column, so allele coverage is a
property of the ten-allele binding-feature panel (`ALLELE_CONTACT_WEIGHTS`,
`src/features.py:287-301`), not of the benchmark. Its Verification cell states
three corpus-composition figures in passing - 248 distinct HLA alleles,
16,984 of 35,597 active rows carrying a panel allele, 774 active HLA-C rows -
and says outright that they "reproduce exactly from the TRACKED
`data/immunogenicity_dataset_v5.csv`; what they lack is a provenance-generating
script, not a tracked source." This script is that script. It does not change
what D29 asserts; it makes the three numbers independently bindable instead of
prose that nothing in the tree computes.

Source (canonical and already committed):
  data/immunogenicity_dataset_v5.csv    the shipped v5 corpus

Method and the two scoping facts each figure actually depends on:
  - "Active" means is_quarantined != True, matching _filter_quarantined in
    src/train_classifier.py and every other consumer in this repository.
  - The distinct-allele count is scoped to ACTIVE rows, not the full corpus.
    This matters: nunique(hla_allele) over the full 51,185-row corpus is 263,
    not 248 - the 15 extra values appear only on quarantined rows. D29's "248"
    is the active-row figure; this script computes and labels both so the
    scoping choice is visible rather than silently picked.
  - HLA-C detection has to cover BOTH nomenclatures present in this column:
    the modern `HLA-C*06:02`-style molecular typing AND the legacy serotype
    form `HLA-Cw7` used on 38 of the 774 active HLA-C rows. A prefix check for
    `HLA-C*` alone finds only 736; the locus is the letter immediately after
    "HLA-" regardless of what follows it (`*NN:NN`, `wNN`, or nothing), which
    is what the regex here extracts.
  - The ten-allele panel is ALLELE_CONTACT_WEIGHTS's key set
    (src/features.py:287-298) - the same ten alleles the mode 10/30/31/33/35
    bind_* columns are computed against. It is imported directly from
    src/features.py rather than re-typed here, so the two can never drift.

--output has no default: results/ artifacts are git-tracked behind an explicit
.gitignore negation allowlist, so a bare invocation prints the table and
writes nothing rather than silently rewriting a committed file. Passing
--output also writes a provenance sidecar recording input digests, which the
integrity harness verifies.

Reproduce:
  python scripts/compute_d29_corpus_composition.py \
      --output results/d29_corpus_composition.csv
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import ALLELE_CONTACT_WEIGHTS  # noqa: E402

DATASET_SRC = "data/immunogenicity_dataset_v5.csv"
TRACKED_OUTPUT = "results/d29_corpus_composition.csv"

PANEL_ALLELES = sorted(ALLELE_CONTACT_WEIGHTS.keys())

_LOCUS_RE = re.compile(r"^HLA-([A-E])")


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _active(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows the training pipeline treats as trainable.

    Mirrors _filter_quarantined in src/train_classifier.py. Kept as a named
    helper so the definition appears exactly once.
    """
    return frame[frame["is_quarantined"] != True]  # noqa: E712 - NaN-safe, unlike `is False`


def _locus(allele: str) -> str | None:
    """Extract the HLA locus letter (A/B/C/E) regardless of typing nomenclature.

    Handles both molecular typing ("HLA-C*06:02") and legacy serotype form
    ("HLA-Cw7"): the locus is always the single letter directly after "HLA-".
    """
    match = _LOCUS_RE.match(allele)
    return match.group(1) if match else None


def compute_composition() -> pd.DataFrame:
    corpus = pd.read_csv(DATASET_SRC, low_memory=False)
    active = _active(corpus)

    distinct_alleles_active = active["hla_allele"].nunique()
    distinct_alleles_all_rows = corpus["hla_allele"].nunique()

    panel_mask = active["hla_allele"].isin(PANEL_ALLELES)
    panel_rows = int(panel_mask.sum())

    locus = active["hla_allele"].astype(str).map(_locus)
    hla_c_rows = int((locus == "C").sum())

    rows: list[dict[str, object]] = [
        {
            "metric": "distinct_hla_alleles_active",
            "value": int(distinct_alleles_active),
            "kind": "count",
        },
        {
            "metric": "distinct_hla_alleles_all_rows",
            "value": int(distinct_alleles_all_rows),
            "kind": "count",
        },
        {"metric": "active_rows_total", "value": int(len(active)), "kind": "count"},
        {"metric": "panel_size", "value": len(PANEL_ALLELES), "kind": "count"},
        {
            "metric": "active_rows_with_panel_allele",
            "value": panel_rows,
            "kind": "count",
        },
        {
            "metric": "active_rows_with_panel_allele_pct",
            "value": round(100.0 * panel_rows / len(active), 1),
            "kind": "percent",
        },
        {"metric": "active_hla_c_rows", "value": hla_c_rows, "kind": "count"},
    ]
    return pd.DataFrame(rows)


def _write_sidecar(output_path: str) -> str:
    """Record input digests alongside the artifact.

    Written at artifact-creation time rather than left as a gap for the
    integrity harness to report, matching the convention set by
    scripts/compute_section33_decoy_binding_join.py.
    """
    sidecar_path = f"{output_path}.provenance.json"
    payload = {
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "script": "scripts/compute_d29_corpus_composition.py",
        "artifact": output_path,
        "sha256": _sha256_file(output_path),
        "inputs": {
            DATASET_SRC: _sha256_file(DATASET_SRC),
        },
        "active_row_definition": "is_quarantined != True",
        "panel_alleles": PANEL_ALLELES,
    }
    with open(sidecar_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return sidecar_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the claims_register D29 corpus-composition counts."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            f"Output CSV path (optional). No default: {TRACKED_OUTPUT} is a "
            "git-tracked artifact, so this script refuses to guess a destination "
            "- omit this flag to print the table without writing anything."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out = compute_composition()
    print(out.to_string(index=False))
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        # lineterminator pinned to LF: the recorded sha256 must match the git
        # blob on any platform, not just the one that generated it (D24-resid).
        out.to_csv(args.output, index=False, lineterminator="\n")
        sidecar = _write_sidecar(args.output)
        print(f"\nwrote {args.output}")
        print(f"wrote {sidecar}")


if __name__ == "__main__":
    main()
