"""Reproduce the decoy-vs-binding-matrix join behind manuscript Section 3.3.

Section 3.3 asserts fourteen figures about the allele_matched_nonbinder decoy
population and its coverage in the tracked binding matrix. Before this script
they were prose that nothing could check: no tracked code performed the join
that produces them. This script performs it and emits every figure as a named
row, so each becomes a reproducible, bindable quantity.

Sources (both canonical and already committed):
  data/immunogenicity_dataset_v5.csv      the shipped v5 corpus
  models/peptide_binding_matrix_v5.csv    peptide -> 10 per-allele bind_* scores

Method: max_presentation = row-wise max over the ten bind_* columns of the
matrix; the corpus is left-joined onto it by peptide. "Active" means
is_quarantined != True, matching _filter_quarantined in src/train_classifier.py
and every other consumer in this repository.

Two scoping notes that the figures depend on, stated because getting either
wrong silently changes the answer:
  - The positives median is reported twice on purpose. The 0.712 figure is over
    ACTIVE rows (n = 6,431); the all-rows figure is 0.705 over n = 7,037,
    because that wider base includes 606 quarantined rows which the decoy sample
    does not. Section 3.3 quotes both and says why.
  - The 218-row matrix-resident decoy sample is NOT a random draw from the
    3,112. The binding matrix predates every decoy file and was never rebuilt,
    so matrix membership selects for peptides already in the earlier corpus and
    biases that comparison upward. The manuscript says so; this script does not
    correct for it, and no consumer of this output should read the 0.761/0.740
    medians as an unbiased estimate.

--output has no default: results/ artifacts are git-tracked behind an explicit
.gitignore negation allowlist, so a bare invocation prints the table and writes
nothing rather than silently rewriting a committed file. Passing --output also
writes a provenance sidecar recording input digests, which the integrity
harness verifies.

Reproduce:
  python scripts/compute_section33_decoy_binding_join.py \
      --output results/section33_decoy_binding_join.csv
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os

import pandas as pd

DATASET_SRC = "data/immunogenicity_dataset_v5.csv"
MATRIX_SRC = "models/peptide_binding_matrix_v5.csv"
TRACKED_OUTPUT = "results/section33_decoy_binding_join.csv"

DECOY_ORIGIN = "allele_matched_nonbinder"
REAL_NEG_ORIGINS = ("tested_negative", "iedb_api")


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


def compute_join() -> pd.DataFrame:
    corpus = pd.read_csv(DATASET_SRC, low_memory=False)
    matrix = pd.read_csv(MATRIX_SRC)

    bind_cols = [c for c in matrix.columns if c.startswith("bind_")]
    if not bind_cols:
        raise SystemExit(f"no bind_* columns found in {MATRIX_SRC}")
    matrix = matrix.drop_duplicates(subset="peptide").set_index("peptide")
    max_presentation = matrix[bind_cols].max(axis=1)

    active = _active(corpus)
    decoys = active[active["negative_origin"] == DECOY_ORIGIN]

    # Decoys that DO resolve to a matrix entry - the 218-row biased sample.
    decoy_scores = decoys["peptide"].map(max_presentation)
    resolved = decoys[decoy_scores.notna()]
    resolved_scores = decoy_scores.dropna()
    distinct_scores = max_presentation.loc[
        sorted(set(resolved["peptide"]) & set(max_presentation.index))
    ]

    positives_active = active[active["label"] == 1]
    positives_all = corpus[corpus["label"] == 1]
    pos_active_scores = positives_active["peptide"].map(max_presentation).dropna()
    pos_all_scores = positives_all["peptide"].map(max_presentation).dropna()

    in_matrix = set(max_presentation.index)
    zero_vec_positives = positives_active[~positives_active["peptide"].isin(in_matrix)]

    rows: list[dict[str, object]] = [
        {"metric": "decoy_rows_active", "value": len(decoys), "kind": "count"},
        {"metric": "decoy_rows_in_matrix", "value": len(resolved), "kind": "count"},
        {
            "metric": "decoy_distinct_peptides_in_matrix",
            "value": len(distinct_scores),
            "kind": "count",
        },
        {
            "metric": "decoy_max_presentation_median_rows",
            "value": round(float(resolved_scores.median()), 3),
            "kind": "median",
        },
        {
            "metric": "decoy_max_presentation_median_distinct",
            "value": round(float(distinct_scores.median()), 3),
            "kind": "median",
        },
        {
            "metric": "decoy_max_presentation_min",
            "value": round(float(distinct_scores.min()), 3),
            "kind": "range",
        },
        {
            "metric": "decoy_max_presentation_max",
            "value": round(float(distinct_scores.max()), 3),
            "kind": "range",
        },
        {
            "metric": "positive_max_presentation_median_active",
            "value": round(float(pos_active_scores.median()), 3),
            "kind": "median",
        },
        {
            "metric": "positive_rows_active_in_matrix",
            "value": int(len(pos_active_scores)),
            "kind": "count",
        },
        {
            "metric": "positive_max_presentation_median_all_rows",
            "value": round(float(pos_all_scores.median()), 3),
            "kind": "median",
        },
        {
            "metric": "positive_rows_all_in_matrix",
            "value": int(len(pos_all_scores)),
            "kind": "count",
        },
        {
            "metric": "positive_rows_quarantined_in_matrix",
            "value": int(len(pos_all_scores) - len(pos_active_scores)),
            "kind": "count",
        },
        {
            "metric": "decoy_rows_absent_from_matrix",
            "value": int(len(decoys) - len(resolved)),
            "kind": "count",
        },
        {
            "metric": "decoy_absent_from_matrix_pct",
            "value": round(100.0 * (len(decoys) - len(resolved)) / len(decoys), 1),
            "kind": "percent",
        },
        {
            "metric": "zero_vector_active_positives",
            "value": int(len(zero_vec_positives)),
            "kind": "count",
        },
        {
            "metric": "zero_vector_active_positives_distinct_viruses",
            "value": int(zero_vec_positives["virus"].nunique()),
            "kind": "count",
        },
    ]

    # Coverage control: the two real-negative classes, which Section 3.3 reports
    # as 0-absent against the decoys' 93.0%. This asymmetry is the actual claim,
    # so both arms are emitted rather than just the decoy arm.
    for origin in REAL_NEG_ORIGINS:
        arm = active[active["negative_origin"] == origin]
        absent = int((~arm["peptide"].isin(in_matrix)).sum())
        rows.append({"metric": f"{origin}_rows_active", "value": len(arm), "kind": "count"})
        rows.append(
            {"metric": f"{origin}_rows_absent_from_matrix", "value": absent, "kind": "count"}
        )

    return pd.DataFrame(rows)


def _write_sidecar(output_path: str) -> str:
    """Record input digests alongside the artifact.

    The sibling Table 3b generator ships without one, which is a gap the
    integrity harness reports as "no checksum recorded". Written here so this
    artifact is verifiable from the moment it exists.
    """
    sidecar_path = f"{output_path}.provenance.json"
    payload = {
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "script": "scripts/compute_section33_decoy_binding_join.py",
        "artifact": output_path,
        "sha256": _sha256_file(output_path),
        "inputs": {
            DATASET_SRC: _sha256_file(DATASET_SRC),
            MATRIX_SRC: _sha256_file(MATRIX_SRC),
        },
        "active_row_definition": "is_quarantined != True",
        "decoy_origin": DECOY_ORIGIN,
    }
    with open(sidecar_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return sidecar_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the Section 3.3 decoy-vs-binding-matrix join."
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
    out = compute_join()
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
