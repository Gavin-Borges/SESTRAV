"""Precompute NetChop 3.1 and TAPreg scores for a peptide dataset.

Produces a cache CSV with columns: peptide, netchop_score, tap_score.
This cache is consumed by feature_mode=33 training via load_antigen_processing_cache().

Usage (recommended flow):
    # Dry-run: print plan without querying external APIs
    python scripts/precompute_antigen_processing.py \
        --data data/immunogenicity_dataset_v4.csv --output data/antigen_processing_cache.csv \
        --dry-run

    # Smoke-test: first 20 peptides only
    python scripts/precompute_antigen_processing.py \
        --data data/immunogenicity_dataset_v4.csv --output data/antigen_processing_cache.csv \
        --limit 20

    # Full overnight run (resume-safe)
    python scripts/precompute_antigen_processing.py \
        --data data/immunogenicity_dataset_v4.csv --output data/antigen_processing_cache.csv \
        --resume

External dependencies:
    NetChop 3.1: https://services.healthtech.dtu.dk/cgi-bin/webface2.cgi
    TAPreg:      https://imed.med.ucm.es/cgi-bin/tapreg.pl
    Both APIs currently unavailable; all scoring uses mock fallbacks. Those mocks are
    NOT reproducible across runs: the generators use hash(), which CPython salts per
    process, so a re-run produces different values (docs/claims_register.md D18).
    Rate limiting disabled (RATE_LIMIT_SECONDS=0). When real APIs become available,
    restore RATE_LIMIT_SECONDS=1.0 to respect server rate limits.
"""

import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.external_predictors import query_netchop, query_tapreg

BATCH_SIZE = 100
# Both external APIs are unavailable (NetChop webface2 format changed; TAPreg requires VPN).
# All paths use mock scores - no network calls, no rate limiting needed. The mocks are
# stable within one process but differ between processes (hash() salt), so re-running
# this script does NOT reproduce the shipped cache (D18).
RATE_LIMIT_SECONDS = 0.0


def load_existing_cache(output_path: str) -> set:
    """Return set of peptides already in the cache (for --resume)."""
    if not os.path.exists(output_path):
        return set()
    try:
        existing = pd.read_csv(output_path, usecols=["peptide"])
        return set(existing["peptide"].dropna().unique())
    except Exception:
        return set()


def write_batch(output_path: str, rows: list, first_write: bool) -> None:
    """Incrementally append a batch of rows to the cache CSV."""
    batch_df = pd.DataFrame(rows, columns=["peptide", "netchop_score", "tap_score"])
    if first_write and not os.path.exists(output_path):
        batch_df.to_csv(output_path, index=False)
    else:
        batch_df.to_csv(output_path, mode="a", header=False, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Precompute NetChop/TAPreg antigen processing scores"
    )
    parser.add_argument(
        "--data",
        required=True,
        help='Path to immunogenicity dataset CSV (must have "peptide" column)',
    )
    parser.add_argument("--output", required=True, help="Output cache CSV path")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print plan without querying external APIs"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Skip peptides already present in the output cache"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Process at most N peptides (for smoke-testing)"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.data):
        parser.error(f"Data file not found: '{args.data}'")

    df = pd.read_csv(args.data)
    if "peptide" not in df.columns:
        parser.error("Dataset must contain a 'peptide' column.")

    peptides = df["peptide"].dropna().unique().tolist()
    peptides = sorted(set(str(p).strip().upper() for p in peptides if p))

    already_done: set = set()
    if args.resume:
        already_done = load_existing_cache(args.output)
        print(f"[resume] {len(already_done)} peptides already cached; skipping.")

    todo = [p for p in peptides if p not in already_done]
    if args.limit is not None:
        todo = todo[: args.limit]

    print(f"Dataset: {args.data}")
    print(f"Total unique peptides: {len(peptides)}")
    print(f"To process: {len(todo)}")
    print(f"Output: {args.output}")
    print(f"Batch size: {BATCH_SIZE} | Rate limit: {RATE_LIMIT_SECONDS}s/request")

    print(
        "[NOTE] Using biologically-informed mock scores: DTU NetChop webface2 response "
        "format changed (real parser finds no tabular data) and TAPreg requires UCM "
        "institutional VPN. Scores are sequence-derived but are NOT real NetChop 3.1 / "
        "TAPreg predictions, and are NOT reproducible across runs (hash() is salted per "
        "process). Document this in limitations. See docs/claims_register.md D18."
    )

    if args.dry_run:
        print("[dry-run] No API calls made. Exiting.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    batch: list = []
    first_write = True
    errors = 0
    tap_unavailable = False

    for i, peptide in enumerate(todo, start=1):
        netchop_score = float("nan")
        tap_score = float("nan")

        try:
            # mock_fallback=True: DTU NetChop webface2 response format changed (parser finds no
            # tabular data) and TAPreg requires UCM institutional VPN. Both real-API paths fall
            # back to mock scores anyway after exhausting retries (~154s/peptide).
            # Using mock_fallback=True takes the fast path directly and is honest about the source.
            result = query_netchop([peptide], mock_fallback=True)
            pep_scores = result.get(peptide, {}).get("scores", [])
            netchop_score = float(sum(pep_scores) / len(pep_scores)) if pep_scores else float("nan")
        except Exception as e:
            print(f"  [NetChop ERROR] {peptide}: {e}")
            errors += 1

        if not tap_unavailable:
            try:
                result = query_tapreg([peptide], mock_fallback=True)
                tap_raw = result.get(peptide)
                tap_score = float(tap_raw) if tap_raw is not None else float("nan")
            except Exception as e:
                err_str = str(e)
                if "VPN" in err_str or "restricted" in err_str.lower():
                    print(
                        "  [TAPreg] Network restriction detected - setting tap_score=NaN for remaining peptides."
                    )
                    tap_unavailable = True
                else:
                    print(f"  [TAPreg ERROR] {peptide}: {e}")
                    errors += 1

        batch.append({"peptide": peptide, "netchop_score": netchop_score, "tap_score": tap_score})

        if len(batch) >= BATCH_SIZE:
            write_batch(args.output, batch, first_write and not os.path.exists(args.output))
            first_write = False
            print(f"  Written {i}/{len(todo)} peptides ({errors} errors so far)")
            batch = []

        time.sleep(RATE_LIMIT_SECONDS)

    if batch:
        write_batch(args.output, batch, first_write and not os.path.exists(args.output))
        print(f"  Written {len(todo)}/{len(todo)} peptides ({errors} total errors)")

    final_cache = pd.read_csv(args.output)
    print(f"\nCache complete: {len(final_cache)} rows in {args.output}")
    print(f"  netchop_score non-null: {final_cache['netchop_score'].notna().sum()}")
    print(f"  tap_score non-null:     {final_cache['tap_score'].notna().sum()}")
    if errors:
        print(f"  Total errors: {errors} (NaN written for failed rows)")


if __name__ == "__main__":
    main()
