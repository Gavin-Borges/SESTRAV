"""Fetch MHC Class I T-cell epitope records from the IEDB REST API.

Replaces the manual XLSX download workflow. For every supported virus, one
command produces a v4-compatible CSV with assay quality metadata that feeds
directly into build_dataset_v4.py.

The IEDB query-api (https://query-api.iedb.org/api/v1/) is a public PostgREST
endpoint requiring no authentication. Filters used per request:
  - mhc_restriction__class = I (MHC Class I only)
  - epitope__object_type   = Linear peptide
  - host__name             = ilike.*Homo sapiens* (human subjects only)
  - epitope__source_organism = ilike.*<pattern>* (virus-specific)

Labels are assigned from assay__qualitative_measurement:
  Positive / Positive-High / Positive-Intermediate / Positive-Low -> 1
  Negative                                                         -> 0
  All other values (inconclusive, etc.)                            -> excluded

Assay quality weights reflect the reliability of each assay type as an
immunogenicity signal; they are stored as a column in the output CSV so that
downstream training can apply sample_weight in sklearn estimators.

Usage
-----
    python scripts/fetch_iedb_tcell.py --virus EBV
    python scripts/fetch_iedb_tcell.py --virus HIV --output data/iedb_HIV_tcell.csv
    python scripts/fetch_iedb_tcell.py --all --output-dir data/iedb/
    python scripts/fetch_iedb_tcell.py --list-viruses

References
----------
Vita R, et al. The Immune Epitope Database (IEDB): 2018 update.
  Nucleic Acids Res. 2019;47(D1):D339-D343. doi:10.1093/nar/gky1006
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request  # noqa: F401 - urlopen is called via urllib.request.urlopen below
from urllib.parse import quote

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _ssl_fix  # noqa: F401, E402 - patch SSL before any network calls
from _dataset_utils import normalize_peptides, write_provenance

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IEDB_API_BASE = "https://query-api.iedb.org/api/v1/tcell_export"
PAGE_SIZE = 1000
REQUEST_DELAY = 1.0  # seconds between paginated requests

# Maps user-facing virus keys to case-insensitive IEDB source_organism substrings.
# Verified against live API 2026-06-19: EBV="human gammaherpesvirus 4",
# HPV16="Human papillomavirus 16", HBV="Hepatitis B virus",
# HIV="Human immunodeficiency virus 1", CMV="Human herpesvirus 5" (not "cytomegalovirus").
ORGANISM_MAP: dict[str, str] = {
    "EBV": "gammaherpesvirus 4",
    "HPV16": "papillomavirus 16",
    "HPV18": "papillomavirus 18",
    "HPV": "papillomavirus",
    "HBV": "hepatitis B virus",
    "HCV": "hepatitis C",
    "HIV": "immunodeficiency virus 1",
    "SARSCOV2": "respiratory syndrome coronavirus 2",
    "IAV": "influenza A",
    "CMV": "herpesvirus 5",  # IEDB taxon name: "Human herpesvirus 5"
    "DENV": "dengue virus",
    "RSV": "respiratory syncytial virus",
}

VIRUS_DISPLAY: dict[str, str] = {
    "EBV": "EBV",
    "HPV16": "HPV16",
    "HPV18": "HPV18",
    "HPV": "HPV",
    "HBV": "HBV",
    "HCV": "HCV",
    "HIV": "HIV-1",
    "SARSCOV2": "SARS-CoV-2",
    "IAV": "IAV",
    "CMV": "CMV",
    "DENV": "DENV",
    "RSV": "RSV",
}

# Assay response quality weights. Higher = more direct/reliable cytotoxic T-cell signal.
# Cytolytic assays (cytotoxicity, degranulation) are Tier 1: direct killing.
# Cytokine release assays (IFNg, TNFa) are Tier 1-2: canonical CD8+ functional readout.
# Proliferation / activation are Tier 3: less specific.
# Binding assays are Tier 4: structural but not functional immunogenicity.
ASSAY_QUALITY_MAP: dict[str, float] = {
    "cytotoxicity": 1.0,
    "degranulation": 1.0,
    "granzyme b release": 1.0,
    "perforin release": 1.0,
    "ifng release": 1.0,
    "tnfa release": 0.9,
    "il-2 release": 0.9,
    "gm-csf release": 0.9,
    "ccl4/mip-1b release": 0.8,
    "granulysin release": 0.8,
    "il-10 release": 0.7,
    "proliferation": 0.7,
    "activation": 0.7,
    "t cell-apc binding": 0.6,
    "qualitative binding": 0.5,
    "dissociation constant kd": 0.4,
    "on rate": 0.4,
    "off rate": 0.4,
    "3d structure": 0.3,
}
_DEFAULT_QUALITY = 0.5

_VALID_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")

# Fields requested from the API (reduces payload size).
_SELECT_FIELDS = ",".join(
    [
        "epitope__name",
        "epitope__source_molecule",
        "epitope__source_organism",
        "mhc_restriction__name",
        "assay__response_measured",
        "assay__qualitative_measurement",
        "reference__pmid",
    ]
)

# ---------------------------------------------------------------------------
# URL / HTTP helpers
# ---------------------------------------------------------------------------


def _build_url(organism_pattern: str, offset: int) -> str:
    """Return a paginated IEDB tcell_export URL for one organism pattern."""
    enc_organism = quote(organism_pattern, safe="*")
    enc_host = "ilike.*Homo%20sapiens*"
    parts = [
        "mhc_restriction__class=eq.I",
        "epitope__object_type=eq.Linear%20peptide",
        f"epitope__source_organism=ilike.*{enc_organism}*",
        f"host__name={enc_host}",
        "order=assay_id",
        f"limit={PAGE_SIZE}",
        f"offset={offset}",
        f"select={_SELECT_FIELDS}",
    ]
    return IEDB_API_BASE + "?" + "&".join(parts)


def _fetch_page(url: str, *, max_retries: int = 4) -> list[dict]:
    """Fetch one page from the IEDB API with exponential backoff on 429/503.

    Returns [] only after exhausting retries or on a non-retriable error.
    """
    import urllib.error

    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:  # nosec B310
                if resp.status == 200:
                    return json.loads(resp.read().decode("utf-8"))
                print(f"  Warning: HTTP {resp.status}", file=sys.stderr)
                return []
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 503) and attempt < max_retries - 1:
                delay = 2**attempt
                print(
                    f"  Rate-limited (HTTP {exc.code}), retrying in {delay}s "
                    f"(attempt {attempt + 1}/{max_retries})...",
                    file=sys.stderr,
                )
                time.sleep(delay)
            else:
                print(f"  Warning: HTTP {exc.code} after {attempt + 1} attempt(s)", file=sys.stderr)
                return []
        except Exception as exc:
            print(f"  Warning: request failed ({exc})", file=sys.stderr)
            return []
    return []


# ---------------------------------------------------------------------------
# Label / quality / allele normalization - pure functions, fully testable
# ---------------------------------------------------------------------------


def _normalize_allele(raw: str | None) -> str:
    """Normalize IEDB allele strings to HLA-X*GG:PP format where possible.

    Handles:
      HLA-A*02:01  -> HLA-A*02:01  (already canonical)
      HLA-A*0201   -> HLA-A*02:01  (8-digit, insert colon)
      HLA-B*57:01  -> HLA-B*57:01
      HLA-A2       -> HLA-A2       (supertypic - returned unchanged)
      class I      -> class I      (too ambiguous - returned unchanged)
    """
    if not raw or not isinstance(raw, str):
        return ""
    allele = raw.strip()
    # Already canonical 4+2 digit format
    if re.match(r"HLA-[A-C]\*\d{2}:\d{2}", allele):
        return allele
    # 8-digit no colon: HLA-A*0201 -> HLA-A*02:01
    m = re.match(r"HLA-([A-C])\*(\d{2})(\d{2})$", allele)
    if m:
        return f"HLA-{m.group(1)}*{m.group(2)}:{m.group(3)}"
    # Supertypic / ambiguous - return as-is
    return allele


def _assign_label(qualitative_measurement: str | None) -> int | None:
    """Map IEDB qualitative_measurement to binary label.

    Returns 1 (positive), 0 (negative), or None (exclude - inconclusive/missing).
    """
    if not isinstance(qualitative_measurement, str):
        return None
    qm = qualitative_measurement.strip().lower()
    if qm.startswith("positive"):
        return 1
    if qm.startswith("negative"):
        return 0
    return None


def _assay_quality(response_measured: str | None) -> float:
    """Return assay quality weight for a given IEDB response_measured string."""
    if not isinstance(response_measured, str):
        return _DEFAULT_QUALITY
    key = response_measured.strip().lower()
    return ASSAY_QUALITY_MAP.get(key, _DEFAULT_QUALITY)


# ---------------------------------------------------------------------------
# Record processing
# ---------------------------------------------------------------------------


def _process_records(raw_records: list[dict], virus_display: str) -> list[dict]:
    """Convert raw IEDB API records to v4-compatible dicts.

    Applies:
    - 8-11mer length filter (MHC Class I canonical range)
    - Standard amino acid filter (drops X, B, modified residues)
    - Binary label assignment (excludes inconclusive)
    - Allele normalization
    - Assay quality weighting
    """
    processed: list[dict] = []
    skipped_length = skipped_aa = skipped_label = 0

    for rec in raw_records:
        raw_peptide = rec.get("epitope__name")
        if not isinstance(raw_peptide, str):
            skipped_aa += 1
            continue
        peptide = raw_peptide.strip().upper()

        if not (8 <= len(peptide) <= 11):
            skipped_length += 1
            continue

        if not all(c in _VALID_AA for c in peptide):
            skipped_aa += 1
            continue

        label = _assign_label(rec.get("assay__qualitative_measurement"))
        if label is None:
            skipped_label += 1
            continue

        response = rec.get("assay__response_measured") or ""
        processed.append(
            {
                "peptide": peptide,
                "label": label,
                "virus": virus_display,
                "protein": rec.get("epitope__source_molecule") or "",
                "strain": rec.get("epitope__source_organism") or "",
                "hla_allele": _normalize_allele(rec.get("mhc_restriction__name")),
                "source_type": "Virus",
                "database_source": "IEDB",
                "assay_type": response,
                "assay_quality_weight": _assay_quality(response),
                "reference_pmid": str(rec.get("reference__pmid") or ""),
            }
        )

    n = len(processed)
    pos = sum(1 for r in processed if r["label"] == 1)
    neg = n - pos
    print(f"  Retained {n} records ({pos} positive, {neg} negative)")
    print(
        f"  Skipped: {skipped_length} wrong length | {skipped_aa} invalid AA | "
        f"{skipped_label} no label"
    )
    return processed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_iedb_tcell(virus_key: str) -> list[dict]:
    """Fetch all MHC Class I T-cell records for a virus from the IEDB API.

    Args:
        virus_key: One of the keys in ORGANISM_MAP (case-insensitive).

    Returns:
        List of v4-compatible dicts (peptide, label, assay_type, ...).
    """
    key = virus_key.upper()
    if key not in ORGANISM_MAP:
        raise ValueError(f"Unknown virus '{virus_key}'. Known: {sorted(ORGANISM_MAP)}")

    pattern = ORGANISM_MAP[key]
    display = VIRUS_DISPLAY.get(key, key)
    print(f"[IEDB] Fetching {display} (pattern='{pattern}') ...")

    raw: list[dict] = []
    offset = 0
    while True:
        url = _build_url(pattern, offset)
        page = _fetch_page(url)
        if not page:
            break
        raw.extend(page)
        print(f"  Page {offset // PAGE_SIZE + 1}: {len(page)} records (total so far: {len(raw)})")
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
        time.sleep(REQUEST_DELAY)

    print(f"  Raw records from API: {len(raw)}")
    return _process_records(raw, display)


def fetch_and_save(
    virus_key: str, output_path: str, schema_path: str | None = None
) -> pd.DataFrame:
    """Fetch, process, deduplicate, and save a v4-compatible CSV for one virus.

    Deduplication is on (peptide, hla_allele, assay_type): multiple assays for
    the same peptide-allele pair are kept as separate rows so that
    build_dataset_v4.py can apply majority-vote or quality-weighted dedup later.

    Args:
        virus_key: E.g. "EBV", "HIV".
        output_path: Destination CSV path.
        schema_path: Optional path to immunogenicity_dataset_v4_schema.json
                     for validation (skipped if not supplied).

    Returns:
        Processed DataFrame.
    """
    records = fetch_iedb_tcell(virus_key)
    if not records:
        print(f"  No records returned for {virus_key}. Output not written.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    # Normalize peptides (already upper-cased, but validates AA set again)
    df = normalize_peptides(df)

    # Deterministic sort so re-runs are byte-stable
    sort_cols = ["peptide", "hla_allele", "assay_type", "label"]
    df = df.sort_values([c for c in sort_cols if c in df.columns]).reset_index(drop=True)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  Saved {len(df)} rows -> {output_path}")

    write_provenance(
        output_path,
        sources=[IEDB_API_BASE],
        row_count=len(df),
        extra={
            "virus_key": virus_key.upper(),
            "positive_count": int((df["label"] == 1).sum()),
            "negative_count": int((df["label"] == 0).sum()),
            "allele_distribution": df["hla_allele"].value_counts().head(10).to_dict(),
            "assay_types": df["assay_type"].value_counts().to_dict(),
        },
    )
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch MHC Class I T-cell epitopes from the IEDB REST API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--virus",
        metavar="KEY",
        help=f"Virus key to fetch. Choices: {', '.join(sorted(ORGANISM_MAP))}",
    )
    mode.add_argument(
        "--all",
        action="store_true",
        help="Fetch all supported viruses.",
    )
    mode.add_argument(
        "--list-viruses",
        action="store_true",
        help="Print supported virus keys and IEDB organism patterns, then exit.",
    )
    p.add_argument(
        "--output",
        metavar="FILE",
        help="Output CSV path (single-virus mode). Default: data/iedb/<VIRUS>_tcell.csv",
    )
    p.add_argument(
        "--output-dir",
        metavar="DIR",
        default="data/iedb",
        help="Output directory (--all mode). Default: data/iedb/",
    )
    p.add_argument(
        "--schema",
        metavar="FILE",
        default="data/immunogenicity_dataset_v4_schema.json",
        help="v4 schema JSON for post-fetch validation.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.list_viruses:
        print(f"{'Key':<12}  {'IEDB organism pattern':<40}  Display name")
        print("-" * 70)
        for key in sorted(ORGANISM_MAP):
            print(f"{key:<12}  {ORGANISM_MAP[key]:<40}  {VIRUS_DISPLAY.get(key, key)}")
        return 0

    schema = args.schema if os.path.exists(args.schema) else None

    if args.all:
        os.makedirs(args.output_dir, exist_ok=True)
        for key in sorted(ORGANISM_MAP):
            out = os.path.join(args.output_dir, f"{key.lower()}_tcell.csv")
            print(f"\n{'=' * 60}")
            fetch_and_save(key, out, schema)
        return 0

    # Single-virus mode
    key = args.virus.upper()
    out = args.output or f"data/iedb/{key.lower()}_tcell.csv"
    fetch_and_save(key, out, schema)
    return 0


if __name__ == "__main__":
    sys.exit(main())
