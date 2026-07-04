"""Shared helpers for SESTRAV v4 dataset-building scripts.

Enforces three guarantees across all ingest scripts: deterministic peptide
normalization (uppercase, stripped whitespace), JSON-schema validation against
the v4 schema, and provenance sidecars written alongside every output file.
Underscore-prefixed so it is not itself treated as an ingest script.
"""
import json
import os
import subprocess
from datetime import datetime, timezone

import jsonschema

# Standard 20 amino acids; matches the v4 schema's peptide `pattern`.
VALID_AA_PATTERN = r"^[ACDEFGHIKLMNPQRSTVWY]+$"


def git_sha() -> str:
    """Return the current commit SHA, or 'unknown' if unavailable."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


MHC_CLASS_I_MIN_LEN = 8   # MHC Class I canonical peptide length lower bound
MHC_CLASS_I_MAX_LEN = 11  # MHC Class I canonical peptide length upper bound


def normalize_peptides(df, peptide_col="peptide",
                       min_len=MHC_CLASS_I_MIN_LEN, max_len=MHC_CLASS_I_MAX_LEN):
    """Strip/upper-case peptides and drop biologically invalid rows.

    Enforces:
    1. Standard amino acids only (no X, B, modified residues, whitespace).
    2. MHC Class I canonical length: 8-11 amino acids.

    MHC Class I-restricted epitopes are canonically 8-11 residues. VDJdb and
    other sources occasionally include longer peptides that may reflect Class II
    epitopes, nested peptide pools, or annotation errors. These must be excluded
    because binding prediction and TCR-contact features are calibrated for 8-11mers.
    """
    df = df.copy()
    df[peptide_col] = df[peptide_col].astype(str).str.strip().str.upper()

    aa_mask = df[peptide_col].str.match(VALID_AA_PATTERN, na=False)
    n_bad_aa = int((~aa_mask).sum())

    lengths = df[peptide_col].str.len()
    len_mask = (lengths >= min_len) & (lengths <= max_len)
    n_bad_len = int((aa_mask & ~len_mask).sum())

    mask = aa_mask & len_mask
    if n_bad_aa:
        print(f"Dropped {n_bad_aa} rows with non-standard amino acid residues.")
    if n_bad_len:
        print(f"Dropped {n_bad_len} rows outside MHC Class I length range "
              f"({min_len}-{max_len}mer): {df.loc[aa_mask & ~len_mask, peptide_col].tolist()[:5]}")
    return df[mask].reset_index(drop=True)


def validate_against_schema(df, schema_path):
    """Validate a dataframe's records against the v4 JSON schema before write."""
    import math

    with open(schema_path, "r") as f:
        schema = json.load(f)

    # pandas to_dict converts NaN → float('nan'); JSON schema needs null (None).
    records = df.to_dict(orient="records")
    for rec in records:
        for k, v in rec.items():
            if isinstance(v, float) and math.isnan(v):
                rec[k] = None

    jsonschema.validate(instance=records, schema=schema)
    print(f"Schema validation passed ({len(df)} records).")


def write_provenance(output_path, sources, row_count, extra=None):
    """Write a `<output>_provenance.json` sidecar (source, counts, date, git SHA)."""
    prov = {
        "output": os.path.basename(output_path),
        "sources": sources,
        "row_count": int(row_count),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
    }
    if extra:
        prov.update(extra)
    prov_path = os.path.splitext(output_path)[0] + "_provenance.json"
    with open(prov_path, "w") as f:
        json.dump(prov, f, indent=2)
    print(f"Wrote provenance to {prov_path}")
