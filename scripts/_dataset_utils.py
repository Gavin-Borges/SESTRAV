"""Shared helpers for SESTRAV v4 dataset-building scripts.

Enforces three guarantees across all ingest scripts: deterministic peptide
normalization (uppercase, stripped whitespace), JSON-schema validation against
the v4 schema, and provenance sidecars written alongside every output file.
Underscore-prefixed so it is not itself treated as an ingest script.
"""

import hashlib
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


MHC_CLASS_I_MIN_LEN = 8  # MHC Class I canonical peptide length lower bound
MHC_CLASS_I_MAX_LEN = 11  # MHC Class I canonical peptide length upper bound


def normalize_peptides(
    df, peptide_col="peptide", min_len=MHC_CLASS_I_MIN_LEN, max_len=MHC_CLASS_I_MAX_LEN
):
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
        print(
            f"Dropped {n_bad_len} rows outside MHC Class I length range "
            f"({min_len}-{max_len}mer): {df.loc[aa_mask & ~len_mask, peptide_col].tolist()[:5]}"
        )
    return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# HLA allele normalization
# ---------------------------------------------------------------------------

# Maps low-resolution or alias allele strings to the canonical HLA format
# (HLA-X*XX:XX) used throughout SESTRAV. Only aliases that unambiguously
# resolve to one specific allele are included; ambiguous supertypes like
# "HLA-B*57" (covers B*57:01 and B*57:03) are intentionally omitted.
# Derived from published HLA frequency databases and IEDB allele annotations.
_HLA_ALIAS_MAP: dict[str, str] = {
    # A*01:01 aliases
    "HLA-A1": "HLA-A*01:01",
    "HLA-A*01": "HLA-A*01:01",
    "HLA-A0101": "HLA-A*01:01",
    # A*02:01 aliases (dominant A2 subtype in Western cohorts)
    "HLA-A2": "HLA-A*02:01",
    "HLA-A*02": "HLA-A*02:01",
    "HLA-A0201": "HLA-A*02:01",
    # A*03:01 aliases
    "HLA-A3": "HLA-A*03:01",
    "HLA-A*03": "HLA-A*03:01",
    "HLA-A0301": "HLA-A*03:01",
    # A*11:01 aliases
    "HLA-A11": "HLA-A*11:01",
    "HLA-A*11": "HLA-A*11:01",
    "HLA-A1101": "HLA-A*11:01",
    # A*24:02 aliases
    "HLA-A24": "HLA-A*24:02",
    "HLA-A*24": "HLA-A*24:02",
    "HLA-A2402": "HLA-A*24:02",
    # B*07:02 aliases
    "HLA-B7": "HLA-B*07:02",
    "HLA-B*07": "HLA-B*07:02",
    "HLA-B0702": "HLA-B*07:02",
    # B*08:01 aliases
    "HLA-B8": "HLA-B*08:01",
    "HLA-B*08": "HLA-B*08:01",
    "HLA-B0801": "HLA-B*08:01",
    # B*27:05 aliases
    "HLA-B27": "HLA-B*27:05",
    "HLA-B*27": "HLA-B*27:05",
    "HLA-B2705": "HLA-B*27:05",
    # B*35:01 aliases
    "HLA-B35": "HLA-B*35:01",
    "HLA-B*35": "HLA-B*35:01",
    "HLA-B3501": "HLA-B*35:01",
    # B*44:02 aliases
    "HLA-B44": "HLA-B*44:02",
    "HLA-B*44": "HLA-B*44:02",
    "HLA-B4402": "HLA-B*44:02",
}

# Allele strings that are too ambiguous to map safely; rows carrying these
# are quarantined (is_quarantined=True) so they don't pollute training.
_HLA_AMBIGUOUS: frozenset[str] = frozenset(
    [
        # Fully unresolved class annotations
        "HLA class I",
        "HLA-class I",
        "Class I",
        "HLA class II",
        # Half-resolved supertypes NOT in _HLA_ALIAS_MAP - no single dominant subtype
        # can be assumed. B*57:01 and B*57:03 have distinct clinical profiles (e.g.
        # abacavir hypersensitivity). If a source reports only "HLA-B57", quarantine.
        "HLA-B57",
        "HLA-B*57",
    ]
)


def normalize_hla_alleles(df, allele_col: str = "hla_allele"):
    """Resolve low-resolution HLA alias strings to canonical HLA-X*XX:XX format.

    Returns (updated_df, summary_dict). Rows with fully ambiguous alleles
    (e.g. 'HLA class I') have is_quarantined set to True so they are excluded
    from training but preserved for audit.
    """
    df = df.copy()
    original = df[allele_col].copy()

    df[allele_col] = df[allele_col].map(
        lambda a: _HLA_ALIAS_MAP.get(str(a).strip(), str(a).strip())
    )

    n_resolved = int((df[allele_col] != original).sum())

    ambiguous_mask = original.isin(_HLA_AMBIGUOUS)
    n_ambiguous = int(ambiguous_mask.sum())
    if n_ambiguous > 0 and "is_quarantined" in df.columns:
        df.loc[ambiguous_mask, "is_quarantined"] = True

    summary = {
        "aliases_resolved": n_resolved,
        "ambiguous_quarantined": n_ambiguous,
    }
    if n_resolved:
        print(f"  Resolved {n_resolved} low-resolution HLA aliases to canonical format.")
    if n_ambiguous:
        print(f"  Quarantined {n_ambiguous} rows with ambiguous alleles ('HLA class I' etc.).")
    return df, summary


# ---------------------------------------------------------------------------
# Virus name normalization
# ---------------------------------------------------------------------------

# Maps source-specific virus name strings to the canonical short labels used
# throughout SESTRAV. IEDB uses full NCBI taxonomy names for negatives while
# the IEDB T-cell loader uses short codes for positives; without this step
# "Influenza A virus" (IEDB negatives) and "IAV" (IEDB positives) remain as
# separate viruses, splitting their training rows and breaking dedup.
# VDJdb uses its own taxonomy names (belt-and-suspenders over ingest_vdjdb.py).
_VIRUS_NAME_MAP: dict[str, str] = {
    # IEDB full taxonomy -> SESTRAV short label
    "Influenza A virus": "IAV",
    "Influenza B virus": "IBV",
    "Influenza A Virus": "IAV",
    "Human immunodeficiency virus 1": "HIV-1",
    "Human immunodeficiency virus type 1": "HIV-1",
    "Severe acute respiratory syndrome coronavirus 2": "SARS-CoV-2",
    "Hepatitis C virus": "HCV",
    "Hepatitis B virus": "HBV",
    "Dengue virus": "DENV",
    "Dengue virus 2": "DENV",
    "Epstein-Barr virus": "EBV",
    "Human cytomegalovirus": "CMV",
    "Cytomegalovirus": "CMV",
    "Human papillomavirus": "HPV",
    "Human papillomavirus type 16": "HPV",
    "Human papillomavirus type 18": "HPV",
    "Respiratory syncytial virus": "RSV",
    # VDJdb species names (also mapped in ingest_vdjdb.py)
    "InfluenzaA": "IAV",
    "InfluenzaB": "IBV",
    "HPV16": "HPV",
    "HPV18": "HPV",
}


def normalize_virus_names(df, virus_col: str = "virus"):
    """Map source-specific virus name strings to SESTRAV canonical short labels.

    Returns (updated_df, n_resolved). Applied before dedup so that the same
    epitope arriving from different sources with different virus name conventions
    collapses to a single canonical row.
    """
    df = df.copy()
    original = df[virus_col].copy()
    df[virus_col] = df[virus_col].map(lambda v: _VIRUS_NAME_MAP.get(str(v).strip(), str(v).strip()))
    n_resolved = int((df[virus_col] != original).sum())
    if n_resolved:
        changed = original[df[virus_col] != original].value_counts()
        print(f"  Normalized {n_resolved} virus name aliases: {changed.to_dict()}")
    return df, n_resolved


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
    # Record a content hash so the integrity harness can detect silent drift of the
    # artifact. Hash only when the output file exists (it does in real ingest usage,
    # where the artifact is written before its sidecar).
    if os.path.isfile(output_path):
        h = hashlib.sha256()
        with open(output_path, "rb") as af:
            for chunk in iter(lambda: af.read(65536), b""):
                h.update(chunk)
        prov["sha256"] = h.hexdigest()
    if extra:
        prov.update(extra)
    prov_path = os.path.splitext(output_path)[0] + "_provenance.json"
    with open(prov_path, "w") as f:
        json.dump(prov, f, indent=2)
    print(f"Wrote provenance to {prov_path}")
