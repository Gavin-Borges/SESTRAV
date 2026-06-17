# SESTRAV Data Registry

This document catalogs the current datasets, schemas, and evidence freezes used by the SESTRAV pipeline.

## Dataset Schemas

### v3 Schema (Legacy)
*   **Status**: Deprecated for new models.
*   **Columns**: `peptide`, `label`, `virus`, `protein`, `strain`.
*   **Limitations**: No HLA-allele tracking; highly biased towards viral sources (EBV/HPV).

### v4 Schema (Current)
*   **Status**: Active.
*   **Description**: Introduced in the Post-Badge Phase 1 expansion.
*   **Schema File**: `data/immunogenicity_dataset_v4_schema.json`
*   **New Columns**:
    *   `hla_allele`: Crucial for pan-allele prediction and zero-shot capabilities.
    *   `source_type`: (Virus, Tumor, Self) to categorize the expanded origin of data.
    *   `database_source`: Provenance tracking (IEDB, VDJdb, UniProt, TSNAdb).

## Current Data Sources

1.  **IEDB (Immune Epitope Database)**: Historical v3 data, primarily viral (EBV, HPV).
2.  **UniProt (Hard Decoys)**: Computationally generated true negatives (MHC binders that are non-immunogenic self-peptides).
3.  **VDJdb**: Curated TCR-pMHC database providing high-confidence true positives with paired HLA data.
4.  **TSNAdb / dbPepNeo**: Tumor-specific neoantigens providing true positives in the oncology domain.

## Governance & Quality Control
All data ingestion must adhere to the schema and pass the `freeze_mode` assertions defined in the SESTRAV architecture.

---

## v4 Build Log

Record each production `build_dataset_v4.py` run here.  Each run must be
accompanied by a `_provenance.json` sidecar next to the output artifact.

| Date | Git SHA | Total rows | Positives | Negatives | Pos rate | Notes |
|---|---|---|---|---|---|---|
| PENDING | — | — | — | — | — | Blocked on VDJdb download + MHCflurry models |

### Component artifact row counts (fill in after build)

| Script | Output | Rows | HLA alleles |
|---|---|---|---|
| `ingest_vdjdb.py` | `data/vdjdb_v4.csv` | — | — |
| `ingest_tsnadb.py` | `data/tsnadb_v4.csv` | — | — |
| `generate_hard_decoys.py` | `data/hard_decoys.csv` | — | — |
| `build_dataset_v4.py` | `data/immunogenicity_dataset_v4.csv` | — | — |

Build command:
```bash
# 1. Download VDJdb (auto-fetched if omitted):
python scripts/ingest_vdjdb.py --output data/vdjdb_v4.csv

# 2. Provide TSNAdb file manually:
python scripts/ingest_tsnadb.py --input <tsnadb_path> --output data/tsnadb_v4.csv

# 3. Generate hard decoys (requires MHCflurry models):
mhcflurry-downloads fetch models_class1_presentation
python scripts/generate_hard_decoys.py --fasta <uniprot_human.fasta> --output data/hard_decoys.csv

# 4. Merge and validate:
python scripts/build_dataset_v4.py
```
