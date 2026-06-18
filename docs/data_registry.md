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

---

## Architecture Decisions Log

Decisions that affect the v4 dataset schema, feature pipeline, or model architecture.
Recorded here so they cannot be silently reversed without a documented rationale.

| ID | Decision | Rationale | Date | Status |
|---|---|---|---|---|
| AD-1 | Canonical model changes from `feature_mode=30` to `feature_mode=31` | Ablation on v3 data: `full_31` AUC-PR 0.864 vs `combined_30` 0.825. `peptide_length` is the critical mediating variable for zero-imputed 8-mer positions p7/p8. Config change: `feature_mode: 31`; new canonical model: `rf_31feature_integrated.joblib`; old model becomes `rf_30feature_legacy.joblib`. | 2026-06-18 | **LOCKED** |
| AD-2 | Feature schema consolidation to three tiers: Legacy (21), Canonical (31), Extended (33+) | Tracks `feature_mode` 21, 30, 31, 33, 50, 166 were too many to maintain. Legacy (21): historical reproducibility only, no new models. Canonical (31): production track. Extended (33+): research track. Migration table in `docs/feature_glossary.md`. | 2026-06-18 | **LOCKED** |
| AD-3 | VDJdb v4 schema adds `tcr_alpha_cdr3` and `tcr_beta_cdr3` (nullable strings) | TCR CDR3 sequences are permanently lost if not captured at ingestion. Phase 5 TCR repertoire matching is blocked without them. These columns are nullable for non-VDJdb rows. | 2026-06-18 | **LOCKED** |
| AD-4 | TSNAdb entries are NOT included in viral v4 training; stored as separate test cohort | Neoantigen immunogenicity (tolerance escape) is mechanistically distinct from foreign antigen immunogenicity. Mixing without stratification may dilute the viral signal. `--source-types Virus VDJdb` filter in `build_dataset_v4.py`. | 2026-06-18 | **LOCKED** |
| AD-5 | MHCflurry model version pinned in `config.yaml` (`mhcflurry_model_version: "2.0.1"`) | MHCflurry binding features change across model releases. Checksums do not catch model drift. CI must verify installed version matches config before any training run. | 2026-06-18 | **LOCKED** |
| AD-6 | Hard decoys are Week 6 Priority 1, before GNN retraining and virus expansion | Hard decoys fix the root cause of negative data quality (IEDB negatives are mostly poor MHC binders, not TCR rejectors). Order: hard decoys → v4 build → model retraining → GNN. | 2026-06-18 | **LOCKED** |
| AD-7 | SARS-CoV-2 panel: Spike (P0DTC2), N (P0DTC9), M (P0DTC5), ORF3a (P0DTC3). NSP3/NSP12 deferred. | NSP3/NSP12 sub-sequence extraction from polyprotein P0DTD1 requires validated residue coordinate mapping against UniProt canonical topology — a separate task. Panel key: `SARS_CoV2_Wuhan1_panel4`. Add in dedicated session with explicit coordinate validation. | 2026-06-18 | **LOCKED** |

---

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
