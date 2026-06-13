# SESTRAV IEDB Data Curation

## Pipeline Overview
The data curation layer (`src/iedb_data_loader.py` and `src/data_curation_qc.py`) ingests raw dataset exports from the IEDB (Immune Epitope Database) and sanitizes them for machine learning ingestion. It strictly enforces conflict resolution and hard reproducibility.

## Ingestion & Parsing (`src/iedb_data_loader.py`)
The pipeline supports two concurrent IEDB export topologies:
1. **Epitope Table Exports**: Encodes T-cell assay immunogenicity directly in the filename (`*positive*` vs `*negative*`). Handles variable offsets (e.g., the HPV16 16-column layout vs. the standard 32-column layout).
2. **T-cell Assay Exports**: Infers immunogenicity from internal "Qualitative Measure" strings (Positive, Positive-High, Negative, etc.).

### Inference Logic
- **Protein Inference**: Heuristically maps unstructured "Antigen Name" free-text to canonical symbols (e.g., "Protein E7" -> `E7`, "latent membrane protein 2" -> `LMP2A`).
- **Strain Inference**: Extracts explicit strains where necessary (e.g., B95-8, GD1) to prevent leakage during cross-validation.
- **Constraints**: Drops sequences outside of 8-11mer lengths or containing non-canonical amino acids (`B`, `J`, `Z`, `X`).

## Deduplication & Conflict Resolution
The curation layer inherently mitigates conflicting assays (the same peptide testing positive in one lab and negative in another):
- Uses a **majority vote** aggregate (`mean >= 0.5`). 
- If the ratio of ambiguous conflicts exceeds 15% across the dataset, the `IEDBDataCurator` throws a hard `RuntimeError`.

## Strict QC Gates (`src/data_curation_qc.py`)
1. **Explicit Negative Mining**: Pulls T-cell negative assays where MHC Binding Affinity is unusually high (< 500nM), extracting "hard negatives" to force the model to learn structural immunogenicity instead of mere anchor-binding physics.
2. **Freeze Mode Checksums**: Validates the SHA-256 hash of the fully compiled `immunogenicity_dataset_v3.csv` against `config.yaml` to ensure zero drift in upstream processing.

## Validation & Benchmarks
- Automatically excludes the `GOLD_STANDARD_EPITOPES` (16 well-known sequences) to prevent data leakage during base training loops.
- Can optionally merge hard-negative external datasets (e.g., `load_schmidt_2021`) as secondary benchmark targets.
