# SESTRAV Snakemake Flow

## Execution Model
The pipeline is orchestrated via Snakemake (`Snakefile` -> `pipeline.smk`), processing configurations defined in `config.yaml`.
The fundamental workflow manages raw IEDB export curation, peptide sequence expansion, continuous feature extraction (binding matrices, physicochemical traits), and model evaluation (ANN and GNN).
The pipeline natively supports dynamic execution based on `dataset_mode`, `dataset_version`, and `feature_mode` (`21` vs `50`).

## Major DAG Rules & Dependencies

1. **`generate_peptides`** (Stage 1)
   - **Input**: Proteome FASTA files (e.g., HPV/EBV)
   - **Output**: `{proteome_id}_peptides.csv`
   - Extracts length-constrained peptides (typically 8-11mers).

2. **`predict_binding`** (Stage 2)
   - **Input**: `{proteome_id}_peptides.csv`
   - **Output**: `{proteome_id}_binding.csv`
   - Integrates with HLA allele configurations for binding affinity predictions.

3. **`extract_features`** (Stage 3)
   - **Input**: `{proteome_id}_binding.csv`
   - **Output**: `{proteome_id}_features.csv`
   - Applies continuous feature scaling depending on the feature mode configuration.

4. **`score_immunogenicity`** (Stage 4)
   - **Input**: `{proteome_id}_features.csv`
   - **Output**: `{proteome_id}_ranked.csv`, visualization PNGs
   - Predicts and ranks candidate epitopes via the integrated models.

5. **`qc_dataset`**
   - **Input**: `data/immunogenicity_dataset_v3.csv`
   - **Output**: `results/qc/dataset_qc.json`
   - Strict checksum and integrity validation prior to training (`src/data_curation_qc.py`).

6. **`train_ann` & `train_gnn`**
   - **Input**: `data/immunogenicity_dataset_v3.csv`, QC output, binding matrices.
   - **Output**: `models/ann/ann_model.pth`, `models/gnn/gnn_model.pth`.
   - Trains multi-layer perceptrons or molecular graph nets using stratified cross-validation.

7. **`full_validation_report`**
   - Integrates outputs to build gold-standard benchmarking reports and cross-tier summaries.
   - Validates metrics like AUC-ROC, AUC-PR, and ISSR@10.

## Freeze Mode
If `freeze_mode` is enabled in `config.yaml`, the pipeline halts if the ingested dataset checksum does not explicitly match the provenance definition in `config.yaml`, ensuring strictly reproducible deployment environments.
