# AI Repository Manifest: SESTRAV Dev State

## 1. Minimalist Repository Map
```
SESTRAV-Dev/
├── config.yaml                    # Central Pydantic-validated pipeline settings
├── Snakefile & pipeline.smk       # Snakemake workflow specification
├── pyproject.toml                 # Package metadata and requirements definition
├── .claudeignore                  # Exclusions for context token defense
├── CLAUDE.md                      # Operational and environment entry protocol
├── api/                           # FastAPI interface and API configurations
├── app/                           # Frontend/Streamlit demos
├── data/                          # Target proteomes & dataset governance files
│   ├── immunogenicity_dataset_v3.csv
│   └── proteomes/
├── docs/                          # Architecture, compliance, and design docs
│   ├── AI_REPOSITORY_MANIFEST.md  # This document
│   ├── security_compliance.md     # OpenSSF posture tracking
│   └── architecture/              # GNN, AlphaFold, and alignment records
├── models/                        # Pre-trained models and checkpoints
├── scripts/                       # Stage-specific runners and benchmark wrappers
├── src/                           # Pipeline core modules
│   ├── core/                      # Configuration, FeatureStore, ModelRegistry
│   │   ├── config.py
│   │   ├── feature_store.py
│   │   └── model_registry.py
│   ├── gnn/                       # Structural GNN (graph construction & layers)
│   ├── verify/                    # Multi-virus validation & GNN evaluation
│   ├── features.py                # Vectorized feature computation
│   ├── model.py                   # PyTorch models and classifier wrappers
│   └── train_ann.py / train_gnn.py
└── tests/                         # Pytest test suite (>100 test cases)
```

## 2. Core Model Logic & Pipeline Stages
- **Stage 1 (Peptide Generation)**: Extracts all sliding-window $k$-mers ($k \in [8, 11]$) from target antigen fasta sequences ([stage1.py](file:///C:/Users/gavin/.gemini/antigravity/scratch/SESTRAV/SESTRAV-Dev/scripts/stage1.py)).
- **Stage 2 (Binding Prediction)**: Scores HLA-peptide binding affinities across active alleles using `mhcflurry` ([stage2.py](file:///C:/Users/gavin/.gemini/antigravity/scratch/SESTRAV/SESTRAV-Dev/scripts/stage2.py)).
- **Stage 3 (Feature Extraction)**: Computes physico-chemical features, multi-allele binding profiles, and antigen processing scores (ERAP1/2 trimming & TAP transport proxy) ([stage3.py](file:///C:/Users/gavin/.gemini/antigravity/scratch/SESTRAV/SESTRAV-Dev/scripts/stage3.py), [features.py](file:///C:/Users/gavin/.gemini/antigravity/scratch/SESTRAV/SESTRAV-Dev/src/features.py)).
- **Stage 4 (Epitope Scoring)**: Scores and ranks peptides utilizing ensemble models (Random Forest, XGBoost, or PyTorch ANN) under strict validation limits ([stage4.py](file:///C:/Users/gavin/.gemini/antigravity/scratch/SESTRAV/SESTRAV-Dev/scripts/stage4.py)).
- **Stage 5 (GNN / Validation)**: Incorporates 3D structural graph representations of peptides mapped to MHC coordinates, validated via multi-virus cohorts (SARS-CoV-2, Flu A, HCV) ([sestrav_evaluator.py](file:///C:/Users/gavin/.gemini/antigravity/scratch/SESTRAV/SESTRAV-Dev/src/verify/sestrav_evaluator.py)).

## 3. Milestones: v1.0 Baseline vs. v2.0 Modernization
- **v1.0 Baseline**:
  - Sequence-only feature set (21 features).
  - Unstructured dictionary configurations and raw YAML parses.
  - Manual files reading/writing (`pd.read_csv`, `df.to_csv`) with hardcoded paths.
  - Basic Random Forest models without structural graph integration.
- **v2.0 Modernization & GNN Integration**:
  - Multi-allele and antigen processing feature expansion (30 canonical / 50 expanded modes).
  - Strict configuration enforcement via Pydantic `SestravConfig`.
  - Standardized file operations via `FeatureStore` with `freeze_mode` guards.
  - Centralized model loading and serialization checking using `ModelRegistry`.
  - Introduction of Structural GNNs with AlphaFold structural predictions mapping TCR interfaces.
  - Comprehensive testing coverage and OpenSSF Passing badge compliance readiness.
