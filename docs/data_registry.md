# SESTRAV Data Registry

This document catalogs the current datasets, schemas, and evidence freezes used by the SESTRAV pipeline.

## Dataset Schemas

### v3 Schema (Legacy)
*   **Status**: Deprecated for new models.
*   **Columns**: `peptide`, `label`, `virus`, `protein`, `strain`.
*   **Limitations**: No HLA-allele tracking; highly biased towards viral sources (EBV/HPV).

### v4 Schema (Frozen)
*   **Status**: Frozen. Superseded by v5. The v4 artifact (`data/immunogenicity_dataset_v4.csv`) is retained as the paper's published baseline.
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
| Superseded | - | - | - | - | - | v4 was never built standalone; v5 (commit e6aafe2, 2026-07-04) is the active dataset |

### Component artifact row counts (fill in after build)

| Script | Output | Rows | HLA alleles |
|---|---|---|---|
| `ingest_vdjdb.py` | `data/vdjdb_v4.csv` | - | - |
| `ingest_tsnadb.py` | `data/tsnadb_v4.csv` | 365,282 | multi-allele (all TSNAdb) |
| `sample_tsnadb_cohort.py` | `data/tsnadb_crossdomain_cohort.csv` | 5,000 | canonical-10 |
| `eval_tsnadb_crossdomain.py` | `results/tsnadb_crossdomain_benchmark.json` | 9,903 pool | AUC-PR 0.9909 |
| `generate_hard_decoys.py` | `data/hard_decoys.csv` | - | - |
| `build_dataset_v4.py` | `data/immunogenicity_dataset_v4.csv` | - | - |

---

## Architecture Decisions Log

Decisions that affect the v4 dataset schema, feature pipeline, or model architecture.
Recorded here so they cannot be silently reversed without a documented rationale.

| ID | Decision | Rationale | Date | Status |
|---|---|---|---|---|
| AD-1 | Canonical model changes from `feature_mode=30` to `feature_mode=31` | Ablation on v3 data: `full_31` AUC-PR 0.864 vs `combined_30` 0.825. `peptide_length` is the critical mediating variable for zero-imputed 8-mer positions p7/p8. Config change: `feature_mode: 31`; new canonical model: `rf_31feature_integrated.joblib`; old model becomes `rf_30feature_legacy.joblib`. | 2026-06-18 | **LOCKED** |
| AD-2 | Feature schema consolidation to three tiers: Legacy (21), Canonical (31), Extended (33+) | Tracks `feature_mode` 21, 30, 31, 33, 50, 166 were too many to maintain. Legacy (21): historical reproducibility only, no new models. Canonical (31): production track. Extended (33+): research track. Migration table in `docs/feature_glossary.md`. | 2026-06-18 | **LOCKED** |
| AD-3 | VDJdb v4 schema adds `tcr_alpha_cdr3` and `tcr_beta_cdr3` (nullable strings) | TCR CDR3 sequences are permanently lost if not captured at ingestion. Phase 5 TCR repertoire matching is blocked without them. These columns are nullable for non-VDJdb rows. | 2026-06-18 | **LOCKED** |
| AD-4 | TSNAdb entries are NOT included in viral v4 training; stored as separate test cohort | Neoantigen immunogenicity (tolerance escape) is mechanistically distinct from foreign antigen immunogenicity. Mixing without stratification may dilute the viral signal. `--source-types Virus VDJdb` filter in `build_dataset_v4.py`. **Operationalized 2026-06-20**: 5,000-sample cross-domain benchmark against hard decoys - AUC-PR 0.9909, AUC-ROC 0.9887 (see `results/tsnadb_crossdomain_benchmark.json`, paper §3.4.1). | 2026-06-18 | **LOCKED** |
| AD-5 | MHCflurry model version pinned in `config.yaml` (`mhcflurry_model_version: "2.2.1"`) | MHCflurry binding features change across model releases. Checksums do not catch model drift. CI must verify installed version matches config before any training run. | 2026-06-18 | **LOCKED** |
| AD-6 | Hard decoys are Week 6 Priority 1, before GNN retraining and virus expansion | Hard decoys fix the root cause of negative data quality (IEDB negatives are mostly poor MHC binders, not TCR rejectors). Order: hard decoys → v4 build → model retraining → GNN. | 2026-06-18 | **LOCKED** |
| AD-7 | SARS-CoV-2 panel: Spike (P0DTC2), N (P0DTC9), M (P0DTC5), ORF3a (P0DTC3). NSP3/NSP12 deferred. | NSP3/NSP12 sub-sequence extraction from polyprotein P0DTD1 requires validated residue coordinate mapping against UniProt canonical topology - a separate task. Panel key: `SARS_CoV2_Wuhan1_panel4`. Add in dedicated session with explicit coordinate validation. | 2026-06-18 | **LOCKED** |
| AD-8 | Hard decoy script upgraded: use `Class1PresentationPredictor` (presentation_score ≥ 0.5), screen all 10 canonical alleles, exclude IEDB positives, support 8-11-mers, target 10,000 total. | Original script used `Class1AffinityPredictor` (IC50 < 50 nM), screened one allele, and did not exclude training positives. These are qualitatively inferior decoys: presentation score is a better predictor of actual surface display than affinity alone. Multi-allele coverage ensures decoys challenge the classifier across the full breadth of training alleles. `fetch_human_proteome.py` downloads UniProt UP000005640. | 2026-06-18 | **LOCKED** |
| AD-9 | Extended `feature_mode=33` is best v3 model: AUC-PR 0.886 unweighted / 0.840 weighted (+0.022 over full_31). Canonical lightweight track remains `feature_mode=31`. | netchop_score is the most informative single feature (RF importance=0.118), confirming independent proteasomal processing signal. full_33 recommended for production where antigen processing cache is available. | 2026-06-18 | **LOCKED** |

---

## v4 Composition Audit (2026-06-24)

Measured from `data/immunogenicity_dataset_v4.csv` (14,699 total rows).

| Virus | n | Pos rate | Assessment |
|---|---|---|---|
| Self (hard decoys) | 5,000 | 0% | Synthetic negatives - OK |
| SARS-CoV-2 | 4,057 | 69.9% | Dominant (28% of viral rows); COVID lit surge artifact |
| CMV | 1,377 | 58.5% | Usable |
| DENV | 874 | 96.2% | Near-no negatives; positive-reporting bias |
| HCV | 719 | 57.6% | Usable |
| HBV | 650 | 60.5% | Usable |
| IAV | 520 | 70.4% | Usable |
| EBV | 494 | 78.9% | Thin on real negatives |
| HIV-1 | 464 | 81.7% | Thin on real negatives |
| HPV16 | 208 | 73.1% | Thin; label fragmented |
| HPV | 144 | 28.5% | Fragmented (HPV/HPV16/HPV18 are separate labels) |
| RSV | 123 | 14.6% | Near-no positives |
| HPV18 | 36 | - | Too thin |
| ~17 singleton viruses | 1-6 each | ~100% | Unusable noise; quarantined in v5 |

**Peptide length distribution:** 8mer 621, 9mer 9,158, 10mer 3,674, 11mer 1,246.

**Key finding:** LOO cross-virus AUC-ROC is near-random (0.46-0.59) for every held-out virus.
Signal does not currently transfer across viruses. v4 pooled AUC-PR 0.7635 is partially a
SARS-CoV-2 composition artifact. See `results/loo_cross_virus_v4.json`.

---

## v5 Schema (Active)

- **Status**: Active. Built and deployed 2026-07-04 (commit e6aafe2).
- **Schema File**: `data/immunogenicity_dataset_v5_schema.json`
- **v4 frozen**: Do NOT modify `data/immunogenicity_dataset_v4.csv`. v5 is a separate file.

**New columns vs v4:**

| Column | Type | Purpose |
|---|---|---|
| `virus_family` | string/null | Taxonomic family; enables family-aware modeling (Q3) |
| `negative_origin` | enum/null | Origin of negative rows; enables negative-set ablation (Q4) |
| `assay_type` | string/null | IEDB assay classification |
| `assay_quality_weight` | float/null | Quality weight: 1.0 direct functional, 0.7 indirect, 0.5 expanded-culture |
| `assay_quality_tier` | int/null (1/2/3) | Integer companion to `assay_quality_weight` for stratification queries (Amendment 2) |
| `reference_pmid` | string/null | PubMed ID for provenance; required for IEDB rows |
| `iedb_assay_id` | string/null | Persistent IEDB assay/reference identifier; row-level provenance (Amendment 1/2) |
| `infection_phase` | enum/null | HBV acute vs chronic epitope hierarchy (Amendment 2) |
| `antigen_latency_program` | enum/null | EBV lytic vs latency-I/II/III antigen program (Amendment 2) |
| `assay_context` | enum/null | Immunization context: vaccine_induced / natural_infection / unknown (Amendment 4) |
| `cross_reactivity_tested` | bool/null | HPV16/18 epitope tested against both strains vs assumed (Amendment 2) |
| `virus_taxon_id` | int/null | NCBI Taxonomy ID; FAIR/Zenodo controlled vocabulary (Amendment 2) |
| `is_quarantined` | bool | True if virus has < 50 rows or < 10 real negatives |

**Ingestion/build hardening status (Amendments 1-3, 2026-06-25):**
- `scripts/ingest_iedb_negatives.py` hardened: two-row IEDB header detection,
  `response_frequency < 0.1` secondary-negative signal, `mhcgnomes` 4-digit HLA
  normalization (supertypes/non-human/Class II rejected), intra-export dedup on
  `(peptide, hla_allele, reference_pmid)`, `iedb_assay_id` provenance, 3-tier
  assay quality, and `input_sha256` in the provenance sidecar.
- `scripts/build_dataset_v5.py`: `ensure_v5_columns` + `V5_COLUMNS` carry all 6
  new columns; `--dry-run` runs the v4-positive / IEDB-negative conflict audit
  (`data/holding/conflicts_v5_preaudit.csv`) and a <3-distinct-PMID warning for
  the target viruses. Reviewed before any non-dry-run build.
- Tests: `tests/test_ingest_iedb_negatives.py`, `tests/test_build_dataset_v5.py`
  (56 tests; ruff + mypy clean; bandit 0 issues).

**HPV normalization in v5:**
- `virus=HPV16` -> `virus=HPV`, `strain=HPV16`
- `virus=HPV18` -> `virus=HPV`, `strain=HPV18`
- `virus=HPV` with null strain -> `strain=HPV_generic`

**Singleton quarantine threshold:** >= 50 rows AND >= 10 `negative_origin=tested_negative` rows.

---

## v5 Build Log

| Date | Git SHA | Total rows | Active rows | IEDB negatives | Quarantined viruses | Notes |
|---|---|---|---|---|---|---|
| 2026-07-04 | e6aafe2 | 46,386 | 31,999 | 36,689 | 17 singletons (< 50 rows or < 10 real negatives) | Merged 4,219 net-new IEDB API negatives via `scripts/merge_iedb_api_negatives.py`; B*27 EBV conflict quarantine (3 rows: FRKAQIQGL x2, RRARSLSAERY); RF v5 pooled AUC-PR 0.7678 / AUC-ROC 0.9368 (both same-pathogen-labeled at the time). NOTE (2026-07-11): both figures later retracted as decoy-inflated; see claims_register.md (D12) and per_virus_eval_v5_mode31.csv (per-virus within-CV mean 0.751; honest pooled ROC 0.712). This row does not describe the shipped corpus - see the 2026-07-05 row below, logged 2026-08-09 (H7). |
| 2026-07-05 | d3972f7 (provenance `be3e260`) | 51,185 | 35,597 | 36,689 | 17 singletons (unchanged criteria) | **This is the shipped corpus** (`data/immunogenicity_dataset_v5.csv`, checksum `1c596ab7...`), superseding the row above. Merged 9,206 published-panel rows (475 panel/IEDB conflicts) via the same-day provenance build (`be3e260`), then `d3972f7` fixed a quarantine-reset bug (`apply_quarantine()` was clearing row-level HLA quarantines set by `normalize_hla_alleles()`) and added 374 SARS-CoV-2 + 300 CMV allele-matched-nonbinder decoys. Pooled cross-validation on this corpus: RF AUC-ROC 0.9429 ± 0.0036, AUC-PR 0.8312 ± 0.0084 (`models/v5/training_results_mode31.csv`). Logged retroactively 2026-08-09 - this build was never previously entered in this table (H7, `_local/state/conflict_register_2026-08-08.md`). Leakage-inflated per D15 like the row above; not a self-proteome or "Gate 1" metric. |

---

## Phase 2 Architecture Decisions

| ID | Decision | Rationale | Date | Status |
|---|---|---|---|---|
| AD-10 | v5 is a new file; v4 is frozen | v4 is the paper's published artifact; v5 is the research/depth dataset. Numbers must not be confused. | 2026-06-24 | **LOCKED** |
| AD-11 | Per-virus AUC-ROC replaces pooled AUC-PR as headline metric | Pooled AUC-PR is inflated by SARS-CoV-2 dominance and positive-rate imbalance. Per-virus AUC-ROC with matched negatives is the honest measure of immunogenicity discrimination. | 2026-06-24 | **LOCKED** |
| AD-12 | No new viruses until existing set is deep | LOO shows no cross-virus signal transfer. Adding more viruses does not improve per-virus performance; going deeper on existing viruses does. | 2026-06-24 | **LOCKED** |
| AD-13 | Singleton quarantine threshold: >= 50 rows AND >= 10 real tested negatives | Below this threshold, per-virus AUC-ROC is unreliable. The 17 singleton viruses inflate the virus count headline without contributing trainable signal. | 2026-06-24 | **LOCKED** |

---

Build command:
```bash
# 1. Download VDJdb (auto-fetched if omitted):
python scripts/ingest_vdjdb.py --output data/vdjdb_v4.csv

# 2. Provide TSNAdb file manually:
python scripts/ingest_tsnadb.py --input <tsnadb_path> --output data/tsnadb_v4.csv

# 3. Generate hard decoys (requires MHCflurry models):
mhcflurry-downloads fetch models_class1_presentation
python scripts/fetch_human_proteome.py  # download UP000005640 (~100 MB, once only)
python scripts/generate_hard_decoys.py \
    --fasta data/proteomes/human_uniprot_UP000005640.fasta \
    --training-data data/immunogenicity_dataset_v3.csv \
    --output data/hard_decoys.csv \
    --num-decoys 10000

# 4. Merge and validate:
python scripts/build_dataset_v4.py
```
