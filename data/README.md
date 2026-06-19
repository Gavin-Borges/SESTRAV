# SESTRAV Data Directory

This directory contains all input datasets, reference sequences, and archives for the SESTRAV pipeline.

---

## Directory Structure at a Glance

```
data/
├── immunogenicity_dataset_v3.csv      [tracked] canonical training dataset
├── immunogenicity_dataset_v4_schema.json [tracked] v4 schema spec
├── antigen_processing_cache.csv       [tracked] MHCflurry/antigen proc. features
├── README.md                          [tracked] this file
│
├── allele_aware/                      [tracked] pan-allele training subsets
├── archive/                           [tracked] superseded dataset versions
├── consensus_sequences/               [tracked] reference FASTA for peptide generation
├── proteomes/                         [tracked] curated virus proteome panels
├── verify/proteomes/                  [tracked] test fixture FASTAs (verification targets)
│
├── raw/                               [gitignored] raw IEDB .xlsx and legacy CSVs
├── iedb/                              [gitignored] IEDB REST API output (fetch_iedb_tcell.py)
├── self_similarity_cache.csv          [gitignored] human proteome k-mer lookup cache
├── self_similarity_cache_provenance.json [gitignored] cache provenance sidecar
└── external/                          [gitignored] external-tool benchmarking data
```

**Gitignored data** (large, regenerable, or sensitive) is never committed. Run the relevant
script to regenerate locally. See the `scripts/` docstrings for exact commands.

---

## Primary Training Dataset

| File | Version | Records | Description |
|------|---------|---------|-------------|
| `immunogenicity_dataset_v3.csv` | v2.0.0-alpha | 720 peptides (506 pos / 214 neg) | **Canonical training dataset.** Used by all training, evaluation, and validation modules. Source: curated IEDB records (IEDB-20260521-EXPANDED-v2.1). SHA256 checksum in `config.yaml` under `dataset_governance`. |

> **This is the file all pipeline and training commands should reference.** The root-level
> `immunogenicity_dataset.csv` has been removed; the canonical path is
> `data/immunogenicity_dataset_v3.csv`.

### Dataset Governance

Version governance metadata (version, provenance timestamp, source databases, checksum, QC
thresholds) is embedded in `config.yaml` under the `dataset_governance` key. In
`freeze_mode: true`, the pipeline enforces a SHA256 checksum match against this file before
proceeding.

### Dataset Version History

| Version | File | Records | Notes |
|---------|------|---------|-------|
| v1 (archived) | `archive/immunogenicity_dataset_v1_archived.csv` | ~407 peptides | Original dataset; 5.58:1 class ratio |
| v2 (archived) | `archive/immunogenicity_dataset_v2.csv` | ~720 peptides | Intermediate expansion; superseded |
| v2.0.0-alpha (canonical) | `immunogenicity_dataset_v3.csv` | 720 peptides | Current canonical; 2.36:1 class ratio |

---

## Archive

`archive/` contains previous dataset versions retained for provenance and reproducibility:

| File | Description |
|------|-------------|
| `immunogenicity_dataset_v1_archived.csv` | v1 dataset (5.58:1 class ratio; BPS 542/CMB 522 original) |
| `immunogenicity_dataset_v2.csv` | v2 intermediate dataset (superseded by v3) |

---

## Allele-Aware Subset

`allele_aware/` contains a pan-allele training subset with HLA pseudo-sequence features.
Files are versioned by source date (format: `IEDB-YYYYMMDD-*`).

---

## Consensus Sequences

`consensus_sequences/` contains reference protein sequences used for binding matrix generation:

| File | Description |
|------|-------------|
| `EBV_*.fasta` | EBV protein consensus sequences (EBNA1, LMP1, etc.) |
| `HPV16_*.fasta` | HPV-16 protein consensus sequences (E2, E5, E6, E7) |
| `HPV18_*.fasta` | HPV-18 protein consensus sequences (E6, E7) |

These are used by `src/generate_binding_matrix.py` to pre-compute the peptide binding matrix.

---

## Proteome FASTA Files (tracked)

`proteomes/` contains the curated multi-virus panel FASTA files used as Stage 1 input:

| File | Proteome ID | Description |
|------|-------------|-------------|
| `HPV16_18_panel8.fasta` | `HPV16_18_panel8` | HPV-16 and HPV-18: E2, E5, E6, E7 (8 total) |
| `EBV_B95_8_panel8.fasta` | `EBV_B95_8_panel8` | EBV B95-8: EBNA1, EBNA3A, EBNA3B, LMP1, LMP2A, gp350, BZLF1, BRLF1 |
| `HBV_ayw_panel4.fasta` | `HBV_ayw_panel4` | HBV ayw strain: HBsAg, HBcAg, HBeAg, HBx |
| `HCV_1a_panel4.fasta` | `HCV_1a_panel4` | HCV genotype 1a: Core, NS3, NS4B, NS5B |

Additional panels fetched on demand via `scripts/fetch_viral_proteomes.py`:

| Proteome ID | Description |
|-------------|-------------|
| `HIV1_HXB2_panel4` | HIV-1 HXB2: Gag, Pol, Env, Nef |
| `SARSCOV2_wuhan1_panel4` | SARS-CoV-2 Wuhan-1: Spike, N, M, ORF3a |
| `IAV_PR8_panel4` | IAV PR8: HA, NA, NP, M1 |
| `CMV_AD169_panel4` | CMV AD169: pp65, gB, IE1, pp50 |

> See [`docs/antigen_accessions.md`](../docs/antigen_accessions.md) for full UniProt accession
> IDs, gene names, and biological accuracy caveats.

---

## Verification Fixture FASTAs (tracked)

`verify/proteomes/` contains small reference FASTAs used as test fixtures by
`tests/test_sestrav_evaluator.py` and defined in `src/verify/targets.json`. These are
deliberately small (one representative protein per virus) and are tracked to keep the test
suite self-contained.

| File | Description |
|------|-------------|
| `sars_cov_2.fasta` | SARS-CoV-2 spike protein (verification target) |
| `influenza_a.fasta` | IAV HA protein (verification target) |
| `hcv.fasta` | HCV NS3 protein (verification target) |

---

## IEDB API Output — gitignored

`iedb/` receives CSV files written by `scripts/fetch_iedb_tcell.py`. Each file is named
`{virus}_tcell.csv` with a `{virus}_tcell_provenance.json` sidecar. This directory is
gitignored because the data is regenerable and can be large.

To regenerate:
```
python scripts/fetch_iedb_tcell.py --virus EBV --output-dir data/iedb/
python scripts/fetch_iedb_tcell.py --all --output-dir data/iedb/
```

---

## Self-Similarity Cache — gitignored

`self_similarity_cache.csv` is written by `scripts/precompute_self_similarity.py`. It
contains human proteome exact-match scores for training peptides. Gitignored because it
requires the 100 MB human proteome FASTA (`scripts/fetch_human_proteome.py`) to generate.

To regenerate:
```
python scripts/fetch_human_proteome.py   # one-time download
python scripts/precompute_self_similarity.py \
    --fasta data/proteomes/human_uniprot_UP000005640.fasta \
    --peptides data/immunogenicity_dataset_v3.csv
```
