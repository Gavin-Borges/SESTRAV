# SESTRAV Usage Guide

Quick-start reference for the `sestrav` command-line interface.

## Installation

```bash
# Core install from source (not yet published to PyPI)
git clone https://github.com/Gavin-Borges/SESTRAV.git
cd SESTRAV
pip install .

# With GNN structural scorer
pip install ".[gnn]"

# With Snakemake pipeline runner
pip install ".[pipeline]"

# With Streamlit demo app (optional expansion panels)
pip install ".[demo]"

# Developer install (includes ruff, mypy, pytest)
pip install -e ".[dev]"
```

After installing, download the MHCflurry presentation models (required for binding prediction):
```bash
mhcflurry-downloads fetch models_class1_presentation
```

---

## Five-Command Quickstart

### 1. Check your environment

```bash
sestrav info
```

Expected output:
```
============================================================
SESTRAV Environment Info
============================================================
  sestrav version : 2.0.3
  mhcflurry       : 2.2.1
  torch           : 2.13.0
  CUDA            : not available
  feature_mode    : 31
  model_path      : models/rf_31feature_integrated.joblib
  mhcflurry_pin   : 2.2.1
  allele panel    : 10 alleles (HLA-A*02:01, HLA-A*01:01, HLA-A*03:01...)
  viruses/panels  : HPV16_18_panel8, EBV_B95_8_panel8, HBV_ayw_panel4, HCV_1a_panel4
============================================================
```

### 2. Score a viral proteome

> **No model binary ships with this repository, so the `--model` path below does not
> exist in a fresh clone.** `git ls-files models/` returns zero `.joblib` files: model
> artifacts are deliberately untracked (see `ARCHITECTURE.md`), because a checkpoint
> committed without provenance is the mislabeled-artifact failure this project has had
> twice (`docs/claims_register.md` D16, D23). **Train one first** - the canonical
> command is in `README.md` under "Install and Train Models".
>
> **That command writes `models/local/rf_31feature_integrated.joblib`, not the
> `models/` path shown below.** `--model-dir` is required and has no default, and the
> canonical invocation points it at `models/local/` (which is gitignored) so that a
> local retrain cannot overwrite published artifacts. Bridge the two in one of two
> ways: pass `--model models/local/rf_31feature_integrated.joblib` to the commands
> below, or train with `--model-dir models --allow-overwrite` when replacing the
> published artifacts is what you actually mean.
>
> **A `--model` path that does not exist is now an error, not a fallback.** Stage 4
> previously fell through to an inline prototype classifier trained on pseudo-labels
> derived from `binding_score`, and its calibrated, thresholded output was
> indistinguishable from a real run once written to CSV. The same applies to the
> FastAPI service and the Streamlit demo, which both load a model at startup.

```bash
sestrav predict \
  --fasta data/proteomes/EBV_B95_8_panel8.fasta \
  --model models/rf_31feature_integrated.joblib \
  --output results/ebv_run/
```

This generates `results/ebv_run/EBV_B95_8_panel8_ranked.csv` with columns:
`peptide, protein_id, immunogenicity_score, rank, allele, presentation_score`

To restrict to specific alleles or lengths:
```bash
sestrav predict \
  --fasta data/proteomes/EBV_B95_8_panel8.fasta \
  --model models/rf_31feature_integrated.joblib \
  --alleles HLA-A*02:01 HLA-B*07:02 \
  --lengths 9 10 \
  --output results/ebv_a02_run/
```

### 3. Validate on labeled data (cross-validation)

```bash
sestrav validate \
  --dataset data/immunogenicity_dataset_v5.csv \
  --model-dir models/local \
  --feature-mode 31 \
  --binding-matrix models/peptide_binding_matrix_v5.csv \
  --sample-weights \
  --report results/validation_report_v5.md
```

`--model-dir` is required and has no default: `sestrav validate` retrains, so it writes
the same artifact set as `python -m src.train_classifier` and needs the same deliberate
destination. `models/local/` is gitignored. A run aborts before training if it would
replace artifacts already present in the target directory; add `--allow-overwrite` when
replacing them is the intent.

**If you already trained into `models/local/` for step 2, this command aborts** rather
than clobbering those artifacts. Point it at a fresh directory
(`--model-dir models/scratch/<run-name>`) or add `--allow-overwrite`.

The published figures for this model are a per-virus within-CV mean AUC-ROC of **0.658** on the v5
dataset (35,597 active rows / 51,185 total; the canonical same-pathogen discrimination metric,
`results/per_virus_eval_v5_mode31.csv`) and a pooled CV AUC-PR of **0.6055**
(`models/v5/training_results_mode31.csv`). The pooled AUC-PR is a base-rate artifact and is not
reported as a headline; see `docs/model_evaluation_summary.md`. The fold-mean of the same five
per-fold AUC-PR values - as opposed to the pooled figure above - is a distinct **0.6058**, the
figure cited elsewhere in this repository against the mode-33/35 antigen-processing ablation
ladder. **Splitter disclosure (required
whenever these figures are quoted, `docs/claims_register.md` D15 - remediated 2026-08-10):** they
come from a **peptide-grouped** splitter, so no peptide appears on both sides of a fold boundary.
The prior ungrouped figures (per-virus mean 0.751, pooled AUC-PR 0.8312) are retracted as
leakage-inflated.

> **Corrected 2026-08-17: do not expect the command above to reproduce those figures.** This section
> previously stated that they came from `--cv-group-by peptide`, "now the CLI default", and that
> passing `--cv-group-by none` would reproduce the retracted ungrouped values. Both statements are
> false for `sestrav validate`: it accepts no `--cv-group-by` flag at all (see the argument table
> below), and `src/cli.py` never passes `cv_group_by`, so `train_models` falls through to its `None`
> default and uses the **ungrouped** `MultiStratifiedKFold` - the splitter whose peptide leakage
> `docs/claims_register.md` D15 retracted a headline figure over. The 0.658 / 0.6055 figures come
> from `python -m src.train_classifier`, whose own parser does default to `--cv-group-by peptide`.
> A `sestrav validate` run will therefore report **higher, leakage-inflated** cross-validation
> numbers; do not cite them as comparable to any peptide-grouped figure in this repository. Whether
> to change what the shipped command computes is a behaviour change and is tracked separately rather
> than patched here.

### 4. Benchmark against gold standard

```bash
sestrav benchmark \
  --predictions results/ebv_run/EBV_B95_8_panel8_ranked.csv \
  --output results/benchmark_report.md
```

Compares your ranked output to the SESTRAV gold-standard epitope list and reports
AUC-PR, AUC-ROC, ISSR@10, and ISSR@25.

### 5. Run the full governed pipeline (Snakemake)

For release-grade, reproducible runs that enforce checksums and freeze_mode:
```bash
snakemake --snakefile pipeline.smk --cores 4
```

The CLI (`sestrav predict`) is for interactive/exploratory use. For publication-quality
results, use the Snakemake pipeline with `freeze_mode: true` in `config.yaml`.

---

## All Subcommands

### `sestrav info`

No arguments required. Prints environment summary and warns on version mismatches.

### `sestrav predict`

| Argument | Required | Description |
|----------|----------|-------------|
| `--fasta` | Yes | Path to FASTA proteome file |
| `--model` | Yes | Path to trained `.joblib` model |
| `--output` | No | Output directory (default: `results/cli_predict/`) |
| `--alleles` | No | HLA alleles, space-separated (default: 10-allele panel) |
| `--lengths` | No | Peptide lengths (default: 8 9 10 11) |
| `--feature-mode` | No | Feature mode override (default: from `config.yaml`) |

### `sestrav validate`

| Argument | Required | Description |
|----------|----------|-------------|
| `--dataset` | Yes | Path to labeled immunogenicity CSV |
| `--binding-matrix` | Cond. | Required for `--feature-mode 30` or `31` |
| `--folds` | No | CV folds (default: 5) |
| `--feature-mode` | No | Feature mode override |
| `--sample-weights` | No | Apply EBV/HPV16 bias-correction weights |
| `--report` | No | Path to write markdown report |
| `--model-dir` | Yes | Directory to write trained models and metrics. No default: use a scratch directory such as `models/local`, and pass `models` only when replacing the published artifacts |
| `--allow-overwrite` | No | Replace training artifacts already present in `--model-dir` (without it the run aborts before training) |

### `sestrav benchmark`

| Argument | Required | Description |
|----------|----------|-------------|
| `--predictions` | Yes | Path to ranked predictions CSV |
| `--score-column` | No | Score column name (auto-detected if omitted) |
| `--output` | No | Path to write markdown benchmark report |

---

## Reproducing the Published Results

The exact published benchmark (AUC-PR 0.828, OOF RF, frozen 2026-05-20) uses:

```bash
# 1. Run full Snakemake pipeline with freeze_mode
snakemake --snakefile pipeline.smk --cores 4 --config freeze_mode=true

# 2. Reproduce the external benchmark comparison
python scripts/benchmark_runner.py --tier A --run-id reproduce_v3 --skip-freeze-check
```

See `results/external_benchmark_comparison.md` for the external benchmark methodology,
`docs/model_evaluation_summary.md` for all benchmark results, and
`docs/claims_register.md` for the scope boundary and required qualifier on each
certified claim. The full contamination analysis lives in an internal validation
sign-off document that is not published in this repository.

> **Evaluation note:** SESTRAV RF is evaluated out-of-fold, while external tools are fully
> scored on the same peptides. On the certified Tier A head-to-head
> (`results/table3_tier_a_metrics.csv`), SESTRAV RF (AUC-PR 0.828) posts the highest point
> AUC-PR - a statistical near-tie with BigMHC (0.822), the MHCflurry binding-only baseline
> (0.800), MixMHCpred 2.2 (0.795), and DeepImmuno (0.698).
>
> **This comparison was previously described here as "conservative by construction" for
> SESTRAV. That is withdrawn, but not replaced with the opposite claim.** This benchmark's
> 720-peptide corpus has zero duplicate peptides, so the exact-peptide cross-validation
> leakage found elsewhere in this project is a structural no-op here (D16). A different,
> unquantified risk applies instead - 32.1% of the 704-peptide scored pool has a
> substring-level near-duplicate elsewhere in the pool, never filtered for this benchmark -
> so the near-tie must not be read as biased in either direction
> (`docs/claims_register.md` D22). Separately, the 0.828 figure
> is a 30-feature, unweighted, 200-tree measurement from 2026-05, not the canonical
> `mode_31` result (D16).

---

## Configuration

Key settings in `config.yaml`:

| Setting | Default | Description |
|---------|---------|-------------|
| `feature_mode` | `31` | Feature tier: 21 (legacy), 30 (legacy), 31 (canonical), 33 (extended) |
| `model_path` | `models/rf_31feature_integrated.joblib` | Production model path. Not tracked in git - produced by training, see step 2 above |
| `mhcflurry_model_version` | `2.2.1` | Pinned MHCflurry version (do not change without rebuilding dataset) |
| `freeze_mode` | `true` | Enforce strict checksums and governance guardrails |
| `alleles` | 10-allele panel | HLA alleles for binding prediction |
| `peptide_lengths` | `[8, 9, 10, 11]` | Peptide lengths for generation |
| `antigen_processing_cache_path` | `null` | Cache CSV for feature_mode=33 (built by `scripts/precompute_antigen_processing.py`) |
