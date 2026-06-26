# Published Peptide Panels - Curation Workspace

This directory holds manually-curated peptide panels transcribed from published
immunology papers. Each CSV is consumed by `scripts/ingest_published_panels.py`,
which normalizes it into the v5 dataset schema and writes a provenance sidecar.

The goal of these panels is to add **real tested negatives** (non-responder peptides
from the same cohort/assay) to the Amendment 6 target viruses, where IEDB negatives
are scarce. This is the binding constraint on the per-virus AUC-ROC exit criteria.

## CSV format (what you fill in)

One row per tested peptide. Columns:

| Column | Required | Meaning |
|---|---|---|
| `peptide` | yes | 8-11 standard amino-acid residues (ACDEFGHIKLMNPQRSTVWY). Rows outside 8-11 or with non-standard AA are dropped. |
| `label` | yes | `1` = immunogenic / responder peptide; `0` = tested negative / non-responder. |
| `hla_allele` | optional | Per-row HLA restriction. Normalized to 4-digit Class I (e.g. `HLA-A*02:01`) via mhcgnomes; unparseable -> row dropped. Omit the column and use `--hla-allele` instead when the whole panel shares one allele. |
| `protein` | optional | Source protein name (e.g. `E6`, `LMP2`, `core`). |
| `strain` | optional | Virus strain, if reported. |

Everything else (virus, pmid, assay context, infection phase, latency program,
taxon id, quality tier) is panel-level metadata passed on the command line, not in
the CSV. `TEMPLATE.csv` has the header row to copy.

### The decisive inclusion rule

A panel is only worth curating if the paper reports peptides that were **tested and
scored negative** in the same cohort/assay - not just a list of positive epitopes.
Before transcribing, confirm the paper has a non-responder / negative peptide table.
If it only lists positives, it adds no negatives and does not move the metric.

Other gate requirements (per the v5 inclusion gate): human host, MHC class I,
resolvable HLA, citable PMID, assay type recorded.

## Confirmed panels and exact ingest commands

PMIDs below are confirmed live by prior sessions. Verify each paper actually contains
negative/non-responder peptide data before curating. Set `--assay-quality-tier`
per assay: 1 = direct ex vivo functional (weight 1.0), 2 = indirect (0.7),
3 = expanded-culture (0.5).

### HPV - Kenter (PMID 19890126, NEJM 2009, vaccine-induced)
`--virus` must match the label `evaluate_per_virus.py` uses for HPV. The dataset
currently evaluates HPV as `HPV`; confirm before running (the historical Kanban note
used `HPV16`). HPV is the only virus where the cross-reactivity flag applies.

```
python -m scripts.ingest_published_panels \
  --input data/published_panels/kenter_hpv16.csv \
  --virus HPV \
  --pmid 19890126 \
  --assay-context vaccine_induced \
  --no-cross-reactivity-tested \
  --assay-quality-tier 1 \
  --assay-type "T cell IFN-gamma ELISpot" \
  --dry-run
```

### HCV - Thimme (PMID 11714747, J Exp Med 2001, acute needlestick cohort)
```
python -m scripts.ingest_published_panels \
  --input data/published_panels/thimme_hcv.csv \
  --virus HCV \
  --pmid 11714747 \
  --assay-context natural_infection \
  --infection-phase acute \
  --assay-quality-tier 1 \
  --assay-type "T cell IFN-gamma ELISpot" \
  --dry-run
```

### HBV - Webster/Bertoletti (PMID 15140968, J Virol 2004, chronic, structural + nonstructural)
```
python -m scripts.ingest_published_panels \
  --input data/published_panels/bertoletti_hbv.csv \
  --virus HBV \
  --pmid 15140968 \
  --assay-context natural_infection \
  --infection-phase chronic \
  --assay-quality-tier 1 \
  --assay-type "T cell IFN-gamma ELISpot" \
  --dry-run
```

### EBV - Rickinson/Hislop group (PMID candidates RESOLVED; not Khanna)
EBV is the most negative-starved target (only ~14 tested negatives) and the **binding
Amendment 6 constraint** - none of the HPV/HCV/HBV panels above supply EBV negatives,
so this is the single highest-priority panel. The long-unresolved "Khanna" lead was a
dead end (Khanna/QIMR papers are single-antigen epitope-discovery, not responder/
non-responder cohort panels). The comprehensive cohort panels are from the Birmingham/
Rickinson group:

- **PMID 11927633** (Hislop AD et al., J Exp Med 2002; PMC2193726, open access) -
  PRIMARY. Longitudinal acute->persistent EBV cohort; covers BOTH lytic and latent
  programs; documents non-responder outcomes (matches the Thimme/Bertoletti design).
  Verify the per-donor/per-timepoint negatives are tabulated cleanly before ingest.
- **PMID 24146041** (Abbott RJM et al., J Immunol 2013; PMC5580796) - COMPLEMENT,
  lytic only. ~200+ overlapping peptides across 15 lytic proteins; richest pool of
  true negative peptide *sequences* (whole antigens gp350/gp42 devoid of epitopes),
  but per-peptide negatives must be derived as (full library minus positive table)
  from the supplement.

Both need full-text/supplement verification by Gavin for clean negative extraction.
EBV ingest also takes `--latency-program` (lytic / latent-I..III) and
`--virus-taxon-id 10376`. Curate Hislop into `khanna_ebv.csv` (or rename to
`hislop_ebv.csv`):

```
python -m scripts.ingest_published_panels \
  --input data/published_panels/khanna_ebv.csv \
  --virus EBV \
  --pmid 11927633 \
  --assay-context natural_infection \
  --latency-program <lytic|latent-I|latent-II|latent-III> \
  --virus-taxon-id 10376 \
  --assay-quality-tier 1 \
  --assay-type "T cell IFN-gamma ELISpot" \
  --dry-run
```

## Workflow

1. Transcribe the paper's tested peptides into the matching per-panel CSV (positives
   `label=1`, non-responders `label=0`).
2. Run the command above with `--dry-run` first; check the printed positive/negative
   counts and the drop stats (bad peptide / bad HLA rows).
3. Remove `--dry-run` to write `data/published_panel_<VIRUS>_<PMID>_v5.csv` plus its
   `_provenance.json` sidecar.
4. Rebuild the v5 dataset (`scripts/build_dataset_v5.py`) so the new panel is merged,
   then re-run the per-virus evaluation to measure the Amendment 6 effect.

## Notes

- Run the ingest script as a MODULE (`python -m scripts.ingest_published_panels`), not
  as a path (`python scripts/ingest_published_panels.py`): it imports from the `scripts`
  package and the path form fails with ModuleNotFoundError. The module form is validated.
- The per-panel CSVs here are header-only by design. The script does not handle a
  zero-data-row input gracefully (it errors during filtering rather than reporting 0
  rows), so a `--dry-run` only works once you have added at least one valid peptide row.
- These CSVs are tracked (not gitignored): they are reproducible curation inputs with
  citable provenance, like the proteome `*_provenance.json` files.
- Keep transcription faithful to the source table; do not infer or pad peptides.
- ASCII only (no em-dashes / smart quotes) per the repo encoding rule.
