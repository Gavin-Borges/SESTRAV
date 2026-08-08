---
name: Data Contribution
about: Propose adding a new virus panel, epitope dataset, or benchmark cohort
title: "[DATA] "
labels: data-contribution, needs-review
assignees: ''
---

## Data Contribution Proposal

### What data are you contributing?

<!-- Describe the dataset: source database, virus/pathogen, assay type, size -->

**Source:** (e.g., IEDB accession, VDJdb export, TSNAdb, in-house)
**Pathogen / panel:** (e.g., CMV pp65, IAV NP, SARS-CoV-2 Spike)
**Alleles covered:** (e.g., HLA-A*02:01, HLA-B*07:02)
**Dataset size:** (approximate: # peptides, # positive / negative labels)
**Assay type(s):** (e.g., IFN-γ ELISpot, multimer staining, proliferation)

### Biological Accuracy Checklist (Four Questions)

Before any data enters the pipeline, the contribution must pass the biological accuracy protocol.

**Q1 - Mechanism:** Is the immunogenicity assay type appropriate for T-cell epitope labeling?
<!-- Yes / No / Partial - explain -->

**Q2 - Scope:** Which alleles, peptide lengths, and virus strains does this data cover?
<!-- Specify constraints and any notable gaps -->

**Q3 - Limitation:** Are there known biases in this dataset (e.g., 9-mer enrichment, allele skew, donor HLA restriction)?
<!-- List known limitations -->

**Q4 - Fairness:** Does this data overlap with existing training or benchmark sets?
<!-- State any known overlap with the current training set (data/immunogenicity_dataset_v5.csv)
     and with the Tier A benchmark intersection. If you can, report the number of exactly
     duplicated peptide + HLA allele pairs. Maintainers re-run a full overlap and
     contamination check before any contributed data is merged, so an approximate answer
     here is fine - do not block your submission on it. -->

### Data Format

Does the data conform to the v4 schema? (`data/immunogenicity_dataset_v4_schema.json`)

- [ ] `peptide` column present (standard amino acids only)
- [ ] `label` column present (1 = immunogenic, 0 = non-immunogenic)
- [ ] `virus` column present
- [ ] `hla_allele` column present (high-resolution format: e.g., `HLA-A*02:01`)
- [ ] `source_type` column present (e.g., `Virus`, `VDJdb`, `TSNAdb`)
- [ ] `database_source` column present

### Provenance

- **Primary citation:** <!-- DOI or PubMed ID -->
- **License:** <!-- CC-BY, open access, academic use, etc. -->
- **Download URL or access instructions:**

### Integration Path

Which ingest script handles this source, or does a new one need to be written?

- [ ] Existing script (specify): `scripts/ingest_____py`
- [ ] New ingest script needed - I will write it following the pattern in `scripts/ingest_vdjdb.py`
- [ ] Manual download required - instructions:

### Additional Notes

<!-- Any other context, known issues, or contacts at the data source -->
