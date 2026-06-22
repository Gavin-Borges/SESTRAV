# Zenodo Dataset Deposition — SESTRAV Immunogenicity Dataset v4

This is a ready-to-use record for minting the dataset DOI referenced in the manuscript
(§5 Availability). The dataset CSV is intentionally **not** tracked in Git, so the DOI is a
**standalone Zenodo dataset deposition** (manual upload), not a GitHub-release archive.

**What remains manual (lead maintainer):** create the Zenodo deposition under the project
account, upload the three files below, paste the metadata, confirm the license, and publish.
Then replace the DOI placeholder in `docs/paper.md` §5 and this file's header.

> **DOI:** `10.5281/zenodo.XXXXXXX` *(to be minted on publication)*

---

## 1. Files to upload (deposition bundle)

| File | Role | SHA-256 |
|---|---|---|
| `data/immunogenicity_dataset_v4.csv` | Dataset (14,699 rows) | `122da358c06a638d1d59e7b0aa77f2ea54dc1583c0235b35c3ed33fef976f6c5` |
| `data/immunogenicity_dataset_v4_schema.json` | Column schema / validation contract | `003dc88f3536457d6043609bce64c3b2f2d18f715b9054ef7ee34c30e283881c` |
| `data/immunogenicity_dataset_v4_provenance.json` | Build provenance (sources, git SHA, counts) | `7f8d8215a66c6f28fb04d0ac844fa4ea26324ecc16f65759953660b12547d8bb` |

Verify before upload:
```bash
sha256sum -c <<'EOF'
122da358c06a638d1d59e7b0aa77f2ea54dc1583c0235b35c3ed33fef976f6c5  data/immunogenicity_dataset_v4.csv
003dc88f3536457d6043609bce64c3b2f2d18f715b9054ef7ee34c30e283881c  data/immunogenicity_dataset_v4_schema.json
7f8d8215a66c6f28fb04d0ac844fa4ea26324ecc16f65759953660b12547d8bb  data/immunogenicity_dataset_v4_provenance.json
EOF
```

## 2. Zenodo metadata (paste into the deposition form)

- **Upload type:** Dataset
- **Title:** SESTRAV Immunogenicity Dataset v4 — viral T-cell epitope labels with central-tolerance hard decoys
- **Authors / creators:**
  - Borges, Gavin — University of Rhode Island — ORCID 0009-0001-2404-5217
  - Eljamal, Abdelrahman — University of Rhode Island
  - Schellenberg, Iris — University of Rhode Island
  - Jouaneh, Charles — University of Rhode Island
  - Byers, Emine — University of Rhode Island
- **License:** CC-BY-4.0 *(recommended — compatible with VDJdb's CC-BY; IEDB records are freely
  redistributable. Confirm before publishing; use CC0-1.0 only if all upstream terms permit.)*
- **Version:** v4
- **Keywords:** immunoinformatics, epitope prediction, MHC class I, T-cell immunogenicity,
  vaccine design, IEDB, VDJdb, hard decoys, central tolerance
- **Related/alternate identifiers:**
  - `isSupplementTo` → https://github.com/Gavin-Borges/SESTRAV (software)
  - `isSupplementTo` → manuscript DOI *(add on journal acceptance)*
  - `cites` → IEDB (Vita et al. 2019); VDJdb (Bagaev et al. 2020)

### Description (paste verbatim)

> Curated MHC class I immunogenicity dataset for viral T-cell epitope prediction, assembled for
> the SESTRAV v4 release. 14,699 peptide records (6,687 positive / 8,012 negative; 45.5% positive)
> spanning 12 viruses (CMV, DENV, EBV, HBV, HCV, HIV-1, HPV16, HPV18, HPV generic, IAV, RSV,
> SARS-CoV-2). Positives are experimentally annotated CD8⁺ T-cell epitopes drawn from IEDB T-cell
> assays and VDJdb; negatives include 5,000 central-tolerance "hard decoy" self-proteome peptides
> (MHC class I binders) that decouple binding affinity from immunogenicity in the negative class.
> Labels were resolved across 14 sources by quality-weighted majority vote (1,843 label conflicts
> resolved). Each record carries peptide, HLA allele, label, virus, protein, and source metadata.
> Schema and full build provenance (source files, generating commit, counts) are included.
> Built 2026-06-20 from repository commit a9cc823.

## 3. Post-publish checklist

- [ ] DOI minted; replace `10.5281/zenodo.XXXXXXX` in this file and `docs/paper.md` §5.
- [ ] Add the DOI badge to `README.md`.
- [ ] Add `cff-version` `identifiers:` entry (DOI) to `CITATION.cff`.
- [ ] If using a Zenodo *concept* DOI, cite the concept DOI in the paper (resolves to latest version).
