# Zenodo Dataset Deposition - SESTRAV Immunogenicity Dataset v5

This is a ready-to-use record for minting the dataset DOI referenced in the manuscript
(Data Availability section). The three files below **are** tracked in the SESTRAV Git repository
(since `dcbb1b1`, 2026-06-26); this deposition additionally mints a permanent, citable Zenodo DOI for
the dataset, independent of repository/release history. *(Corrected 2026-08-17: this previously said
the CSV was "intentionally not tracked in Git" - that was false as of the same commit that added it.)*

**What remains manual (lead maintainer):** create the Zenodo deposition under the project
account, upload the three files below, paste the metadata, confirm the license, and publish.
Then replace the DOI placeholder in `docs/paper.md` Data Availability section and this file's header.

*Note (2026-07-28): `docs/paper.md` was fully replaced with the Bioinformatics-format
manuscript. Its DOI placeholder now reads `[PLACEHOLDER - reserve and paste before
submission]` inside the Data Availability section (previously a bare `10.5281/zenodo.XXXXXXX`
in a numbered `## 5. Availability` section) - same open item, new location and wording.*

> **DOI:** `10.5281/zenodo.XXXXXXX` *(to be minted on publication)*

---

## 1. Files to upload (deposition bundle)

**Platform note (added 2026-08-17):** the hashes below are computed against the canonical Git blob
(LF line endings), which is what any properly configured clone - Linux, CI, or a Windows clone with
`.gitattributes`' `eol=lf` pin honored - produces on checkout. `.gitattributes` now pins `eol=lf` for
all three files. If `sha256sum -c` fails, re-run `git checkout HEAD -- <file>` first: an already
checked-out working copy is not retroactively rewritten by a `.gitattributes` change alone.

| File | Role | SHA-256 |
|---|---|---|
| `data/immunogenicity_dataset_v5.csv` | Dataset (51,185 rows total; 35,597 active / non-quarantined) | `6928cba8bc2de66128adba3358be26a41353b18010b502979eff36111132b0c4` |
| `data/immunogenicity_dataset_v5_schema.json` | Column schema / validation contract | `f0a1f69baa3c6feb380effbf9b73c69f76cc3e839096ea40ab4ebcd38db640d7` |
| `data/immunogenicity_dataset_v5_provenance.json` | Build provenance (sources, git SHA, counts) | `c1fbf4ca6a63d39067922984431a140866899d7db19938376f7122cc50b48b3d` |

Verify before upload:
```bash
sha256sum -c <<'EOF'
6928cba8bc2de66128adba3358be26a41353b18010b502979eff36111132b0c4  data/immunogenicity_dataset_v5.csv
f0a1f69baa3c6feb380effbf9b73c69f76cc3e839096ea40ab4ebcd38db640d7  data/immunogenicity_dataset_v5_schema.json
c1fbf4ca6a63d39067922984431a140866899d7db19938376f7122cc50b48b3d  data/immunogenicity_dataset_v5_provenance.json
EOF
```

## 2. Zenodo metadata (paste into the deposition form)

- **Upload type:** Dataset
- **Title:** SESTRAV Immunogenicity Dataset v5 - viral T-cell epitope labels with central-tolerance hard decoys and IEDB tested-negative expansion
- **Authors / creators:**
  - Borges, Gavin - University of Rhode Island - ORCID 0009-0001-2404-5217
  - Eljamal, Abdelrahman - University of Rhode Island
  - Schellenberg, Iris - University of Rhode Island
  - Jouaneh, Charles - University of Rhode Island
  - Byers, Emine - University of Rhode Island
- **License:** CC-BY-4.0 *(recommended - compatible with VDJdb's CC-BY; IEDB records are freely
  redistributable. Confirm before publishing; use CC0-1.0 only if all upstream terms permit.)*
- **Version:** v5 (5.0.0)
- **Keywords:** immunoinformatics, epitope prediction, MHC class I, T-cell immunogenicity,
  vaccine design, IEDB, VDJdb, hard decoys, central tolerance
- **Related/alternate identifiers:**
  - `isSupplementTo` -> https://github.com/Gavin-Borges/SESTRAV (software)
  - `isSupplementTo` -> manuscript DOI *(add on journal acceptance)*
  - `cites` -> IEDB (Vita et al. 2019); VDJdb (Bagaev et al. 2020)

### Description (paste verbatim)

> Curated MHC class I immunogenicity dataset for viral T-cell epitope prediction, assembled for
> the SESTRAV v5 release. 51,185 total peptide records (8,712 positive / 42,473 negative), of
> which 35,597 are active (non-quarantined) rows retained for per-virus training and metrics; the
> remaining rows are flagged `is_quarantined = true` (virus with fewer than 50 rows or fewer than
> 10 real tested negatives) and kept in the file for traceability. Positives are experimentally
> annotated CD8+ T-cell epitopes drawn from IEDB T-cell assays and VDJdb; negatives include
> central-tolerance "hard decoy" self-proteome peptides (MHC class I binders) that decouple
> binding affinity from immunogenicity, plus a large IEDB tested-negative expansion. v5 extends
> v4 (6,687 positives, 5,000 hard decoys, 3,012 other negatives) with 36,689 IEDB negatives and
> 9,206 published-panel rows; 6,397 duplicate rows were dropped on merge and 1,331 label conflicts
> were resolved by quality-weighted majority vote. Each record carries peptide, HLA allele, label,
> virus, protein, and source metadata, plus v5 provenance columns (virus_family, negative_origin,
> assay_type, assay_quality_weight/tier, reference_pmid, iedb_assay_id, virus_taxon_id) and
> per-virus biology context (infection_phase, antigen_latency_program, assay_context,
> cross_reactivity_tested). Schema and full build provenance (source files, generating commit,
> counts) are included. Built 2026-07-05 from repository commit be3e260.

## 3. Post-publish checklist

- [ ] DOI minted; replace `10.5281/zenodo.XXXXXXX` in this file and the `[PLACEHOLDER - reserve
      and paste before submission]` in `docs/paper.md` Data Availability section.
- [ ] Add the DOI badge to `README.md`.
- [ ] Add `cff-version` `identifiers:` entry (DOI) to `CITATION.cff`.
- [ ] If using a Zenodo *concept* DOI, cite the concept DOI in the paper (resolves to latest version).
