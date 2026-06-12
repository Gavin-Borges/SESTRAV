# SESTRAV Antigen Accessions Reference (v1)

This document serves as the standard reference for **Stage 1 (sliding-window) inputs** in the SESTRAV pipeline. During Stage 1, the pipeline ingests viral proteome sequences and scans them using sliding windows of defined lengths (typically 8-11mer peptides) to generate the initial candidate epitope pool. 

To ensure exact sequence reproducibility, the reference accessions below must be used when querying databases or downloading input FASTA files.

---

## 1. Epstein-Barr Virus (EBV) Antigens
All sequences are derived from the **EBV B95-8 strain** (Human herpesvirus 4).

| Protein Name | UniProt Accession ID | Gene Name | Biological Role |
| :--- | :--- | :--- | :--- |
| **gp350** | `P03200` | `BLLF1` | Viral envelope glycoprotein (host cell entry) |
| **EBNA1** | `P03211` | `BKRF1` | Latent nuclear antigen (viral genome replication) |
| **EBNA3A** | `P03204` | `BLRF3`/`BERF1` | Latent nuclear antigen (transcriptional regulator) |
| **EBNA3B (EBNA4)** | `P03205` | `BERF2a`/`BERF2b` | Latent nuclear antigen (transcriptional regulator) |
| **LMP1** | `P03230` | `BNLF1` | Latent membrane protein (oncogenic driver) |
| **LMP2A** | `P13285` | — | Latent membrane protein (blocks lytic reactivation) |
| **BZLF1** | `P03206` | `BZLF1` | Immediate-early lytic transactivator (lytic cycle trigger) |
| **BRLF1** | `P03209` | `BRLF1` | Immediate-early lytic transactivator (activates lytic cycle) |

> [!NOTE]
> **EBNA1 Sequence Variant:** The standard B95-8 accession `P03211` contains the glutamate (Glu) variant at the immunogenic `HPVGEADYFEY` epitope position. Retaining this specific accession is required to align with legacy clinical benchmarks.

---

## 2. Human Papillomavirus (HPV) Antigens
This panel comprises key regulatory and oncogenic proteins from the high-risk strains **HPV16** and **HPV18**.

| Protein Name | UniProt Accession ID | Gene Name | Virus Strain | Biological Role |
| :--- | :--- | :--- | :--- | :--- |
| **E2** | `P03120` | `E2` | HPV16 | Transcription/replication regulator |
| **E5** | `P06927` | `E5` | HPV16 | Oncoprotein (membrane signaling interaction) |
| **E6** | `P03126` | `E6` | HPV16 | Oncoprotein (induces p53 degradation) |
| **E7** | `P03129` | `E7` | HPV16 | Oncoprotein (binds and inactivates pRb) |
| **E2** | `P06790` | `E2` | HPV18 | Transcription/replication regulator |
| **E5** | `P06929` | `E5` | HPV18 | Oncoprotein (membrane signaling interaction) |
| **E6** | `P06463` | `E6` | HPV18 | Oncoprotein (induces p53 degradation) |
| **E7** | `P06788` | `E7` | HPV18 | Oncoprotein (binds and inactivates pRb) |

---

## 3. FASTA Configurations and Stage 1 Integration
These accessions are consolidated into default FASTA inputs used in the Snakemake sliding-window generation:
*   **EBV Proteome Panel (8 proteins):** [EBV_B95_8_panel8.fasta](../data/proteomes/EBV_B95_8_panel8.fasta)
*   **HPV Proteome Panel (8 proteins):** [HPV16_18_panel8.fasta](../data/proteomes/HPV16_18_panel8.fasta) (comprising 4 HPV16 and 4 HPV18 proteins)

Any sliding-window queries initiated via `Snakefile` or `pipeline.smk` default to these configurations. Custom proteins added for Stage 1 must follow this mapping format and be documented accordingly.
