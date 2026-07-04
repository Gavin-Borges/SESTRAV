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
| **LMP2A** | `P13285` | - | Latent membrane protein (blocks lytic reactivation) |
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

## 3. Hepatitis B Virus (HBV) Antigens
All sequences are derived from the **HBV genotype D subtype ayw** reference strains (France/Tiollais/1979 isolate where available).

| Protein Name | UniProt Accession ID | Gene | Strain | Biological Role |
| :--- | :--- | :--- | :--- | :--- |
| **HBcAg** | `P03147` | `C` | genotype D subtype adw (UK/adyw/1979) | Capsid/core antigen (nucleocapsid assembly) |
| **HBx** | `P03165` | `X` | genotype D subtype ayw (France/Tiollais/1979) | Transcriptional transactivator (oncogenic driver) |
| **HBsAg-S** | `P03138` | `S` | genotype D subtype ayw (France/Tiollais/1979) | Large envelope protein (surface antigen, host entry) |
| **HBpol** | `P03157` | `P` | genotype C subtype ad (Japan/S-179/1988) | DNA polymerase / reverse transcriptase |

> [!NOTE]
> Sequences are downloaded via `scripts/fetch_viral_proteomes.py` with provenance recorded in `data/proteomes/HBV_ayw_panel4_provenance.json`.

> [!WARNING]
> **HBV Genotype Coverage Caveat.** This panel uses genotype D (ayw) reference sequences - the best-curated Swiss-Prot entries available. Genotype B and C strains dominate East and Southeast Asia, where HBV-related hepatocellular carcinoma burden is highest. Genotypes B and C show 8-12% nucleotide divergence from genotype D, producing peptide-level differences that may affect epitope prediction accuracy. Predicted epitopes derived from this panel should be treated with reduced confidence when applied to genotype B/C-predominant patient populations. Genotype-specific expansion targeting genotypes B and C is planned for v2.2.

---

## 4. Hepatitis C Virus (HCV) Antigens
This panel uses the best-available reviewed Swiss-Prot entries for each NS protein. Only one reviewed genotype 1a entry exists (P26664, H77 genome polyprotein); NS3/NS5A/NS5B are represented by the best-characterized available accessions.

| Protein Name | UniProt Accession ID | Strain | Biological Role |
| :--- | :--- | :--- | :--- |
| **Core** | `P26664` | genotype 1a (H77) - full polyprotein | Nucleocapsid protein; polyprotein contains all NS regions |
| **NS3** | `O92972` | genotype 1b (HC-J4) - best-reviewed NS3 | Serine protease / RNA helicase |
| **NS5A** | `O92975` | Hepacivirus hominis fragment | Membrane-associated phosphoprotein (replication complex) |
| **NS5B** | `O92976` | Hepacivirus hominis fragment | RNA-directed RNA polymerase |

> [!NOTE]
> No reviewed genotype 1a-specific NS3/NS5A/NS5B accessions exist in Swiss-Prot. O92972 (1b HC-J4) is the most-reviewed NS3 source; O92975/O92976 are TrEMBL fragments. Sequences downloaded via `scripts/fetch_viral_proteomes.py`; provenance in `data/proteomes/HCV_1a_panel4_provenance.json`.

---

## 5. HIV-1 Antigens
All sequences are derived from the **HIV-1 clade B reference strain HXB2** (GenBank K03455).

| Protein Name | UniProt Accession | Gene | Biological Role |
| :--- | :--- | :--- | :--- |
| **Gag** | `P04591` | `gag` | Gag polyprotein (MA, CA/p24, SP1, NC, SP2, p6) - dominant CD8+ target |
| **Pol** | `P04585` | `pol` | Pol polyprotein (PR, RT, RNase H, IN) - reverse transcriptase target |
| **Nef** | `P04601` | `nef` | Negative factor - accessory protein; major early CD8+ immunodominant antigen |
| **Env** | `P04578` | `env` | Envelope glycoprotein gp160 (gp120 + gp41) |

> [!NOTE]
> Sequences downloaded via `scripts/fetch_viral_proteomes.py` with provenance in `data/proteomes/HIV1_HXB2_panel4_provenance.json`.

> [!WARNING]
> **HIV-1 Clade Coverage Caveat.** HXB2 is clade B, which predominates in North America and Western Europe. Clade C (Sub-Saharan Africa, South Asia) is the most globally prevalent and shows approximately 10% amino-acid divergence from clade B in Gag and Env. Predictions derived from this panel carry additional uncertainty when applied to clade C, D, or A patient populations. SESTRAV does not currently model HIV hypervariable loop regions (V1-V5 of gp120). Clade-specific expansion planned for v2.2.

---

## 6. SARS-CoV-2 Antigens
All sequences are derived from the **SARS-CoV-2 Wuhan-1 reference strain** (Hu-1; GenBank MN908947).

| Protein Name | UniProt Accession | Gene | Length | Biological Role |
| :--- | :--- | :--- | :--- | :--- |
| **Spike** | `P0DTC2` | `S` / ORF2 | 1273 AA | Surface glycoprotein - viral entry; primary humoral + CD8 target |
| **N** | `P0DTC9` | `N` / ORF9 | 419 AA | Nucleocapsid phosphoprotein - highly conserved; dominant CD8 target |
| **M** | `P0DTC5` | `M` / ORF5 | 222 AA | Membrane glycoprotein - conserved CD8 target across betacoronavirus family |
| **ORF3a** | `P0DTC3` | `ORF3a` | 275 AA | Accessory pore-forming protein - documented CD8 responses in IEDB |

> [!NOTE]
> Sequences downloaded via `scripts/fetch_viral_proteomes.py` with provenance in `data/proteomes/SARSCOV2_wuhan1_panel4_provenance.json`.

> [!WARNING]
> **Omicron Variant Divergence.** BA.2 and later Omicron subvariants carry >30 Spike mutations relative to Wuhan-1, including substitutions at documented CD8+ T-cell epitopes (e.g., positions 417, 452, 501). Predictions for Spike-derived peptides are most reliable for ancestral and Delta-lineage strains; cross-variant accuracy requires per-peptide conservation analysis not yet implemented. N and M proteins are substantially more conserved across variants (< 5 mutations) and predictions are expected to generalize better.

---

## 7. Influenza A Virus (IAV) Antigens
All sequences are derived from the **A/Puerto Rico/8/1934 (PR8) strain**, subtype H1N1, the canonical laboratory reference and vaccine manufacturing strain.

| Protein Name | UniProt Accession | Gene | Length | Biological Role |
| :--- | :--- | :--- | :--- | :--- |
| **NP** | `P03466` | `NP` | 498 AA | Nucleoprotein - dominant, cross-subtype conserved CD8+ target; universal vaccine candidate |
| **M1** | `P03485` | `M` | 252 AA | Matrix protein 1 - highly conserved across H1N1 and H3N2; well-characterized CD8 target |
| **HA** | `P03437` | `HA` | 566 AA | Hemagglutinin - H1 subtype-specific; principal neutralizing antibody target |
| **PB1-F2** | `P0C0U1` | `PB1-F2` | 90 AA | Pathogenicity factor encoded in PB1 +1 reading frame; documented CD8 target in severe disease |

> [!NOTE]
> NP and M1 are the primary cross-strain conserved targets underlying universal influenza vaccine proposals (Sridhar et al. 2013, *Nat Med*). PB1-F2 is only 90 AA in PR8 and generates a limited peptide pool (≤83 9-mer windows); treat any predictions for this protein with reduced confidence due to low training coverage. Sequences downloaded via `scripts/fetch_viral_proteomes.py` with provenance in `data/proteomes/IAV_PR8_panel4_provenance.json`.

> [!WARNING]
> **Subtype Generalization Caveat.** HA predictions are H1-subtype specific and will not generalize to H3N2, H5N1, or avian influenza subtypes without retraining on subtype-appropriate sequences. NP and M1 predictions are expected to generalize across H1N1 and H3N2 but have not been formally validated on non-PR8 sequences in SESTRAV.

---

## 8. Human Cytomegalovirus (CMV) Antigens
All sequences are derived from the **CMV strain AD169** (Human betaherpesvirus 5), the most widely studied laboratory-adapted CMV strain.

| Protein Name | UniProt Accession | Gene | Length | Biological Role |
| :--- | :--- | :--- | :--- | :--- |
| **pp65** | `P06725` | `UL83` | 561 AA | Lower matrix phosphoprotein - immunodominant CD8+ target; dominates CMV-specific T-cell pool (up to 10-20% of CD8+ T-cells in seropositive adults) |
| **IE1** | `P13202` | `UL123` | 491 AA | Immediate-early antigen 1 - dominant CD8+ target during primary/reactivation lytic phase |
| **pp50** | `P16785` | `UL44` | 433 AA | DNA polymerase processivity factor - documented CD8+ T-cell target in transplant recipients |
| **gB** | `P06473` | `UL55` | 906 AA | Envelope glycoprotein B - CD8+ T-cell responses documented in primary infection and post-transplant |

> [!NOTE]
> CMV drives the largest pathogen-specific CD8+ T-cell pool of any common human infection (Sylwester et al. 2005, *J Exp Med*). pp65 and IE1 are the primary targets used in CMV-specific T-cell monitoring (tetramer assays, ELISPOT). Sequences downloaded via `scripts/fetch_viral_proteomes.py` with provenance in `data/proteomes/CMV_AD169_panel4_provenance.json`.

> [!WARNING]
> **AD169 Laboratory Adaptation Caveat.** The AD169 strain is highly passage-adapted and has lost the UL/b' genomic region (RL1-RL13, UL1-UL20) present in low-passage clinical isolates. Clinical CMV isolates express additional immunogenic proteins absent from AD169. Predictions from this panel may underrepresent clinical strain epitope diversity. Future versions will supplement with Merlin strain (ATCC VR-1745), which retains the UL/b' region.

---

## 9. FASTA Configurations and Stage 1 Integration
These accessions are consolidated into default FASTA inputs used in the Snakemake sliding-window generation:
*   **EBV Proteome Panel (8 proteins):** [EBV_B95_8_panel8.fasta](../data/proteomes/EBV_B95_8_panel8.fasta)
*   **HPV Proteome Panel (8 proteins):** [HPV16_18_panel8.fasta](../data/proteomes/HPV16_18_panel8.fasta) (comprising 4 HPV16 and 4 HPV18 proteins)
*   **HBV Proteome Panel (4 proteins):** [HBV_ayw_panel4.fasta](../data/proteomes/HBV_ayw_panel4.fasta)
*   **HCV Proteome Panel (4 proteins):** [HCV_1a_panel4.fasta](../data/proteomes/HCV_1a_panel4.fasta)

*Week 6 expansion panels (FASTAs generated after running `scripts/fetch_viral_proteomes.py --panels HIV1_HXB2_panel4 SARSCOV2_wuhan1_panel4 IAV_PR8_panel4 CMV_AD169_panel4`):*
*   **HIV-1 HXB2 Panel (4 proteins):** [HIV1_HXB2_panel4.fasta](../data/proteomes/HIV1_HXB2_panel4.fasta)
*   **SARS-CoV-2 Wuhan-1 Panel (4 proteins):** [SARSCOV2_wuhan1_panel4.fasta](../data/proteomes/SARSCOV2_wuhan1_panel4.fasta)
*   **IAV PR8 Panel (4 proteins):** [IAV_PR8_panel4.fasta](../data/proteomes/IAV_PR8_panel4.fasta)
*   **CMV AD169 Panel (4 proteins):** [CMV_AD169_panel4.fasta](../data/proteomes/CMV_AD169_panel4.fasta)

Any sliding-window queries initiated via `Snakefile` or `pipeline.smk` default to these configurations. Custom proteins added for Stage 1 must follow this mapping format and be documented accordingly.
