# SESTRAV v2: Wet-Lab Validation Protocol (Pre-Registration)

**Version:** 1.0.0
**Target Phase:** Phase 7 (Post-Funding Clinical Validation)

## Objective
To prospectively validate the structural GNN predictions from the SESTRAV v2 architecture in a rigorous *in vitro* setting using PBMCs (Peripheral Blood Mononuclear Cells) sourced from human donors.

## 1. Donor Selection Criteria
We will recruit $N=10$ donors matching the following parameters:
- **Demographics:** Healthy adults (18-55 years).
- **Infection History:** PCR-confirmed history of SARS-CoV-2 or EBV infection (depending on target antigen panel).
- **HLA Typing:** High-resolution HLA-I genotyping must confirm the presence of at least one of the 10 canonical alleles mapped in `mhc_pseudo_sequences.json` (e.g., `HLA-A*02:01`, `HLA-B*07:02`).

## 2. Peptide Synthesis
- Synthesize top $N=50$ SESTRAV-predicted immunogenic peptides (GNN probability $>0.85$).
- Synthesize $N=50$ high-binding but low-immunogenicity decoys (Binding $< 500$ nM, GNN probability $<0.2$).
- Purity must be $>95\%$ via HPLC.

## 3. ELISpot Assay Protocol
1. **PBMC Isolation:** Isolate via density gradient centrifugation. Cryopreserve until use.
2. **Stimulation:** Thaw PBMCs and rest overnight. Plate at $2 \times 10^5$ cells/well in IFN-$\gamma$ coated ELISpot plates.
3. **Peptide Pulsing:** Pulse wells with 1 $\mu$g/mL of synthesized peptides. Include positive controls (CEF pool) and negative controls (DMSO vehicle).
4. **Incubation:** Incubate for 18-24 hours at 37°C, 5% CO$_2$.
5. **Development & Readout:** Develop plates per manufacturer protocols and count Spot Forming Units (SFU) using an automated ELISpot reader.

## 4. Success Criteria ($R_{10}$ Enrichment)
The GNN predictions will be deemed successful if they demonstrate a statistically significant enrichment of immunogenic hits compared to the pure binding-affinity baseline.
- **Metric:** Top-10 Enrichment Ratio ($R_{10}$)
- $R_{10} = \frac{\text{SFU}_{\text{SESTRAV Top 10}}}{\text{SFU}_{\text{Binding-Only Top 10}}} \ge 2.0$
- **Significance:** $p < 0.05$ (Mann-Whitney U test between SESTRAV predictions and decoy baseline).
