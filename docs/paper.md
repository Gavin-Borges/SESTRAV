---
title: 'SESTRAV: Structural Epitope Screening and T-cell Receptor AI Validation'
tags:
  - Python
  - immunology
  - machine learning
  - graph neural networks
  - vaccines
  - bioinformatics
authors:
  - name: Gavin Borges
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
 - name: University of Rhode Island
   index: 1
date: 10 June 2026
bibliography: paper.bib
---

# Summary

The identification of highly immunogenic T-cell epitopes is a critical bottleneck in the design of next-generation viral vaccines and targeted immunotherapies. Traditional experimental screening (e.g., ELISpot assays) is prohibitively expensive and slow, restricting the scope of characterized antigens. While computational predictors like MHCflurry have revolutionized MHC-peptide binding prediction, binding affinity alone is poorly correlated with true *in vivo* immunogenicity (the ability of a peptide-MHC complex to successfully activate a T-cell receptor).

**SESTRAV** (Structural Epitope Screening and T-cell Receptor AI Validation) is an end-to-end, highly automated machine learning pipeline designed to predict actual T-cell immunogenicity from raw viral proteomes. Moving beyond sequence-based tabular baselines, SESTRAV introduces a state-of-the-art **Graph Neural Network (GNN)** architecture that maps 34 critical HLA pocket residues into a 3D biophysical coordinate space using AlphaFold/PDB embeddings.

# Statement of need

Existing tools in the computational immunology space suffer from several critical shortcomings:
1. **Lack of Structural Zero-Shot Generalization:** Legacy tools require distinct training sets for every HLA allele. SESTRAV's graph representation of the MHC-I binding pocket allows it to generalize to entirely uncharacterized alleles purely based on the physical chemistry of the structural nodes.
2. **Provenance and Reproducibility:** Datasets extracted from the Immune Epitope Database (IEDb) and VDJdb are constantly evolving. SESTRAV introduces strict dataset governance, utilizing Snakemake automation to ensure byte-for-byte reproducibility, automated extraction, and rigorous quality-control gates before model training.
3. **Rigorous Statistical Promotion:** Instead of ad-hoc metric evaluations, SESTRAV strictly gates the promotion of its GNN models using $N=10,000$ paired statistical bootstrapping, measuring Expected Calibration Error (ECE), out-of-fold generalizability, and viral breakout mutation sensitivity.

# Architecture & Features

- **Automated Data Ingestion:** A hardened ingestion API that queries IEDB and VDJdb over secure connections, applying strict payload sanitization to prevent data contamination.
- **Structural GNN:** A PyTorch Geometric implementation using `GINEConv` layers and mixed-precision operations that processes vectorized graph representations of peptide-MHC complexes natively on the GPU, avoiding CPU Dataloader starvation.
- **Snakemake DAG:** Full orchestration from raw `.fasta` proteomes to cross-validated model weights and calibration plots, ensuring end-to-end provenance mapping.

# Acknowledgements

The SESTRAV pipeline integrates structural embeddings derived from the AlphaFold Protein Structure Database and the Protein Data Bank (CC-BY 4.0).

# References
