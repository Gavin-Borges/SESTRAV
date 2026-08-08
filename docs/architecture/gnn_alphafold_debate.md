# Phase 3: Three-Persona Debate Protocol
**Topic:** Structural Biology Integration into SESTRAV GNN using AlphaFold Edges

**Prompt:** "Weigh the tradeoffs between different graph convolution layers (e.g., GCN vs. GAT) for handling AlphaFold-derived structural edges."

---

## Persona 1: The Structural Computational Biologist
**Goal:** Maximize biological fidelity of the T-cell receptor (TCR) interface representation.

**Argument:** 
"Sequential backbone adjacency (1D topology) is biologically naive. A peptide bound to an MHC cleft folds into a distinct 3D conformation where residues $i$ and $i+3$ might be physically closer than $i$ and $i+1$, especially at the central bulge (p4-p8). We must integrate AlphaFold PDB outputs to construct an Angstrom-distance adjacency matrix.
If we use AlphaFold distances, we absolutely need a **Graph Attention Network (GAT)**. A GAT can learn to attend heavily to spatially proximate TCR-facing residues (e.g., p5 and p7) while ignoring buried anchor residues. A standard GCN treats all edges with the same static localized weight, which dilutes the structural context."

## Persona 2: The ML Performance Engineer
**Goal:** Maximize computational throughput, memory efficiency, and training stability.

**Argument:**
"Integrating AlphaFold edges sounds great until you look at the tensor dimensions and memory bandwidth on our RTX 4070 Ti Super. AlphaFold inference per peptide is incredibly slow. Even if we pre-compute the PDBs, transitioning from a static chain graph to dense/sparse pairwise distance matrices for millions of synthetic batch variations will shatter our memory throughput.
Furthermore, **GCNs** are far more robust here. GATs require multi-head attention aggregations which massively bloat VRAM usage and are notoriously unstable on small biological graphs (8-11 nodes). A GCN using a continuous, thresholded edge-weight (e.g., $1 / \text{distance}$) will capture the structural proximity without the exponential parameter cost of a GAT."

## Persona 3: The Pipeline Maintainer (SESTRAV Architect)
**Goal:** Ensure pipeline reproducibility, strict dependency bounds, and `freeze_mode` compliance.

**Argument:**
"Our `SestravConfig` and `FeatureStore` currently expect deterministic 1D sequence properties. If we integrate AlphaFold, we introduce a massive external runtime dependency. 
If we do this, it cannot be run online during `train_gnn.py`. We must build an offline structural cache. `GraphBuilder` should load pre-computed `.pt` sparse tensors from a `structural_cache/` directory.
As for GCN vs. GAT, I side with the ML Engineer for V2. Let's build a **GCN with thresholded distance edges** first. We already proved GCNs work with PyTorch sparse tensors without pulling in `torch-geometric`. Writing a custom multi-head GAT layer in raw PyTorch sparse math will introduce immense technical debt and potential silent bugs. Let's stick to GCNs until we prove the baseline Angstrom-distance matrix actually improves AUC-PR."

---

### Synthesis & Strategic Recommendation

1. **Architecture Choice:** **GCN with Edge Weights.** We will extend `GraphPredictor` to accept an edge-weight tensor ($1/d_{ij}$ derived from AlphaFold) rather than implementing a full GAT. This respects our base-PyTorch constraint and VRAM limits.
2. **Data Flow:** We will not run AlphaFold at runtime. We will create a standalone script that pre-computes distance matrices for the dataset and saves them to `data/structural_cache/`.
   > **Implemented as (added 2026-08-08):** this became `scripts/run_pandora_structures.py`, which generates Cb-Cb distance tensors via PANDORA/MODELLER rather than AlphaFold. `data/structural_cache/` is wired through `config.yaml` (`structural_cache_dir`) and consumed by `src/train_gnn.py`. The originally proposed filename `src/generate_structural_cache.py` was never created; this note records the rename so the design record stays traceable to the code that actually exists.
3. **GraphBuilder Update:** `GraphBuilder.build_chain_adj` will be extended to `GraphBuilder.build_spatial_adj(peptide, cache_dir)` which loads the distance matrix and thresholds it into a sparse adjacency tensor.
