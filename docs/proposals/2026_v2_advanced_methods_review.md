# SESTRAV v2.0 Advanced Methods Review

**A visionary-but-rigorous exploration of four frontier methodologies for next-generation epitope immunogenicity prediction**

**Status:** Research proposal / horizon-scan. Nothing here is implemented, certified, or promoted. This document is scoped to inform the v2.0 forward architecture and is held to the same evidentiary standard as the rest of the repository: every quantitative claim about SESTRAV traces to a certified artifact, and every external claim traces to a primary source. Where the honest answer is "not demonstrated here," that is the answer given.
**Audience:** SESTRAV maintainer and collaborators; computational immunologists benchmarking against the field; clinical oncologists evaluating the uncertainty-quantification layer.
**Date:** 2026-10-10
**Relation to existing records:** extends `docs/proposals/2026_feature_upgrade_roadmap.md` (Phases 1–3, still proposal), `docs/architecture/gnn_alphafold_debate.md`, `docs/architecture/gnn_models.md`, `ARCHITECTURE.md` §6, and `src/conformal.py`. It does not supersede them; it is the literature-and-method layer beneath the roadmap's feature-ranking.

> **Reading note on SESTRAV's own numbers.** This review is honest about where the framework stands. The canonical production scorer (`mode_31` RF) reports pooled **peptide-grouped** cross-validation AUC-PR **0.6055** and mean within-virus AUC-ROC **0.658** on the v5 corpus (35,597 active rows, 9 viruses), with leave-one-virus-out mean AUC-ROC **0.463** (`docs/claims_register.md` D15, re-baselined 2026-08-10). The gated GINEConv + ESM-2 research track **fails** its promotion Gate 1 (measured AUC-PR 0.6298–0.6458 across eight v5 runs vs the 0.65 threshold; `ARCHITECTURE.md` §6.3). These are the modest-AUC regime the clinical uncertainty section (§3) is designed for, not an embarrassment to be hidden. A method that only helps at AUC 0.90 is useless to SESTRAV; a method that delivers a *guaranteed* clinical bound at AUC 0.65 is the whole point.

---

## Table of contents

1. [Foundation models and protein language model embeddings](#section-1--foundation-models-and-protein-language-model-embeddings)
2. [Biomolecular structure prediction and tri-molecular modeling](#section-2--biomolecular-structure-prediction-and-tri-molecular-modeling)
3. [Conformal prediction and clinical uncertainty quantification](#section-3--conformal-prediction-and-clinical-uncertainty-quantification)
4. [Regulatory and translational roadmap](#section-4--regulatory-and-translational-roadmap)
5. [Consolidated references](#consolidated-references)

---

## Section 1 — Foundation models and protein language model embeddings

### 1.1 The landscape: what each model is, and what it is for

Protein language models (PLMs) are transformers pre-trained by self-supervision on hundreds of millions of natural protein sequences. The learned per-residue hidden states ("embeddings") are dense vectors that encode, implicitly, the statistics of which residues co-occur and co-vary across evolution. The three families named in the SESTRAV v2.0 brief occupy distinct points in the capability/cost space.

| Model | Params / embedding dim | Objective & modality | Training corpus | License / access (2026) | Primary source |
|---|---|---|---|---|---|
| **ESM-2** `t6_8M` | 8M / **320-d** | Masked-LM, sequence-only | UniRef50 (≈65M clustered seqs) | MIT, open weights | Lin et al., *Science* 379:1123 (2023) |
| **ESM-2** `t12_35M` | 35M / **480-d** | " | " | MIT | " |
| **ESM-2** `t30_150M` | 150M / **640-d** | " | " | MIT | " |
| **ESM-2** `t33_650M` | 650M / **1280-d** | " | " | MIT | " |
| **ESM-2** `t36_3B` | 3B / **2560-d** | " | " | MIT | " |
| **ESM-3** (`esm3-open-small`) | 1.4B (frontier: 98B) / 1536-d | **Multimodal generative** (sequence + structure tokens + function), chain-of-thought over tracks | 2.78B proteins, 771B tokens | Non-commercial license, gated on Hugging Face | Hayes et al., *Science* 387:850–858 (2025) |
| **Ankh** (base/large; Ankh3 2025) | 450M–1.15B / 768–1536-d | **Encoder–decoder (T5-style)**, protein-specific optimization, efficiency-first | UniRef50 | Open (academic), Hugging Face | Elnaggar et al., arXiv:2301.06568 (2023); Ankh3 arXiv:2505.20052 (2025) |

Three practical consequences for SESTRAV:

- **ESM-2 is the pragmatic default and is already wired in.** `ARCHITECTURE.md` §6.2 names `facebook/esm2_t12_35M_UR50D` (480-d) as the canonical node-feature source, precomputed and cached by `scripts/precompute_esm2_embeddings.py` and *never run per batch*. `src/gnn/models.py::GraphEncoderV2` defaults to `node_dim=320` (i.e. `t6_8M`); the capacity choice is therefore a knob, not a commitment, which is exactly the right posture for a scaling study.
- **ESM-3 buys a structure modality without a folding run.** Its structure track emits discretized local-geometry tokens; in principle these let a sequence-only pipeline inject coarse structural priors (see §2) without calling a co-folding model. The cost is a non-commercial license — a hard blocker for any eventual clinical-product path (see §4), and a reason to treat ESM-3 features as *research-track only*.
- **Ankh is the efficiency play.** It reaches ESM-2-650M-class downstream performance at a fraction of the parameter count, and its encoder–decoder form makes conditional generation (e.g. anchor-fixed peptide design) natural. For a solo-maintained project on a single RTX-4070-Ti-Super (the hardware constraint recorded in `docs/architecture/gnn_alphafold_debate.md`), Ankh-base is the most defensible large-model choice.

### 1.2 Application pattern: pMHC and TCR–pMHC

PLM embeddings enter immunogenicity models in three canonical ways. SESTRAV already uses the first; the second and third are the v2.0 frontier.

```
 (a) Per-residue node features  ── the SESTRAV v2.3 pattern ──────────────┐
     peptide "SIINFEKL" ──ESM-2──▶ H ∈ R^{L×d}  (L residues × d dims)      │
                                     │ one row per residue = graph node    │
                                     ▼                                      │
 (b) Pooled global descriptor                                              │
     mean/attention pool over L  ──▶ h ∈ R^{d}  (one vector per peptide)   │
                                     │ concatenate with tabular features   │
                                     ▼                                      │
 (c) Cross-chain / tri-molecular representation                           │
     [peptide ⊕ MHC-pseudoseq ⊕ TCR-CDR3] ──ESM-2──▶ joint attention ─────┘
```

- **pMHC.** The peptide is embedded directly; the MHC allele is represented by the **34-residue NetMHCpan pseudo-sequence** (the groove-contacting positions), for which `src/verify/mhc_pseudo_sequences.json` already ships the lookup. Both chains can be ESM-2-embedded and joined. Representative sequence-only pMHC/TCR models that establish this pattern: NetMHCpan-4.1 (Reynisson et al., *NAR* 2020), MHCflurry 2.0 (O'Donnell et al., *Cell Systems* 2020), BigMHC (Albert et al., *Nat. Mach. Intell.* 2023), and HLAB/ESM-fine-tuned binders.
- **TCR–pMHC.** The TCR contributes primarily through its CDR3β (and CDR3α when paired data exist). ESM-/BERT-style TCR encoders — TCR-BERT (Wu et al., 2021), catELMo (Zhang et al., 2023), NetTCR-2.0 (Montemurro et al., *Commun. Biol.* 2021), STAPLER, and the 2026 ImmSET sequence predictor (arXiv:2603.26994) — show that contextual embeddings of CDR3 improve specificity prediction over k-mer baselines, though all of them remain bottlenecked by the scarcity of paired, epitope-labeled TCR data.

### 1.3 The core question: do ESM-2 residue embeddings capture the physicochemistry at TCR-contact positions p4–p8 *better than* handcrafted scales?

This is the question that decides whether the ESM-2 track earns its complexity budget. SESTRAV's handcrafted representation (`mode_31`) places **20 physicochemical descriptors on the TCR-contact residues p4–p8** (following Chowell et al., *PNAS* 2015), plus 10 fixed-panel MHC binding scores and peptide length. The candidate replacement is a contextual per-residue ESM-2 embedding. The comparison is not rhetorical — it is measurable, and SESTRAV has already started measuring it.

**What the handcrafted scales are, and their structural limitation.**

- **Atchley factors** (Atchley et al., *PNAS* 102:6395, 2005): five orthogonal factors from a factor analysis of 54 amino-acid indices — roughly (1) polarity/hydrophobicity, (2) secondary-structure propensity, (3) molecular size/volume, (4) codon/refractivity, (5) charge. **Context-free:** valine at p5 gets the identical five numbers in every peptide.
- **Kidera factors** (Kidera et al., 1985): ten factors, same context-free property.
- **Miyazawa–Jernigan** (Miyazawa & Jernigan, *Macromolecules* 1985; *JMB* 1996): a 20×20 **statistical contact potential** — an energy for each residue *pair* in contact. It encodes pairwise interaction preferences, not per-position descriptors, so it is naturally a feature on *edges* or contacts, not nodes. SESTRAV does not currently use it; it is the obvious handcrafted baseline for any contact-graph edge feature.

The shared limitation: a static scale assigns a residue the same vector regardless of its neighbours. It cannot represent the fact that the TCR-facing chemistry at p5 depends on the identity of the anchors at p2/pΩ that set the peptide's register and bulge, nor on higher-order covariation.

**What ESM-2 embeddings add, in principle.** ESM-2 states are **contextual**: attention makes the embedding of p5 a function of the entire peptide (and, for longer inputs, flanking protein context). Linear-probe analyses of ESM-1b/ESM-2 recover hydrophobicity, charge, volume, and secondary-structure propensity as low-rank directions in embedding space (Rives et al., *PNAS* 2021; Lin et al. 2023), so the physicochemistry the Atchley factors hand-encode is *present and then some* — co-evolutionary signal that no static scale contains. On its face, ESM-2 strictly dominates.

**Why "in principle" is not "in SESTRAV," and the four caveats that decide it.**

1. **Short peptides are near-out-of-distribution.** ESM-2 is trained on full-length proteins; a bare 9-mer gives the attention stack almost no context to exploit, so much of the contextual advantage evaporates. Feeding the **source-protein window** around the epitope (not the bare 9-mer) is the design fix, and it is not yet done.
2. **The objective is evolutionary plausibility, not immunogenicity.** Masked-LM optimizes for what is a *natural* protein, which is close to — but not identical to — what a TCR *sees as foreign*. The immunogenicity-relevant signal may be present but **entangled** with abundant evolutionary signal, which is hard to learn from ~1,000–35,000 labeled rows.
3. **Anchor vs TCR-contact is not represented explicitly.** The handcrafted `mode_31` representation *deliberately* restricts physicochemistry to p4–p8 — it bakes in immunological prior knowledge. A raw ESM-2 embedding treats all positions alike and must *relearn* the anchor/contact distinction from scarce labels.
4. **SESTRAV's own ablation is the decisive datapoint, and it is not flattering to the PLM.** The N4 edge ablation (`ARCHITECTURE.md` §6.3; `docs/architecture/gnn_models.md`) found that removing chain-edge message passing *helped* (mean paired ΔAUC-PR +0.0175, 8/8 seeds), because **the ESM-2 node features already carry whole-peptide context, so local graph mixing is redundant.** More pointedly: the full GINEConv + ESM-2 track scores AUC-PR 0.6298–0.6458 across eight v5 runs and **fails Gate 1 (≥0.65)**, i.e. it does **not** beat the handcrafted `mode_31` RF (0.6055) by the promotion margin. On this dataset, at this label budget, contextual ESM-2 embeddings have **not** demonstrated superiority over the Atchley-style handcrafted representation.

**Honest verdict.** The literature prior favors ESM-2 embeddings; the SESTRAV evidence, so far, does not. The most likely reconciliation is that the *advantage is real but label-starved and context-starved*: it will appear when (i) the source-protein window replaces the bare peptide, (ii) the label corpus grows (more viruses, more IEDB), and (iii) a larger ESM-2 (650M) is dimensionality-reduced rather than fed raw into a small-n model. That is a falsifiable, pre-registerable claim — exactly the form §1.6 proposes to test. **Until then, ESM-2 embeddings are a research-track hypothesis, not an upgrade.**

### 1.4 A direct, pre-registerable probe experiment

To answer "do ESM-2 embeddings capture p4–p8 physicochemistry better than Atchley?" without conflating it with model-architecture choices, run a **representation-only linear-probe ablation** holding the classifier fixed:

- **R0 (baseline):** Atchley 5-factor vectors on p4–p8 (20-d) → logistic regression.
- **R1:** Kidera 10-factor on p4–p8 → logistic regression.
- **R2:** ESM-2 `t12` per-residue embeddings at p4–p8, mean-pooled → logistic regression.
- **R3:** ESM-2 `t33_650M` at p4–p8, PCA→32-d → logistic regression (the "ESM-2 PCA" feature already named in `2026_feature_upgrade_roadmap.md`).
- **R4:** ESM-2 `t33_650M` of the **±15-residue source-protein window**, pooled at the epitope span → logistic regression (tests caveat #1).

Score all five under `src.ml_utils.PeptideGroupedKFold` (closing the D15 leakage class), report ΔAUC-PR with bootstrap CIs, and pre-register the promotion band (e.g. R2/R3 must beat R0 by ≥ a band chosen from the paired-bootstrap null, mirroring the GNN gate preregistration in `docs/gnn_gate_retry_preregistration.md`). A clean probe isolates the *representation* question from the *fusion-architecture* question in §1.5. **This decides whether the embeddings carry the signal at all before any GNN is built on top of them.**

### 1.5 Concrete architectures: combining ESM-2 with graph neural networks

Four architectures, in increasing order of biological ambition and engineering cost. **A1 exists today;** A2–A4 are the forward menu.

#### A1 — Peptide chain-graph + ESM-2 nodes (the shipped v2.3 research track)

```
        peptide (L residues, L ≤ 11)
              │
   ESM-2 (frozen, cached)        SESTRAV physicochemical / binding (mode_31)
              │                                  │
     x ∈ R^{L×320..1280}                  physico ∈ R^{n_feat}
              │                                  │
   ┌──────────▼───────────┐                      │
   │  GINEConv  (d→256)    │  edge_attr ∈ R^{L'×3}│   one-hot {self, →, ←}
   │  BN, ReLU             │  chain graph         │
   │  GINEConv  (256→128)  │                      │
   └──────────┬───────────┘                      │
      pool (mean | attention)                     │
              │                                    │
         g ∈ R^{128}                      physico_block: Linear→64, BN, ReLU, Drop
              └───────────────┬────────────────────┘
                          concat (192)
                              │
                    fusion: Linear 192→128, ReLU, Drop, Linear→1  ──▶ logit
```

This is `GraphPredictorV2` / `GraphEncoderV2` verbatim (`src/gnn/models.py`). Two design choices already in the code are worth preserving in any successor: **frozen, pre-cached ESM-2** (reproducibility + VRAM; the embeddings are deterministic artifacts, not a runtime dependency — the `freeze_mode` constraint from `config.yaml` demands this), and **attentional pooling** (`AttentionalAggregation`, Li et al. 2016) as an alternative to mean pooling so that **anchor and TCR-contact residues are not diluted** by the rest of the peptide — the exact dilution worry Persona 1 raised in the AlphaFold debate.

**Known ceiling (from §1.3, caveat 4):** at `self-loop-only` the GINEConv stack degenerates to a per-node MLP + pooling and performs *as well or better*. A1's graph structure, over a bare peptide, is not adding value. A2–A4 exist to give the graph something non-trivial to do.

#### A2 — pMHC heterograph (give the graph a second chain)

Add MHC nodes from the 34-residue pseudo-sequence and **peptide↔MHC contact edges**, so message passing finally models an interaction rather than a 1-D chain:

```
   peptide nodes  ●─●─●─●─●─●─●─●─●        (ESM-2 node features)
                  │ ╲ │ ╳ │ ╱ │   ⟍         peptide–MHC edges
   MHC pseudoseq  ○─○─○─○─○─○─...─○ (34)   (ESM-2 node features)

   edge types: {chain, self, pep–MHC-contact}  → GINEConv edge_dim = k
   pooling: separate pep-pool ⊕ MHC-pool ⊕ cross-attention readout
```

Edges can be seeded from a static groove-contact map (cheap, allele-generic) or from predicted contacts (§2). This is the minimal change that makes allele identity a *structural* input — directly addressing the "allele-blind production model" limitation flagged in `README.md` and claims-register D30. **Recommended first A-series upgrade:** highest signal-to-engineering ratio.

#### A3 — TCR–pMHC tri-molecular graph (the full vision)

```
   TCR CDR3α/β  ◆◆◆◆◆◆           (ESM-2 / TCR-BERT node features)
                   ⟍  contact edges (from co-folding, §2)
   peptide      ●●●●●●●●●  p4–p8 are the hotspot
                   ╱  contact edges
   MHC          ○○○…○ (34)
```

If 3-D coordinates are available (§2), use an **E(3)-equivariant GNN** (EGNN, Satorra et al. 2021) or a geometric message-passing layer so that interface geometry — not just contact topology — drives the embedding, and edge features carry the physical contact metrics of §2 (including a **Miyazawa–Jernigan contact potential** per residue pair, the natural handcrafted edge feature). This is the architecture that would make §2's structural features first-class rather than tabular add-ons. It is also the most data-hungry and the most exposed to co-folding error (§2.3), so it is a **tier-2 research target**, gated behind A2 showing lift.

#### A4 — Attention-only fusion (no explicit graph)

A graph over 8–11 nodes is a small object; a **cross-attention transformer** (a Pairformer-lite block, à la AF3's Pairformer, or a Perceiver-IO readout) over the concatenated `[peptide ⊕ MHC ⊕ TCR]` token stream can subsume A2/A3 without hand-specified edges, letting attention learn the contacts. Cost: more parameters, more overfitting risk at n≈10³–10⁴. Worth benchmarking as the "no-graph-inductive-bias" control against A2/A3 — if A4 ties A3, the explicit graph is not earning its keep (the same lesson N4 already taught at the chain level).

#### Fusion & capacity guidance (applies to A1–A4)

- **Freeze the PLM, project the embedding.** Linear or 2-layer MLP projection `d→{64,128}` on frozen ESM-2; reserve LoRA/top-layer fine-tuning for when labels exceed ~10⁵. At n≈10³–10⁴, fine-tuning a 650M model is a near-certain overfit.
- **Reduce dimensionality before the small-n head.** Raw 1280-d (650M) or 2560-d (3B) into a model trained on ~1,000 positives invites memorization; PCA/UMAP or a learned bottleneck is mandatory, consistent with the "ESM-2 PCA" roadmap item.
- **Keep the handcrafted `mode_31` block in the fusion.** It is interpretable (SHAP-attributable), cheap, and currently the stronger arm. The ESM-2 track should have to prove it *adds* to `mode_31`, not replace it — mirroring SESTRAV's existing ablation discipline.
- **Calibration and gating are non-negotiable.** Any ESM-2/GNN successor still faces the five promotion gates in `src/verify/promote_gnn.py` (grouped AUC-PR ≥ 0.65, per-fold std ≤ 0.02, latency ≤ 2× RF, ECE < 0.05, escape sensitivity ≥ 80%). A frontier embedding that cannot clear ECE < 0.05 is useless to the §3 uncertainty layer.

---

## Section 2 — Biomolecular structure prediction and tri-molecular modeling

### 2.1 The open-access co-folding models (2024–2026)

A genuine generational shift happened in 2024–2025: all-atom **co-folding** models that jointly place proteins, peptides, nucleic acids, and ligands, with open (or open-ish) weights.

| Model | Release | Weights license (2026) | Headline capability | Primary source |
|---|---|---|---|---|
| **AlphaFold-3** | Server May 2024; code + weights **Nov 2024** | Code **CC-BY-NC-SA 4.0**; weights **gated, academic non-commercial, on request** | SOTA all-atom complexes; diffusion decoder; Pairformer | Abramson et al., *Nature* 630:493 (2024) |
| **Chai-1** | Sept 2024 | **Apache-2.0** (code + weights) | AF3-class accuracy, runs **without MSA**; PoseBusters 77% vs AF3 76% | Chai Discovery (2024), chaidiscovery.com; github.com/chaidiscovery/chai-lab |
| **Boltz-1** | Nov 2024 | **MIT** | First fully-open AF3 reproduction | Wohlwend et al., bioRxiv/MIT (2024) |
| **Boltz-2** | June 2025 | **MIT** | First co-folding model to **jointly predict structure + binding affinity**, ~FEP accuracy, ~1000× faster (developer benchmark) | boltz.bio/boltz2; MIT + Recursion (2025) |

**The licensing axis is strategically decisive for SESTRAV.** AF3's non-commercial weight license and ESM-3's non-commercial license are acceptable for the *research track* but are hard blockers on any eventual commercial/clinical-product path (§4). **Boltz-2 (MIT) and Chai-1 (Apache-2.0) are the only options compatible with an open, redistributable pipeline** — and Boltz-2's native affinity head is directly relevant to immunogenicity (binding stability is a known mediator). If structural features ever enter a certifiable SESTRAV release, they should come from Boltz-2 or Chai-1, not AF3.

### 2.2 Feasibility and accuracy for pMHC and pMHC–TCR

The feasibility question has a sharp, well-documented answer that gets worse as more chains are added:

- **pMHC (2 chains, class I).** Tractable and reasonably accurate. The MHC-I fold is rigid and over-represented in the PDB; the groove is modeled well; peptide backbone placement is good for canonical 9-mers and degrades for bulged/longer peptides. This is why SESTRAV's own structural cache was built with **PANDORA/MODELLER homology modeling** (`scripts/run_pandora_structures.py`), seconds per model, rather than AF — adequate for pMHC groove geometry at proteome scale.
- **pMHC–TCR (tri-molecular).** This is **hard**, and the honesty here matters. Standard AlphaFold-Multimer is **inconsistent** on TCR–pMHC (Bradley, *eLife* 12:e82813, 2023), which motivated **TCRdock**'s hybrid-template pipeline to constrain the docking geometry. The 2024–2026 co-folding models improve on this: a 2025 benchmark (bioRxiv 2025.11.30.691400) reports **AlphaFold-3 best-in-class median DockQ 0.636 (class I) / 0.679 (class II)** — "medium-to-acceptable" quality (DockQ > 0.49 acceptable, > 0.80 high), with **CDR3 pLDDT correlating with docking accuracy** (a usable built-in confidence signal). But the decisive immunological caveat is this:

> **Peptide-swap sensitivity is below the useful resolution.** The difference between an immunogenic neoepitope and its non-immunogenic wild-type counterpart is often a single TCR-contact substitution at p4–p8. Current co-folding models do not reliably resolve that the TCR docks differently, or at all, for such a pair — the inter-model and inter-seed variance exceeds the structural effect of the mutation. So a predicted TCR–pMHC complex is useful for *geometry and confidence triage*, and much weaker as a *per-peptide immunogenicity discriminator*.

### 2.3 Do physical contact metrics add orthogonal signal over sequence presentation scores?

This is the question that decides whether co-folding earns a feature slot. The candidate structural features:

| Feature | Definition | Handcrafted analogue |
|---|---|---|
| Interface ΔSASA (BSA) | buried surface area on complex formation | — |
| Hydrogen-bond count | donor–acceptor pairs across the interface | — |
| Shape complementarity *Sc* | Lawrence & Colman, *JMB* 234:946 (1993) | — |
| Rosetta `dG_separated` / ddG | interface energy; per-mutation ΔΔG | — |
| TCR-contact residue count | # peptide residues within contact of CDR loops | mirrors the p4–p8 prior |
| Electrostatic complementarity | interface charge matching | Miyazawa–Jernigan (pairwise) |

**The orthogonality argument (why they *might* help).** Presentation scores (NetMHCpan %rank, MHCflurry) answer "is the peptide displayed?" Immunogenicity additionally requires "is the displayed peptide *seen as foreign and engaged* by a TCR?" Interface metrics target the second question, which presentation scores are silent on. If they are predictive *and* uncorrelated with %rank, they are exactly the signal SESTRAV was built to add ("MHC binding is a weak proxy for immunogenicity," `README.md`).

**The skeptic's argument (why they probably don't, yet — and SESTRAV's own evidence agrees).**

1. **Derived-from-prediction noise.** Interface metrics computed on a *predicted* complex inherit its error. When §2.2 says the model can't reliably place the peptide/CDR3 for a WT/mutant pair, the resulting ΔSASA and H-bond counts are **noise dressed as signal**. Orthogonality to presentation scores is then *expected and useless* — it is the orthogonality of a random feature.
2. **SESTRAV already found no lift from structure-adjacent signal.** The N4 ablation (§1.3) showed chain-edge message passing adds nothing on top of ESM-2, and ESM-2 is itself a structure-aware representation. The structural-edge path (`build_spatial_adj`, PANDORA distances) is **disabled by default** (`use_spatial_adj: false`) and was never shown to lift AUC-PR — `gnn_alphafold_debate.md` records the decision to *not* run AF at runtime precisely because the baseline Angstrom-distance matrix had not been proven to help.
3. **Cost at proteome scale is prohibitive for the discovery stage.** AF3/Boltz-2 run in minutes–hours per complex on GPU. A proteome × allele × candidate-TCR grid is millions of complexes. This is incompatible with SESTRAV's proteome-scale Stage-1 generation.

**The reconciliation — a staged, falsifiable design.** The correct role for co-folding is **not** a proteome-scale feature; it is a **top-k re-ranking stage**:

```
 Stage 1–4 (sequence, cheap)         Stage 5 (structural, expensive, NEW)
 proteome ──▶ mode_31 RF ──▶ rank ──▶ top-k (k≈50–200) ──▶ Boltz-2 co-fold
                                                            │
                                   interface features + affinity head
                                                            │
                                   conformal re-rank within the pool (§3)
```

and the test of whether it helps is a **pre-registered orthogonality + lift ablation**:

- Compute interface features on Boltz-2/Chai-1 models for the labeled Tier-A / v5 corpora.
- Report (a) **partial correlation** of each structural feature with the label, *controlling for* MHCflurry %rank (orthogonality), and (b) **ΔAUC-PR** from adding the structural block to `mode_31` under `PeptideGroupedKFold` with bootstrap CIs (lift).
- Pre-register the promotion band; a feature must clear **both** bars (orthogonal **and** lifting) to enter a track — orthogonal-but-not-lifting is the random-feature trap in caveat #1.
- Control the negative set for decoy inflation (the pooled-AUC-ROC 0.9368 retraction, `README.md` Paradigm 2, is the cautionary precedent).

**Honest verdict.** Co-folding is genuinely useful for **structure-based triage of a shortlist** (geometry sanity, confidence filtering, affinity via Boltz-2) and for **populating A3's contact edges (§1.5)**. The claim that interface metrics add orthogonal *predictive* signal over presentation scores is **plausible but unproven in SESTRAV**, and the existing evidence (N4, disabled spatial edges) counsels skepticism until the staged ablation above clears both bars. Use Boltz-2/Chai-1 (open licenses), never AF3, on any path that might become a product.

---

## Section 3 — Conformal prediction and clinical uncertainty quantification

### 3.1 The clinical problem, stated precisely

A clinical oncologist selecting a personalized peptide pool has an **asymmetric** loss: **missing a truly immunogenic epitope (a false negative) is far costlier than including a dud (a false positive).** A dropped true epitope is a lost therapeutic opportunity that no downstream step can recover; an extra synthesized peptide is marginal cost. The oncologist's request is therefore not "give me calibrated probabilities" but:

> *"Give me a candidate pool that provably retains at least (1 − α) of the truly immunogenic peptides — a guaranteed false-negative-rate bound — even though your classifier's AUC is only 0.60–0.70."*

The central, non-obvious fact that makes this achievable: **a distribution-free FNR guarantee does not depend on AUC.** AUC governs *efficiency* (how large the pool must be), not *validity* (whether the bound holds). This is the single most important thing the SESTRAV uncertainty layer can offer a clinician at modest AUC.

### 3.2 What SESTRAV ships today, and why it is *not* this guarantee

SESTRAV's `src/conformal.py` implements **Cross Venn-Abers prediction (CVAP)** (Vovk, Petej & Fedorova, *NeurIPS* 2015). The module's own docstring is scrupulously honest, and it must be quoted rather than paraphrased:

- "**CVAP has NO proven validity guarantee.** Vovk et al. introduce the cross variant explicitly as the 'without guarantees of validity' half of their title."
- "`[lower, upper]` is **NOT a confidence interval** for the true conditional probability and has **no nominal coverage level.**"
- "Rows here are grouped by peptide and are **NOT exchangeable at row level.**"

So CVAP gives **calibrated multiprobability intervals** (useful for *ranking* and *triage within a pool*), but it is **not** the FNR-controlling selector the clinician asked for, and `docs/claims_register.md` D40 records exactly this scope limit. **The method that delivers the guarantee is inductive conformal prediction / conformal risk control — and it is not yet in the repository.** This section is the formal basis for adding it.

### 3.3 Framework A — Inductive (split) conformal prediction as a one-sided selector

**Setup.** Train the scorer `s(·)` on a proper training split (higher `s` = more immunogenic-looking). Reserve a **calibration set of confirmed positives** `D_cal⁺ = {(x_i, y_i = 1)}_{i=1}^{n}`, disjoint in peptide from training (grouped, per §3.5). Define the nonconformity of a positive as its score `s_i = s(x_i)` — a *low* score is *nonconforming* for a positive (the model is "surprised" to see a positive scoring low).

**The selection rule.** Retain every candidate whose score is at least a threshold `τ̂`:
```
    pool  S(τ̂) = { x : s(x) ≥ τ̂ }
    miss (false negative)  ⟺  a true positive has s(x) < τ̂
```
Choose `τ̂` as the empirical `α`-quantile of calibration-positive scores, with the conformal finite-sample correction:
```
    k = ⌊ α · (n + 1) ⌋            (k = 0 ⟹ τ̂ = −∞, retain everything)
    τ̂ = s_(k)                      the k-th smallest calibration-positive score
```

**Exact guarantee.** For a new exchangeable positive `X_{n+1}`, the rank of `s(X_{n+1})` among the `n+1` positive scores is uniform, so
```
    P( s(X_{n+1}) < τ̂ )  ≤  k / (n + 1)  ≤  α
    ⟺   P( retain a true positive )  =  TPR  ≥  1 − α       (finite-sample, distribution-free)
```
i.e. the **per-positive false-negative rate is bounded by α**, marginally over the draw of calibration and test positives. No distributional assumption, no AUC assumption. (This is the one-sided specialization of the standard split-conformal quantile lemma; Vovk et al. 2005; Angelopoulos & Bates, arXiv:2107.07511, 2023.)

**Why AUC is irrelevant to validity.** Nothing in the derivation references `s`'s discriminative power. A coin-flip scorer still yields a valid bound — it simply sets `τ̂` so low that `S(τ̂)` is almost the whole proteome. AUC enters only through pool size (§3.6).

### 3.4 Framework B — Conformal Risk Control (the rigorous generalization)

Split-CP controls a *coverage* event. The oncologist's FNR is an *expected loss*, and the exactly-right tool is **Conformal Risk Control (CRC)** (Angelopoulos, Bates, Fisch, Lei & Schuster, *ICLR* 2024; arXiv:2208.02814), which generalizes split conformal to any **bounded, monotone** loss — and whose **canonical worked example is the false-negative rate.**

**Setup.** Let a threshold `λ` define the retained set `S_λ(x) = {x : s(x) ≥ λ}`; smaller `λ` ⟹ larger pool ⟹ fewer misses, so the per-example FNR loss
```
    L_i(λ) = 1{ y_i = 1  and  s(x_i) < λ }   (restricted to / normalized over positives)
```
is **non-increasing in λ** and bounded by `B = 1`. Define the empirical risk on the `n` calibration positives `R̂_n(λ) = (1/n) Σ_i L_i(λ)`.

**The CRC threshold.** Choose the smallest-pool (largest-λ) threshold that keeps the inflated empirical risk under `α`:
```
    λ̂ = inf { λ :  (n / (n+1)) · R̂_n(λ)  +  B / (n+1)  ≤  α }
```

**Guarantee (Angelopoulos et al. 2024, Thm 1).** Under exchangeability of the calibration + test positives,
```
    E[ L_test(λ̂) ]  =  E[ FNR ]  ≤  α
```
with the bound **tight up to an O(1/n) factor.** Again distribution-free and AUC-agnostic. CRC reduces to split-CP when `L` is the miscoverage indicator, so Framework A is the special case — but CRC is the form to implement, because it extends cleanly to **group-FNR** (control the miss rate *per virus* or *per HLA supertype*, via one `λ` per group) and to compound losses.

**Non-monotone or multi-objective variants → Learn-Then-Test (LTT).** If the clinician wants to control FNR *and* cap pool size *and* bound a fairness gap simultaneously (a non-monotone, multi-risk problem), the right machinery is **LTT** (Angelopoulos, Bates, Candès, Jordan & Lei, arXiv:2110.01052), which recasts risk control as **multiple hypothesis testing** over a grid of configurations with FWER control. LTT is the general chassis; CRC is the efficient special case for the single monotone FNR objective.

### 3.5 The exchangeability problem — and SESTRAV's grouped, shifted reality

The guarantees above assume exchangeability, which SESTRAV **violates in two ways that must be handled explicitly**, or the "guarantee" is fiction:

1. **Peptide grouping (non-exchangeable rows).** As `src/conformal.py` note 5 states, rows share peptides and are not row-level exchangeable. **Fix:** calibrate at the **peptide (group) level** — one representative or one aggregated nonconformity per peptide group — and compute the quantile/CRC risk over groups, not rows. This is the conformal analogue of the `PeptideGroupedKFold` discipline already enforced across the repo (D15). Substring near-duplicates (D22) should be clustered into the same group.
2. **Covariate shift (IEDB → clinic).** The calibration distribution (curated IEDB viral epitopes) differs from the deployment distribution (a specific patient's neoantigens). Marginal exchangeability then fails. **Fix:** **weighted conformal prediction** (Tibshirani, Barber, Candès & Ramdas, *NeurIPS* 2019) with importance weights `w(x) = p_clinic(x)/p_cal(x)` (estimated by a domain classifier), which restores the guarantee under covariate shift. For the neoantigen setting specifically, **this is the difference between a bound that holds in the clinic and one that only holds on IEDB.**

Additionally, because positives are the minority class, use **label-conditional (Mondrian) conformal calibration** (Vovk 2012): calibrate the threshold on positives only, which is already how Frameworks A/B are posed — this is what makes the FNR (a positive-class quantity) the controlled risk rather than a pooled error.

### 3.6 The AUC → pool-size tradeoff, quantified

Validity is free; **efficiency is what the oncologist pays for.** At target `TPR = 1 − α`, the classifier's ROC fixes the false-positive rate it must accept, `FPR = ROC⁻¹(1 − α)`, and the pool's expected size (as a fraction of candidates, prevalence `π`) is
```
    E[ |S| / N ]  =  π·(1 − α)  +  (1 − π)·FPR(1 − α)
```
Under a bi-normal ROC with separation `d'` (where `AUC = Φ(d'/√2)`), `FPR(1−α) = Φ( Φ⁻¹(1−α) − d' )`. The weaker the classifier, the larger the pool for the same guarantee:

| AUC (`d'`) | Target TPR = 1 − α = 0.90 ⟹ required FPR | Pool size at π = 0.2 |
|---|---|---|
| 0.60 (0.36) | ≈ 0.80 | ≈ 0.82 of candidates |
| 0.65 (0.54) | ≈ 0.73 | ≈ 0.76 |
| 0.70 (0.74) | ≈ 0.65 | ≈ 0.70 |
| 0.85 (1.47) | ≈ 0.33 | ≈ 0.44 |

*(Illustrative bi-normal values; the real curve comes from the empirical OOF ROC, not a parametric fit.)* The reading for SESTRAV at its honest AUC-ROC ≈ 0.658: **a 90%-retention (FNR ≤ 0.10) pool is large — roughly three-quarters of screened candidates — but it is *provably* a 90%-retention pool.** That is a legitimate, defensible deliverable: the oncologist chooses `α` to trade synthesis/screening budget against the FN risk they are willing to bear, and the bound is honest at every point on that curve. The complementary move — *improving* the classifier (§1, §2) — is what *shrinks* the pool at fixed `α`; the two efforts are orthogonal and both worthwhile.

### 3.7 Proposed clinical workflow and module

```
 train scorer (proper split)
        │
        ▼
 peptide-GROUPED calibration positives ──(covariate-shift weights, §3.5)
        │
        ▼
 clinician picks α   e.g. FNR ≤ 0.10
        │
        ▼
 CRC threshold λ̂ (§3.4)  ──▶  candidate pool S(λ̂)   [GUARANTEED E[FNR] ≤ α]
        │
        ▼
 CVAP intervals (existing src/conformal.py)  ──▶  rank & triage WITHIN S
        │
        ▼
 Boltz-2 structural re-rank of top-k (§2)  ──▶  wet-lab screen (Wet_Lab_Protocol_v1.md)
        │
        ▼
 report realized FNR on a held-out test split  ──▶  claims register entry
```

**Concrete implementation.** Extend `src/conformal.py` (which is deliberately a pure, I/O-free library) with a `conformal_risk_control(scores, labels, groups, alpha, B=1.0)` function returning `λ̂` and the realized grouped risk, plus a `select_pool(scores, lambda_hat)`. Keep CVAP for the ranking role it is valid for; add CRC for the selection role it is valid for — and preserve the module's culture of stating *what is guaranteed and what is not* in the docstring. Pre-register `α` and the test-split FNR report (mirroring `docs/gnn_gate_retry_preregistration.md`), and wire the calibrator into `pipeline.smk` (today `scripts/fit_conformal_calibrator.py` is reachable from no rule — ROADMAP line ~91). **This is the single highest-value clinical-translation upgrade in this review:** it converts a modest-AUC scorer into an instrument with a guarantee a clinician can act on.

---

## Section 4 — Regulatory and translational roadmap

> **Scope and honesty note.** SESTRAV is, and declares itself to be, a **dry-lab research tool** that makes **no biological-efficacy claim** (`README.md`, `docs/limitations_statement_v1.md`). Nothing below reclassifies it as a medical device or a qualified biomarker. The purpose of this section is to map the regulatory terrain a SESTRAV-derived component would have to cross **if** it were ever embedded in an individualized-neoantigen-vaccine (INT) design pipeline, and to identify which of SESTRAV's existing governance assets already align with that terrain. The regulatory posture summarized here reflects the public record through early-to-mid 2026; status fields are flagged where a document is still draft or where the author is not certain of the latest revision.

### 4.1 Where AI immunogenicity prediction sits in the regulatory map

A tool like SESTRAV can enter a regulated workflow by three distinct doors, and the applicable framework differs sharply by door:

| Door | What the tool is | Governing framework |
|---|---|---|
| **(a) Drug-development support tool** | AI model producing evidence used in an IND/BLA decision (e.g. neoantigen ranking that informs which peptides enter a personalized product) | FDA AI-credibility framework (§4.2); EMA AI reflection paper (§4.4) |
| **(b) Qualified biomarker / drug-development tool (DDT)** | A computational biomarker formally qualified for a stated context of use | FDA Biomarker Qualification Program / BEST; EMA Qualification of Novel Methodologies (§4.3) |
| **(c) Software as a Medical Device (SaMD)** | Software that itself drives a clinical decision | IEC 62304 / ISO 13485 / ISO 14971; FDA SaMD + PCCP; EU AI Act + MDR (§4.5) |

The INT neoantigen-selection step today lives mostly behind **door (a)** — it is part of the sponsor's product-definition / manufacturing logic, assessed within the drug application, **not** separately qualified as a biomarker and **not** (yet) regulated as standalone SaMD. This is the single most important orientation fact for SESTRAV: the realistic near-term path is **credibility evidence inside someone else's drug application**, not independent device clearance.

### 4.2 FDA posture on AI in drug/biologic development (2024–2026)

The anchor document is the FDA draft guidance **"Considerations for the Use of Artificial Intelligence To Support Regulatory Decision-Making for Drug and Biological Products"** (CDER/CBER/CDRH/OCP; docket **FDA-2024-D-4689**; Federal Register **7 January 2025**, comment period closed 7 April 2025). It remains a **Level-1 draft ("not for implementation")** as of this writing — no final version had issued at the author's horizon; verify current status on the docket. It builds on two 2023 discussion papers (*"Using Artificial Intelligence and Machine Learning in the Development of Drug and Biological Products"*, >800 comments). Its core is a **7-step, risk-based credibility-assessment framework** for any AI model whose output supports a regulatory decision on drug/biologic **safety, effectiveness, or quality**:

```
 1. State the question of interest           (what regulatory question the AI informs)
 2. Define the Context of Use (COU)           (exactly how/where the model output is used)
 3. Assess AI model RISK = f(model influence, decision consequence)
        model influence     = weight of AI evidence in the decision
        decision consequence = severity if the decision is wrong
 4. Develop a credibility-assessment PLAN      (evidence proportional to risk)
 5. Execute the plan
 6. Document results and deviations
 7. Determine model adequacy for the COU; define lifecycle maintenance
```

The two pillars SESTRAV must internalize are **context of use (COU)** and **model risk as the product of influence and consequence.** A neoantigen ranker that merely *prioritizes* candidates for wet-lab screening (human-in-the-loop, low influence) sits far lower on the risk axis than one that *selects* the final clinical peptide set unreviewed (high influence, high consequence). **SESTRAV should always document its COU at the low-influence end** — "prioritization for downstream experimental screening," exactly the framing already in `README.md` — because that is both truthful and the lowest-evidence-burden position. One scope nuance to note explicitly: the draft **excludes pure drug-discovery and operational-efficiency uses** that do not affect patient safety, product quality, or study-result reliability. A pure proteome-screening aid may fall *outside* the framework entirely; output that shapes which epitopes a clinical product encodes falls *inside* it. The draft was informed by a **Duke-Margolis expert workshop (December 2022)** and the 2023 discussion-paper comments (an additional FDA/UMD M-CERSI workshop link is plausible but unverified here).

### 4.3 FDA Biomarker Qualification and computational biomarkers

- **BEST (Biomarkers, EndpointS, and other Tools) Resource** (FDA–NIH, 2016, maintained): the controlled vocabulary classifying biomarkers as *susceptibility/risk, diagnostic, monitoring, prognostic, predictive, pharmacodynamic/response,* and *safety.* A predicted-immunogenicity score, if pursued as a biomarker, would most naturally be framed **predictive** (predicts response to the vaccine) — a high bar.
- **Biomarker Qualification Program (BQP)** (21st Century Cures Act §3011; 21 USC 360bbb-8): a three-stage context-of-use path — **Letter of Intent → Qualification Plan → Full Qualification Package** — after which a qualified biomarker can be relied on across multiple programs within its stated COU, which has two components (the BEST category + the specific drug-development use). There is, to the author's knowledge, **no precedent for a qualified *in-silico* immunogenicity biomarker**; this would be novel territory and is **not** a near-term SESTRAV goal.
- **ISTAND pilot → permanent program.** For AI tools that *do not* fit the biomarker definition, FDA's **ISTAND** ("Innovative Science and Technology Approaches for New Drugs") qualification program is the more natural door — it explicitly covers AI-based algorithms and novel digital measures used in drug development. ISTAND **became a permanent FDA qualification program (FDA Voices, 31 July 2025)**. For a SESTRAV-like algorithm, however, the **January 2025 AI credibility framework (§4.2)** remains the more directly applicable lens than either BQP or ISTAND.
- **CDRH PCCP guidance** — *"Marketing Submission Recommendations for a Predetermined Change Control Plan for Artificial Intelligence-Enabled Device Software Functions"* (draft April 2023, **finalized December 2024**). This is the mechanism by which an AI model may be **updated post-authorization** (retraining, threshold changes) within a pre-specified envelope without a new submission. For any future SaMD path, SESTRAV's existing **promotion-gate machinery** (`src/verify/promote_gnn.py`) and **config-mutation discipline** are a near-perfect conceptual match for a PCCP's "modification protocol + impact assessment."

### 4.4 EMA posture (2024–2026)

- **"Reflection paper on the use of artificial intelligence in the lifecycle of medicines"** (EMA, adopted by **CHMP and CVMP**; ref. **EMA/CHMP/CVMP/83833/2023**; draft July 2023, **final September 2024**). Principles: a **risk-based** approach keyed to the AI's impact on benefit–risk; **human oversight** throughout; **data integrity and governance**; **technical robustness and transparency**; and that the **marketing-authorization holder remains fully responsible** for any AI used, including third-party and open-source models. The MAH-responsibility clause means a sponsor adopting a SESTRAV-like tool inherits accountability for its validation and provenance — which **raises the value of SESTRAV's cryptographic dataset governance and freeze-mode** to a prospective adopter.
- **HMA–EMA Big Data / AI workplan 2023–2028** and the **EMA Qualification of Novel Methodologies** pathway (CHMP/SAWP qualification advice and opinions) — the European analogue of FDA biomarker qualification, and the more plausible of the two formal-qualification routes for a computational method.
- **EU AI Act (Regulation (EU) 2024/1689)**, in force since August 2024 with phased application: AI that is a medical device or a safety component of one is **high-risk**, triggering risk-management, data-governance, logging, transparency, and human-oversight obligations that **stack on top of** MDR/IVDR. A research-prioritization tool is out of scope; a clinical-decision tool is squarely in scope.

### 4.5 Software governance, auditability, and data integrity

This is where SESTRAV is **already strong**, and where the report can be concrete rather than aspirational. The relevant standards and how SESTRAV maps to them:

| Standard / principle | What it requires | SESTRAV's existing alignment |
|---|---|---|
| **21 CFR Part 11** | Trustworthy electronic records & signatures; audit trails; access control | Git history + signed commits (DCO), release provenance |
| **ALCOA+** (data integrity) | Attributable, Legible, Contemporaneous, Original, Accurate, + Complete, Consistent, Enduring, Available | **Freeze-mode** + cryptographic dataset provenance (`*.provenance.json`), checksum manifests (`models/model_artifact_checksums.json`), `claims_register.md` |
| **GAMP 5, 2nd ed.** (ISPE, 2022) | Risk-based computerized-system validation; critical thinking; supplier assessment | Snakemake DAG reproducibility; `src/ci/validate_release.py`; hashed, `--require-hashes` dependencies |
| **FDA CSA** — *Computer Software Assurance for Production and Quality System Software* (draft Sept 2022, **final September 2025**; supersedes §6 of the 2002 *General Principles of Software Validation*; scope = production/quality-system software, **not** SaMD) | Risk-based assurance emphasizing testing effort proportional to risk, over exhaustive documentation | Library-scope CI coverage gating (`.coveragerc.library`), integration/benchmark gates |
| **IEC 62304 / ISO 13485 / ISO 14971** | Medical-device SW lifecycle, QMS, risk management — **only if SaMD** | N/A today (not a device); the lifecycle/risk scaffolding would need to be added for door (c) |
| **GMLP — "Good Machine Learning Practice for Medical Device Development: Guiding Principles"** (FDA + Health Canada + MHRA, Oct 2021; 10 principles) and **"Transparency for ML-Enabled Medical Devices"** (June 2024) | Data representativeness, leakage control, performance monitoring, transparency to users | **Peptide-grouped CV (D15 remediation)**, leakage auditing (`scripts/audit_cv_leakage.py`), honest claims register, model cards |
| **Model Cards** (Mitchell et al., FAccT 2019); **Datasheets for Datasets** (Gebru et al., 2021) | Documented intended use, metrics, limitations, dataset provenance | `docs/model_cards/*`, `docs/data_registry.md`, `docs/data/iedb_curation.md` |
| **TRIPOD+AI** (Collins et al., *BMJ* 2024); **DECIDE-AI, SPIRIT-AI, CONSORT-AI, PROBAST-AI** | Transparent reporting / risk-of-bias for clinical prediction models and AI trials | Reporting discipline of `README.md`/`docs/validation_summary.md` already approximates TRIPOD+AI; full compliance would be a documentation pass |

**Device vs non-device boundary.** Whether the SaMD stack (IEC 62304/ISO 13485/ISO 14971) applies at all turns on the device definition. A tool used internally to support drug development, or one meeting the **non-device Clinical Decision Support criteria** of the 21st Century Cures Act (FDA final *Clinical Decision Support Software* guidance, 28 Sept 2022) — transparent basis, independent clinician review, not time-critical — is **generally not** regulated as a device; the device standards then apply only as voluntary best practice. SESTRAV, as a research-prioritization tool with a human reviewing every ranked output, sits firmly on the non-device side.

**The governance headline:** SESTRAV's most distinctive assets — freeze-mode immutability, cryptographic provenance, a public claims register that *records its own retractions*, peptide-grouped leakage control, and gated model promotion — are precisely the **data-integrity (ALCOA+), leakage-control (GMLP), and change-control (PCCP) properties** that regulators are converging on. SESTRAV's honesty culture is not merely good scientific hygiene; it is **pre-adapted to the audit posture of 21 CFR Part 11 and the EMA MAH-responsibility clause.** The gap to any regulated use is not governance philosophy — it is the device-lifecycle scaffolding (IEC 62304/ISO 14971) that only becomes relevant at door (c), which SESTRAV explicitly does not pursue.

### 4.6 The INT clinical context and the precedent for the computational step

Individualized neoantigen therapies are the clinical setting that makes this review live — and, as of 2026, they carry **both** a landmark positive signal **and** a cautionary failure. Presenting them as uniformly "promising" would be dishonest; the record is mixed and recent.

- **mRNA-4157 / V940 (intismeran autogene; Moderna + Merck).** Individualized mRNA encoding **up to 34 neoantigens**, given with pembrolizumab. The randomized Phase 2b **KEYNOTE-942 / mRNA-4157-P201** trial (adjuvant, resected high-risk stage III/IV melanoma) reported a **~44% reduction in risk of recurrence or death** vs pembrolizumab alone (RFS HR ≈ 0.56) — though the small Phase-2 95% CI (≈ 0.31–1.08) crossed 1. FDA **Breakthrough Therapy Designation (2023)**; EMA **PRIME (6 April 2023)**. **Landmark update (announced 19 August 2026):** the Phase 3 **INTerpath-001** (adjuvant resected high-risk melanoma, n ≈ 1,137, 2:1) **met its primary RFS endpoint and a key secondary DMFS endpoint** at a prespecified interim analysis — reported as **the first positive Phase 3 result for an mRNA-based cancer therapy.** This is **topline only**: overall survival remained immature, full data were pending presentation, and no approval had issued at the author's horizon. The product embeds a **neoantigen-prediction-and-ranking pipeline** to select the encoded neoepitopes — the exact computational role a SESTRAV-class tool would play. INTerpath-002 (adjuvant NSCLC) and further indications are ongoing.
- **Autogene cevumeran / BNT122 / RO7198457 (BioNTech + Genentech).** In **pancreatic ductal adenocarcinoma** (Rojas et al., *Nature* 618:144–150, 2023; a 16-patient adjuvant Phase 1 with atezolizumab + mFOLFIRINOX), the vaccine induced de novo neoantigen-specific T cells in **8/16** patients, and responders had significantly longer RFS (not reached vs 13.4 months, *P* = 0.003) — a **correlation in a small single-arm study**, not controlled efficacy; a randomized Phase 2 (NCT05968326) is ongoing.
- **The cautionary data point — colorectal cancer.** The adjuvant monotherapy trial **BNT122-01 (NCT04486378)** in ctDNA-positive resected stage II/III CRC was **discontinued (late 2025 / early 2026)** after the independent DSMB observed a **numerical overall-survival imbalance (more deaths in the vaccine arm)** and crossed a predefined futility boundary. INT is **not** a solved modality; indication, setting, and the quality of neoantigen selection all matter, which raises — not lowers — the value of a rigorously validated, uncertainty-quantified selection step.

**The precedent that matters for SESTRAV:** in every current INT program, the computational neoantigen-selection algorithm is treated as **part of the investigational product's definition and manufacturing process, assessed within the drug application (door (a))** — it is **not** separately qualified as a biomarker, and it is **not** regulated as standalone SaMD. The evidentiary expectation is therefore **credibility and reproducibility of the selection pipeline** (CMC-adjacent), not independent clinical validation of the algorithm in isolation. This is good news for a SESTRAV-style tool: the realistic contribution is **a transparent, auditable, reproducible prioritization engine with honest performance bounds and — via §3 — a *guaranteed* false-negative-rate on the candidate pool**, which is exactly the kind of credibility evidence the FDA 7-step framework (§4.2) asks a sponsor to supply for a low-to-moderate-influence AI component.

### 4.7 Translational roadmap: concrete, staged, honest

1. **Stay at door (a), low influence.** Keep the documented COU at "prioritization for downstream wet-lab screening," never "autonomous clinical selection." This is truthful and minimizes evidentiary burden.
2. **Package SESTRAV's governance as a credibility dossier.** Freeze-mode, provenance JSONs, claims register, leakage audit, model cards, and promotion gates already constitute ~80% of an FDA-7-step credibility-assessment plan and a GAMP-5 validation package. A single `docs/regulatory_credibility_dossier.md` cross-walking existing artifacts to the 7 steps would make SESTRAV *citable* by a sponsor.
3. **Adopt the §3 conformal FNR guarantee.** A distribution-free, pre-registered FNR bound on the candidate pool is the most regulator-legible uncertainty statement a modest-AUC tool can make, and it directly answers the GMLP "performance monitoring" and transparency principles.
4. **Add TRIPOD+AI / Model-Card completeness passes** — low-cost documentation work that aligns existing honesty with the emerging reporting standards.
5. **Treat PCCP as the template for the promotion gates.** If a regulated path ever opens, `promote_gnn.py`'s gate logic is the skeleton of a Predetermined Change Control Plan; document it as such.
6. **Keep licenses product-compatible** (the §2 lesson): Boltz-2 (MIT) / Chai-1 (Apache-2.0), **not** AF3 or ESM-3 (non-commercial), on any artifact that might enter a sponsor's pipeline — a licensing defect discovered late is a translational blocker.

---

## Consolidated references

**Protein language models & embeddings (§1)**
- Lin, Z. et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model (ESM-2 / ESMFold). *Science* 379:1123–1130.
- Rives, A. et al. (2021). Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. *PNAS* 118:e2016239118.
- Hayes, T. et al. (2025). Simulating 500 million years of evolution with a language model (ESM-3). *Science* 387:850–858.
- Elnaggar, A. et al. (2023). Ankh: Optimized protein language model unlocks general-purpose modelling. arXiv:2301.06568. (Ankh3: Alsamkary et al., arXiv:2505.20052, 2025.)
- Atchley, W.R. et al. (2005). Solving the protein sequence metric problem (Atchley factors). *PNAS* 102:6395–6400.
- Kidera, A. et al. (1985). Statistical analysis of the physical properties of the 20 naturally occurring amino acids. *J. Protein Chem.* 4:23–55.
- Miyazawa, S. & Jernigan, R.L. (1985; 1996). Estimation of effective interresidue contact energies. *Macromolecules* 18:534; *J. Mol. Biol.* 256:623.
- Chowell, D. et al. (2015). TCR contact residue hydrophobicity is a hallmark of immunogenic CD8+ T-cell epitopes. *PNAS* 112:E1754–E1762.
- Reynisson, B. et al. (2020). NetMHCpan-4.1 and NetMHCIIpan-4.0. *Nucleic Acids Res.* 48:W449.
- O'Donnell, T.J. et al. (2020). MHCflurry 2.0. *Cell Systems* 11:42–48.
- Albert, B.A. et al. (2023). Deep neural networks predicting MHC presentation and immunogenicity (BigMHC). *Nat. Mach. Intell.* 5:861–872.
- Montemurro, A. et al. (2021). NetTCR-2.0. *Commun. Biol.* 4:1060.
- Wu, K. et al. (2021). TCR-BERT. bioRxiv/NeurIPS LMRL.
- Satorras, V.G. et al. (2021). E(n)-equivariant graph neural networks (EGNN). *ICML*.
- Li, Y. et al. (2016). Gated graph sequence neural networks (attentional aggregation). *ICLR*.

**Structure prediction & co-folding (§2)**
- Abramson, J. et al. (2024). Accurate structure prediction of biomolecular interactions with AlphaFold 3. *Nature* 630:493–500. (Code CC-BY-NC-SA 4.0, Nov 2024; weights gated, academic non-commercial.)
- Chai Discovery (2024). Introducing Chai-1. chaidiscovery.com; github.com/chaidiscovery/chai-lab (Apache-2.0).
- Wohlwend, J. et al. (2024). Boltz-1: democratizing biomolecular interaction modeling. bioRxiv (MIT license).
- Passaro, S. et al. / MIT + Recursion (2025). Boltz-2: towards accurate and efficient binding-affinity prediction. boltz.bio/boltz2 (MIT; structure + affinity).
- Bradley, P. (2023). Structure-based prediction of T-cell receptor:peptide-MHC interactions (TCRdock). *eLife* 12:e82813.
- Benchmarking TCR-pMHC structure prediction (2025). bioRxiv 2025.11.30.691400 (AF3 median DockQ 0.636 class I / 0.679 class II).
- Lawrence, M.C. & Colman, P.M. (1993). Shape complementarity at protein/protein interfaces. *J. Mol. Biol.* 234:946–950.

**Conformal prediction & uncertainty (§3)**
- Vovk, V., Petej, I. & Fedorova, V. (2015). Large-scale probabilistic predictors with and without guarantees of validity (Venn-Abers / CVAP). *NeurIPS* 28.
- Vovk, V., Gammerman, A. & Shafer, G. (2005). *Algorithmic Learning in a Random World* (inductive conformal prediction).
- Vovk, V. (2012). Conditional validity of inductive conformal predictors (Mondrian / label-conditional). *ACML*.
- Angelopoulos, A.N. & Bates, S. (2023). A gentle introduction to conformal prediction and distribution-free uncertainty quantification. arXiv:2107.07511.
- Angelopoulos, A.N., Bates, S., Fisch, A., Lei, L. & Schuster, T. (2024). Conformal Risk Control. *ICLR 2024*; arXiv:2208.02814 (FNR as canonical worked example).
- Angelopoulos, A.N., Bates, S., Candès, E.J., Jordan, M.I. & Lei, L. (2021). Learn then Test: calibrating predictive algorithms to achieve risk control. arXiv:2110.01052.
- Tibshirani, R.J., Barber, R.F., Candès, E.J. & Ramdas, A. (2019). Conformal prediction under covariate shift. *NeurIPS* 32.

**Regulatory & translational (§4)**
- FDA (Jan 2025, draft). Considerations for the Use of Artificial Intelligence To Support Regulatory Decision-Making for Drug and Biological Products. CDER/CBER/CDRH/OCP.
- FDA–NIH (2016, maintained). BEST (Biomarkers, EndpointS, and other Tools) Resource.
- FDA Biomarker Qualification Program (21st Century Cures Act §3011; 21 USC 360bbb-8).
- FDA/CDRH (final Dec 2024). Marketing Submission Recommendations for a Predetermined Change Control Plan for AI-Enabled Device Software Functions.
- FDA (final Sept 2025; draft Sept 2022). Computer Software Assurance for Production and Quality System Software (CSA); supersedes §6 of the 2002 General Principles of Software Validation.
- FDA (final 28 Sept 2022). Clinical Decision Support Software (non-device CDS criteria).
- FDA Voices (31 July 2025). ISTAND made a permanent qualification program.
- EMA/CHMP/CVMP (final Sept 2024; ref. EMA/CHMP/CVMP/83833/2023). Reflection paper on the use of artificial intelligence in the lifecycle of medicines. See also HMA–EMA AI workplan 2023–2028.
- Regulation (EU) 2024/1689 (EU AI Act, in force Aug 2024).
- ISPE (2022). GAMP 5: A Risk-Based Approach to Compliant GxP Computerized Systems, 2nd ed.
- 21 CFR Part 11 — Electronic Records; Electronic Signatures (effective 1997).
- IEC 62304; ISO 13485:2016; ISO 14971:2019 (medical-device software lifecycle, QMS, risk management).
- FDA / Health Canada / MHRA (Oct 2021). Good Machine Learning Practice for Medical Device Development: Guiding Principles (10 principles); Transparency for ML-Enabled Medical Devices (June 2024); PCCP for ML-Enabled Medical Devices (Oct 2023).
- Mitchell, M. et al. (2019). Model Cards for Model Reporting. *FAccT*; DOI 10.1145/3287560.3287596.
- Gebru, T. et al. (2021). Datasheets for Datasets. *Commun. ACM* 64(12):86–92; DOI 10.1145/3458723.
- Collins, G.S. et al. (2024). TRIPOD+AI statement. *BMJ* 385:e078378; DOI 10.1136/bmj-2023-078378.
- Cruz Rivera, S. / Liu, X. et al. (2020). SPIRIT-AI and CONSORT-AI extensions. *Nat. Med.* 26:1351–1363.
- Vasey, B. et al. (2022). DECIDE-AI reporting guideline. *Nat. Med.* 28:924–933.
- Weber, J.S. et al. (2024). Individualised neoantigen therapy mRNA-4157 (V940) plus pembrolizumab in resected melanoma (KEYNOTE-942). *Lancet* 403:632–644. (Phase 3 INTerpath-001 positive topline announced 19 Aug 2026; full data pending.)
- Rojas, L.A. et al. (2023). Personalized RNA neoantigen vaccines stimulate T cells in pancreatic cancer (autogene cevumeran). *Nature* 618:144–150; DOI 10.1038/s41586-023-06063-y. (BNT122-01 colorectal trial discontinued 2025/26 for an adverse OS imbalance.)

> **Provenance of this section's citations.** §1–§3 citations were verified against primary sources during drafting. The §4 regulatory names/dates/statuses were confirmed in a dedicated 2026-10-10 research pass: **settled** — CSA final (Sept 2025), CDRH PCCP final (Dec 2024, AI-DSF scope), EMA reflection paper final (Sept 2024), INTerpath-001 positive Phase 3 topline (19 Aug 2026, topline only), BNT122-01 colorectal discontinued (2025/26); **still draft** — the FDA AI drug/biologic credibility guidance (Jan 2025, docket FDA-2024-D-4689). Some journal DOIs and ISO numbers for long-established documents are drawn from reference knowledge rather than a re-fetched primary page (direct page-fetch was unavailable during the pass) and are flagged accordingly. A reader citing this document for a regulatory submission should re-confirm every load-bearing date/DOI against fda.gov / ema.europa.eu / the primary journal, since these move quarterly.
