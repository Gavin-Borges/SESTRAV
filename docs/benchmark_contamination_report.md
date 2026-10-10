# Presentation vs. Immunogenicity Confounding in MHC-I Epitope Prediction

### A methodological report on benchmark contamination, hard-negative construction, and the correct framing of an AUC collapse under assay-confirmed negatives

*Prepared 2026-10-10. Literature current to 2026.*

---

## 0. TL;DR for the impatient reviewer

1. **A drop from ~0.80 AUC to ~0.46 AUC when non-binding decoys are replaced by assay-confirmed, MHC-binding non-immunogenic peptides is the *expected* signature of a model that was silently predicting *presentation*, not *immunogenicity*.** It is the single most diagnostic experiment you can run, and the result you report is consistent with the published literature (Paul et al. 2020; the TESLA consortium 2020; independent re-benchmarks reporting AUCs of ~0.5–0.6).
2. **This is a *paradigm-relevant* finding — but only after you have ruled out the mundane explanations for a sub-0.5 AUC** (label-polarity bug, test set too small to distinguish 0.46 from 0.50, metric/threshold error, residual leakage in the opposite direction). The honest, defensible version of your paper *is* the most compelling version. Do not "spin" a bug as a paradigm; instead, run the decomposition experiment (§6) that *proves* which one you have.
3. **The correct frame is not "our model failed" but "the field's label was confounded."** Your 0.80 measured a binder-vs-non-binder task wearing an immunogenicity label. Report the ~0.46 with a bootstrap confidence interval, state plainly whether it is distinguishable from chance, and position it as corroboration of a documented, growing consensus rather than a lone negative result.

---

## 1. The core confound: presentation is necessary but not sufficient for immunogenicity

Antigen recognition by a CD8⁺ T cell is a **multi-step cascade**: proteasomal processing → TAP transport → MHC-I binding/stability → cell-surface **presentation** → **TCR recognition** → functional **activation**. Prediction tools have become excellent at the *early* steps (binding and presentation) because those steps generate abundant, cheap, high-quality training data — in-vitro affinity assays and, especially, mass-spectrometry (MS) eluted-ligand (EL) datasets. The *late* steps (TCR recognition, immunogenicity) are data-starved and, critically, are **conditioned on presentation already having happened**.

The consequence: **a peptide can be a strong binder/well-presented ligand and still be non-immunogenic** (self-tolerance, hole in the TCR repertoire, insufficient TCR-contact features). Conversely, a peptide cannot be immunogenic without being presented. This asymmetry is the root of the confound.

- Paul, Grifoni, Peters & Sette (2020) show directly that **prediction accuracy depends on which outcome you are predicting** — binding vs. natural processing vs. immunogenicity are measurably different tasks with different achievable performance ceilings [R5].
- Reviews of neoantigen prioritization note that **filtering candidates on MHC binding alone yields mostly non-immunogenic "binders"** [R11, R14].
- Structural/feature work (Chowell et al. 2015) localizes the immunogenicity signal to **TCR-contact residues (notably P4–P6)**, which are largely orthogonal to the anchor residues (P2/P9) that govern binding [R6, R7]. If your features and negatives do not separate these two signals, your model will learn the easier one.

**Implication for negative-set design:** if your negatives are non-binders, the decision boundary that maximizes AUC is simply "binds vs. does not bind." The model never has to learn anything about TCR recognition. Your 0.80 is the binding classifier's score; your 0.46 is what remains once binding is held constant.

---

## 2. Why non-binding decoys inflate AUC — the mechanism behind 0.80 → 0.46

This is a textbook instance of what Kapoor & Narayanan (2023) formalize as **leakage via a non-representative / confounded negative class** within their eight-type taxonomy of data leakage in ML-based science [R4]. The mechanism:

| Design | What the classifier can exploit | Apparent AUC | What is actually measured |
|---|---|---|---|
| Positives = immunogenic binders; **Negatives = random/non-binding decoys** | Binding/presentation signal (huge effect size) | **0.80+** | Presentation, mislabeled as immunogenicity |
| Positives = immunogenic binders; **Negatives = assay-confirmed non-immunogenic *binders*** | Only TCR-recognition signal (small, data-starved) | **~0.5** | Immunogenicity proper |

Two corroborating observations from the re-benchmarking literature:

- **Decoy sampling scheme changes apparent difficulty.** Benchmarks that draw proteome-wide random decoys report higher separability than those using length-matched or same-protein decoys; as "easy" decoys are exhausted, the margin of learned methods over a pure presentation baseline shrinks [R8, R9].
- **Train/test overlap compounds the inflation.** Re-analysis of MHC-I antigen-processing predictors found ~60% of one test set already present in a baseline's training data, so reported EL performance "is likely inflated" [R9]. Easy negatives + overlapping positives is the worst case.

The "true negative is unknown" problem (only a subset of candidates is ever experimentally tested for T-cell reactivity) is a long-standing hazard in epitope benchmarking and is exactly why comprehensively mapped systems (e.g., the vaccinia model system) were built [R8].

---

## 3. Landmark and recent papers that exposed over-optimistic evaluation

### 3.1 Consortium / community benchmarks
- **TESLA — Wells et al., *Cell* 2020** [R1]. The Tumor neoantigen SELection Alliance pooled 36 teams against shared tumor data, then experimentally tested 608 predicted epitopes. It isolated the *features that actually track immunogenicity* (binding affinity, presentation, agretopicity, hydrophobicity, foreignness) and demonstrated how much of the apparent predictive power across pipelines was attributable to presentation rather than T-cell recognition. The open benchmark (Synapse) remains a reference standard.

### 3.2 Tools whose *architecture* encodes the confound as a design principle
- **BigMHC — Albert et al., *Nature Machine Intelligence* 2023** (DOI 10.1038/s42256-023-00694-6) [R2]. BigMHC is deliberately **two-stage**: train on MS presentation data, then **transfer-learn** neoepitope immunogenicity. The explicit rationale — "not all presented neoantigens are immunogenic" — is itself an acknowledgment that presentation and immunogenicity must be modeled as *separate* tasks. Any single-stage model that reports immunogenicity AUC against non-binders is collapsing exactly the distinction BigMHC was designed to keep apart.
- **PRIME 2.0 / MixMHCpred2.2 — Gfeller et al., *Cell Systems* 2023** (14(1):72–83) [R3]. PRIME2.0 is trained with a curated set in which **the non-immunogenic peptides are themselves binders** (Table S4), i.e., binding is held approximately constant across classes. This is the reference example of correct hard-negative construction (§4) and pairs an explicit presentation model (MixMHCpred2.2) with an explicit recognition model (PRIME2.0).

### 3.3 Independent re-benchmarks reporting near-chance immunogenicity performance
- **Paul, Grifoni, Peters & Sette, *Front. Immunol.* 2020** (DOI 10.3389/fimmu.2019.03151) [R5]: accuracy is outcome-dependent; EL/binding ≠ immunogenicity.
- **"Beyond MHC binding" review, *Exploration of Immunology* 2024** [R11]: independent evaluation of immunogenicity tools yields **AUCs ≈ 0.58–0.61** — "limited ability to predict a cytotoxic response." *(Snippet-verified; confirm full text.)*
- **ITSNdb — *Unraveling tumor-specific neoantigen immunogenicity prediction*, 2023** [R14]: on 199 curated tumor neoantigens, relative binding affinity (DAI) **did not separate** immunogenic from non-immunogenic peptides; deep learners beat random only modestly and were inconsistent across validation sets. *(Snippet-verified.)*
- **"Universal shortcut bias in neoantigen prediction" (ImmUni), 2025/2026** [R10]: argues deep immunogenicity predictors have learned **shortcut correlations** (intra-HLA label imbalance / presentation signal) and that "benchmark accuracy alone fails to reflect genuine biological generalization." *(Snippet-verified; confirm venue/DOI.)*
- **Blinded prospective evaluation (reported as Culka et al., Genentech, 2025):** a blinded neoantigen study in which even leakage-corrected methods performed **no better than chance** in the zero-shot setting. *(Cited second-hand only — you must locate and verify the primary source before citing it.)*

### 3.4 General ML-leakage methodology
- **Kapoor & Narayanan, *Patterns* 2023** [R4]: the canonical cross-disciplinary treatment — 8-type leakage taxonomy, "model info sheets," and the finding that leakage-corrected models frequently fail to beat simple baselines. Your negative-set confound maps onto their "illegitimate features / non-representative test distribution" categories.

---

## 4. State of the art for **hard negative controls**

**Definition (the one to put in your Methods):** a *hard negative* for an immunogenicity task is a peptide that is **presentation-matched to the positives** — high-affinity binder (e.g., predicted/measured IC₅₀ < 500 nM, or high EL rank, or MS-eluted) to the *same or matched HLA allele(s)* — **and assay-confirmed non-immunogenic** (tested in a T-cell reactivity assay and shown negative). The point is to make presentation **uninformative** so the classifier is forced onto TCR-recognition signal.

### Sources of assay-confirmed binder-negatives
1. **Curated tested-negatives from IEDB** — peptides with negative T-cell assay outcomes, filtered to retain only predicted/known binders. This is how PRIME2.0's non-immunogenic set was built (binding ≳ 500 nM threshold) [R3, R12].
2. **MS-eluted ligands that are non-reactive** — peptides demonstrably presented (EL evidence) but negative in reactivity screens. Presentation is then *certain*, not predicted.
3. **Multimer/dextramer and combinatorial screens** (e.g., large pMHC multimer panels) — provide both reactive and non-reactive presented peptides against the same allele.
4. **Neoepitope benchmarks with position-aware annotation** — e.g., neo-epitope immunogenicity work that annotates which mutated positions are TCR-contacting vs. anchor, explicitly "preventing an important confounding factor" by preselecting on binding [R12].

### Pitfalls to pre-empt (reviewers will ask)
- **False negatives.** "Not detected" ≠ "not immunogenic" (repertoire/donor-dependent). State donor coverage and assay sensitivity; prefer sets tested across multiple donors. Protein-LM immunogenicity work flags false negatives as *the* field weakness [R13].
- **Allele coverage.** Hard negatives must be matched on HLA allele distribution to the positives, or allele frequency becomes the new shortcut (the ImmUni "intra-HLA imbalance" failure mode) [R10].
- **Length/anchor matching.** Match length distributions and anchor usage so the model cannot separate classes on peptide length or anchor chemistry.
- **Quantitative balance of presentation.** Don't just require "binder"; match the *distribution* of EL/affinity across classes, or residual presentation signal will leak.

---

## 5. Partitioning protocols and what premier journals now expect

Easy negatives are one leak; **sequence homology across folds** is the other. A peptide in test that is a near-neighbor of a training peptide leaks the label regardless of negative quality.

### Standard algorithmic partitioning
- **CD-HIT** — greedy incremental clustering with a word-count filter; fast, widely used. Typical identity thresholds in protein/peptide work: **40%** for stringent generalization tests, up to **70–90%** for near-duplicate removal; **assign whole clusters to a single split** [R16, R17].
- **MMseqs2** — more sensitive and far more scalable than CD-HIT for large sets; `mmseqs cluster` / `linclust` at a chosen `--min-seq-id`, again with **cluster-level (grouped) assignment** to train/val/test [R15, R16].
- **Cluster-then-split / grouped cross-validation** is the expected default: cluster first, then partition at the cluster level so no cluster straddles folds.

### The known gap for short peptides — identity splitting is *not enough*
- **PepBenchmark (2026 preprint)** argues identity-based clustering **misses k-mer leakage**: two short peptides can share an immunogenic k-mer motif yet fall below any global identity threshold, so MMseqs2/CD-HIT alone cannot resolve it. Recommended fix: a **hybrid split — k-mer-based splitting first, then MMseqs2 similarity clustering** on the remainder; and **report the max identity (and shared-k-mer rate) of each test peptide to its nearest training neighbor** as a leakage audit [R18]. *(Single preprint — present as emerging best practice, not settled consensus.)*

### What to put in Methods to satisfy *Bioinformatics*, *Briefings in Bioinformatics*, *Nature Machine Intelligence*
These venues increasingly require, explicitly or via reviewer norms:
1. **Homology-aware splits** with the tool, version, threshold, and cluster-assignment rule stated.
2. A **leakage audit**: distribution of nearest-neighbor identity between test and train (ideally the k-mer-overlap audit above).
3. **Presentation-controlled negatives** (or an explicit statement and sensitivity analysis if not), given the Paul et al. / TESLA precedent.
4. **Confidence intervals** on all metrics (bootstrap over the test set), not point AUCs.
5. **A baseline**: a pure presentation predictor (NetMHCpan EL) reported on the *same* splits — per Kapoor & Narayanan, "does it beat the trivial baseline?" is now a standard reviewer question [R4].
6. Open data/code (TESLA/Synapse-style) and a model info sheet [R4].

---

## 6. Framing the 0.80 → 0.46 result: paradigm correction vs. engineering failure

**The decision rule:** the *only* thing that distinguishes "important paradigm correction" from "we shipped a bug" is **whether the drop is caused by removing the presentation confound, or by an artifact.** You can — and must — determine which with a short, decisive set of experiments. Do these *before* writing the framing; the data will tell you which paper you have.

### 6.1 First, rule out the mundane causes of a sub-0.5 AUC
An AUC of 0.46 is *below* chance, which is a mild red flag that deserves three checks (all cheap):
- **Label-polarity / sign bug.** A systematically inverted AUC lands near `1 − x`. Verify your label encoding and score orientation end-to-end; confirm 0.46 is not an inverted 0.54.
- **Statistical distinguishability.** Compute a **bootstrap 95% CI** and a DeLong test against 0.5. If the CI spans 0.5 (likely, given plausible hard-negative set sizes), the honest claim is **"indistinguishable from chance,"** not "0.46." Do **not** over-interpret 0.46 < 0.50 as anti-predictive unless the CI excludes 0.5.
- **Metric/threshold/imbalance error.** Confirm you're reporting AUROC (threshold-free) and also AUPRC with the class prior stated; make sure no accuracy-at-0.5 artifact is driving the narrative.

If the drop survives these, it is real.

### 6.2 The decisive decomposition experiment (this *is* your headline figure)
Run all four cells of this 2×2 on identical, homology-split folds:

| | Easy negatives (non-binders) | Hard negatives (assay-confirmed binders) |
|---|---|---|
| **Your model** | ~0.80 (reproduce) | ~0.46 |
| **Pure presentation baseline (NetMHCpan-4.1 EL)** | **~0.80 (predict: matches you)** | **~0.50 (predict: collapses too)** |

**Interpretation that writes your paper:**
- If the EL baseline *also* scores ~0.80 on easy negatives and ~0.5 on hard negatives, you have **proven** that the original 0.80 was presentation signal and that *no current method* (yours or the field's baseline) carries immunogenicity signal once presentation is controlled. **That is a paradigm correction, not an engineering failure** — your model is behaving exactly as a correct model should; the *benchmark* was the problem.
- If your model collapses but a presentation-controlled competitor (e.g., PRIME2.0) retains some signal (AUC meaningfully > 0.5) on your hard negatives, then part of the drop is *your* model/features, and you must say so. Honest either way.

This single experiment converts a potentially embarrassing negative result into a controlled, mechanistic claim — and it is precisely the experiment TESLA and Paul et al. motivate.

### 6.3 How to frame it (the honest version is the strong version)
Lead with the mechanism, not the metric:

> "Prior reported performance for peptide immunogenicity prediction is substantially attributable to an information shortcut: the use of non-binding decoys as negatives reduces the task to presentation prediction. When negatives are restricted to assay-confirmed, HLA-matched *binders* and folds are partitioned by sequence homology, discrimination falls to chance (AUROC 0.46, 95% CI […]), matching the behavior of a pure presentation baseline on the same data. We therefore argue that presentation and immunogenicity must be benchmarked as separate tasks, and we release a presentation-controlled, homology-partitioned benchmark to enable this."

Why this framing is correct and will survive review:
1. It is **literally true** and matches TESLA [R1], Paul et al. [R5], the independent ~0.5–0.6 AUC re-benchmarks [R11, R14], and the shortcut-bias analyses [R10].
2. It reframes the number as a **property of the benchmark**, not a defect of your estimator. A below-chance AUC under a *fixed, confounded* convention is evidence *about the convention*.
3. It is **constructive**: you ship the corrected benchmark + partitioning protocol + baseline, which is a contribution venues reward (cf. BigMHC's two-stage design [R2], PRIME2.0's binder-negatives [R3]).
4. It **pre-empts the "your model is just bad" reviewer** by showing the field's own baselines collapse identically.

### 6.4 What *not* to do
- Do **not** present 0.46 as a positive result in disguise, pick the negative set that maximizes the number, or omit the easy-negative comparison. That would be the engineering-failure-dressed-as-paradigm move, and it is both unethical and trivially caught: a reviewer will ask for the presentation baseline and the CI.
- Do **not** claim "0.46 < 0.5 means the model learned an anti-signal" unless the CI excludes 0.5 and you can explain the biology. "Chance-level" is the safe, honest claim.
- Do **not** quietly fix the confound and re-report 0.8 on a new easy-negative set. The *contribution is the correction*, so make the collapse the centerpiece.

---

## 7. Concrete recommendations checklist

- [ ] Reproduce 0.80 on the original easy-negative split (establish the baseline you are correcting).
- [ ] Build hard negatives: HLA-matched, EL/IC₅₀-confirmed binders, assay-confirmed non-immunogenic, length/anchor/allele-distribution matched to positives.
- [ ] Re-partition with MMseqs2 (state version + `--min-seq-id`) or CD-HIT at a stringent identity; assign **whole clusters** to folds; add a **k-mer-overlap audit** of test-vs-train.
- [ ] Report AUROC **and** AUPRC with **bootstrap 95% CIs** and a DeLong test vs. 0.5.
- [ ] Run the 2×2 decomposition (§6.2) including a **NetMHCpan EL presentation baseline** and ≥1 presentation-controlled competitor (PRIME2.0).
- [ ] Rule out label-polarity/metric bugs explicitly in the supplement.
- [ ] Release data + code + model info sheet; frame as a benchmark correction (§6.3).

---

## References

> **Verification status.** R1–R7 are confirmed against primary/official pages or well-established records. R8, R9, R15–R17 are confirmed as real works via search but specific numeric thresholds should be re-checked against the source. R10, R11, R12, R13, R14, R18 are **snippet-verified only** (I could not open the full text in this environment) — confirm authors/venue/DOI before citing. The Genentech "Culka et al. 2025" item in §3.3 is **second-hand only** and must be located before use.

1. **Wells DK, et al.** "Key Parameters of Tumor Epitope Immunogenicity Revealed Through a Consortium Approach Improve Neoantigen Prediction." *Cell* 183(3):818–834 (2020). (TESLA consortium.) https://pmc.ncbi.nlm.nih.gov/articles/PMC7652061
2. **Albert BA, et al. (Karchin lab).** "Deep neural networks predict class I major histocompatibility complex epitope presentation and transfer learn neoepitope immunogenicity." *Nature Machine Intelligence* (2023). DOI 10.1038/s42256-023-00694-6. Preprint: https://www.biorxiv.org/content/10.1101/2022.08.29.505690
3. **Gfeller D, et al.** "Improved predictions of antigen presentation and TCR recognition with MixMHCpred2.2 and PRIME2.0 reveal potent SARS-CoV-2 CD8⁺ T-cell epitopes." *Cell Systems* 14(1):72–83 (2023). https://pmc.ncbi.nlm.nih.gov/articles/PMC9811684
4. **Kapoor S, Narayanan A.** "Leakage and the reproducibility crisis in machine-learning-based science." *Patterns* (2023). Project page: https://reproducible.cs.princeton.edu/ ; text: https://ar5iv.labs.arxiv.org/html/2207.07048
5. **Paul S, Grifoni A, Peters B, Sette A.** "Major Histocompatibility Complex Binding, Eluted Ligands, and Immunogenicity: Benchmark Testing and Predictions." *Frontiers in Immunology* (2020). DOI 10.3389/fimmu.2019.03151. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7012937/
6. **Li G, et al.** "DeepImmuno: deep learning-empowered prediction and generation of immunogenic peptides for T-cell immunity." *Briefings in Bioinformatics* 22(6):bbab160 (2021). https://academic.oup.com/bib/article/22/6/bbab160/6261914
7. **Chowell D, et al.** "TCR contact residue hydrophobicity is a hallmark of immunogenic CD8⁺ T cell epitopes." *PNAS* 112(14) (2015). DOI 10.1073/pnas.1500973112. https://www.pnas.org/doi/10.1073/pnas.1500973112
8. **Paul S, et al.** "Benchmarking predictions of MHC class I restricted T cell epitopes in a comprehensively studied model system." *PLOS Computational Biology* (2020). DOI 10.1371/journal.pcbi.1007757. https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007757
9. "Improving MHC class I antigen-processing predictions using representation learning and cleavage site-specific kernels" (MHCrank). *PMC* (2022). Reports ~60% train/test overlap inflating EL performance; decoy-scheme effects. https://pmc.ncbi.nlm.nih.gov/articles/PMC9499997
10. "Cross-task interpretability through unified modeling reveals a universal shortcut bias in neoantigen prediction" (ImmUni). *PMC* (2025/2026). https://pmc.ncbi.nlm.nih.gov/articles/PMC13347938/ *(verify venue/DOI)*
11. "Beyond MHC binding: immunogenicity prediction tools to refine neoantigen selection in cancer patients." *Exploration of Immunology* (2024), Article 100391. Independent AUC ≈ 0.58–0.61. https://www.explorationpub.com/Journals/ei/Article/100391 *(snippet-verified)*
12. "Prediction of neo-epitope immunogenicity reveals TCR recognition determinants and provides insight into immunoediting." *Cell Reports Medicine* (2021). Non-immunogenic set preselected on binding; TCR-contact vs. anchor annotation. https://www.sciencedirect.com/science/article/pii/S2666379121000057 *(snippet-verified)*
13. "A modular protein language modelling approach to immunogenicity prediction." *PMC* (2024). Flags false negatives as a core field weakness. https://pmc.ncbi.nlm.nih.gov/articles/PMC11581412/ *(snippet-verified)*
14. "Unraveling tumor-specific neoantigen immunogenicity prediction: a comprehensive analysis" (ITSNdb). *PMC* (2023). DAI does not separate immunogenic vs. non-immunogenic. https://pmc.ncbi.nlm.nih.gov/articles/PMC10411733 *(snippet-verified)*
15. **Steinegger M, Söding J.** MMseqs2: fast and sensitive clustering/search of large protein sequence sets. (Tool reference for homology-aware splitting.)
16. **PepBenchmark** and related peptide-ML benchmarks: standard use of MMseqs2/CD-HIT clustering + near-duplicate removal (>90%) before splitting. arXiv:2604.10531 *(preprint)*
17. **CD-HIT** (Li & Godzik; Fu et al.) — greedy identity clustering; typical thresholds 40% (stringent) to 70–90% (dedup), whole-cluster split assignment.
18. **PepBenchmark (2026 preprint).** k-mer leakage not caught by identity clustering; recommends hybrid k-mer + MMseqs2 split and nearest-neighbor identity audit. arXiv:2604.10531 *(single preprint — emerging practice)*
