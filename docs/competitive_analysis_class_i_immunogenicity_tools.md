# Competitive Analysis: Class I Epitope Immunogenicity & Presentation Predictors

**Scope.** A comparative intelligence review of five leading computational tools for MHC
class I epitope **immunogenicity / presentation** prediction — **BigMHC**, **T-SCAPE**,
**MixMHCpred 2.2**, **DeepImmuno**, and **PRIME 2.0** — against the two dominant
binding/presentation reference points (NetMHCpan-4.1 EL, MHCflurry 2.0) and against the
evaluation protocol SESTRAV adopts (a completed nine-virus leave-one-virus-out cycle).

**Status.** Analytical / competitive-intelligence document. It introduces **no new
SESTRAV performance claim**: every SESTRAV number quoted here is already carried by a
tracked artifact and cited inline (`docs/paper.md`, `README.md`,
`results/table3_tier_a_metrics.csv`, `docs/claims_register.md`). External-tool statements
are cited to the primary literature with a confidence marker where the fact rests on a
secondary source or a preprint table flagged as uncertain. This document follows the
register's **Four Questions** discipline — mechanism, scope, limitation, fairness — and
discloses the asymmetries in SESTRAV's own shared-benchmark numbers rather than resolving
them in SESTRAV's favour.

**Last updated:** 2026-10-10.

---

## 1. Executive summary

1. **All five tools are sequence/physicochemistry models.** None of the five consumes
   three-dimensional pMHC coordinates. "Structural" framings (SESTRAV's own name
   included) refer to physicochemical proxies at TCR-contact positions, not
   crystallographic or predicted atomic geometry. The architectures span one PWM/motif
   model (MixMHCpred 2.2), one shallow neural network over a binding score (PRIME 2.0),
   one residual CNN (DeepImmuno), one large transfer-learned attention+LSTM ensemble
   (BigMHC), and one multidomain adversarial-adaptation network (T-SCAPE).

2. **Negative-set construction is the single biggest methodological fault line.**
   Presentation models (MixMHCpred 2.2; BigMHC-EL) are trained against **random proteomic
   decoys** at high decoy ratios. Immunogenicity models diverge: PRIME 2.0 and (the
   T-cell-activation head of) T-SCAPE use **assay-confirmed non-immunogenic binders plus
   random peptides**; BigMHC-IM and DeepImmuno rely on **IEDB assay-confirmed negatives**
   that are heavily cancer-neoepitope-weighted. The hardest and most informative negative
   class — *strong binders that are confirmed non-immunogenic* — is scarce in every
   training corpus.

3. **No tool except SESTRAV reports a completed held-out-pathogen cycle.** PRIME 2.0 runs
   leave-one-**allele**-out and leave-one-**study**-out; BigMHC and T-SCAPE control
   leakage by **peptide-sequence homology**, not by source organism; DeepImmuno uses
   random 10-fold CV plus two small curated independent sets. Pathogen-level holdout, where
   reported in the field at all, has been a single-pathogen supplementary check
   (Bravi et al. 2023; the TRAP workflow, Lee et al. 2023) — not a cycle.

4. **DeepImmuno degrades on independent, deduplicated, length/allele-diverse benchmarks**
   for three compounding reasons: redundancy in its IEDB training/evaluation corpus, a
   narrow input domain (9–10-mers and a fixed pseudosequence panel) that silently drops
   peptides, and a beta-binomial label that encodes IEDB assay-count priors more than
   transferable biology. On SESTRAV's shared field it is the weakest tool (AUC-PR 0.698,
   AUC-ROC **0.542** — effectively chance).

5. **The "binding ceiling" is real and narrow.** On a single shared 704–720-peptide field,
   the best dedicated immunogenicity tool beats a raw MHCflurry 2.0 presentation baseline
   by only **≈ +0.02 AUC-PR** and **≈ +0.03 AUC-ROC** — inside bootstrap noise — while
   MixMHCpred 2.2 and DeepImmuno sit **at or below** the binding baseline. The newest
   multidomain model, T-SCAPE, reports a class I immunogenicity AUROC of only **≈ 0.568**.
   Independent benchmarking (Buckley et al. 2022) found that no existing model
   substantially beat random or improved appreciably on HLA-ligand prediction for
   SARS-CoV-2. Immunogenicity prediction has, to date, bought very little above binding.

---

## 2. Master comparative matrix

Tools in columns; the five requested analysis parameters in rows. Cells are compressed;
Sections 3–7 give the full treatment and citations. "EL" = eluted-ligand presentation;
"IM" = immunogenicity.

| Parameter | **BigMHC** | **T-SCAPE** | **MixMHCpred 2.2** | **DeepImmuno** | **PRIME 2.0** |
|---|---|---|---|---|---|
| **1. Architecture** | Ensemble of 7 deep nets: anchor block + single-head self-attention + bidirectional LSTM (window 8); one-hot peptide + MHC pseudosequence; **transfer learning EL → IM**. Sequence-only. | Multidomain net with **adversarial domain adaptation**: shared "integrative" encoder (domain-invariant) + task-specific encoders, source discriminator, orthogonality constraints; pretrain → fine-tune. Sequence/multi-source. | **Motif / position-weight-matrix** presentation model with mixture-model deconvolution of MS immunopeptidomes. Not a neural net. Sequence-only. | **Residual CNN** over peptide + HLA pseudosequence, each residue encoded by AAindex-PCA physicochemical features; sibling **GAN** generator. Sequence/physicochem. | **Shallow neural network** on top of the MixMHCpred 2.2 binding score + amino-acid frequencies at TCR-facing positions + length. Sequence-only. |
| **2. Negatives / controls** | EL: MS eluted ligands vs **random proteomic decoys** (~1:20). IM: IEDB assay data, **predominantly neoepitopes** (5,279 / 6,873); negatives = assay-confirmed non-immunogenic. | T-cell-activation head fine-tuned on assay data (12,782 points); pretraining draws EL, binding affinity, TCR-pMHC, source-organism signals. Negatives include assay-confirmed non-immunogenic. *(detail from preprint — medium confidence)* | MS **eluted ligands** as positives; **random human-proteome peptides** as background (%Rank defined against random peptides). A presentation tool, not an immunogenicity tool. | IEDB immunogenicity assays; **beta-binomial "immunogenic potential"** weights each pMHC by positive/negative assay counts; negatives = assay-confirmed non-immunogenic. | **Assay-confirmed immunogenic + non-immunogenic + random peptides**; v2.0 corpus deliberately spans a realistic binding range (not ligand-enriched). |
| **3. Cross-pathogen validation** | **None.** Infectious-disease eval separated from training only by **removing overlapping peptide-MHC instances**, not by organism. Neoepitope-centric. | **None by organism.** Leakage controlled by **9-mer homology** (training peptides within <2 mismatches of any benchmark peptide removed). Neoantigen + infectious-disease *discovery* sets. | **None.** Evaluated on presentation of eluted ligands (allele-level), not pathogen holdout. | **Random 10-fold CV** on IEDB + two small curated independent sets (TESLA neoantigens, SARS-CoV-2). Cancer-neoantigen-heavy. | **10-fold + leave-one-ALLELE-out + leave-one-STUDY-out.** No pathogen holdout. |
| **4. Independent-set degradation** | Headline IM gains are **top-k precision under extreme imbalance** (AUPPVn 2–25× NetMHCpan-4.1 on a 45M-decoy background) — sensitive to corpus composition and the hardest-negative gap. | New (2025); limited third-party replication. Absolute class I IM AUROC is low (**≈ 0.568**), so headroom to degrade is small. | As a presentation model it is stable **for presentation**; it simply was never an immunogenicity predictor, so it "degrades" only when repurposed as one. | **Yes — the clearest case.** Narrow domain drops peptides (86.5% coverage on the shared field); IEDB redundancy inflates CV; on independent fields it collapses toward chance (AUC-ROC 0.542). | No deduplicated pathogen-holdout stress test reported; leave-one-allele/study-out are weaker generalization probes than organism holdout. |
| **5. Margin over binding ("ceiling")** | Shared field: AUC-PR **0.822** vs binding-only 0.800 (**+0.022**, n.s.); AUC-ROC 0.659 vs 0.633; **best ISSR@10 (0.917)**. | Self-reported class I IM AUROC **≈ 0.568** vs MHCnuggets 0.557 (**+0.011**, P<0.001); claims gains over NetMHCpan-4.1 / MHCflurry-2.0 / MixMHCpred-2.2 on discovery sets. *(preprint — medium confidence)* | Shared field: AUC-PR **0.795**, AUC-ROC 0.626 — **below** the binding-only baseline. For *presentation* it exceeds NetMHCpan-4.1/MHCflurry-2.0/HLAthena. | Shared field: AUC-PR **0.698**, AUC-ROC **0.542** — **well below** binding-only. | Not reproducible from a certified SESTRAV artifact (compared on capabilities only). Author claim: improvement over MixMHCpred is "**likely not due to better binding**" — i.e. a genuine but modest above-binding signal. |

**Shared-field reference values** (`results/table3_tier_a_metrics.csv`; 720-peptide Tier A
set, external tools fully scored, SESTRAV out-of-fold on the 704 it scores):

| Tool | AUC-PR | AUC-ROC | ISSR@10 | n scored | Coverage |
|---|---|---|---|---|---|
| SESTRAV RF (30-feat, OOF, 2026-05) | 0.828 | 0.726 | 0.843 | 704 | 97.8% |
| **BigMHC** | 0.822 | 0.659 | **0.917** | 720 | 100% |
| Binding-only (MHCflurry 2.0) | 0.800 | 0.633 | 0.861 | 720 | 100% |
| **MixMHCpred 2.2** | 0.795 | 0.626 | 0.847 | 720 | 100% |
| **DeepImmuno** | 0.698 | 0.542 | 0.710 | 623 | 86.5% |

> **Fairness note (carried from `docs/claims_register.md` D16/D22, HD1).** SESTRAV's 0.828
> is a 2026-05 **30-feature** out-of-fold figure, *not* the canonical `mode_31` production
> model, and its 704-peptide pool carries a disclosed, **unquantified 32.1% substring-homology
> risk**. Its edge over BigMHC is a **statistical near-tie** (paired ΔAUC-PR +0.018, 95% CI
> −0.022…+0.058, p=0.37) and SESTRAV ranks **4th of 5 on ISSR@10**. The point of the table
> is the *compression of the whole field around the binding baseline*, not a SESTRAV win.
> T-SCAPE and PRIME 2.0 are **not** in this certified field (T-SCAPE postdates it; PRIME has
> no reproducible metric here) and their rows above are sourced externally.

---

## 3. Parameter 1 — Architectural formulation

| Tool | Family | Structural coordinates? | Defining mechanism |
|---|---|---|---|
| BigMHC | Deep NN ensemble (attention + BiLSTM) | No — MHC **pseudosequence**, one-hot | Transfer learning from large-scale EL presentation to IM |
| T-SCAPE | Multidomain NN, adversarial adaptation | No — multi-source sequence features | Domain-invariant shared encoder + task-specific encoders |
| MixMHCpred 2.2 | PWM / motif mixture model | No | MS-driven motif deconvolution per allele |
| DeepImmuno | Residual CNN (+ GAN) | No — AAindex-PCA physicochemistry | Physicochemical CNN with beta-binomial-weighted labels |
| PRIME 2.0 | Shallow NN over a binding score | No | TCR-recognition propensity layered on presentation |

**BigMHC** (Albert et al., *Nat Mach Intell* 2023) is an ensemble of seven deep networks.
Each network one-hot encodes the peptide and the MHC pseudosequence, routes the anchor
residues (first and last four) through a dedicated two-layer "anchor block," applies
single-headed self-attention, and reads the sequence with a bidirectional LSTM using a
window length of eight (the minimum pMHC peptide length) rather than the canonical one.
Its defining property is **transfer learning**: the ensemble is first trained on
mass-spectrometry eluted-ligand presentation (**BigMHC-EL**) and then fine-tuned on
immunogenicity (**BigMHC-IM**). No pMHC 3D structure is used.

**T-SCAPE** (Kim et al., *Sci Adv* 2025; Seok lab / Galux) is a **multidomain** model that
uses **adversarial domain adaptation**. A shared "integrative" encoder is pushed to learn
features that are invariant across heterogeneous data domains — MHC presentation,
peptide-MHC binding affinity, TCR-pMHC interaction, source-organism identity, and T-cell
activation — while task-specific encoders capture domain-specific signal; a source
discriminator and orthogonality constraints keep the representations separated. Training is
two-phase (pretraining, then fine-tuning of a Task-IM decoder on a smaller activation set
of 12,782 points). Notably, despite the Seok lab's structural-modelling pedigree, T-SCAPE
is **sequence/data-driven, not a 3D structural model**; it even predicts anti-drug-antibody
potential of therapeutic antibodies without MHC inputs.

**MixMHCpred 2.2** (Gfeller et al., *Cell Syst* 2023) is **not a neural network**. It is a
motif model: position weight matrices learned by mixture-model deconvolution of mass-spec
immunopeptidomes, scored as a %Rank relative to random human-proteome peptides. It predicts
**presentation**, i.e. it *is* a binding/presentation tool rather than an immunogenicity
tool.

**DeepImmuno** (Li et al., *Brief Bioinform* 2021) is a residual **CNN**. Its distinctive
choice is input encoding: each residue of the peptide and the HLA pseudosequence is encoded
with a reduced-PCA representation of 566 AAindex physicochemical descriptors rather than
one-hot, to avoid sparsity. A sibling generative adversarial network (**DeepImmuno-GAN**)
synthesizes candidate immunogenic peptides. No 3D structure.

**PRIME 2.0** (Gfeller et al., *Cell Syst* 2023) is architecturally the thinnest: a shallow
neural network whose **first input node is the MixMHCpred 2.2 binding score** (−log %Rank),
combined with amino-acid frequencies at TCR-facing positions and peptide length. By
construction it layers a TCR-recognition propensity on top of a presentation prediction.

**Takeaway.** Complexity ranges across three orders of magnitude of parameters (a PWM, a
shallow NN, a CNN, a ~hundreds-of-millions-parameter attention+LSTM ensemble, a multidomain
adversarial net), yet **none** models the physical pMHC-TCR interface. The field's gains
have come from *more data and better negatives*, not from geometry.

---

## 4. Parameter 2 — Dataset construction and negative controls

The negative class determines what "immunogenicity" each model actually learns. The field
uses four kinds of negatives, in rough order of difficulty:

1. **Random proteomic non-binders** (easy) — used by presentation models and as EL decoys.
2. **Shuffled / randomly generated peptides** (easy) — padding negatives.
3. **Allelotype-matched non-binding proteome decoys** (intended-hard but often not) —
   the class SESTRAV found to be contaminating (Section 6).
4. **Assay-confirmed non-immunogenic binders** (genuinely hard) — scarce everywhere.

| Tool | Positive source | Negative construction | Decoy ratio / notes |
|---|---|---|---|
| BigMHC-EL | MS eluted ligands | **Random proteomic decoys** | 45,409 ligands vs 900,592 random (~1:20); AUROC 0.9733, AUPRC 0.8779 |
| BigMHC-IM | IEDB + assay validations | Assay-confirmed non-immunogenic | **Predominantly neoepitopes** (5,279 / 6,873 ≈ 77%) |
| T-SCAPE | Multi-source (activation head: assay) | Assay non-immunogenic + multi-domain pretraining negatives | 12,782 fine-tuning points *(preprint — medium confidence)* |
| MixMHCpred 2.2 | MS eluted ligands | **Random human-proteome peptides** | Presentation %Rank baseline |
| DeepImmuno | IEDB immunogenicity assays | Assay-confirmed non-immunogenic, **beta-binomial weighted** | Labels = immunogenic *potential*, not binary |
| PRIME 2.0 | Assay-confirmed immunogenic | **Assay non-immunogenic + random peptides**, realistic binding range | v2.0 explicitly less ligand-enriched than v1.0 |

**The structural problem shared by all five.** The biologically decisive discrimination is
*strong binder, immunogenic* vs *strong binder, non-immunogenic*. Two design choices keep
models from learning it:

- **Easy negatives inflate headline numbers.** When negatives are random non-binders, a
  model can score well on immunogenicity simply by re-learning binding — which is exactly
  what the "binding ceiling" (Section 7) reflects.
- **Hard negatives are rare and biased.** Assay-confirmed non-immunogenic binders are
  under-represented, and where they are abundant (e.g. HIV-1 in IEDB) they are skewed by
  clinical selection toward high-affinity B-restricted candidates confirmed non-responsive
  — a distribution SESTRAV shows is *unlearnable* from the other pathogens (Section 6).

**Fairness correction carried from the register (D28).** An earlier SESTRAV draft claimed
TRAP's holdout negatives mixed assay-confirmed peptides with unassayed thymic self-ligands;
that was **false and is retracted**. TRAP's pathogen holdouts (Lee et al. 2023) and Bravi
et al. (2023) both drew test negatives from **assay-confirmed records**, as SESTRAV does.
The differentiator is the *completed cycle*, not the negative source (Section 5).

---

## 5. Parameter 3 — Cross-pathogen validation

| Tool | Partition scheme | Pathogen holdout? | Leakage control |
|---|---|---|---|
| BigMHC | Random / instance-disjoint | **No** | Remove overlapping peptide-MHC instances |
| T-SCAPE | Homology-controlled benchmark | **No** | 9-mer homology (<2 mismatches) removed |
| MixMHCpred 2.2 | Per-allele presentation eval | **No** | N/A (presentation) |
| DeepImmuno | Random 10-fold CV + 2 curated sets | **No** | None beyond the split |
| PRIME 2.0 | 10-fold + **leave-one-allele-out** + **leave-one-study-out** | **No** | Allele / study grouping |
| **SESTRAV** | **9-virus leave-one-VIRUS-out cycle** | **Yes (completed)** | Organism-level holdout + assay-confirmed-only test negatives |

Three distinct leakage-control philosophies are visible, and **none of the five uses the
organism as the unit of exclusion**:

- **Instance-disjoint** (BigMHC): removing shared peptide-MHC pairs still lets two
  registrations of the same epitope, or two peptides from the same protein, land on both
  sides of the split.
- **Homology-disjoint** (T-SCAPE): a 9-mer / <2-mismatch filter is stricter, but is defined
  on *sequence*, not *source*. A novel pathogen can share local 9-mers with training data
  and still be an entirely unseen immunological context.
- **Allele / study grouping** (PRIME 2.0): the strongest *within-field* probe available,
  but leave-one-allele-out answers "does this transfer to an unseen HLA?", and
  leave-one-study-out answers "does this transfer to an unseen lab?" — **neither answers
  "does this transfer to an unseen pathogen?"**

Pathogen-level holdout has appeared only narrowly in the broader literature: Bravi et al.
(2023) ran leave-one-**organism**-out for a single-allele generative model as a
supplementary check (mean AUC ≈ 0.68 on assay-confirmed negatives); TRAP (Lee et al. 2023)
reported two single-pathogen splits (SARS-CoV-2-out, vaccinia-out). **Neither completed a
cycle over many pathogens.** SESTRAV's contribution is precisely the *conjunction*: a
completed leave-one-out cycle (as in Bravi) over a **nine-pathogen** panel (CMV, DENV, EBV,
HBV, HCV, HIV-1, HPV, IAV, SARS-CoV-2) with test negatives restricted to assay-confirmed
IEDB records.

---

## 6. Parameter 4 — Why DeepImmuno degrades on independent, deduplicated benchmarks

DeepImmuno's own report is strong (10-fold CV AUROC ≈ 0.85, AUPR ≈ 0.81 on IEDB; favourable
recall of TESLA neoantigens and SARS-CoV-2 peptides). On independent fields the picture
inverts — on SESTRAV's shared Tier A set it is the **weakest tool scored**: AUC-PR 0.698 and
AUC-ROC **0.542** (essentially chance), at only **86.5% coverage**. Three compounding
mechanisms explain the gap; the first is the most load-bearing and the most commonly
misstated.

**(a) Training/evaluation redundancy, not just exact duplicates.** IEDB immunogenicity
records are highly redundant: the same peptide recurs across alleles, studies, and
length registrations, and near-duplicates (the same epitope tested at shifted
registration boundaries) are pervasive. Random k-fold CV over such a corpus places
near-identical peptides on both sides of a fold boundary, so the headline CV metric
measures **memorization of a redundant corpus**, not generalization. A precise caveat is
warranted: one published neoantigen benchmark found only ~1% *exact* peptide overlap with
DeepImmuno's training data, and removing those exact duplicates changed little — i.e. the
effect is driven by **substring/near-duplicate homology and distributional overlap**, not
by a handful of exact repeats. (This is the same class of risk SESTRAV discloses for its
*own* Tier A corpus — 32.1% substring near-duplication, `docs/claims_register.md` D22 — and
it cuts both ways.) The practical consequence: when a benchmark is assembled to be
genuinely independent and homology-filtered, the inflation disappears and DeepImmuno falls
toward the binding floor.

**(b) A narrow input domain that silently drops peptides.** DeepImmuno accepts only **9- and
10-mer** peptides and a **fixed HLA pseudosequence panel**. On the shared Tier A field this
shows up directly: it scores only **623 of 720** peptides (86.5%). An immunogenicity tool
that cannot score 8- and 11-mers, or less common alleles, is not merely lower-coverage — it
is **evaluated on a self-selected, easier sub-field**, and its numbers are not comparable to
full-coverage tools without this caveat.

**(c) A label that encodes IEDB assay bookkeeping.** The beta-binomial "immunogenic
potential" weights each pMHC by its positive/negative assay counts. This is a principled way
to handle noisy labels *within* IEDB, but it ties the model tightly to IEDB's assay-count
distribution — a property of how often something was tested and by whom, which does not
transfer to a new pathogen with no assay history.

**Bottom line.** DeepImmuno's degradation is not a single "duplicate-peptide" bug; it is the
combination of a **redundancy-inflated CV estimate**, a **domain-restricted, coverage-biased
evaluation**, and an **IEDB-specific label**. Independent benchmarking of the whole field is
consistent with this: Buckley et al. (2022) found that on SARS-CoV-2, existing models
(DeepImmuno among the class of tools evaluated in that line of work) did not substantially
beat random or improve appreciably over HLA-ligand prediction.

---

## 7. Parameter 5 — The binding ceiling, quantified

"Binding ceiling" = the performance a model reaches by re-expressing MHC
binding/presentation, above which only genuine TCR-recognition / processing signal can lift
it. The cleanest quantification is a **single shared field with a raw presentation
baseline**.

**On SESTRAV's shared Tier A field** (MHCflurry 2.0 max-presentation score as the
binding-only baseline; `results/table3_tier_a_metrics.csv`):

| Tool | AUC-PR | Δ vs binding-only (0.800) | AUC-ROC | Δ vs binding-only (0.633) |
|---|---|---|---|---|
| SESTRAV RF (30-feat, OOF) | 0.828 | +0.028 point / **+0.038 paired** *(p=0.04, CI [+0.002,+0.071]; narrow & unconfirmed — D22)* | 0.726 | +0.093 |
| BigMHC | 0.822 | +0.022 point / **+0.018 paired** *(near-tie: p=0.37, CI includes 0)* | 0.659 | +0.026 |
| MixMHCpred 2.2 | 0.795 | **−0.005** | 0.626 | −0.008 |
| DeepImmuno | 0.698 | **−0.102** | 0.542 | −0.091 |

Reading this honestly: the **entire dedicated-immunogenicity field spans a ~0.13 AUC-PR band
around a binding baseline**, the best tool's margin over binding is **inside bootstrap
noise**, and two of the four named tools are **at or below** binding. MixMHCpred 2.2 sits
below binding-only because it *is* a presentation model being asked to do immunogenicity —
it has no above-binding component to contribute.

**Self-reported margins (different fields, not mutually comparable; use with care):**

- **BigMHC-IM:** reports significantly higher precision than seven prior models on
  neoepitope immunogenicity; the headline gain is a **top-k precision metric under extreme
  imbalance** (AUPPVn reported 2–25× NetMHCpan-4.1, identifying a few hundred immunogenic
  neoepitopes from a ~45M-decoy background). This is a real but **regime-specific** gain —
  precision at the extreme top of the ranking — and is exactly the metric most sensitive to
  corpus composition and the hard-negative gap.
- **T-SCAPE:** the preprint reports a class I immunogenicity **AUROC ≈ 0.568**, marginally
  above MHCnuggets 0.557 (P<0.001), with claimed gains over NetMHCpan-4.1, MHCflurry-2.0 and
  MixMHCpred-2.2 on neoantigen and infectious-disease discovery sets. An absolute AUROC of
  ~0.57 is itself a statement about the ceiling: even a 2025 multidomain adversarial model
  barely clears chance on class I immunogenicity. *(Preprint table flagged as needing
  verification — medium confidence on the exact figure, high confidence on the order of
  magnitude.)*
- **PRIME 2.0:** no reproducible metric on the shared field (compared on capabilities only
  by SESTRAV). The authors argue PRIME's improvement over MixMHCpred for immunogenicity is
  "**likely not due to better binding predictions**" — i.e. a genuine above-binding TCR
  component, but a modest one.

**Independent anchor.** Buckley et al. (2022) benchmarked nine published models on
assay-confirmed SARS-CoV-2 CD8⁺ epitopes and found **none substantially better than random,
nor appreciably better than HLA-ligand prediction**. This is the field-level statement of
the binding ceiling, from a disinterested source.

---

## 8. The blind spots SESTRAV's 9-virus LOO benchmark exposes

SESTRAV's own headline is *not* that it wins (it does not — it is a near-tie with BigMHC
inside a noisy field, and its best configuration posts a cross-virus mean **AUC-ROC of
0.463**, below chance on six of nine viruses). Its value is **methodological**: the
nine-virus leave-one-virus-out cycle on assay-confirmed negatives is an instrument that
makes three blind spots — invisible under the other tools' partitioning — measurable.

**Blind spot 1 — Within-distribution evaluation masks cross-pathogen collapse.** Every other
tool's partition (random, instance-disjoint, homology-disjoint, allele- or study-grouped)
keeps peptides from the *same pathogens* on both sides of the split. SESTRAV's LOO shows
that within-virus discrimination (mean AUC-ROC 0.658) **does not survive organism holdout**
(mean 0.463): the drop is uniform across all nine pathogens. A random/instance split cannot
see this because it never removes a pathogen. **Implication:** published within-split numbers
for BigMHC, T-SCAPE, DeepImmuno and PRIME 2.0 are upper bounds on, not estimates of,
emerging-pathogen performance.

**Blind spot 2 — Binding-feature dominance inverts on hard negatives.** Because binding
features carry the majority of discriminative weight in binding-dominated models (55.8% of
RF importance in SESTRAV's own case), a model confronted with a negative class made of
**assay-confirmed strong binders** ranks them high and inverts. SESTRAV's HIV-1 fold is the
extreme demonstration: within-CV AUC-ROC 0.663 **collapses to an anti-predictive 0.162**
under LOO, because HIV-1's confirmed negatives are high-affinity B-restricted binders that
the prior learned from eight other viruses actively mis-scores. Tools that never test on a
held-out pathogen with assay-confirmed strong-binder negatives **cannot surface this
failure mode at all**. It is the single most important blind spot for vaccine-design use,
where the question is always "is this binder also immunogenic?"

**Blind spot 3 — Allelotype-matched decoys are a contamination trap, not a hard negative.**
SESTRAV found that "allele_matched_nonbinder" proteome decoys — intended as hard negatives,
and the kind of synthetic decoy several pipelines pad test sets with — are in fact **high
presentation-scoring** (median MHCflurry score 0.74–0.76, comparable to or above true
positives on the measurable sample). Including them in test partitions inflated LOO AUC-ROC
by **0.25–0.50 points** for affected viruses (DENV 0.372→0.870, EBV 0.496→0.824,
IAV 0.488→0.784). Any benchmark that uses such decoys as its negative class is measuring
decoy-pattern recognition, not immunogenicity. This is a direct warning about the EL-style
random/decoy negatives that presentation-derived tools (BigMHC-EL, MixMHCpred) and
decoy-padded immunogenicity evaluations rely on.

**What the instrument does *not* claim.** Consistent with the register's fairness
discipline: SESTRAV's model is allele-blind (the ten-allele panel bounds only the alleles
its *binding features* are computed against, not corpus coverage); its "structural" features
are physicochemical proxies, not coordinates; and its own shared-field lead is unquantified
in direction because of a disclosed 32.1% substring-homology risk. The LOO benchmark exposes
the field's blind spots **without** resolving SESTRAV above the binding ceiling — that
ceiling binds SESTRAV too.

---

## 9. Synthesis

The five tools represent a decade of increasing architectural sophistication — PWM → shallow
NN → CNN → transfer-learned attention ensemble → multidomain adversarial net — applied to a
problem where sophistication has bought remarkably little. Three facts, triangulated from
the tools' own reports, SESTRAV's shared-field measurements, and disinterested benchmarking,
cohere into one conclusion:

1. **The negatives, not the architecture, are the bottleneck.** Easy negatives (random
   decoys) let a model succeed by re-learning binding; hard negatives (assay-confirmed
   non-immunogenic strong binders) are scarce and, where abundant, clinically biased. No
   amount of model capacity fixes a training signal that mostly re-encodes presentation.

2. **Every tool but SESTRAV measures the wrong generalization axis.** Instance-, homology-,
   allele-, and study-disjoint splits all answer narrower questions than the one that matters
   for emerging pathogens. The leave-one-virus-out cycle is the only protocol here that
   isolates cross-pathogen transfer — and when applied, transfer largely fails.

3. **The binding ceiling is empirically ~+0.02–0.03 over a raw presentation baseline on a
   shared field, and ≈0 for presentation-derived tools.** DeepImmuno falls below it;
   MixMHCpred 2.2 sits at it by construction; BigMHC and PRIME 2.0 clear it by a margin
   inside noise; T-SCAPE's absolute class I AUROC (~0.57) shows how low the ceiling sits.

The actionable reading for antigen prioritization: **treat current immunogenicity scores as
lightly-reranked presentation scores**, demand organism-level holdout before trusting any
cross-pathogen claim, and treat decoy-padded benchmark numbers as contaminated until the
negative class is shown to be assay-confirmed. The highest-leverage next step for the field
is not a new architecture but **assay-confirmed, per-pathogen negative panels** and
**routine held-out-pathogen evaluation** — which is the gap SESTRAV's benchmark was built to
make visible.

---

## 10. References

Internal (tracked artifacts):
- `docs/paper.md` — SESTRAV manuscript draft (LOO protocol, Tier A field, discussion).
- `README.md` — External Benchmark Results; capability matrix.
- `results/table3_tier_a_metrics.csv` — certified shared-field metrics.
- `results/external_benchmark_comparison.md` — 2026-05 pairwise report (historical).
- `docs/claims_register.md` — D15/D16/D17/D18/D19/D22/D25/D28/D29/D30/D42 (corrections and
  fairness/scope boundaries carried above).

External (primary literature):
1. Bravi B, et al. A transfer-learning approach to predict antigen immunogenicity and TCR
   specificity. *eLife*. 2023;12:e85126. doi:10.7554/eLife.85126
2. Lee CH, et al. A robust deep learning workflow to predict CD8+ T cell epitopes (TRAP).
   *Genome Med*. 2023;15:70. doi:10.1186/s13073-023-01225-z
4. Reynisson B, et al. NetMHCpan-4.1 and NetMHCIIpan-4.0. *Nucleic Acids Res*.
   2020;48(W1):W449-W454. doi:10.1093/nar/gkaa379
6. Calis JJA, et al. Properties of MHC class I presented peptides that enhance
   immunogenicity (IEDB immunogenicity predictor). *PLoS Comput Biol*. 2013;9(10):e1003266.
7. **BigMHC** — Albert BA, et al. Deep neural networks predict class I MHC epitope
   presentation and transfer learn neoepitope immunogenicity. *Nat Mach Intell*.
   2023;5:861-872. doi:10.1038/s42256-023-00694-6
8. **T-SCAPE** — Kim J, et al. T-SCAPE: T cell immunogenicity scoring via cross-domain aided
   predictive engine. *Sci Adv*. 2025;11(49):eadz8759. doi:10.1126/sciadv.adz8759
   (preprint: bioRxiv 2025.05.11.653308)
9. **MixMHCpred 2.2 / PRIME 2.0** — Gfeller D, et al. Improved predictions of antigen
   presentation and TCR recognition with MixMHCpred2.2 and PRIME2.0. *Cell Syst*.
   2023;14(1):72-83.e5. doi:10.1016/j.cels.2022.12.002
10. Buckley PR, et al. Evaluating performance of existing computational models in predicting
    CD8+ T cell pathogenic epitopes and cancer neoantigens. *Brief Bioinform*.
    2022;23(3):bbac141. doi:10.1093/bib/bbac141
11. O'Donnell TJ, et al. MHCflurry 2.0. *Cell Syst*. 2020;11(1):42-48.e7.
    doi:10.1016/j.cels.2020.06.010
23. **DeepImmuno** — Li G, et al. DeepImmuno: deep learning-empowered prediction and
    generation of immunogenic peptides for T-cell immunity. *Brief Bioinform*.
    2021;22(6):bbab160. doi:10.1093/bib/bbab160

> **Confidence markers.** High confidence: architectures and negative-set designs of BigMHC,
> MixMHCpred 2.2, DeepImmuno, PRIME 2.0; all SESTRAV shared-field numbers (tracked).
> Medium confidence: T-SCAPE's exact fine-tuning counts and the AUROC 0.568 figure (2025
> preprint table flagged as needing verification against the final *Sci Adv* version); the
> precise fraction of T-SCAPE activation-head negatives that are assay-confirmed. These are
> marked in place and should be confirmed against the published versions before external use.
