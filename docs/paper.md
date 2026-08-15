---
title: "SESTRAV: A Leave-One-Virus-Out Immunogenicity Benchmark Reveals Systematic Test Partition Contamination in Cross-Pathogen MHC Class I Prediction"
target_journal: Bioinformatics (Oxford Academic)
last_updated: 2026-08-10
status: manuscript draft - Acknowledgements/CRediT, Funding, and Zenodo DOI sections are open placeholders pending final author and funding confirmation. The v5 cross-validation figures are current against the 2026-08-10 peptide-grouped re-baseline, and the headline metrics are bound to tracked source artifacts. The Section 3.2 calibration figures were corrected on 2026-08-12 (D24): the pair previously reported there was computed before the 2026-08-10 peptide-grouped re-baseline and did not survive it, and it cited a gitignored staging file that did not in fact contain it. Calibration is now assessed by scripts/assess_calibration.py, bound to results/calibration_assessment_v5_mode31.csv, and reported as a limitation rather than a benefit. This is not a blanket all-clear: qualifications are disclosed in place rather than corrected away, and at least these three must be carried by the reader - the Tier A comparison (0.828) is a 2026-05 30-feature figure whose 720-peptide corpus has zero duplicate peptides, so peptide-grouping is a no-op on it and the D15 exact-duplicate leakage finding does not apply; it instead carries a disclosed, unquantified substring-homology risk (D16/D22, Section 3.5); the feature_mode=33 antigen-processing values are MOCK and not reproducible (D18, caveat after Table 1); and the pooled mixed-background figure's negative background is three-quarters out-of-panel vaccinia rows, not the self-proteome decoys an earlier version of this draft named (D19, Results 3.2 and 3.4). See docs/claims_register.md for the full register
---

## Abstract

SESTRAV (Structural Epitope Scoring via TCR Recognition and Vaccinology) is a
six-stage governed computational workflow for MHC class I CD8+ T-cell
immunogenicity prediction, providing reproducible, auditable infrastructure for
antigen prioritization across emerging pathogens. Binding affinity to MHC class I
is necessary but not sufficient for T-cell activation, yet cross-pathogen transfer
is rarely benchmarked. Where held-out-pathogen evaluation has been reported it was a
supplementary single-allele analysis [1] or a two-pathogen split whose negative set
mixes assay-confirmed peptides with presumed-tolerised self ligands [2]. We report a
nine-pathogen, pan-allele leave-one-virus-out (LOO) benchmark with test negatives
restricted to assay-confirmed IEDB negatives. SESTRAV trains a Random Forest on 35,597 quality-filtered
peptide-HLA pairs - 13,358 from nine human-pathogenic target viruses plus 22,239
real IEDB-confirmed negative-background pairs from additional non-target species,
dominated by Orthopoxvirus vaccinia (21,432 pairs, 60.2% of the active pool) -
combining 20 physicochemical descriptors at predicted TCR-contact positions
(p4-p8) with allele-stratified MHCflurry 2.0 presentation scores; an
edge-conditioned graph convolutional network with ESM-2 embeddings is a research
component. We identify
and correct a systematic LOO test partition flaw in which allelotype-matched
non-binding proteome decoys inflate AUC-ROC by 0.25-0.50 points for affected
viruses, and report results restricted to real IEDB-confirmed negatives. Under
this protocol, mean LOO AUC-ROC is 0.463 across nine pathogens, with three above
chance (CMV 0.633, HBV 0.556, HCV 0.528); HIV-1 is anti-predictive (0.162) and
human papillomavirus an active transfer failure (0.468), reflecting
binding-feature dominance over true immunogenicity. On a shared 720-peptide Tier
A test set, a 30-feature out-of-fold Random Forest configuration (AUC-PR 0.828,
unweighted, 200 trees, 2026-05) posted the highest point AUC-PR among
fully-scored external tools - BigMHC (0.822, a statistical near-tie), the
MHCflurry binding-only baseline (0.800), MixMHCpred 2.2 (0.795), and DeepImmuno
(0.698). This benchmark's 720-peptide corpus has zero duplicate peptides, so the
exact-peptide cross-validation leakage found and corrected elsewhere in this manuscript
does not apply here; a separate, unquantified substring-homology risk does apply and has
not been corrected for (32.1% of the scored pool has a near-duplicate peptide elsewhere
in the pool), so the +0.028 margin over binding-only should be read as unquantified in
either direction rather than as a comparison favoring either SESTRAV or the
external tools. SESTRAV is available at https://github.com/Gavin-Borges/SESTRAV
and installs from source.

---

## 1. Introduction

When the SARS-CoV-2 proteome was sequenced in January 2020, candidate peptide-MHC
class I complexes for T-cell vaccine design had to be computationally prioritized
before any virus-specific immunogenicity data existed. A typical coronavirus proteome
generates on the order of 50,000 distinct 8-11 amino acid peptide candidates eligible
for MHC class I presentation, a number that exceeds high-throughput experimental T-cell
screening capacity by two to three orders of magnitude [3]. Identifying
which of these peptides will elicit measurable CD8+ T-cell responses - peptide
immunogenicity, as distinct from peptide-MHC binding affinity - is a prerequisite for
rational antigen selection in vaccine development, pandemic preparedness surveillance,
and personalized neoantigen therapy. The clinical utility of any immunogenicity
predictor therefore depends critically on its performance not only for pathogens with
abundant prior T-cell response data, but also for pathogens encountered for the first
time.

The dominant approach to MHC class I epitope prediction is binding affinity scoring.
NetMHCpan 4.1 [4] and related pan-allele tools accurately predict
whether a peptide occupies the MHC class I groove, but binding is necessary and not
sufficient for T-cell activation: a bound peptide must also survive proteasomal
processing and TAP-mediated transport, adopt a conformation recognized by circulating
T-cell receptors, and trigger productive TCR signaling at physiologically relevant
kinetics [5]. Dedicated immunogenicity prediction tools attempt to
model this additional selection. The IEDB immunogenicity predictor [6]
introduced TCR-contact amino acid composition features and showed modest improvement
over binding-only baselines, but was evaluated on a single assay compilation without
systematic cross-pathogen testing. BigMHC [7] applied deep learning
transfer from large-scale MHC binding data to immunogenicity scoring and achieved
strong performance on a curated benchmark, though its immunogenicity training set is
predominantly neoepitopes (5,279 of 6,873 experimentally validated examples) and no
leave-one-pathogen-out evaluation was reported. T-SCAPE
[8] developed a statistical immunogenicity scoring framework
demonstrating improvement over binding tools, but similarly confined its validation to
within-pathogen experimental data. Across these tools, the evaluation paradigm is
consistent: models are trained and tested on data drawn from the same set of pathogens.

This within-pathogen evaluation design does not assess the scenario most relevant to
vaccine development practice - a pathogen for which no prior T-cell response data
exists. For emerging or rare viruses, a predictor must generalize from immunogenicity
patterns learned on other pathogens, a zero-shot transfer task that within-pathogen
cross-validation cannot measure. Stratifying test peptides by fold rather than by
pathogen conflates discriminating among peptides of an already-characterized virus with
discriminating entirely unseen peptides from a new one.

Cross-validation in this field is most often random, or stratified by HLA allele or by
study: PRIME2.0 reports leave-one-allele-out and leave-one-study-out cross-validation
[9]. Pathogen-level holdout has been reported, but narrowly. Bravi et al. performed
leave-one-organism-out cross-validation for a single-allele generative model as a
supplementary robustness check, reporting a mean AUC of 0.68 on assay-confirmed
negatives [1]. TRAP trained separate models excluding SARS-CoV-2 and vaccinia virus
respectively, but its negative set combines assay-confirmed pathogen negatives with
thymically expressed self ligands that were never assay-tested [2]. Neither reports a
multi-pathogen, pan-allele panel with test negatives restricted to assay-confirmed
records, which is the protocol we adopt here.

That transfer across pathogens is difficult is itself established rather than novel:
Buckley et al. benchmarked nine published models on a compiled panel of assay-confirmed
SARS-CoV-2 CD8+ T-cell epitopes and found that none performed substantially better than
random, or improved appreciably on HLA ligand prediction [10]. The present work
generalises that result from one pathogen to a nine-pathogen panel, and its distinct
contribution is methodological - the quantification of a test-partition contamination
effect described below.

Here we present SESTRAV (Structural Epitope Scoring via TCR Recognition and
Vaccinology), a computational workflow for MHC class I CD8+ T-cell immunogenicity
prediction. SESTRAV was trained on labelled data for nine human-pathogenic target
viruses, augmented with a real IEDB-confirmed negative background from additional
non-target species quantified in Section 2.1, drawn from the Immune Epitope
Database (IEDB; [3]), VDJdb, and the Los Alamos National Laboratory HIV molecular
immunology database. The production model
integrates physicochemical descriptors at TCR-contacting residue positions with
allele-stratified MHC presentation scores from MHCflurry 2.0 [11];
a custom edge-conditioned graph convolutional network incorporating ESM-2 per-residue
protein language model embeddings [12] is evaluated as a research
component. We evaluate SESTRAV under within-virus stratified cross-validation and, as
the primary generalization benchmark, a virus-level LOO protocol in which a separate
Random Forest was retrained for each of the nine pathogens on data from the remaining
eight viruses, together with the out-of-panel negative background and human
self-proteome hard decoys, then evaluated on the held-out pathogen using exclusively
real IEDB assay-confirmed negatives. We report two
findings: first, we identify and correct a systematic test partition flaw in which
allelotype-matched non-binding viral proteome decoys in LOO test sets artificially
inflate AUC-ROC by 0.25-0.50 points for the affected pathogens; second, after
correction, mean LOO AUC-ROC is 0.463 across nine pathogens, with three of nine above
the chance baseline of 0.5 (CMV 0.633, HBV 0.556, HCV 0.528). HIV-1 is anti-predictive
(0.162), reflecting binding-feature dominance over true immunogenicity: real IEDB-tested
HIV-1 immunogenicity negatives are predominantly strong MHC binders scored highly by
the classifier. Human papillomavirus is an active transfer failure (0.468). These results
characterize the data requirements for reliable cross-virus immunogenicity generalization
and quantify the test partition conditions under which transfer assessments are valid.

Section 2 describes dataset construction, the LOO protocol, and model architecture.
Sections 3 and 4 report within-virus and cross-virus performance, an analysis of
cross-virus generalization, a comparison against published tools on a curated labelled
benchmark, and limitations.

---

## 2. Methods

The SESTRAV workflow proceeds from raw data acquisition through dataset assembly,
feature construction, model training, and evaluation under three paradigms
(within-virus cross-validation, leave-one-virus-out generalization, and an
external-tool benchmark); an overview of the full pipeline is shown in Figure 1.

*Figure 1. SESTRAV pipeline: dataset assembly to model evaluation. Three primary
data sources (IEDB T-cell assay records, VDJdb TCR-pMHC pairs, and the LANL HIV
database) feed the v5 dataset assembly stage, which applies a quarantine train/test
split, HLA allele normalization, and injection of hard and viral decoy negatives,
yielding 35,597 active records (51,185 total). Each peptide is represented by three
complementary feature groups: physicochemical descriptors plus peptide length, 10
per-allele MHCflurry binding scores, and ESM-2 (t12) protein-language-model
embeddings. These features train two model families: the production Random Forest
(RF mode-31) and a graph neural network (GNN). Trained models are assessed under
three evaluation paradigms: within-virus cross-validation, leave-one-virus-out (LOO)
cross-virus generalization, and an external-tool benchmark. Arrows indicate data
flow (directed acyclic graph).*

### 2.1 Dataset Construction

Positive immunogenic peptide-MHC class I (pMHC-I) pairs were curated from three
established repositories. The primary source was the Immune Epitope Database (IEDB; [3]), queried for MHC class I-restricted CD8+ T-cell
assay records with a positive qualitative measurement outcome (any "Positive" grade).
Additional positives were retrieved from VDJdb [13], a database linking
immunodominant viral epitopes to paired TCR alpha and beta CDR3 sequences. HIV-1
epitopes were further supplemented from the LANL HIV Molecular Immunology Database
[14], a curated repository of best-defined cytotoxic T-lymphocyte
responses.

The dataset spans nine target viruses: cytomegalovirus (CMV), dengue virus (DENV),
Epstein-Barr virus (EBV), hepatitis B virus (HBV), hepatitis C virus (HCV), HIV-1,
human papillomavirus (HPV; serotypes HPV16 and HPV18 normalised to a single HPV label
with strain annotation), influenza A virus (IAV), and SARS-CoV-2. Coverage was
restricted to ten HLA class I alleles representing common population-level supertypes:
HLA-A*01:01, HLA-A*02:01, HLA-A*03:01, HLA-A*11:01, HLA-A*24:02, HLA-B*07:02,
HLA-B*08:01, HLA-B*27:05, HLA-B*35:01, and HLA-B*44:02. These nine viruses are the
target pathogens curated for positive evidence and evaluated in the within-virus
and leave-one-virus-out benchmarks (Sections 3.2-3.4); the active training pool
additionally includes real IEDB-confirmed negative-background records from
non-target species, quantified below.

Confirmed negatives were sourced from IEDB T-cell assay records with a negative
qualitative measurement outcome. To address class imbalance for viruses with a
positive-to-negative ratio exceeding 2:1, viral proteome hard decoys were generated
by sliding-window enumeration of 9-mers over published reference proteomes (SARS-CoV-2:
Wuhan-Hu-1 reference genome, NC_045512.2, four canonical proteins; CMV: strain AD169;
and reference sequences for IAV, EBV, HIV-1, and DENV). Only 9-mers with no exact
match in any IEDB positive record were retained as negatives, confirming their
non-immunogenic status by absence of prior evidence. Human self-proteome hard decoys
provided additional negative examples orthogonal to viral sequence space.

All peptide sequences were normalised to uppercase, restricted to the 20 canonical
amino acids, and filtered to the MHC class I canonical length range of 8-11 residues;
sequences outside this range or containing non-standard residues were discarded. HLA
allele designations were resolved to standard four-digit format (HLA-X*XX:XX):
low-resolution aliases (e.g. HLA-A2, HLA-B27) were mapped deterministically to the
dominant subtype (HLA-A*02:01, HLA-B*27:05, respectively) using a curated alias
table derived from published HLA frequency databases and IEDB allele annotations.
Entries bearing irresolvably ambiguous allele designations, including "HLA class I"
and supertype labels such as "HLA-B57" for which subtype identity cannot be
assumed without additional evidence, were flagged is_quarantined = True and
withheld from model training. Source-specific virus taxonomy strings (e.g.
"Influenza A virus" in IEDB bulk exports, "InfluenzaA" in VDJdb) were mapped to
canonical short labels before record deduplication on the composite key
(peptide, hla_allele); when the same entry appeared in multiple sources,
the highest-priority record was retained in the order: curated positives, published
validation panels, self-proteome decoys, IEDB negatives.

A per-virus depth threshold excluded pathogens with fewer than 50 labelled rows or
fewer than 10 confirmed-negative assay records from model training, preventing poorly
supported species from biasing learned representations. The resulting v5 dataset
comprised 51,185 total rows, of which 35,597 remained active after quarantine
filtering, with a canonical allele rate of 96.8%.

Of these 35,597 active rows, 13,358 (37.5%) are peptide-HLA pairs from the nine
target viruses listed above; the remaining 22,239 (62.5%) are real IEDB-confirmed
records from additional, non-target species retained as negative-class background
rather than as evaluated pathogens. This background is dominated by Orthopoxvirus
vaccinia (21,432 rows, 60.2% of the active pool; entirely tested-negative assay
records, not synthetic decoys), with smaller contributions from six further species
(human betaherpesvirus 6B, yellow fever virus, respiratory syncytial virus, Lassa
virus, coxsackievirus B, and Zika virus; 807 rows combined, 2.3% of the active pool,
overwhelmingly negative-labelled with 18 positive-labelled exceptions across three
of the six). None of these non-target species is included in the leave-one-virus-out
evaluation (Section 2.5) or the per-virus tables in Section 3; they contribute only
to the pooled within-CV background reported in Section 3.2.

Population coverage of the ten-allele panel was estimated from allele frequencies
in the Allele Frequency Net Database (AFND; [15]) across the five WHO super-populations
(African, AFR; admixed American, AMR; East Asian, EAS; European, EUR; South Asian, SAS).
Under a Hardy-Weinberg diploid assumption, the phenotypic frequency of an allele with
haplotype frequency h is 1 - (1 - h)^2, and panel coverage - the fraction of individuals
carrying at least one panel allele - is 1 - product_i (1 - phenotype_freq_i) over the ten
alleles. Panel coverage is highest in EUR (0.919) and lowest in AFR (0.621), with a global
mean of 0.789 (Table S1), reflecting the panel's bias toward alleles common in European
reference populations and motivating future expansion to alleles prevalent in African and
East Asian populations.

*Table S1. Estimated population coverage of the ten-allele panel across five WHO
super-populations. Coverage is the fraction of individuals carrying at least one of the
ten panel alleles, computed from AFND haplotype frequencies [15] under a Hardy-Weinberg
diploid assumption.*

| WHO super-population | Panel coverage |
|---|---|
| European (EUR)       | 0.919 |
| South Asian (SAS)    | 0.847 |
| Admixed American (AMR) | 0.813 |
| East Asian (EAS)     | 0.742 |
| African (AFR)        | 0.621 |
| Global mean          | 0.789 |


### 2.2 Feature Engineering and Model Architecture

Each peptide was represented by a 31-dimensional feature vector (mode-31 in the
SESTRAV feature taxonomy). Twenty physicochemical descriptors encoded four amino
acid properties at five predicted TCR contact positions (p4-p8) following the
residue-numbering convention of Chowell et al. [16]: Kyte-Doolittle
hydrophobicity [17] (scale range -4.5 to +4.5), binary
aromaticity (1 for F, W, Y, or H; 0 for all other residues), van der Waals atomic
volume [18], and formal charge at physiological pH (K, R = +1; D, E = -1;
all others = 0). Contact positions p4-p6 corresponded to residue indices 3, 4, and 5
from the N-terminus; p7 and p8 were C-terminal-relative at indices L-3 and L-2, where
L is peptide length. Positions that coincided with an earlier contact position or fell
at the C-terminal anchor residue were zero-imputed to avoid double-counting. A total
of 4 properties x 5 positions yielded the 20 physicochemical features.

Ten MHC binding affinity features represented the MHCflurry 2.0 [11] antigen presentation probability score (a probability in [0, 1]) for each of the
ten target HLA alleles. Scores were pre-computed offline for all unique peptides and
stored in a peptide-allele binding matrix; peptides absent from the matrix received a
score of zero. A single peptide length feature (integer amino acid count, range 8-11)
completed the 31-feature vector.

Pre-computed ESM-2 per-residue embeddings (facebook/esm2_t12_35M_UR50D; [19]; 480 dimensions) were cached offline as a peptide-to-tensor dictionary (30,687
unique peptides; each entry a zero-padded tensor of shape 11 x 480). These embeddings
serve as node features for the GNN variant of SESTRAV and were not used by the Random
Forest classifier.

The primary immunogenicity classifier was a Random Forest implemented in scikit-learn
[20], configured with 200 decision trees (n_estimators = 200),
balanced class weights (class_weight = "balanced") to compensate for the
positive-to-negative class ratio, and a fixed random seed (random_state = 42) for
reproducibility; max_features defaulted to the square root of the feature dimension
(sqrt(31)). Model performance was estimated by stratified five-fold cross-validation
in which folds were balanced across virus identity, HLA allele distribution, and
negative origin category to ensure representative label proportions in each split.
A curated set of well-characterised canonical IEDB epitopes (gold-standard set) was
excluded from the training pool in all cross-validation runs, preventing known
high-confidence epitopes from inflating performance estimates. After cross-validation,
the production model was retrained on the complete active training pool. LOO
evaluation is described in Section 2.5.


### 2.3 Model Architectures

The Random Forest canonical track is described in full in Section 2.2. Single-threaded
execution (n_jobs = 1) was set to ensure platform-independent reproducibility across
CPU configurations; the production model was retrained on all 35,597 active training
rows after cross-validation completed.

A supplementary gradient-boosted tree classifier (XGBoost; [21]) was trained on
identical feature matrices and cross-validation partitions. The configuration comprised
200 boosting rounds (n_estimators = 200), binary logistic loss
(objective = binary:logistic), the inverse negative-to-positive row count as the
positive class scale weight (scale_pos_weight = n_neg / max(n_pos, 1), recomputed
from the training pool), AUC-PR as the monitored evaluation metric
(eval_metric = aucpr), single-threaded execution (nthread = 1), and a fixed random
seed (random_state = 42). XGBoost results serve as a supplementary comparison track;
both classifiers shared the same feature mode, cross-validation partitions, and
gold-standard exclusion set.

The GNN structural track implemented a two-layer graph convolutional network with
edge-conditioned message passing as a custom PyTorch [22] nn.Module hierarchy.
Peptides were represented as directed chain graphs in which residue positions were
nodes and edges encoded backbone connectivity through three one-hot classes: self-loop
(position i to itself), forward (i to i+1), and backward (i+1 to i). Node features
were the pre-computed ESM-2 per-residue embeddings (facebook/esm2_t12_35M_UR50D; [19]; 480 dimensions per residue) drawn from the offline cache described
in Section 2.2; only real residue positions were materialised as graph nodes, with
zero-padded cache entries excluded. The graph encoder applied two convolutional layers,
each with batch normalisation and ReLU activation, projecting node representations from
480 dimensions through a 256-dimensional hidden layer to a 128-dimensional output;
global mean pooling over the node dimension yielded the graph-level embedding. All
linear layer parameters were initialised with Xavier uniform initialisation and
zero-initialised biases. A parallel physicochemical feature block projected the
31-dimensional mode-31 input through a 64-dimensional linear layer with batch
normalisation, ReLU activation, and dropout (rate 0.3). The 128- and 64-dimensional
representations were concatenated and decoded by a two-layer fusion MLP (192 to 128
to 1 units) with intermediate ReLU activation and dropout. Internal verification
gates 2 and 3, confirming forward-pass correctness and end-to-end gradient flow to
all trainable parameters, passed for this architecture. GNN training on the v5 dataset
will be reported in a subsequent release; all results in this manuscript reflect the
RF mode-31 canonical classifier.


### 2.4 Evaluation Methodology

Model performance was estimated using five-fold cross-validation, **peptide-grouped**
as of 2026-08-10 (`src.ml_utils.PeptideGroupedKFold`, stratified on the same
label/negative-origin/HLA-supertype/length composite as the splitter it replaced, but
holding every row for a given peptide in exactly one fold). All performance metrics
reported for the v5 within-virus, per-virus, and binding-confound results (Tables 1-3b)
are out-of-fold (OOF) under this splitter: predictions for each sample were generated
by a model trained on the remaining folds, none of which contain that sample's peptide.

**This methodology was not always peptide-grouped, and the discovery and correction of
that gap is itself a reported finding of this work.** The splitter originally used
stratified by class and length but did not group by peptide, and the dataset is
deduplicated on `(peptide, hla_allele)` rather than on peptide alone; because every
mode-31/mode-30 feature is a pure function of the peptide string, 71.1% of held-out
rows under that splitter shared their exact peptide with the training fold and were
feature-identical to a training row. Held at the production configuration, this
inflated pooled AUC-PR by +37.0%, the per-virus mean by +14.0%, and the pooled
same-pathogen figure by +19.0% (`docs/claims_register.md` D15, disclosed 2026-08-09).
The splitter was repaired and every certified v5 CV number in this manuscript
re-measured under it (D15, remediated 2026-08-10); the figures reported throughout
Section 3 are the corrected values, with the retracted prior figures noted alongside
them for transparency. The production model retrained on the complete active pool
after cross-validation completed is used for inference and deployment only; it is not
the source of any reported evaluation figure. This OOF protocol is in contrast
to the evaluation conditions under which external tools were assessed, as detailed
below.

Comparisons with external prediction tools - BigMHC [7], MixMHCpred 2.2 [9],
DeepImmuno [23], and an MHCflurry 2.0 [11] binding-only baseline - were conducted on a
shared 720-peptide Tier A benchmark set frozen on 2026-05-20, using a **2026-05, 30-feature,
unweighted** SESTRAV configuration that predates both `feature_mode=31` and the
peptide-grouped splitter described above (`docs/claims_register.md` D16). Each external
tool was fully scored on the shared set from its public release checkpoint. SESTRAV's
arm reflects out-of-fold predictions from that earlier, ungrouped splitter, so the
comparison is not conservative for SESTRAV as previously claimed here: the leakage runs
toward SESTRAV, making its arm optimistic rather than handicapped relative to the
fully-scored external tools. This Tier A arm is **deliberately not re-measured** under
the peptide-grouped splitter: only 414 of its 704 peptides resolve to an active row of
the current v5 corpus, so a re-run would score a smaller, non-comparable field rather
than refresh this one (D16). The head-to-head is reported in Section 3.5.

Leave-one-virus-out evaluation is described in full in Section 2.5.

The primary performance metric was AUC-ROC, computed on balanced positive-and-negative
subsets to decouple ranking ability from class composition effects. AUC-PR was the
co-primary metric, reported alongside the dataset-specific positive rate in every
comparison, because AUC-ROC becomes misleading when the positive fraction substantially
exceeds 0.1 [24]. Precision at 10% recall was the secondary endpoint,
corresponding to a practical vaccine peptide shortlisting threshold at which
specificity is highly constrained. All per-virus AUC-ROC and AUC-PR point estimates
carry 95% confidence intervals derived from 1,000-replicate bootstrap resampling of
the held-out partition (`results/per_virus_eval_v5_mode31.csv`); Tables 2 and 2b report
the point estimates only, and the reader should consult that file for the interval on
any individual cell. Pairwise comparisons between predictors on the same held-out
partition used paired bootstrap resampling (N=10,000; `src/statistical_bootstrap.py`)
rather than DeLong's paired test [25]: DeLong's test assumes normal AUC distributions
and sample independence, both of which are violated for the nested and
peptide-correlated model comparisons made here, and the paired bootstrap avoids that
assumption while still preserving the joint covariance between the two predictors'
scores. Metric differences across independently constructed cross-validation
partitions were not subjected to significance testing because the partitions are not
exchangeable. Sixteen named canonical gold-standard
epitopes were excluded from the training pool in all cross-validation folds, as
specified in Section 2.1. This exclusion is narrow: it prevents contamination of
cross-validation estimates by those sixteen well-characterised epitopes, and it does
not make the external benchmark field unseen. Of the 704 peptides in the Tier A
comparison, 414 are present in the training corpus.

For the Tier A comparison specifically (Section 3.5, Table 4), which retains the
pre-remediation, ungrouped splitter for the reasons given above: its 720-row training
corpus has zero duplicate peptides (`docs/claims_register.md` D16), so the exact-peptide
leakage mechanism described in this section is a structural no-op there and grouping by
peptide would not change the result. A different mechanism, substring homology between
distinct peptides, has not been checked or corrected for on this corpus and affects 32.1%
of the scored pool; whether it affected the reported score is not established, since the
historical fold assignment cannot be recovered. Both the exact-duplicate finding above
and the Tier A scope determination are documented in the project's claims register
(entries D15 and D22 respectively).


### 2.5 Leave-One-Virus-Out Evaluation

Standard k-fold cross-validation allocates training and test samples from the same
pathogen, thereby conflating per-virus discrimination with cross-virus transfer. To
assess whether immunogenicity signals learned from one set of viral pathogens transfer
to an entirely unseen pathogen - the realistic deployment scenario for emerging-virus
surveillance - a virus-level leave-one-out (LOO) evaluation was conducted. This design
is conceptually related to leave-one-out cross-validation at the sample level, but
here the unit of exclusion is the complete labelled set for one virus, not a single
peptide. Pathogen-level holdout has precedent in this field but not at panel scale on
assay-confirmed negatives alone; Section 1 states the position and cites it [1,2].

**Eligibility criteria.** A pathogen was eligible for inclusion in the LOO evaluation
if its representation in the v5 active dataset met a minimum class-balance requirement:
at least 20 confirmed-positive and 10 confirmed IEDB-tested negative records, both
counted after applying the quarantine filter (rows with is_quarantined = True excluded).
This threshold was chosen to ensure that per-virus AUC-ROC estimates were computed on
a sufficient number of examples in each class to be statistically interpretable. Nine
pathogens satisfied these criteria: cytomegalovirus (CMV), dengue virus (DENV),
Epstein-Barr virus (EBV), hepatitis B virus (HBV), hepatitis C virus (HCV), HIV-1,
human papillomavirus (HPV), influenza A virus (IAV), and SARS-CoV-2.

**Fold construction.** Nine LOO folds were defined, one per eligible pathogen. In each
fold, the training set comprised all active rows from the eight non-held-out pathogens
together with the out-of-panel negative-background rows described in Section 2.1, and
was augmented with the 5,000 human self-peptide decoys (rows with source_type = "Self")
drawn from the v5 dataset. Those 5,000 decoys are a single set, included in training in
all nine folds and never assigned to any test set. The out-of-panel bloc is the larger
part of the viral training pool and is not incidental: in the CMV fold, for example,
22,239 of the 34,243 active viral training rows (65%) come from species outside the
nine-pathogen panel, chiefly Orthopoxvirus vaccinia (the fold total is n_train_viral in
results/loo_cross_virus_v5_clean.csv; the out-of-panel count is the off_panel scope of
results/calibration_assessment_v5_mode31.csv and is derivable from
data/immunogenicity_dataset_v5.csv). Descriptions of LOO training as fitting on
"the remaining eight viruses" therefore refer to the panel composition, not to the
whole training pool.

The test set for each fold comprised the active rows from the held-out pathogen, with
one deliberate exclusion described below: allele-matched non-binder decoys derived from
the held-out virus were withheld from the test partition. That exclusion is substantial
in six of the nine folds, ranging from 300 rows (CMV, EBV) to 1,000 (DENV), and is zero
for HBV, HCV and HPV (n_allele_nb_excluded in results/loo_cross_virus_v5_clean.csv).
No random subsampling was applied.

**Contamination control.** To prevent data leakage from canonical benchmark epitopes
into the training procedure, any peptide appearing in the GOLD_STANDARD_EPITOPES set
was excluded from the training pool prior to model fitting. This exclusion was applied
only to the training set; gold-standard peptides present in the held-out test virus
were retained in the test set to ensure that recovery of established epitopes could be
assessed in Results. LOO test partition negatives were restricted to IEDB
assay-confirmed negative records from the held-out virus; viral proteome hard decoys
derived from held-out-virus proteomes were withheld from the LOO test partition to
prevent soft leakage arising from decoy-pattern recognition learned during multi-virus
training.

**Model and hyperparameters.** Each LOO fold was fit using the RF mode-31 Random
Forest classifier. The 31-feature canonical set comprised 20 physicochemical
descriptors (four properties - Kyte-Doolittle hydrophobicity, aromaticity, Zamyatnin
van der Waals volume, and formal charge at pH 7 - each computed at five TCR contact
positions p4-p8 in the peptide-MHC complex), 10 per-allele MHCflurry presentation
scores for the panel alleles (HLA-A*01:01, A*02:01, A*03:01, A*11:01, A*24:02,
B*07:02, B*08:01, B*27:05, B*35:01, B*44:02), and peptide length as an integer
feature. Mode-31 was selected over extended feature modes (33, 35) so that LOO evaluation
would not depend on antigen-processing caches that may not be available for novel
pathogens. That choice is now additionally supported by a fact established after it was
made: the antigen-processing values in the shipped cache are mock, not real NetChop 3.1 /
TAPreg output (`docs/claims_register.md` D18; see the caveat following Table 1). Mode-31
is retained on both grounds. Hyperparameters were held
constant across all nine folds and matched those used for within-virus stratified
cross-validation: 200 trees, balanced class weighting (class_weight = "balanced"),
and a fixed random seed of 42. No fold-specific hyperparameter tuning was performed.

**Evaluation.** The primary metric was the area under the receiver operating
characteristic curve (AUC-ROC), computed on each held-out test set using the model's
predicted positive-class probability. The area under the precision-recall curve
(AUC-PR) was recorded as a secondary metric; owing to pronounced class imbalance in
several viral test sets, AUC-PR was treated as supplementary. LOO AUC-ROC values were
compared against the corresponding within-virus AUC-ROC, defined as the mean AUC-ROC
from a five-fold stratified cross-validation trained and evaluated exclusively on the
rows of the same pathogen. This within-virus baseline contextualises the cross-pathogen
transfer penalty. All code implementing the LOO framework and the withheld test sets for
each fold are available at https://github.com/Gavin-Borges/SESTRAV.

---

## 3. Results

### 3.1 Dataset Construction and Quality

The SESTRAV v5 immunogenicity dataset contains 51,185 total rows assembled from three
primary sources: the Immune Epitope Database (IEDB), VDJdb, and the Los Alamos HIV
Molecular Immunology Database. Following quality filtering, 35,597 rows are designated
active for model training; quarantine filtering removes rows for several reasons, most of
them population-scope decisions - self-proteome and unidentified-source rows are excluded
by design (8,811 and 4,577 rows respectively), and off-target or insufficiently-depth
viruses are quarantined below fixed per-virus thresholds (Section 2.1). The 8,811
self-proteome rows counted here are those carrying virus = "Self", which is a wider set
than the 5,000 UniProt-derived hard decoys added to LOO training in Section 2.5
(source_type = "Self"): the remaining 3,811 are human self-peptides drawn from IEDB
assay records. All 8,811 are quarantined, so neither subset enters within-virus
cross-validation. Within the nine
target viruses' own labelled data, allele-ambiguity is the only quarantine mechanism that
applies: a row is quarantined if its HLA field cannot be resolved to a single canonical
four-digit subtype. For example, 27 of EBV's 31 quarantined rows report only the fully
unresolved class annotation "HLA class I", and 4 report the ambiguous supertype "HLA-B57",
which spans B*57:01 and B*57:03, subtypes with distinct clinical profiles that cannot be
assumed from the supertype alone (`data/immunogenicity_dataset_v5.csv`, `is_quarantined`;
`scripts/_dataset_utils.py`).
Of all labelled peptide-HLA pairs in the active set, 96.8% carry alleles resolving to
canonical four-digit HLA nomenclature. Full construction methodology is described in
Section 2.1.


### 3.2 Within-Virus Cross-Validation Performance

The canonical mode-31 Random Forest classifier (20 physicochemical descriptors at
TCR-contact positions p4-p8, 10 allele-stratified MHCflurry 2.0 presentation scores,
and peptide length) was cross-validated on 35,555 rows under peptide-grouped 5-fold
cross-validation (`src.ml_utils.PeptideGroupedKFold`, re-baselined 2026-08-10, closing
`docs/claims_register.md` D15 - see below). That pool is the 35,597 active v5 rows less the
42 gold-standard holdout rows withheld from training by the contamination control of Section
2.2 (EBV 37, HPV 5); the production model was subsequently refitted on all 35,597. The pooled mixed-background out-of-fold
performance was AUC-ROC 0.814 and AUC-PR 0.606 (models/v5/training_results_mode31.csv),
reflecting discrimination between immunogenic viral peptides and a mixed negative
background when all nine viruses contribute jointly to training. **That background is
dominated by out-of-panel negatives, and contains no self-proteome decoys.** Of its
27,534 scored negatives (`models/v5/rf_oof_predictions_mode31.csv`), 21,432 (77.8%) are
*Orthopoxvirus vaccinia* rows - a pathogen outside the nine-virus target panel, though
these are genuine IEDB assay-confirmed negatives (`negative_origin = tested_negative`),
not synthetic decoys - and a further 3,112 (11.3%) are allelotype-matched non-binding
viral-proteome decoys (`negative_origin = allele_matched_nonbinder`). Only 1,851 are
same-pathogen IEDB REST-API negatives; across the nine target viruses the real-negative
pool contributing here is 2,201 rows. The 5,000 human self-proteome decoys in the v5
corpus carry `is_quarantined = True` and are removed by `_filter_quarantined`
(`src/train_classifier.py`) before training and scoring, so none of them reaches this
figure. Because this pooled figure is computed against a background three-quarters
composed of a single out-of-panel pathogen, it is not directly comparable to the
per-virus within-CV or real-neg-only discrimination values reported below, and it is
referred to throughout as the pooled mixed-background AUC-ROC to keep the two quantities
distinct. A feature ablation within the same Random Forest family (Table 1) shows that
the ten allele-stratified binding scores add a substantial margin over a binding-free
physicochemical floor (mode-21: AUC-ROC 0.712, AUC-PR 0.505), while adding
antigen-processing (mode-33) and self-similarity (mode-35) features leaves both metrics
essentially unchanged (within one standard deviation of the mode-31 mean). We therefore
retain mode-31 as the production configuration.

**Peptide-level cross-validation leakage (D15) and its remediation.** Every mode-31
feature is a pure function of the peptide string (no allele-specific column enters the
31-feature vector), while the v5 corpus is deduplicated on (peptide, hla_allele), not on
peptide alone; consequently the same peptide can appear on both sides of a fold boundary
under a splitter that does not group by peptide. A leakage audit
(`scripts/audit_cv_leakage.py`, `results/cv_leakage_audit.csv`) measured that 71.1% of
held-out rows under the (pre-remediation) production splitter had their exact peptide
present in that fold's training set, and that holding the RF configuration fixed while
switching only the splitter moved pooled mode-31 AUC-PR from 0.835 to 0.609 (-0.225). This
was disclosed 2026-08-09 and remediated 2026-08-10: `src/ml_utils.py` gained a
peptide-grouped splitter class, and every certified v5 CV number in this manuscript,
including Tables 1-3b below, is now measured under it. The figures given throughout this
section are therefore the corrected, generalization-honest estimates, not the
leakage-inflated figures an earlier version of this manuscript reported.

Table 1. v5 Random Forest feature ablation (pooled peptide-grouped 5-fold cross-validation,
single Random Forest model family; models/v5/training_results_ablation.csv, re-baselined
2026-08-10). AUC-ROC and AUC-PR are cross-validation means.

| Feature mode | Features                                          | AUC-ROC | AUC-PR |
|--------------|---------------------------------------------------|---------|--------|
| 21           | physicochemical + length (binding-free floor)     | 0.712   | 0.505  |
| 31           | + 10 allele binding scores (production)           | 0.814   | 0.606  |
| 33           | + antigen processing (MOCK features - see below)  | 0.817   | 0.609  |
| 35           | + self-similarity                                 | 0.814   | 0.607  |

On the v5 build, 12.6% of active (non-quarantined) training rows (4,502 of 35,597) have no
entry in the antigen-processing cache and receive an imputed `netchop_score` and
`tap_score`; for modes 33 and 35 that imputation is now fit inside each CV fold on training
rows only (`--no-fold-impute` reproduces the whole-cache-median pre-Phase-0 behavior). The
self-similarity features are mode-35 only and are not imputed at all: the same share of
active rows is absent from the self-similarity cache and is filled with a constant 0.0,
meaning no known self-match, which is fold-independent and so carries no leakage. The
near-zero contribution of modes 33 and 35 over mode-31 should be read with this caveat in
mind.

**The mode-33 antigen-processing features are mock, not real tool output, and the mode-33
row of Table 1 must be read accordingly.** The `netchop_score` and `tap_score` values in
`data/antigen_processing_cache.csv` are generated locally by
`src/external_predictors.py` - the DTU NetChop 3.1 web interface changed its response
format and the TAPreg endpoint requires institutional VPN access, so
`scripts/precompute_antigen_processing.py` calls both predictors with `mock_fallback=True`,
which short-circuits before any network call. **They are also not reproducible**: the
generators key on Python's built-in `hash()`, which CPython salts per process, and the
seed behind the shipped cache was never recorded. They are not pure noise either - a
hand-coded rule supplies the dominant term (increments for hydrophobic and basic residues,
a proline penalty) with a small hash-derived jitter - which is precisely why no
biological inference may be drawn from their feature importance: the generator *assumes*
the hydrophobic and basic C-terminal cleavage preference that such an inference would be
claiming to discover. Mode 33 is therefore reported here as a structural ablation only,
not as validated antigen-processing prediction, and `mode_31` remains the production
configuration. This is documented in full as `docs/claims_register.md` **D18**; the
remedy - replacing the mock cache rather than repairing it - is tracked as Phase 1 step 8
of `docs/proposals/2026_feature_upgrade_roadmap.md` and has not been carried out.

Per-virus within-CV AUC-ROC values, computed by training and evaluating exclusively on
each individual virus under peptide-grouped 5-fold cross-validation, are more variable,
ranging from 0.482 (HPV) to 0.805 (DENV) (Table 2; results/per_virus_eval_v5_mode31.csv).
This spread reflects differences in cohort size, negative-class composition,
and the inherent difficulty of within-virus immunogenicity discrimination for each
pathogen. The pooled mixed-background model (AUC-ROC 0.814) exceeds most per-virus
within-CV values; this pooled comparison is inflated by the out-of-panel *vaccinia*
bloc that makes up 77.8% of its negative background - **not**, as an earlier version of
this section stated, by self-proteome decoys, which are quarantined out of training and
scoring entirely (`docs/claims_register.md` D19). Note the allelotype-matched non-binders
do **not** explain this gap in either direction: they are present on both sides of the
comparison and constitute a far larger share of most per-virus negative sets (Table 3b:
0.988 for DENV, 0.920 for HIV-1) than of the pooled background (0.113), so if anything
they inflate the per-virus side more. No re-slice isolating them is reported here. The magnitude is measured in the same file the pooled
figure is drawn from: re-slicing the identical out-of-fold predictions with the vaccinia
rows removed moves AUC-ROC from 0.814 to 0.670 (`rf_cv_mean_no_vaccinia`,
models/v5/training_results_mode31.csv). On that same re-slice AUC-PR moves in the
opposite direction, 0.606 to 0.733, because removing 78% of the negatives raises the
positive base rate; that is a prevalence effect and must not be read as better
discrimination. The re-slice re-partitions the same predictions rather than refitting the
model, and is not the corpus-refit counterpart reported in results/cv_leakage_audit.csv.
These figures should be read alongside the decoy-free real-neg-only values in Table 2b. **We retract the inference, made in an earlier
version of this section, that the (larger, pre-remediation) gap between the pooled and
per-virus figures indicated cross-viral training provides complementary discriminatory
signal.** Both the pooled figure and the per-virus within-CV mean were, prior to
2026-08-10, computed under the same ungrouped, peptide-leaking splitter described in
Section 2.4, but the leakage differed in magnitude between them (pooled +37.0% vs.
per-virus mean +14.0%, `docs/claims_register.md` D15); the gap this paragraph previously
read as a biological finding was substantially explained by that differential inflation,
not by a validated cross-viral signal - the two figures reported here are both measured
under the same peptide-grouped splitter, so no comparable differential-leakage artifact
remains. HPV and HCV remain the lowest-performing viruses in within-CV evaluation; both
have comparatively sparse confirmed-negative records, which limits negative-class
separation.

Table 2. Per-virus within-CV AUC-ROC (peptide-grouped 5-fold, per-virus-only training,
mode-31 RF, v5 dataset, re-baselined 2026-08-10; results/per_virus_eval_v5_mode31.csv).

| Virus      | Within-CV AUC-ROC |
|------------|-------------------|
| DENV       | 0.805*            |
| CMV        | 0.743             |
| EBV        | 0.711             |
| IAV        | 0.697             |
| HIV-1      | 0.663             |
| HBV        | 0.656             |
| SARS-CoV-2 | 0.616             |
| HCV        | 0.548             |
| HPV        | 0.482             |

*DENV within-CV is decoy-inflated; see Table 2b and text below. Prior (retracted,
ungrouped, leakage-inflated) values: HIV-1 0.894, DENV 0.859, IAV 0.856, CMV 0.819,
EBV 0.790, HBV 0.708, SARS-CoV-2 0.699, HCV 0.575, HPV 0.561.

The per-virus within-CV values above are computed against each virus's full negative set,
which for several viruses is dominated by synthetic viral-proteome and allele-matched
non-binding decoys rather than assay-confirmed negatives. Because these decoys are
trivially separable by the MHCflurry-dominated feature set, within-CV AUC-ROC can be
inflated for viruses with large decoy populations. To provide a decoy-free estimate of
within-virus discrimination, we recomputed each per-virus AUC-ROC restricting the negative
class to real IEDB-tested negatives only (negative_origin in {tested_negative, iedb_api}),
reported in Table 2b alongside the within-CV value and the count of real negatives.

Table 2b. Per-virus within-CV AUC-ROC (full negative set, with decoys where present) versus
the honest real-neg-only AUC-ROC (negatives restricted to real IEDB-tested negatives;
mode-31 RF, v5 dataset, re-baselined 2026-08-10; results/per_virus_eval_v5_mode31.csv).
Gap = real-neg-only minus within-CV; n_real_neg is the number of real IEDB-tested
negatives available for the decoy-free estimate (data-shape counts, unaffected by the
splitter fix).

| Virus      | Within-CV AUC-ROC | Real-neg-only AUC-ROC | Gap    | n_real_neg |
|------------|-------------------|-----------------------|--------|------------|
| CMV        | 0.743             | 0.693                 | -0.050 | 272        |
| HBV        | 0.656             | 0.656                 |  0.000 | 229        |
| IAV        | 0.697             | 0.569                 | -0.128 | 119        |
| HIV-1      | 0.663             | 0.432                 | -0.231 | 60         |
| EBV        | 0.711             | 0.556                 | -0.156 | 72         |
| SARS-CoV-2 | 0.616             | 0.585                 | -0.031 | 980        |
| HCV        | 0.548             | 0.548                 |  0.000 | 320        |
| HPV        | 0.482             | 0.482                 |  0.000 | 137        |
| DENV       | 0.805             | 0.438                 | -0.368 | 12         |

Restricting negatives to assay-confirmed IEDB negatives removes the decoy-driven inflation.
The three viruses whose training negatives contain no decoys (HBV, HCV, HPV) show identical
within-CV and real-neg-only AUC-ROC, so their within-CV values were already decoy-free (and
already weak, 0.48 to 0.66). For the decoy-bearing viruses the honest metric is lower by
0.03 to 0.37 (Table 2b), with CMV retaining the strongest honest discrimination (0.693 on 272 real
negatives) and IAV (0.569) and EBV (0.556) retaining weak-to-moderate signal below their
decoy-inflated headlines. HIV-1's real-neg-only AUC-ROC (0.432) falls below chance,
consistent with its anti-predictive LOO result (0.162, Table 3) - both metrics now agree
that the model's within-CV ranking for this virus is driven substantially by decoy
separation rather than genuine positive/real-negative discrimination. DENV is the extreme
case: its within-CV AUC-ROC of 0.805 collapses to 0.438 - indistinguishable from chance -
on its 12 real IEDB-tested negatives, demonstrating that the DENV within-CV headline is
essentially a positives-versus-synthetic-decoy score rather than a trustworthy
discrimination result. The DENV within-CV value of 0.805 should therefore not be
interpreted as evidence of within-virus discrimination; DENV is classified as not validated
for within-virus ranking, and the real-neg-only estimate is itself statistically noisy at
n=12. We accordingly lead per-virus quality assessment with the real-neg-only AUC-ROC and
its real-negative count, flagging DENV (n=12), HIV-1 (n=60), and EBV (n=72) as
low-real-negative and provisional.

Score calibration. The production scorer applies an isotonic calibration layer fitted on
out-of-fold predictions. We assess it by cross-fitting that layer over peptide-grouped folds,
so that no row is calibrated by a model that saw its peptide
(results/calibration_assessment_v5_mode31.csv). Pooled over all 35,555 out-of-fold rows the
layer appears to reduce the Expected Calibration Error (ECE) from 0.060 to 0.005, but that
pooled figure is an artifact of cancellation and is not evidence of reliability. The pool
mixes two populations that are miscalibrated in opposite directions: in every score bin the
nine target viruses are under-confident while the 22,239 out-of-panel rows (62.5% of the pool,
carrying 18 positives between them) are over-confident, so the two errors offset. Assessed
separately neither population is well calibrated - the out-of-panel rows reach ECE 0.140 - and
the pooled value is therefore lower than either of its components rather than intermediate
between them. On the nine target viruses (n = 13,316), which is the population these results
are about, calibration does not improve at all: ECE moves from 0.227 to 0.235, and of the nine
only HBV improves individually. We therefore report calibration as a current limitation rather
than a benefit. The out-of-panel ECE is itself a degenerate statistic at a positive rate of
0.0008, where it reduces to little more than the mean predicted score; we quote it only to
characterise the pooled figure, not as a calibration result in its own right. This is the
pooled-versus-panel distinction recorded for the mixed-background discrimination figure
(docs/claims_register.md D19), one level deeper: there the pooled figure was dominated by an
easier subpopulation, whereas here it is lower than both subpopulations. Finally, isotonic
calibration is monotone non-decreasing rather than strictly monotonic: it preserves the weak
ordering of scores but collapses distinct scores into ties, so it is not AUC-preserving. The
AUC-ROC values reported in this section are computed on raw out-of-fold scores and are
unaffected by calibration for that reason.


### 3.3 Leave-One-Virus-Out Generalization and Test Partition Validity

To evaluate cross-virus transfer under a realistic deployment scenario, a leave-one-virus-out
(LOO) benchmark was conducted (Section 2.5). For each of the nine eligible pathogens, a
mode-31 RF was retrained from scratch on the remaining eight viruses, the out-of-panel
negative background, and 5,000 self-proteome hard decoys, then evaluated on the held-out
virus (Section 2.5 gives the pool composition). A critical methodological
finding emerged in the construction of LOO test partitions: viral proteome decoys carrying
the label negative_origin = allele_matched_nonbinder (3,112 rows in v5) were included in
test partitions under the initial LOO protocol. Despite the label, these decoys are not
low-affinity: of the 218 such rows present in the tracked binding matrix
(`models/peptide_binding_matrix_v5.csv`, covering 168 distinct peptides), the maximum
per-allele MHCflurry presentation score has median 0.761 over rows and 0.740 over distinct
peptides (range 0.503-0.982) - comparably high to, and on this sample
higher than, the 0.712 median for true positives (n = 6,431 active rows; the all-rows
figure is 0.705 over n = 7,037, but that base includes 606 quarantined rows, which the
decoy sample does not). Those 218 rows are not a random draw from the 3,112: the
binding matrix predates every decoy file and was never rebuilt, so matrix membership
selects for peptides already present in the earlier corpus and biases this particular
comparison upward. The mechanism by which their inclusion inflated LOO AUC-ROC is
therefore not simply "trivially rejected as low-affinity
negatives", and is not further characterised here; the measured effect itself is
well-established from the certified before/after comparison. One structural asymmetry is
recorded here because it constrains any future mechanistic account: 2,894 of the 3,112
decoys (93.0% of active rows) have no entry in the binding matrix at all and therefore
receive an all-zero binding-feature vector, against 0 of 22,467 tested_negative and 0 of
1,963 iedb_api active rows. This is a gap in matrix coverage rather than in measured
affinity, since the decoys were selected on a presentation-score threshold at generation.
It does not by itself explain the inflation, and we do not present it as doing so: 1,624
active positives carry the same all-zero signature, and in five of the six decoy-bearing
held-out folds that signature is associated with the positive class in the training pool
rather than the negative one. Those five are not independent replications. Every one of the
1,624 is an HIV-1 peptide, so they are one virus's rows appearing in every training pool
except their own, and the sole dissenting fold is HIV-1 itself, where the association
disappears only because holding HIV-1 out removes every zero-vector positive from training.
HIV-1 is also the second-largest of the affected decoy populations listed below, is the
anti-predictive outlier discussed in Section 4.2, and carries the second-largest measured
inflation of any virus (+0.474 AUC-ROC, behind only DENV at +0.497), so it is not a neutral
control. The two largest inflations therefore sit on opposite sides of this signature. Their inclusion in test
partitions is associated with an inflation in LOO AUC-ROC of 0.25-0.50 points for
viruses with large allele_nb populations:
DENV (1,000 of 1,012 initial test negatives), EBV (300 of 380), IAV (445 of 564), and
HIV-1 (693 of 753). All reported results use test partitions restricted to real
IEDB assay-confirmed negatives only (negative_origin in {tested_negative, iedb_api};
see Methods 2.5 and results/loo_cross_virus_v5_clean.json).

After correction, mean LOO AUC-ROC across nine pathogens is 0.463, with three of nine
viruses above the chance baseline of 0.5 (Table 3; Figure 2). LOO AUC-ROC is uniformly
lower than the corresponding within-virus cross-validation AUC-ROC across all nine
pathogens (Figure 2). Cross-virus transfer on real
IEDB-confirmed negative test sets is limited for the RF mode-31 feature set at current
v5 data scales. CMV achieves the highest corrected LOO AUC-ROC (0.633), consistent with
the presence of EBV (family Herpesviridae) in its LOO training corpus. HBV (0.556) and
HCV (0.528) are modestly above chance with clean negative sets (229 and 320 real tested
negatives respectively). The remaining six viruses cluster near or below the chance
baseline.

HIV-1 is anti-predictive (LOO AUC-ROC 0.162) despite substantially higher within-CV
performance (0.663, peptide-grouped as of 2026-08-10; previously reported as 0.894 under
an ungrouped splitter, retracted as leakage-inflated - `docs/claims_register.md` D15). The
HIV-1 clean test partition contains 60 real IEDB-tested immunogenicity
negatives out of 2,576 total test observations. These 60 negatives are predominantly
confirmed strong MHC binders found to be non-immunogenic in T-cell assay - precisely the
hard case the model fails. The RF mode-31 classifier assigns high scores to strong MHC
binders because binding features dominate the 31-feature set, inverting the
positive-to-negative rank ordering for HIV-1. This result is the clearest evidence in the
v5 evaluation that binding-score features are insufficient for cross-virus immunogenicity
discrimination when the non-immunogenic class consists of assay-characterized strong
binders rather than binding-negative decoys.

DENV has only 12 real IEDB-tested negatives after quarantine and allele_nb exclusion; the
corrected AUC-ROC of 0.372 carries a wide confidence interval at this negative count and
should be interpreted with caution. EBV, IAV, and SARS-CoV-2 have sufficient clean
negative populations for reliable LOO estimates (80, 119, and 980 respectively), and all
three are at or below chance (0.496, 0.488, 0.462).

Table 3. Corrected LOO and within-CV AUC-ROC for nine viruses (mode-31 RF, v5 dataset,
Amendment 7 clean test partitions). Delta = LOO - within-CV. Within-CV AUC-ROC
re-baselined 2026-08-10 under a peptide-grouped splitter closing D15 (`docs/claims_register.md`);
LOO AUC-ROC is unaffected (each virus is already held out of training entirely, so it was
never exposed to the peptide-leakage this closes).

| Virus      | LOO AUC-ROC | Within-CV AUC-ROC | Delta  | n_clean_neg | n_nb_excl |
|------------|-------------|-------------------|--------|-------------|-----------|
| CMV        | 0.633       | 0.743             | -0.110 | 272         | 300       |
| HBV        | 0.556       | 0.656             | -0.100 | 229         | 0         |
| HCV        | 0.528       | 0.548             | -0.020 | 320         | 0         |
| EBV        | 0.496       | 0.711             | -0.215 | 80          | 300       |
| IAV        | 0.488       | 0.697             | -0.209 | 119         | 445       |
| HPV        | 0.468       | 0.482             | -0.014 | 137         | 0         |
| SARS-CoV-2 | 0.462       | 0.616             | -0.154 | 980         | 374       |
| DENV       | 0.372*      | 0.805             | -0.433 | 12*         | 1000      |
| HIV-1      | 0.162       | 0.663             | -0.501 | 60          | 693       |
| Mean       | 0.463       | 0.658             |        |             |           |

*Statistically noisy at n=12 clean negatives. Prior (retracted, ungrouped, leakage-inflated)
Within-CV column: CMV 0.819, HBV 0.708, HCV 0.575, EBV 0.790, IAV 0.856, HPV 0.561,
SARS-CoV-2 0.699, DENV 0.859, HIV-1 0.894, Mean 0.751.

*Figure 2. Leave-one-virus-out (LOO) cross-virus generalization versus within-virus
cross-validation. AUC-ROC of the SESTRAV RF mode-31 model on the v5 dataset (IEDB
assay-confirmed clean test partitions), shown per pathogen for two evaluation
paradigms. Within-virus cross-validation trains and tests on peptides from the same
pathogen; leave-one-virus-out withholds the target pathogen entirely from training
and uses it only for testing. The dashed horizontal line marks random-chance
performance (AUC-ROC = 0.5), and pathogens are ordered by descending LOO AUC-ROC.
Across all nine pathogens, LOO performance is uniformly lower than within-virus CV,
and for six of nine pathogens LOO falls at or below chance, indicating that
within-virus predictive signal does not transfer to unseen pathogens. HIV-1 is a
notable case: substantially higher within-virus performance (AUC-ROC = 0.663) collapses to
anti-predictive performance under LOO (AUC-ROC = 0.162).*


### 3.4 Interpretation of LOO Results

Cross-virus immunogenicity transfer is limited by two compounding factors identified in
this evaluation. First, the binding-score-dominated mode-31 feature set does not
distinguish strong MHC binders that are immunogenicity-negative from strong binders that
are immunogenicity-positive, which is the biologically relevant discrimination problem
for cross-virus generalization. The pooled mixed-background performance of RF mode-31 is
high relative to the per-virus real-negative-only figures (pooled mixed-background
AUC-ROC 0.81; Section 3.2) because these pooled test sets
include both IEDB-tested positives and a mixed background of IEDB-tested and
allelotype-matched non-binding negatives - dominated by the out-of-panel *vaccinia*
tested-negative bloc - drawn from similar HLA and peptide distributions as the training
data.
(Self-proteome decoys do **not** enter this pooled background: they are quarantined out
of `train_classifier.py`'s pipeline entirely, per `docs/claims_register.md` D19. They are
included in LOO *training* only, as described in Section 2.5, which is a different
protocol from the pooled evaluation referenced here.) Cross-virus test sets expose the model to
pathogen-specific immunogenicity patterns for which the binding prior provides misleading
signal, as demonstrated most acutely by HIV-1.

Second, the absolute count of real IEDB-tested negatives limits the reliability of LOO
estimates for under-resourced viruses. DENV's corrected AUC-ROC is unreliable at n=12;
the true cross-virus transfer score for DENV from its Flaviviridae training relative (HCV)
cannot be assessed until additional assay-confirmed DENV negatives are curated from the
published literature. This quantifies a specific data gap: the limiting factor is not
positive immunogenicity data (all nine viruses have adequate positive coverage) but
negative immunogenicity data from binding-positive, assay-confirmed non-responders.

CMV (0.633) remains the most plausible candidate for genuine family-transfer: EBV
provides herpesvirus-specific physicochemical signal in its training set, and the 272
clean negatives provide a reliable estimate. Whether this represents true phylogenetic
transfer or shared HLA restriction pattern coverage requires confirmation on an
independent external CMV cohort. HPV (0.468) remains an active generalization failure
consistent with the physicochemical distinctiveness of papillomavirus-derived peptides
relative to the RNA-virus-dominated training corpus.

**Binding-confound decomposition.** The two compounding factors above can be separated
quantitatively by evaluating the same model under three nested regimes that each remove
one source of non-immunogenic signal (Table 3b, re-baselined 2026-08-10). Regime 1 is
within-virus cross-validation scored against all negatives, real and decoy (mean AUC-ROC
0.658). Regime 2 restricts the within-virus negatives to real IEDB assay-confirmed
negatives, removing the trivially separable binding-negative decoys (mean 0.551). Regime 3
is the corrected cross-virus LOO protocol, which additionally removes the target pathogen
from training (mean 0.463, unaffected by this re-baseline - see Table 3). The decline is
monotone for every pathogen. The regime 1 to regime 2 drop - the decoy-inflation term (mean
0.107) - is not uniform: it correlates with the fraction of decoys in each virus's negative
set (Pearson r = 0.888), and for the three viruses whose negative sets contain no decoys at
all (HBV, HCV, HPV) it is exactly zero, a built-in control confirming that the inflation is
decoy-composition-driven rather than an artifact of the metric. The residual regime 2 to
regime 3 drop - the cross-virus transfer gap (mean 0.088) - is largest for HIV-1 (0.269),
the same pathogen whose within-virus confident predictions invert under LOO. Read together,
the two terms partition the apparent within-virus discrimination into a component explained
by negative-set composition and a component that fails to transfer across pathogens,
leaving cross-virus immunogenicity transfer at approximately chance once both confounds are
removed. We therefore present the LOO collapse not as an incidental weakness but as a
designed diagnostic: an evaluation protocol that isolates how much of a seemingly strong
immunogenicity score is attributable to binding-correlated confounds rather than
transferable biology.

Table 3b. Binding-confound decomposition across three nested evaluation regimes (mode-31
RF, v5, re-baselined 2026-08-10 under a peptide-grouped splitter - R1 and R2 are the D15
remediation target; R3 is unaffected). Decoy inflation = regime 1 - regime 2; transfer gap
= regime 2 - regime 3.

| Virus      | R1 within, all neg | R2 within, real neg | R3 cross-virus LOO | Decoy inflation | Transfer gap | Decoy frac |
|------------|--------------------|---------------------|--------------------|-----------------|--------------|------------|
| CMV        | 0.743              | 0.693               | 0.633              | 0.050           | 0.060        | 0.524      |
| DENV       | 0.805              | 0.438               | 0.372*             | 0.368           | 0.065        | 0.988      |
| EBV        | 0.711              | 0.556               | 0.496              | 0.156           | 0.060        | 0.806      |
| HBV        | 0.656              | 0.656               | 0.556              | 0.000           | 0.099        | 0.000      |
| HCV        | 0.548              | 0.548               | 0.528              | 0.000           | 0.020        | 0.000      |
| HIV-1      | 0.663              | 0.432               | 0.162              | 0.231           | 0.269        | 0.920      |
| HPV        | 0.482              | 0.482               | 0.468              | 0.000           | 0.014        | 0.000      |
| IAV        | 0.697              | 0.569               | 0.488              | 0.128           | 0.081        | 0.789      |
| SARS-CoV-2 | 0.616              | 0.585               | 0.462              | 0.031           | 0.123        | 0.276      |
| Mean       | 0.658              | 0.551               | 0.463              | 0.107           | 0.088        |            |

*DENV LOO is statistically noisy at n=12 clean negatives. (Regime-2 and decomposition
values derive from the auc_roc_real_neg_only column of results/per_virus_eval_v5_mode31.csv
and results/loo_cross_virus_v5_clean.csv; the decomposition table above is committed as
results/loo_binding_confound_decomposition.csv and is regenerated by
scripts/compute_loo_binding_confound.py.)


### 3.5 External Tool Comparison

A 30-feature Random Forest configuration (unweighted, 200 trees, 2026-05) was compared
against three published prediction tools - BigMHC, MixMHCpred 2.2, and DeepImmuno - and
a binding-only baseline using MHCflurry 2.0 presentation scores. All comparisons used a
shared 720-peptide Tier A test set assembled from viral immunogenicity studies and frozen
on 2026-05-20. The RF predictions were generated by out-of-fold 5-fold cross-validation;
the external tools were fully scored on the shared set from their public release
checkpoints. **This RF arm was previously attributed here to the canonical mode-31
production model and described as conservative by construction; both attributions are
corrected below (`docs/claims_register.md` D15, D16).** The evaluated configuration is a
30-feature, unweighted, 200-tree model from 2026-05, predating the introduction of
`feature_mode=31` by 26 days - it is not the production `mode_31` model. Its training
corpus (the 720-row root `immunogenicity_dataset.csv`) has zero duplicate peptides, so
the exact-peptide leakage mechanism affecting this manuscript's v5 figures (Section 2.4,
`docs/claims_register.md` D15) is a structural no-op here and does not apply to this
benchmark (D22). A separate mechanism does apply and has not been corrected for: an
all-pairs substring-containment scan of the 704-peptide scored pool found 226 peptides
(32.1%) with a near-duplicate elsewhere in the pool, length differences of 1-3 residues,
consistent with the same epitope tested at different registration boundaries. Whether
this affected the reported AUC-PR is not established - the historical cross-validation
fold assignment is not recoverable - but the risk was never measured or filtered for this
benchmark, unlike the H2 gold-standard evaluation, which does filter its own corpus for
substring overlap (`docs/claims_register.md` D22).

**Table 4. External tool comparison on the shared Tier A test set (AUC-PR; SESTRAV out-of-fold, 30-feature RF configuration, substring-homology risk disclosed but not corrected for (D22); external tools fully scored)**

| Tool | AUC-PR | Evaluation type | n scored | Coverage |
|---|---|---|---|---|
| SESTRAV RF (30-feature, unweighted, 2026-05) | 0.828 | Out-of-fold CV, ungrouped | 704 | 97.8% |
| BigMHC | 0.822 | Fully scored | 720 | 100% |
| Binding-only (MHCflurry) | 0.800 | Fully scored | 720 | 100% |
| MixMHCpred 2.2 | 0.795 | Fully scored | 720 | 100% |
| DeepImmuno | 0.698 | Fully scored | 623 | 86.5% |

On the shared Tier A test set, the SESTRAV RF configuration above (AUC-PR 0.828) posted
the highest point AUC-PR among a field of fully-scored external tools: BigMHC (0.822), the
MHCflurry binding-only baseline (0.800), MixMHCpred 2.2 (0.795), and DeepImmuno (0.698).
The margin over the binding-only baseline (+0.028) was previously described as notable
given an out-of-fold handicap; that framing is withdrawn, because out-of-fold scoring on
a corpus with zero duplicate peptides is not a handicap in the way it would be on a
duplicate-heavy corpus. Nor is it established to be an advantage: the substring-homology
risk disclosed above (32.1% of the scored pool) is unquantified in its effect on this
score, so neither the near-tie with BigMHC nor the margin over binding-only should be
read as biased in a known direction. A peptide-grouped splitter is now used for every other v5 CV figure in this
manuscript (Section 2.4), but is deliberately not applied to this Tier A field: n=414 of
the 704 peptides resolve to an active, non-quarantined row of the current v5 production
corpus; of the remaining 290, 236 are absent from v5 entirely and 54 appear only in
quarantined rows, so a re-run would score a smaller, non-comparable field rather than
correct this one.

To distinguish genuine performance separation from sampling noise, we performed a paired
bootstrap analysis (10,000 resamples) of the AUC-PR difference on the peptide subset scored
by both SESTRAV and each comparator (n = 704 paired peptides, 490 positive / 214 negative).
Against BigMHC, the strongest external tool, SESTRAV's advantage was not statistically
significant (paired AUC-PR difference +0.018, 95% CI -0.022 to +0.058, p = 0.37); on the
shared benchmark the two are a statistical near-tie. Against the MHCflurry binding-only
baseline, the advantage cleared zero by a narrow margin (paired AUC-PR difference +0.038,
95% CI +0.002 to +0.071, p = 0.04). **Both tests were computed on the same OOF arm
discussed above, whose substring-homology risk is disclosed but not corrected for; the
binding-only result in particular has a 95% CI lower bound sitting just above zero within
a 0.069-wide interval, so this significance finding should be treated as narrow and
unconfirmed rather than robust. That lower bound should not be read at four decimal
places: the two comparisons reported here draw from a single seeded random stream in call
order, and across call orders and reseedings of the same data the bound moves over a range
comparable in size to the margin by which it clears zero, without any conclusion changing.
A substring-aware re-run - retraining with near-duplicate peptides excluded or grouped
across folds, which would move the point estimate itself and not merely the interval - has
not been performed and would be needed to settle whether the finding is a genuine effect
(`docs/claims_register.md` D22).** (Paired-
bootstrap values are reproducible via the frozen Tier A scores; the summary is committed as
results/tier_a_paired_bootstrap.csv and is regenerated by
scripts/compute_tier_a_paired_bootstrap.py.)

---

## 4. Discussion

### 4.1 The Workflow Advantage

The primary contribution of SESTRAV is the governed pipeline architecture rather
than the classifier in isolation. The six-stage workflow - proteome-scale peptide
enumeration, allele-stratified MHC binding matrix generation via MHCflurry 2.0,
TCR-contact physicochemical feature extraction, viral decoy generation,
immunogenicity inference, and freeze-mode output with cryptographic provenance -
represents, to our knowledge, the first systematic attempt to make
immunogenicity prediction reproducible at the pipeline level rather than at the
model level alone. Freeze-mode output records SHA-256 checksums of all input
data files, the MHCflurry model version, and feature matrix dimensions in a JSON
provenance sidecar at each training run, enabling exact replay of any historical
build regardless of upstream database updates. This audit trail addresses a
reproducibility gap that affects the immunogenicity prediction field broadly:
most published tools provide a trained model but not the data curation workflow
that produced it, and IEDB updates silently alter the positive and negative class
composition of any dataset assembled without version-locked provenance.

The virus-level leave-one-out evaluation protocol introduced here is itself a
benchmark contribution independent of the classification results. No published
MHC class I immunogenicity predictor, to our knowledge, has reported a
systematic LOO protocol in which a separate model is retrained for each
held-out pathogen and evaluated exclusively on that pathogen's labelled data.
Standard within-pathogen stratified cross-validation - the dominant evaluation
paradigm in the field - does not address the scenario most relevant to vaccine
development practice, namely a pathogen for which no prior T-cell response data
exist. Stratifying folds by peptide rather than by pathogen conflates
discriminating among peptides of an already-characterised virus with the
genuinely harder problem of generalizing to entirely unseen molecular space. The
LOO protocol separates these tasks and quantifies cross-pathogen generalizability
as a primary endpoint.

The "structural" framing in the SESTRAV name and throughout this manuscript
requires a specific qualification: features encode physicochemical proxies for
structural discrimination at predicted TCR contact positions, not
three-dimensional coordinates derived from crystallography or computational
modelling. No structure prediction or molecular dynamics simulation is
performed; structural claims throughout are dependent on the contact position
assignment of Chowell et al. [16], which was derived from canonical 9-mer
crystal structure analyses and carries additional uncertainty when applied to
8-mer and 10-mer binding registers.

### 4.2 Limitations

Several limitations constrain the conclusions that can be drawn from the v5
evaluation. The primary training labels are drawn from IEDB population-average
T-cell assay outcomes. These labels capture the aggregate probability that a
peptide elicits a measurable CD8+ T-cell response across a diverse donor pool;
they are neither allele-specific nor donor-specific. A positive-labelled peptide
may elicit responses in only a subset of HLA-matched donors, while a
negative-labelled peptide may be recognised in individuals with specific TCR
repertoire compositions not represented in the screened cohort. The model
therefore estimates a population-level signal and should not be interpreted as
predicting individual-level response probability.

The TCR contact position assignment follows the Chowell et al. [16]
convention of positions p4-p8 for canonical 9-mers. For 8-mer and 10-mer
peptides, the C-terminal-relative index mapping used in SESTRAV - where p7 and
p8 are computed as offsets from the peptide C-terminus - is an approximation
without crystal structure validation for non-canonical binding registers, and
its accuracy is expected to decline as register geometry deviates from the
canonical 9-mer template.

Binding signal dominates the mode-31 feature set. The ten MHCflurry 2.0
antigen presentation scores account for the majority of discriminative
information in the Random Forest: 55.8% of total importance, against 41.7% for the
20 physicochemical descriptors at TCR contact positions and 2.5% for peptide length
(`models/v5/feature_importances.csv`). The dominance is cleanest stated by rank - the
ten binding features occupy ranks 1 to 10 without exception, and every physicochemical
descriptor falls below all of them - but the margin at that boundary is narrow: the
weakest binding feature (`bind_B3501`, 4.27%) exceeds the strongest physicochemical one
(`p8_vdw_volume`, 3.87%) by only 1.10x. The physicochemical block is also strongly
bimodal, so its mean per-feature importance describes no feature in the set and should
not be used to characterise the gap. Mode-31 remains best characterised as a
discriminator of strong MHC binders from weaker ones, with a substantial secondary
physicochemical component, rather than as an independent structural immunogenicity
predictor.
That characterisation is a reading of the importance structure, not a measurement of
the increment above a pure binding baseline: Table 1 contains no binding-only arm, so
that increment is not quantified here.

Class imbalance varies substantially across the nine target viruses, ranging from a
0.6:1 positive-to-negative ratio for IAV to 3.3:1 for HIV-1
(`results/per_virus_eval_v5_mode31.csv`). AUC-ROC is
relatively insensitive to class balance across this range, but precision-recall
metrics and positive predictive value would diverge considerably between virus
cohorts, and performance figures should be interpreted with the per-virus class
compositions in mind.

HPV is an active generalization failure at LOO AUC-ROC 0.468, one of six viruses
below the 0.5 chance baseline. This below-chance
result indicates that
physicochemical features learned from the RNA-virus-dominated training corpus
not only fail to transfer to HPV but may actively mislead the classifier for
the papillomavirus peptide landscape. The HPV within-virus AUC-ROC, re-baselined
2026-08-10 under a peptide-grouped splitter, is **0.482** (previously reported as
0.561 under an ungrouped, leakage-inflated splitter, `docs/claims_register.md` D15) -
itself close to chance, so HPV is no longer a clean case of within-virus signal that
fails to transfer; both the within-virus and cross-virus estimates are weak.
HPV is designated an active generalization failure and a priority case for
dedicated training data collection.

The LOO evaluation identified a systematic test partition flaw. Viral proteome decoys
labelled negative_origin = allele_matched_nonbinder were included in test partitions in
the v4 LOO protocol. These decoys measure as high-affinity, not low-affinity, on the
peptides for which a binding-matrix score is available (Section 3.3), so the mechanism
by which they inflated AUC-ROC is not yet characterised; the effect itself is measured
and certified. For DENV, 1,000 of 1,012 test negatives were allele_matched_nonbinder
rows, and AUC-ROC moved from the corrected 0.372 to the reported 0.870 once they were
included. After restricting test partition negatives
to real IEDB assay-confirmed negatives (Amendment 7, results/loo_cross_virus_v5_clean.json),
the apparent family-transfer signals for DENV (0.870 to 0.372), EBV (0.824 to 0.496),
and IAV (0.784 to 0.488) are eliminated; all were artifacts of this partition flaw.
The corrected LOO AUC-ROC values for viruses with already-clean test partitions -
HBV (0.556), HCV (0.528), HPV (0.468) - are modestly shifted but substantially
unchanged, confirming that those values were not inflated.

The HIV-1 anti-predictive result (LOO AUC-ROC 0.162) is a severe LOO-specific
amplification of a milder inversion already present within cross-validation against
assay-confirmed negatives, not a uniquely LOO artifact. That within-CV inversion is
narrow, and we state its width rather than only calling it provisional: on the n=60
real-negative definition used throughout this paper its bootstrap 95% confidence
interval is [0.370, 0.496] (4,000 resamples, both classes, seed 42), clearing chance by
under 0.005, and on the narrower `iedb_api`-only cut (n=49, AUC-ROC 0.432) the interval
[0.360, 0.505] includes 0.5 outright. The qualifier also matters: against the full
negative set, which is 92% decoys, no inversion occurs (AUC-ROC 0.663, 95% CI
[0.642, 0.682]; `results/per_virus_eval_v5_mode31.csv`).
The RF mode-31 model is binding-dominated: the ten MHC binding-affinity
features account for 55.8% of total feature importance, against 41.7%
for the twenty physicochemical descriptors and 2.5% for peptide length
(`models/v5/feature_importances.csv`). Within cross-validation on the full training pool
the model does separate HIV-1 positives from the pooled negative set (mean out-of-fold
score: positives 0.439 over n=2,516, all negatives 0.316 over n=753;
`models/v5/rf_oof_predictions_mode31.csv`), but that separation is decoy-driven rather
than evidence of genuine discrimination for this virus: 693 of those 753 negatives are
allele-matched non-binder decoys (`negative_origin = allele_matched_nonbinder`), which
are unassayed rather than experimentally confirmed, and against the 60 assay-confirmed
negatives the ordering reverses (mean
0.479, above the positives' 0.439; counts in
`results/per_virus_eval_v5_mode31.csv`). Restated as a rank statistic those same 60 rows
give the below-chance real-negative-only AUC-ROC of 0.432 reported above - the same fact
in another form, not independent corroboration - and HIV-1 is one of the three viruses
flagged above as low-real-negative and provisional at this sample size. The LOO failure
is therefore not a property of the HIV-1 data in isolation, but neither does the
within-CV figure rescue this virus. The LOO collapse emerges when HIV-1 is excluded from
training: the model inherits from the remaining eight viruses a prior that high
MHCflurry binding predicts immunogenicity, whereas HIV-1
IEDB-confirmed negatives are enriched for high-affinity B-allele binders (B*08:01,
B*27:05, and B*35:01 all show higher mean binding scores in HIV-1 confirmed-negative
records than in confirmed-positive records). We therefore hypothesise that the
anti-prediction reflects binding-feature dominance compounded by a likely clinical
selection bias in HIV-1 IEDB studies, which frequently test high-affinity
B-restricted candidate epitopes and confirm non-response, producing a negative set
atypical of the other eight viruses and unlearnable from the remaining pool in LOO.
This mechanism is offered as a hypothesis rather than a demonstrated fact: the score
inversion is far more severe in the LOO setting (0.162) than in the within-training
out-of-fold data (0.432 against assay-confirmed negatives), rather than absent from the
latter, and the per-allele binding enrichments in the HIV-1 negative set,
though consistent in direction, are individually non-significant at the available
negative sample size.

### 4.3 Future Directions

The most impactful immediate improvement to LOO generalization is replacing
decoy-padded LOO test partitions with real assay-confirmed negatives from
published per-virus T-cell screening panels. Priority sources include Webster
et al. (PMC415806) for HBV and Riemer et al. (PMC2937992) for HPV; HCV
validation cohorts with full negative characterisation remain an active search
target. Replacing synthetic decoys with screened negatives in the test
partitions for HBV, HCV, and SARS-CoV-2 would convert the current provisional
LOO estimates into independently validated performance figures.

Peptide stability at the MHC class I surface - quantified as complex half-life
and available through NetMHCstabpan [26] - is the
highest-priority candidate feature addition to the mode-31 feature set.
Presentation probability and pMHC stability are biologically orthogonal: a
weakly presented peptide may form a long-lived complex and sustain T-cell
visibility, while a strongly predicted binder with rapid turnover may not. Adding
stability half-life as an 11th binding feature is the most tractable extension
requiring no new model architecture.

GNN v5 training on the full dataset is deferred to a subsequent release pending
GPU provisioning. This experiment will evaluate whether per-residue ESM-2 t12
embeddings [19] and functionally motivated graph edges connecting
the p4-p8 contact subgraph provide discriminative lift beyond the RF mode-31
baseline. Either outcome - lift or no lift - is informative and will be reported
with ablation results rather than characterised as a research component without
supporting numbers.

HLA allele expansion to twelve additional alleles weighted by disease burden,
including B*46:01 and A*30:01 for nasopharyngeal carcinoma and HBV cohorts
prevalent in Asian populations, would address a systematic coverage gap in the
current ten-allele panel. Template-based pMHC structural modelling via PANDORA
[27] would provide three-dimensional anchor-position
geometry for contact weight refinement without full structure prediction, enabling
empirically grounded replacement of the current Chowell approximation for
non-canonical peptide lengths.

A dedicated per-virus versus pooled architecture comparison experiment is
required before the production model architecture can be stated as optimal.
The current results establish that cross-viral pooling provides complementary
signal for viruses with a family representative in training, but do not
demonstrate that pooling outperforms a dedicated per-virus model for viruses
with sufficient data - a question that must be answered empirically.

### 4.4 Conclusions

SESTRAV v5 demonstrates that cross-pathogen immunogenicity transfer on assay-
confirmed test negatives is limited for the RF mode-31 feature set at current
IEDB data scales: mean LOO AUC-ROC of 0.463 with three of nine viruses above
chance. The central methodological contribution is the identification and
correction of a systematic test partition flaw - allelotype-matched non-binding
decoys in LOO test sets - that inflated prior transfer estimates by 0.25-0.50
AUC-ROC points. CMV (0.633) and HBV (0.556) are modest transfer successes;
HPV is an active failure; HIV-1 (0.162) reveals binding-feature dominance as
the primary barrier to generalization on strong-binder immunogenicity negatives.
The primary contribution of this work is not a state-of-the-art immunogenicity
predictor but a governed, auditable pipeline infrastructure that makes data
curation, model training, and rigorous LOO evaluation fully reproducible at the
workflow level. We characterise the test partition conditions required for valid
cross-virus transfer assessment, demonstrate why binding-dominated features fail
for immunogenicity generalization, and provide this infrastructure as a community
foundation for future work incorporating expanded assay-confirmed negative cohorts
and structural GNN features.

---

## Acknowledgements

The authors thank the Immune Epitope Database (IEDB), VDJdb, and the LANL HIV Molecular
Immunology Database for maintaining the publicly available immunological data resources on
which this work depends.

---

## Author Contributions

[PLACEHOLDER - confirm with H1 authorship/order. CRediT taxonomy template:]
G.B. conceived the study, designed the LOO benchmark and contamination-correction protocol,
implemented the pipeline and integrity harness, performed the analysis, and wrote the manuscript
(Conceptualization, Methodology, Software, Formal analysis, Investigation, Data curation, Writing -
original draft, Visualization). [Co-author initials] contributed to [Methodology / Validation /
Writing - review and editing / Supervision - fill per co-author]. All authors read and approved the
final manuscript.

---

## Funding

[PLACEHOLDER: Funding statement required by Bioinformatics (Oxford Academic). If self-funded with no
specific grant, use: "This research received no specific grant from any funding agency in the public,
commercial, or not-for-profit sectors."]

---

## Conflict of Interest

The authors declare no conflict of interest.

---

## Data Availability

The SESTRAV v5 immunogenicity dataset, pre-trained RF mode-31 model, LOO evaluation
scripts, and corrected results files are available at https://github.com/Gavin-Borges/SESTRAV.
The archived release corresponding to this manuscript is deposited at Zenodo
(DOI: [PLACEHOLDER - reserve and paste before submission]). The IEDB source data are available at
https://www.iedb.org. VDJdb data are available at https://vdjdb.cdr3.net. LANL HIV Molecular
Immunology Database data are available at https://www.hiv.lanl.gov/content/immunology.

---

## References

1. Bravi B, Di Gioacchino A, Fernandez-de-Cossio-Diaz J, Walczak AM, Mora T, Cocco S, et al. A transfer-learning approach to predict antigen immunogenicity and T-cell receptor specificity. eLife. 2023;12:e85126. doi:10.7554/eLife.85126

2. Lee CH, Huh J, Buckley PR, Jang M, Pereira Pinho M, Fernandes RA, et al. A robust deep learning workflow to predict CD8+ T cell epitopes. Genome Med. 2023;15:70. doi:10.1186/s13073-023-01225-z

3. Vita R, Mahajan S, Overton JA, Dhanda SK, Martini S, Cantrell JR, et al. The Immune Epitope Database (IEDB): 2018 update. Nucleic Acids Res. 2019;47(D1):D339-D343. doi:10.1093/nar/gky1006

4. Reynisson B, Alvarez B, Paul S, Peters B, Nielsen M. NetMHCpan-4.1 and NetMHCIIpan-4.0: improved predictions of MHC antigen presentation by concurrent motif deconvolution and integration of MS MHC eluted ligand data. Nucleic Acids Res. 2020;48(W1):W449-W454. doi:10.1093/nar/gkaa379

5. Rock KL, Goldberg AL. Degradation of cell proteins and the generation of MHC class I-presented peptides. Annu Rev Immunol. 1999;17:739-779. doi:10.1146/annurev.immunol.17.1.739

6. Calis JJA, Maybeno M, Greenbaum JA, Weiskopf D, De Silva AD, Sette A, et al. Properties of MHC class I presented peptides that enhance immunogenicity. PLoS Comput Biol. 2013;9(10):e1003266. doi:10.1371/journal.pcbi.1003266

7. Albert BA, Yang Y, Shao XM, Singh D, Smith KN, Anagnostou V, et al. Deep neural networks predict class I major histocompatibility complex epitope presentation and transfer learn neoepitope immunogenicity. Nat Mach Intell. 2023;5:861-872. doi:10.1038/s42256-023-00694-6

8. Kim J, Jung N, Lee J, Cho NH, Noh J, Seok C. T-SCAPE: T cell immunogenicity scoring via cross-domain aided predictive engine. Sci Adv. 2025;11(49):eadz8759. doi:10.1126/sciadv.adz8759

9. Gfeller D, Schmidt J, Croce G, Guillaume P, Bobisse S, Genolet R, et al. Improved predictions of antigen presentation and TCR recognition with MixMHCpred2.2 and PRIME2.0 reveal potent SARS-CoV-2 CD8+ T-cell epitopes. Cell Syst. 2023;14(1):72-83.e5. doi:10.1016/j.cels.2022.12.002

10. Buckley PR, Lee CH, Ma R, Woodhouse I, Woo J, Tsvetkov VO, et al. Evaluating performance of existing computational models in predicting CD8+ T cell pathogenic epitopes and cancer neoantigens. Brief Bioinform. 2022;23(3):bbac141. doi:10.1093/bib/bbac141

11. O'Donnell TJ, Rubinsteyn A, Laserson U. MHCflurry 2.0: improved pan-allele prediction of MHC class I-presented peptides by incorporating antigen processing. Cell Syst. 2020;11(1):42-48.e7. doi:10.1016/j.cels.2020.06.010

12. Rives A, Meier J, Sercu T, Goyal S, Lin Z, Liu J, et al. Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. Proc Natl Acad Sci USA. 2021;118(15):e2016239118. doi:10.1073/pnas.2016239118

13. Bagaev DV, Vroomans RMA, Samir J, Stervbo U, Rius C, Dolton G, et al. VDJdb in 2019: database extension, new analysis infrastructure and a T-cell receptor motif compendium. Nucleic Acids Res. 2020;48(D1):D1057-D1062. doi:10.1093/nar/gkz874

14. Mamrosh JL, David-Fung ES, Korber BTM, Brander C, Barouch D, de Boer R, et al. HIV Molecular Immunology 2025. Los Alamos, NM: Los Alamos National Laboratory, Theoretical Biology and Biophysics; 2025. LA-UR-25-30629. doi:10.2172/3007488

15. Gonzalez-Galarza FF, McCabe A, Santos EJMD, Jones J, Takeshita L, Ortega-Rivera ND, et al. Allele frequency net database (AFND) 2020 update: gold-standard data classification, open access genotype data and new query tools. Nucleic Acids Res. 2020;48(D1):D783-D788. doi:10.1093/nar/gkz1029

16. Chowell D, Krishna S, Becker PD, Cocita C, Shu J, Tan X, et al. TCR contact residue hydrophobicity is a hallmark of immunogenic CD8+ T cell epitopes. Proc Natl Acad Sci USA. 2015;112(14):E1754-E1762. doi:10.1073/pnas.1500973112

17. Kyte J, Doolittle RF. A simple method for displaying the hydropathic character of a protein. J Mol Biol. 1982;157(1):105-132. doi:10.1016/0022-2836(82)90515-0

18. Zamyatnin AA. Protein volume in solution. Prog Biophys Mol Biol. 1972;24:107-123. doi:10.1016/0079-6107(72)90005-3

19. Lin Z, Akin H, Rao R, Hie B, Zhu Z, Lu W, et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. Science. 2023;379(6637):1123-1130. doi:10.1126/science.ade2574

20. Pedregosa F, Varoquaux G, Gramfort A, Michel V, Thirion B, Grisel O, et al. Scikit-learn: machine learning in Python. J Mach Learn Res. 2011;12:2825-2830.

21. Chen T, Guestrin C. XGBoost: a scalable tree boosting system. In: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining; 2016 Aug 13-17; San Francisco, CA. New York: ACM; 2016. p. 785-794. doi:10.1145/2939672.2939785

22. Paszke A, Gross S, Massa F, Lerer A, Bradbury J, Chanan G, et al. PyTorch: an imperative style, high-performance deep learning library. In: Advances in Neural Information Processing Systems 32. Red Hook, NY: Curran Associates; 2019. p. 8024-8035.

23. Li G, Iyer B, Prasath VBS, Ni Y, Salomonis N. DeepImmuno: deep learning-empowered prediction and generation of immunogenic peptides for T-cell immunity. Brief Bioinform. 2021;22(6):bbab160. doi:10.1093/bib/bbab160

24. Saito T, Rehmsmeier M. The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLoS One. 2015;10(3):e0118432. doi:10.1371/journal.pone.0118432

25. DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. Biometrics. 1988;44(3):837-845. doi:10.2307/2531595

26. Rasmussen M, Fenoy E, Harndahl M, Kristensen AB, Nielsen IK, Nielsen M, et al. Pan-specific prediction of peptide-MHC class I complex stability, a correlate of T cell immunogenicity. J Immunol. 2016;197(4):1517-1524. doi:10.4049/jimmunol.1600582

27. Marzella DF, Parizi FM, van Tilborg D, Renaud N, Koelman JFM, Hekkelman ML, de Ridder D, Abreu R, de Bruijn R, Xue LC, Bonvin AMJJ. PANDORA: A Fast, Anchor-Restrained Modelling Protocol for Peptide:MHC Complexes. Front Immunol. 2022;13:878762. doi:10.3389/fimmu.2022.878762
