---
title: "Mechanistic basis of per-virus success and failure in cross-pathogen CD8+ T-cell immunogenicity prediction"
subtitle: "A molecular-immunology companion to the SESTRAV leave-one-virus-out benchmark, drafted for integration into Section 4 (Discussion)"
status: discussion draft - intended to be merged into docs/paper.md Section 4 (Interpretation / Discussion). Inline citations are author-year for portability; a Vancouver reference list follows. References already present in the manuscript's numbered bibliography are flagged in square brackets (e.g. [manuscript ref 16]) so they can be collapsed onto the existing numbers during copy-editing rather than duplicated.
last_updated: 2026-10-10
---

# Mechanistic interpretation of the per-virus LOO results

## 4.5 Why binding-centred features succeed and fail unevenly across viral families

The leave-one-virus-out (LOO) benchmark reported in Section 3 produces a pattern that
is, at first reading, paradoxical: a feature set dominated by MHC class I presentation
scores (55.8% of Random Forest importance; Section 4.2) ranks peptides well within a
virus yet collapses to a mean cross-virus AUC-ROC of 0.463, with one virus (HIV-1)
driven below chance to 0.162 and another (HPV) to an active failure at 0.468, while a
third (CMV, 0.633) is rescued specifically when its herpesvirus relative EBV is present
in the training background. This section argues that the pattern is not a numerical
accident of the classifier but a faithful readout of four distinct pieces of CD8+
T-cell biology, each of which decouples *peptide-MHC binding* from *functional
immunogenicity* in a virus-family-specific way. The decoupling is the central
immunological fact of the field: a peptide must bind MHC class I to be presented, but
binding is necessary and not sufficient for a productive T-cell response, because the
peptide must also be liberated by the proteasome and transported by TAP, must be
displayed at sufficient surface density on an antigen-presenting or infected cell, must
engage a T-cell receptor (TCR) drawn from a repertoire that has not been deleted,
anergised, escaped, or exhausted, and must do so with kinetics that cross the threshold
for productive signalling (Rock and Goldberg, 1999 [manuscript ref 5]; Yewdell and
Bennink, 1999; Calis et al., 2013 [manuscript ref 6]). A presentation-heavy model
learns the first step and is structurally blind to the remainder; each virus below
exposes a different one of the remaining steps as its dominant mode of failure.

A useful organising principle is the positional division of labour within the
peptide-MHC complex. The MHC anchor residues (canonically P2 and the C-terminus for
HLA class I) are read by the binding predictor and determine groove occupancy, whereas
the solvent-exposed, TCR-facing residues (approximately P4-P8 in a 9-mer, the exact
positions SESTRAV encodes following Chowell et al., 2015 [manuscript ref 16]) determine
whether a presented peptide is actually recognised. Any biological process that modifies
immunogenicity while leaving the anchors intact - viral escape mutation at TCR-contact
positions, immunodominance hierarchies, chronic exhaustion, or antigen-presentation
downregulation - will move a peptide's functional label without moving its binding
score, and will therefore be invisible to, or actively misread by, a binding-dominated
feature set. This is the unifying mechanism behind rank inversion, and it is strongest
precisely where immune-driven sequence evolution and immunoregulation are most intense.

## 4.6 HIV-1: rank inversion as the convergence of escape, immunodomination, exhaustion, and antigen-presentation evasion

HIV-1 is the extreme case in the benchmark (LOO AUC-ROC 0.162; real-negative-only
within-CV AUC-ROC 0.432; transfer gap 0.269, the largest of any virus; Tables 3 and 3b).
The model's assay-confirmed HIV-1 negatives are *enriched* for high-affinity,
B-restricted binders (higher mean MHCflurry scores for HLA-B\*08:01, B\*27:05 and B\*35:01
in confirmed-negative than in confirmed-positive records; Section 4.2), so a classifier
whose prior equates strong binding with immunogenicity does not merely fail on HIV-1 - it
inverts. Four non-exclusive mechanisms generate exactly this population of
"binds-well-but-negative" peptides, and all four are unusually pronounced in HIV-1.

**Anchor-preserving, TCR-contact-ablating escape.** HIV-1 replicates with an
error-prone reverse transcriptase under relentless CD8+ selection, and the dominant form
of escape fixes mutations that abrogate TCR recognition while preserving MHC anchor
residues, so that the variant peptide is still presented but is no longer seen by the
cognate repertoire (Goulder and Watkins, 2004; Goulder and Watkins, 2008). Because the
epitope remains a high-affinity ligand, a presentation predictor continues to score it
positive long after it has become functionally silent in the circulating population; the
immunogenicity label flips at the TCR-contact residues the binding score does not read.
Escape can also act upstream of the groove, by mutating proteasomal cleavage sites or
flanking residues so that a still-bindable epitope is no longer liberated or transported
(Draenert et al., 2004), and the aggregate footprint of these processes is large enough
that HLA-associated escape polymorphisms are a major driver of HIV-1 sequence diversity
at the population level (Allen et al., 2005). The LANL-derived HIV-1 positives and the
IEDB-derived HIV-1 negatives therefore sample a moving target in which binding affinity
and immunogenicity have been experimentally decorrelated by the virus itself.

**Immunodominance hierarchies and immunodomination.** Not every presented high-affinity
epitope recruits a response, because responses are hierarchically organised: a handful of
immunodominant specificities suppress the priming and expansion of subdominant ones
through competition for antigen, for antigen-presenting-cell access, and through direct
immunodomination (Yewdell and Bennink, 1999). In HIV-1 the hierarchy is also
protein-biased in a way that is clinically consequential - Gag-specific responses
associate with lower viraemia whereas Env-specific responses do not, despite comparable
presentation (Kiepiela et al., 2007) - so many perfectly presentable Env and accessory-
protein peptides are genuine non-responders not because they cannot be displayed but
because the repertoire is focused elsewhere. A binding model cannot represent this
competition and will score dominated subdominant epitopes as false positives.

**Chronic antigen exposure and T-cell exhaustion.** Untreated chronic HIV-1 infection
drives progressive CD8+ exhaustion, with upregulation of PD-1 and other inhibitory
receptors, loss of proliferative and cytokine-producing capacity, and eventual deletion
(Day et al., 2006; Trautmann et al., 2006; Wherry, 2011). ELISPOT and intracellular
cytokine staining score *function* (IFN-gamma, TNF, degranulation), so an exhausted or
deleted specificity registers as a functional negative even when its cognate epitope is
abundantly presented. The assay thus labels as non-immunogenic peptides that are both
well bound and, in a naive host, would be immunogenic - a state dependence that no
peptide-intrinsic feature can capture.

**Nef- and Vpu-mediated antigen-presentation downregulation.** HIV-1 actively collapses
the surface density of the very complexes a binding score assumes are displayed. Nef
redirects HLA-A and HLA-B molecules to the endolysosomal pathway and downregulates them
from the infected-cell surface (Schwartz et al., 1996; Collins et al., 1998), while
selectively sparing HLA-C and HLA-E to avoid licensing NK-cell killing (Cohen et al.,
1999); HLA-C is subsequently downregulated by Vpu (Apps et al., 2016). The immunological
consequence is a systematic, allele-specific gap between predicted presentation and
realised surface display: HLA-A- and HLA-B-restricted epitopes that a binding predictor
scores highly may never reach the density required to trigger a TCR on an infected cell,
which contributes to HLA-B-restricted peptides appearing disproportionately in the
assay-confirmed-negative set.

**Methodological corollary.** These four mechanisms jointly predict the direction of the
IEDB selection bias the manuscript already flags (Section 4.2): HIV-1 immunogenicity
studies preferentially synthesise and test high-affinity, B-restricted predicted binders
and then document non-response, precisely because escape, immunodomination, exhaustion,
and Nef/Vpu evasion make "strong binder, no response" the biologically interesting and
common outcome in this virus. The HIV-1 negative set is therefore not a random draw from
peptide space but an adversarially curated collection of binding-score false positives,
which is why it is "unlearnable from the remaining pool" under LOO and inverts a binding-
dominated model. We reiterate the manuscript's statistical caution - the within-CV
inversion rests on n=60 assay-confirmed negatives and its bootstrap interval clears
chance by under 0.005 - so the mechanistic account above explains the *direction* and
*reproducibility* of the effect across both the within-CV (0.432) and LOO (0.162)
estimates rather than asserting a precisely calibrated magnitude.

## 4.7 HPV: why presentation-heavy algorithms fail on a non-cytolytic, antigen-sequestering tumour virus

HPV is an active generalization failure in both paradigms (LOO 0.468; within-CV 0.482;
no decoys in its negative set, so neither value is decoy-inflated; n=137 real negatives).
Unlike HIV-1 the failure is not an inversion but a loss of signal, and its roots are the
antigen biology of a small double-stranded DNA tumour virus whose immunogenic targets are
expressed, processed, and surveyed under conditions a peptide-level presentation model
never sees.

**E6/E7 versus L1/L2: expression context dominates immunogenicity.** The two antigen
classes that dominate HPV CD8 datasets are biologically opposite. The structural capsid
proteins L1 and L2 are expressed only late, exclusively in terminally differentiated
suprabasal keratinocytes at the epithelial surface, in an immune-privileged compartment
with no viremia, no cytolysis, and minimal inflammation - an evolved strategy of
stealth persistence that keeps structural-antigen load, and therefore structural-antigen
priming, low (Stanley, 2006; Westrich et al., 2017). The oncoproteins E6 and E7, by
contrast, are constitutively expressed and functionally required in transformed cells,
making them the rational therapeutic-vaccine targets (van der Burg and Melief, 2011;
Kenter et al., 2009; Trimble et al., 2015). A model scoring peptide-MHC presentation
cannot represent this axis at all: an L1 peptide and an E7 peptide of identical predicted
affinity have radically different real-world immunogenicity because of *when, where, and
at what level the source protein is expressed and surveyed*, not because of any property
of the 9-mer. The feature set is orthogonal to the variable that actually governs the
HPV label.

**Active interferon and antigen-presentation evasion.** HPV does not merely express its
oncoproteins quietly; E6 and E7 dismantle innate sensing and presentation. E7 binds and
antagonises the cGAS-STING DNA-sensing pathway (Lau et al., 2015), E6/E7 suppress type I
interferon induction and signalling, and HPV16 E5 retains MHC class I in the Golgi and
reduces its surface transport (Ashrafi et al., 2005; reviewed in Westrich et al., 2017).
The net effect is that even constitutively expressed E6/E7 epitopes are presented at low
surface density in an interferon-starved microenvironment, so predicted presentation
again overstates realised display.

**The HPV-associated tumour microenvironment.** Most HPV CD8 data are generated in
persistent infection or in HPV-associated malignancy (cervical and oropharyngeal
carcinoma), where the microenvironment is immunosuppressive: regulatory T-cell
infiltration, myeloid-derived suppressor cells, and PD-1/PD-L1-mediated adaptive immune
resistance blunt effector function (Lyford-Pike et al., 2013). Functional read-outs
(ELISPOT, ICS) performed on cells drawn from, or primed in, this environment can score a
competent, well-presented epitope as a functional negative. Notably, the same biology
runs in the opposite direction in a subset of HPV+ oropharyngeal cancers, where strong
intratumoural T-cell infiltration tracks with favourable prognosis - underscoring that
the HPV immunogenicity label is set by host and microenvironmental state, not by peptide
chemistry, and is therefore not transferable from an RNA-virus-trained feature
distribution.

**Cohort and assay bias, and physicochemical distinctiveness.** Because HPV epitope
cohorts are enriched for E6/E7 in therapeutic-vaccine and tumour-infiltrating-lymphocyte
studies and for L1/L2 in prophylactic and natural-history studies, the positive and
negative classes are partly defined by study design rather than by intrinsic
immunogenicity, injecting cohort-specific structure that does not generalise. Layered on
top is a representational mismatch the manuscript already names (Section 4.2): the
papillomavirus proteome is compositionally distinct from the RNA-virus-dominated training
corpus, so physicochemical descriptors learned elsewhere do not merely fail to help but
can actively mislead, which is consistent with the below-chance LOO value. The remedy is
not a better binding feature but dedicated, context-annotated HPV negative data, as the
manuscript's Future Directions already propose.

## 4.8 Herpesviridae: why CMV generalizes, and why EBV in the training background is what rescues it

CMV achieves the best corrected LOO transfer in the panel (0.633, on a reliable 272 real
negatives; honest within-CV 0.693), and it does so specifically when EBV - its only
Herpesviridae relative among the nine targets - is present in the training background.
This is the one clean positive transfer result, and its mechanism is instructive because
it is *not* straightforward sequence conservation.

**Shared immunodominance architecture, not shared sequence.** CMV (a
betaherpesvirus) and EBV (a gammaherpesvirus) are deeply divergent in primary
sequence; their proteomes are not meaningfully alignable at the level that would let a
sequence-identity model transfer an epitope directly. What they share is an
*architecture* of the CD8 response. Both are large (>150 ORF) double-stranded DNA viruses
that establish lifelong infection with recurring lytic and latent phases, and both
elicit exceptionally broad, high-magnitude, and reproducible CD8 responses focused on
abundant structural/tegument and immediate-early proteins - pp65 and IE-1 for CMV
(Sylwester et al., 2005; Klenerman and Oxenius, 2016), and the lytic immediate-early and
early antigens (e.g. BZLF1, BMLF1) together with the EBNA3 latency antigens for EBV
(Rickinson and Moss, 1997; Hislop et al., 2007). CMV in particular drives "memory
inflation," in which these specificities accumulate to very large, stable frequencies
over the host's lifetime (Klenerman and Oxenius, 2016). Because SESTRAV encodes
physicochemical properties at TCR-contact positions rather than raw sequence, what
transfers from EBV to CMV is the *statistical regularity of herpesvirus immunodominant
epitopes* - their characteristic TCR-contact hydrophobicity and charge patterning (the
Chowell et al., 2015 hallmark [manuscript ref 16]), their length distribution, and their
restriction skew - rather than any specific peptide. This is why a physicochemical,
allele-blind model can borrow from one herpesvirus to score another even without
alignable sequence.

**Overlapping HLA presentation patterns.** Both viruses lean heavily on HLA-B-restricted
immunodominant epitopes drawn from the same common supertypes the SESTRAV binding panel
covers, so the presentation sub-space EBV populates in training overlaps the sub-space
CMV occupies at test. The manuscript is appropriately cautious that this could be the
whole story - "shared HLA restriction pattern coverage" rather than "true phylogenetic
transfer" (Section 3.4) - and the two explanations are not mutually exclusive: shared
restriction is the channel through which shared immunodominance architecture becomes
visible to an allele-blind physicochemical model. The honest reading is that EBV supplies
the only training examples whose immunodominance geometry and restriction footprint
resemble the held-out CMV epitopes, and removing EBV would be predicted to erase most of
CMV's above-chance transfer. This also explains the asymmetry with the RNA viruses below:
no other target virus has a comparably close relative in the panel, so no comparable
rescue is available to them.

**A caution on cross-reactivity.** Herpesvirus T-cell biology also features genuine
heterologous cross-reactivity - EBV-specific CD8 clones can cross-recognise influenza
epitopes and shape the response during acute infectious mononucleosis (Clute et al.,
2005; Selin et al., 2006) - but this is a property of individual TCR-pMHC pairs, not the
population-average, peptide-level signal SESTRAV learns. The CMV rescue should therefore
be attributed to shared immunodominance/restriction architecture rather than to
molecular-mimicry cross-reactivity, which operates at a level the model does not
represent and would require paired-repertoire data (e.g. VDJdb) to capture.

## 4.9 Respiratory RNA viruses: why physicochemical transfer fails with no family relative in training

Influenza A virus (IAV, LOO 0.488) and SARS-CoV-2 (0.462) both sit at or just below
chance, and critically neither has a family relative in the nine-virus panel - there is
no second orthomyxovirus and no second coronavirus to supply the kind of
architecture-matched training examples EBV supplies for CMV. Their failure is the control
condition that proves the CMV result: when the one transferable signal (a same-family
immunodominance/restriction template) is absent, a physicochemical-plus-binding feature
set has nothing virus-appropriate to generalise from, and it defaults to the pan-training
prior that strong binders are immunogenic - a prior these viruses do not obey.

**Proteome composition and feature-distribution shift.** Immunogenicity is not a pure
function of a peptide's local physicochemistry; it depends on the abundance, kinetics,
and processing of the source protein, which differ systematically between viral families.
Human CD8 responses to IAV are dominated by the internal, conserved proteins - NP, M1,
and the polymerase subunits - rather than the variable surface glycoproteins (Assarsson
et al., 2008; Grant et al., 2016), and SARS-CoV-2 responses are broad and distributed
across ORF1ab, nucleocapsid, membrane, and spike with pronounced inter-individual and
HLA-dependent immunodominance (Grifoni et al., 2020; Tarke et al., 2021). The
physicochemical and length statistics of epitopes drawn from these proteomes differ from
those of the DNA-virus- and retrovirus-dominated training corpus, so the model operates
under covariate shift: descriptors calibrated on the training distribution are evaluated
on peptides drawn from a different one, with no in-family anchor to recalibrate them.

**Heavy glycosylation and processing of the surface antigens.** The principal surface
antigens of both viruses - influenza hemagglutinin and the SARS-CoV-2 spike - are heavily
N-glycosylated, which alters proteasomal processing and epitope liberation in ways a
sequence-derived physicochemical score cannot see, further weakening the link between a
predicted ligand and a processed, presented, immunogenic peptide.

**Active innate and antigen-presentation antagonism.** Both viruses encode potent
interferon and host-shutoff antagonists that depress presentation globally: influenza NS1
suppresses type I interferon induction (Hale et al., 2008), while SARS-CoV-2 Nsp1 shuts
down host translation (Thoms et al., 2020) and ORF8 downregulates MHC class I (Zhang et
al., 2021). As for HIV-1 and HPV, these processes lower realised surface display relative
to predicted presentation, but here there is no same-family training example in which the
model could have learned the family-specific offset between the two.

**Consistency with prior external evidence.** This outcome is not idiosyncratic to
SESTRAV: an independent benchmark of nine published predictors on assay-confirmed
SARS-CoV-2 CD8 epitopes found that none substantially outperformed random or improved
appreciably on HLA-ligand prediction (Buckley et al., 2022 [manuscript ref 10]). The LOO
result generalises that single-pathogen finding to a controlled two-virus comparison and
assigns it a mechanism: zero-shot physicochemical transfer to a respiratory RNA virus
fails not because the feature set is uninformative in general, but because immunogenicity
is governed by proteome-, processing-, and host-state-level variables that are only
learnable from a family-matched exemplar, which these viruses lack in the panel.

## 4.10 Synthesis

Across all four cases the same structural limitation expresses itself through four
different immunological channels. A binding-dominated feature set learns groove occupancy
at the MHC anchors and is blind to everything that sets immunogenicity at the TCR-contact
face and downstream of presentation. Where the virus actively decorrelates binding from
recognition - HIV-1 escape, immunodomination, exhaustion, and Nef/Vpu downregulation -
the model inverts; where the label is set by expression context, interferon state, and
the tumour microenvironment - HPV - the model loses signal; where a family-matched
exemplar supplies the immunodominance and restriction architecture - CMV rescued by EBV -
the model transfers; and where no such exemplar exists - the respiratory RNA viruses -
transfer collapses to the pan-training binding prior. The unifying prediction for model
development is therefore specific rather than generic: cross-virus immunogenicity transfer
will improve not primarily from better presentation features, but from (i) features that
read the TCR-contact determinants of recognition and escape, (ii) pMHC stability as a
correlate of immunogenicity orthogonal to presentation probability (Harndahl et al.,
2012; Rasmussen et al., 2016 [manuscript ref 26]), and (iii) family-matched training
exemplars and assay-confirmed, binding-positive negative sets that let the model learn,
per family, the offset between what is presented and what is seen.

---

# References (Vancouver; author-year used inline above)

> Note for copy-editing: entries marked **[= manuscript ref N]** already appear in the
> `docs/paper.md` bibliography and should be collapsed onto the existing number rather than
> duplicated. The remainder are new citations to be appended to the manuscript reference
> list and renumbered into sequence.

1. Rock KL, Goldberg AL. Degradation of cell proteins and the generation of MHC class I-presented peptides. Annu Rev Immunol. 1999;17:739-779. doi:10.1146/annurev.immunol.17.1.739 **[= manuscript ref 5]**

2. Calis JJA, Maybeno M, Greenbaum JA, Weiskopf D, De Silva AD, Sette A, et al. Properties of MHC class I presented peptides that enhance immunogenicity. PLoS Comput Biol. 2013;9(10):e1003266. doi:10.1371/journal.pcbi.1003266 **[= manuscript ref 6]**

3. Chowell D, Krishna S, Becker PD, Cocita C, Shu J, Tan X, et al. TCR contact residue hydrophobicity is a hallmark of immunogenic CD8+ T cell epitopes. Proc Natl Acad Sci USA. 2015;112(14):E1754-E1762. doi:10.1073/pnas.1500973112 **[= manuscript ref 16]**

4. Buckley PR, Lee CH, Ma R, Woodhouse I, Woo J, Tsvetkov VO, et al. Evaluating performance of existing computational models in predicting CD8+ T cell pathogenic epitopes and cancer neoantigens. Brief Bioinform. 2022;23(3):bbac141. doi:10.1093/bib/bbac141 **[= manuscript ref 10]**

5. Rasmussen M, Fenoy E, Harndahl M, Kristensen AB, Nielsen IK, Nielsen M, et al. Pan-specific prediction of peptide-MHC class I complex stability, a correlate of T cell immunogenicity. J Immunol. 2016;197(4):1517-1524. doi:10.4049/jimmunol.1600582 **[= manuscript ref 26]**

6. Yewdell JW, Bennink JR. Immunodominance in major histocompatibility complex class I-restricted T lymphocyte responses. Annu Rev Immunol. 1999;17:51-88. doi:10.1146/annurev.immunol.17.1.51

7. Goulder PJR, Watkins DI. HIV and SIV CTL escape: implications for vaccine design. Nat Rev Immunol. 2004;4(8):630-640. doi:10.1038/nri1417

8. Goulder PJR, Watkins DI. Impact of MHC class I diversity on immune control of immunodeficiency virus infection. Nat Rev Immunol. 2008;8(8):619-630. doi:10.1038/nri2357

9. Draenert R, Le Gall S, Pfafferott KJ, Leslie AJ, Chetty P, Brander C, et al. Immune selection for altered antigen processing leads to cytotoxic T lymphocyte escape in chronic HIV-1 infection. J Exp Med. 2004;199(7):905-915. doi:10.1084/jem.20031982

10. Allen TM, Altfeld M, Geer SC, Kalife ET, Moore C, O'Sullivan KM, et al. Selective escape from CD8+ T-cell responses represents a major driving force of human immunodeficiency virus type 1 (HIV-1) sequence diversity and reveals constraints on HIV-1 evolution. J Virol. 2005;79(21):13239-13249. doi:10.1128/JVI.79.21.13239-13249.2005

11. Kiepiela P, Ngumbela K, Thobakgale C, Ramduth D, Honeyborne I, Moodley E, et al. CD8+ T-cell responses to different HIV proteins have discordant associations with viral load. Nat Med. 2007;13(1):46-53. doi:10.1038/nm1520

12. Day CL, Kaufmann DE, Kiepiela P, Brown JA, Moodley ES, Reddy S, et al. PD-1 expression on HIV-specific T cells is associated with T-cell exhaustion and disease progression. Nature. 2006;443(7109):350-354. doi:10.1038/nature05115

13. Trautmann L, Janbazian L, Chomont N, Said EA, Gimmig S, Bessette B, et al. Upregulation of PD-1 expression on HIV-specific CD8+ T cells leads to reversible immune dysfunction. Nat Med. 2006;12(10):1198-1202. doi:10.1038/nm1482

14. Wherry EJ. T cell exhaustion. Nat Immunol. 2011;12(6):492-499. doi:10.1038/ni.2035

15. Schwartz O, Marechal V, Le Gall S, Lemonnier F, Heard JM. Endocytosis of major histocompatibility complex class I molecules is induced by the HIV-1 Nef protein. Nat Med. 1996;2(3):338-342. doi:10.1038/nm0396-338

16. Collins KL, Chen BK, Kalams SA, Walker BD, Baltimore D. HIV-1 Nef protein protects infected primary cells against killing by cytotoxic T lymphocytes. Nature. 1998;391(6665):397-401. doi:10.1038/34929

17. Cohen GB, Gandhi RT, Davis DM, Mandelboim O, Chen BK, Strominger JL, et al. The selective downregulation of class I major histocompatibility complex proteins by HIV-1 protects HIV-infected cells from NK cells. Immunity. 1999;10(6):661-671. doi:10.1016/S1074-7613(00)80065-5

18. Apps R, Del Prete GQ, Chatterjee P, Lara A, Brumme ZL, Brockman MA, et al. HIV-1 Vpu mediates HLA-C downregulation. Cell Host Microbe. 2016;19(5):686-695. doi:10.1016/j.chom.2016.04.005

19. Stanley M. Immune responses to human papillomavirus. Vaccine. 2006;24(Suppl 1):S16-S22. doi:10.1016/j.vaccine.2005.09.002

20. Westrich JA, Warren CJ, Pyeon D. Evasion of host immune defenses by human papillomavirus. Virus Res. 2017;231:21-33. doi:10.1016/j.virusres.2016.11.023

21. van der Burg SH, Melief CJM. Therapeutic vaccination against human papillomavirus induced malignancies. Curr Opin Immunol. 2011;23(2):252-257. doi:10.1016/j.coi.2010.12.010

22. Kenter GG, Welters MJP, Valentijn ARPM, Lowik MJG, Berends-van der Meer DMA, Vloon APG, et al. Vaccination against HPV-16 oncoproteins for vulvar intraepithelial neoplasia. N Engl J Med. 2009;361(19):1838-1847. doi:10.1056/NEJMoa0810097

23. Trimble CL, Morrow MP, Kraynyak KA, Shen X, Dallas M, Yan J, et al. Safety, efficacy, and immunogenicity of VGX-3100, a therapeutic synthetic DNA vaccine targeting human papillomavirus 16 and 18 E6 and E7 proteins for cervical intraepithelial neoplasia 2/3: a randomised, double-blind, placebo-controlled phase 2b trial. Lancet. 2015;386(10008):2078-2088. doi:10.1016/S0140-6736(15)00239-1

24. Lau L, Gray EE, Brunette RL, Stetson DB. DNA tumor virus oncogenes antagonize the cGAS-STING DNA-sensing pathway. Science. 2015;350(6260):568-571. doi:10.1126/science.aab3291

25. Ashrafi GH, Haghshenas MR, Marchetti B, O'Brien PM, Campo MS. E5 protein of human papillomavirus type 16 selectively downregulates surface HLA class I. Int J Cancer. 2005;113(2):276-283. doi:10.1002/ijc.20558

26. Lyford-Pike S, Peng S, Young GD, Taube JM, Westra WH, Akpeng B, et al. Evidence for a role of the PD-1:PD-L1 pathway in immune resistance of HPV-associated head and neck squamous cell carcinoma. Cancer Res. 2013;73(6):1733-1741. doi:10.1158/0008-5472.CAN-12-2384

27. Sylwester AW, Mitchell BL, Edgar JB, Taormina C, Pelte C, Ruchti F, et al. Broadly targeted human cytomegalovirus-specific CD4+ and CD8+ T cells dominate the memory compartments of exposed subjects. J Exp Med. 2005;202(5):673-685. doi:10.1084/jem.20050882

28. Klenerman P, Oxenius A. T cell responses to cytomegalovirus. Nat Rev Immunol. 2016;16(6):367-377. doi:10.1038/nri.2016.38

29. Hislop AD, Taylor GS, Sauce D, Rickinson AB. Cellular responses to viral infection in humans: lessons from Epstein-Barr virus. Annu Rev Immunol. 2007;25:587-617. doi:10.1146/annurev.immunol.25.022106.141553

30. Rickinson AB, Moss DJ. Human cytotoxic T lymphocyte responses to Epstein-Barr virus infection. Annu Rev Immunol. 1997;15:405-431. doi:10.1146/annurev.immunol.15.1.405

31. Clute SC, Watkin LB, Cornberg M, Naumov YN, Sullivan JL, Luzuriaga K, et al. Cross-reactive influenza virus-specific CD8+ T cells contribute to lymphoproliferation in Epstein-Barr virus-associated infectious mononucleosis. J Clin Invest. 2005;115(12):3602-3612. doi:10.1172/JCI25078

32. Selin LK, Brehm MA, Naumov YN, Cornberg M, Kim SK, Clute SC, et al. Memory of mice and men: CD8+ T-cell cross-reactivity and heterologous immunity. Immunol Rev. 2006;211:164-181. doi:10.1111/j.0105-2896.2006.00394.x

33. Assarsson E, Bui HH, Sidney J, Zhang Q, Glenn J, Oseroff C, et al. Immunomic analysis of the repertoire of T-cell specificities for influenza A virus in humans. J Virol. 2008;82(24):12241-12251. doi:10.1128/JVI.01563-08

34. Grant EJ, Quinones-Parra SM, Clemens EB, Kedzierska K. Human influenza viruses and CD8+ T cell responses. Curr Opin Virol. 2016;16:132-142. doi:10.1016/j.coviro.2016.01.016

35. Grifoni A, Weiskopf D, Ramirez SI, Mateus J, Dan JM, Moderbacher CR, et al. Targets of T cell responses to SARS-CoV-2 coronavirus in humans with COVID-19 disease and unexposed individuals. Cell. 2020;181(7):1489-1501.e15. doi:10.1016/j.cell.2020.05.015

36. Tarke A, Sidney J, Kidd CK, Dan JM, Ramirez SI, Yu ED, et al. Comprehensive analysis of T cell immunodominance and immunoprevalence of SARS-CoV-2 epitopes in COVID-19 cases. Cell Rep Med. 2021;2(2):100204. doi:10.1016/j.xcrm.2021.100204

37. Hale BG, Randall RE, Ortin J, Jackson D. The multifunctional NS1 protein of influenza A viruses. J Gen Virol. 2008;89(Pt 10):2359-2376. doi:10.1099/vir.0.2008/004606-0

38. Thoms M, Buschauer R, Ameismeier M, Koepke L, Denk T, Hirschenberger M, et al. Structural basis for translational shutdown and immune evasion by the Nsp1 protein of SARS-CoV-2. Science. 2020;369(6508):1249-1255. doi:10.1126/science.abc8665

39. Zhang Y, Chen Y, Li Y, Huang F, Luo B, Yuan Y, et al. The ORF8 protein of SARS-CoV-2 mediates immune evasion through down-regulating MHC-I. Proc Natl Acad Sci USA. 2021;118(23):e2024202118. doi:10.1073/pnas.2024202118

40. Harndahl M, Rasmussen M, Roder G, Dalgaard Pedersen I, Sorensen M, Nielsen M, et al. Peptide-MHC class I stability is a better predictor than peptide affinity of CTL immunogenicity. Eur J Immunol. 2012;42(6):1405-1416. doi:10.1002/eji.201141774
