# SESTRAV Figure Captions

ASCII-only. Numbers cited match the plotted values.

## Figure 1. SESTRAV pipeline: dataset assembly to model evaluation.

Schematic of the SESTRAV workflow, from raw data acquisition through model
evaluation. Three primary data sources (IEDB T-cell assay records, VDJdb
TCR-pMHC pairs, and the LANL HIV database) feed the v5 dataset assembly stage,
which applies a quarantine train/test split, HLA allele normalization, and
injection of hard and viral decoy negatives, yielding 35,597 active records
(51,185 total). Each peptide is then represented by three complementary feature
groups: physicochemical descriptors plus peptide length, 10 per-allele binding
scores computed with MHCflurry, and ESM-2 (t12) protein-language-model
embeddings. These features train two model families: the production random
forest (RF mode-31) and a graph neural network (GNN). Trained models are
assessed under three evaluation paradigms: within-virus cross-validation,
leave-one-virus-out (LOO) cross-virus generalization, and an external-tool
benchmark. Arrows indicate data flow (directed acyclic graph).

## Figure 2. Leave-one-virus-out (LOO) cross-virus generalization versus within-virus cross-validation.

AUC-ROC of the SESTRAV RF mode-31 model on the v5 dataset (IEDB
assay-confirmed clean test partitions), shown per pathogen for two evaluation
paradigms. Blue bars: within-virus cross-validation, where training and test
peptides come from the same pathogen. Red bars: leave-one-virus-out, where the
target pathogen is entirely withheld from training and used only for testing.
The dashed horizontal line marks random-chance performance (AUC-ROC = 0.5).
Pathogens are ordered by descending LOO AUC-ROC. Across all nine pathogens,
LOO performance is uniformly lower than within-virus CV, and for six of nine
pathogens LOO falls at or below chance, indicating that within-virus predictive
signal does not transfer to unseen pathogens. HIV-1 is a notable case: strong
within-virus performance (AUC-ROC = 0.894) collapses to anti-predictive
performance under LOO (AUC-ROC = 0.162), i.e. below chance.

Per-pathogen values plotted (within-virus CV / LOO AUC-ROC):
CMV 0.819 / 0.633; HBV 0.708 / 0.556; HCV 0.575 / 0.528; EBV 0.790 / 0.496;
IAV 0.856 / 0.488; HPV 0.561 / 0.468; SARS-CoV-2 0.699 / 0.462;
DENV 0.859 / 0.372; HIV-1 0.894 / 0.162.
