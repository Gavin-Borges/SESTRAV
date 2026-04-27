# Biological Sanity Filters & Output Validation

This document describes the sanity checks applied to SESTRAV pipeline outputs and known limitations.

## Automated Checks (Applied During Pipeline Run)

### Stage 1: Peptide Generation
- **Standard amino acids only**: All generated peptides contain only the 20 canonical amino acids. Non-standard residues (U, X, B, Z, J) would be rejected by downstream feature computation.
- **Length filtering**: Only 8–11-mer peptides are retained (MHC-I compatible range).
- **No duplicate peptides**: Each unique peptide appears exactly once per virus.

### Stage 2: MHC Binding Prediction
- **MHCflurry presentation score**: Peptides are scored by MHCflurry's antigen presentation model, which integrates binding affinity and processing likelihood.
- **Allele panel support**: Predictions are generated across the configured allele panel in `config.yaml` (10 HLA alleles in the canonical release configuration).

### Stage 3: Feature Extraction
- **Track-aware features computed**: Feature extraction supports both legacy 21-feature and canonical 30-feature model expectations.
- **Feature alignment verified**: Stage 4 checks model feature count and selects compatible columns.
- **Binding features**: Legacy 21-feature models exclude binding features from training, while canonical 30-feature models include binding-matrix-derived features.

### Stage 4: Immunogenicity Scoring
- **Score range**: All scores fall within [0, 1] (RF probability output).
- **Gold-standard recovery**: 15/15 known epitopes are present in the final output, confirmed by automated validation.

## Known Limitations & Honest Assessment

### 1. Score Saturation
**Finding**: ~0.3% of EBV peptides and ~0.2% of HPV peptides receive a score of exactly 1.0000.

**Impact**: These peptides are tied at rank 1, meaning the pipeline cannot distinguish among them. This affects ~69 EBV peptides and ~12 HPV peptides.

**Root cause**: The RF ensemble produces a probability of 1.0 when all 200 trees vote for the positive class. This typically happens for peptides with feature profiles that fall squarely in the "immunogenic" region of feature space.

**Mitigation**: For experimental follow-up, tie-breaking by binding score or secondary ranking is recommended. The `presentation_score` column is preserved in output for this purpose.

### 2. Rank Tie Density
**Finding**: Only 2,957 unique rank values exist among 21,002 EBV peptides (14.1% rank resolution). The largest tie group contains 380 peptides sharing the same rank.

**Root cause**: RF's discrete probability output (200 trees × binary vote = 201 possible probability values) limits granularity.

**Mitigation**: XGBoost produces finer-grained scores. For applications requiring strict ordering, XGBoost or ensemble averaging may be preferred.

### 3. 8-mer Overrepresentation in Top Rankings
**Finding**: 8-mers constitute 25% of the dataset but 46–50% of the top-100 ranked peptides.

**Root cause**: 8-mers have only 3 TCR contact positions (p4, p5, p6) vs. 4 positions for 9–11-mers (p4, p5, p6, p7). Features at absent positions are zero-filled. This effectively creates a distinct feature subspace for 8-mers.

**Biological consideration**: 8-mers are less common as natural MHC-I ligands. Their overrepresentation in top ranks should be treated with caution. The `peptide_length` feature partially accounts for this but may not fully correct the bias.

**Mitigation**: Length-stratified ranking (separate top lists per k-mer) could address this for experimental prioritization.

### 4. Binding-Score Decorrelation
**Finding**: Pearson correlation between `immunogenicity_score` and `presentation_score` is 0.019 (EBV) and 0.058 (HPV) — essentially zero.

**Interpretation**: This is *by design*. SESTRAV uses only sequence-derived physicochemical features for scoring. The near-zero correlation confirms the model captures orthogonal information to binding affinity. A combined score (e.g., binding × immunogenicity) would leverage both dimensions.

### 5. Gold-Standard Recovery Paradox
**Finding**: The binding-only baseline places 15/15 gold-standard epitopes in the top 10%, while SESTRAV RF places only 6/15.

**Interpretation**: All 15 gold-standard epitopes are known strong MHC binders (that's why they were discovered and characterized). Ranking by binding affinity naturally puts them at the top. SESTRAV correctly scores them as immunogenic (all have scores > 0.5) but does not rank them as highly because many other peptides also have high immunogenicity features. This is expected — the gold-standard set cannot evaluate SESTRAV's core proposition (distinguishing immunogenic from non-immunogenic among good binders) because it contains only positives.

## Validation Recommendations for Experimental Follow-Up

1. **Prioritize by combined score**: `rank_combined = immunogenicity_score × presentation_score` captures both dimensions
2. **Use length-stratified lists**: Report top candidates separately for 9-mers (most reliable) and 8/10/11-mers
3. **Cross-reference IEDB**: Check if top candidates have prior experimental data
4. **Verify protein source**: Confirm peptides originate from immunologically relevant proteins (e.g., latent/lytic antigens for EBV, E6/E7 for HPV)
