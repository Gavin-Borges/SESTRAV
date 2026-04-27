"""
SESTRAV Stage 2 — MHC Binding Prediction
Runs MHCflurry Class1PresentationPredictor against a configurable panel of
HLA alleles. Produces a standardized binding CSV consumable by downstream stages
regardless of binding backend.

In multi-allele mode, per-allele presentation scores are retained as separate
columns (bind_A0101 ... bind_B4402) for use with 30-feature models.
"""

import re
from mhcflurry import Class1PresentationPredictor
import pandas as pd

DEFAULT_ALLELES = [
    "HLA-A*02:01", "HLA-A*01:01", "HLA-A*03:01", "HLA-A*24:02",
    "HLA-A*11:01", "HLA-B*07:02", "HLA-B*08:01", "HLA-B*27:05",
    "HLA-B*35:01", "HLA-B*44:02"
]


def _allele_to_col(allele):
    """Convert 'HLA-A*02:01' to 'bind_A0201'."""
    return 'bind_' + re.sub(r'[^A-Za-z0-9]', '', allele.replace('HLA-', ''))


def predict_binding(peptides_df, proteome_id, alleles=None):
    """
    Predict MHC-I presentation using MHCflurry for all unique peptides.

    For each peptide, predictions are made against every allele in the panel.
    The best-presenting allele per peptide is retained (highest presentation_score).

    Args:
        peptides_df: DataFrame with at least 'peptide' and 'protein_id' columns
        proteome_id: label used in output filename
        alleles: list of HLA allele strings (default: 10-allele panel)

    Returns:
        DataFrame with one row per peptide containing:
        peptide, allele, affinity, presentation_score,
        presentation_percentile, protein_id
    """
    if alleles is None:
        alleles = DEFAULT_ALLELES

    predictor = Class1PresentationPredictor.load()
    unique_peptides = peptides_df["peptide"].unique().tolist()

    print(f"[Stage 2] Running MHCflurry on {len(unique_peptides)} peptides × "
          f"{len(alleles)} alleles = {len(unique_peptides) * len(alleles)} predictions")

    # Class1PresentationPredictor.predict() treats an allele list as a genotype
    # and enforces a maximum of 6 alleles.  To support a 10-allele panel we
    # predict one allele at a time and concatenate the results.
    per_allele_frames = []
    for allele in alleles:
        pred_df = predictor.predict(
            peptides=unique_peptides,
            alleles=[allele],
            verbose=0
        )
        # Some MHCflurry outputs omit the allele column when a single allele
        # is passed; restore it so downstream pivoting is stable.
        if 'allele' not in pred_df.columns:
            pred_df['allele'] = allele
        per_allele_frames.append(pred_df)
        print(f"  [Stage 2]  {allele}: {len(pred_df)} predictions done")

    all_predictions = pd.concat(per_allele_frames, ignore_index=True)

    # Build per-allele wide-format columns (bind_A0101 ... bind_B4402)
    per_allele_wide = (
        all_predictions[['peptide', 'allele', 'presentation_score']]
        .copy()
    )
    per_allele_wide['bind_col'] = per_allele_wide['allele'].map(_allele_to_col)
    per_allele_pivot = per_allele_wide.pivot_table(
        index='peptide', columns='bind_col',
        values='presentation_score', aggfunc='first'
    ).reset_index()

    # Best-allele selection (legacy single binding_score behaviour)
    all_predictions = all_predictions.sort_values(
        'presentation_score', ascending=False
    )
    predictions = all_predictions.drop_duplicates('peptide', keep='first')
    predictions = predictions.reset_index(drop=True)

    if "protein_id" in peptides_df.columns:
        protein_map = (
            peptides_df[["peptide", "protein_id"]]
            .dropna(subset=["protein_id"])
            .drop_duplicates()
            .groupby("peptide")["protein_id"]
            .agg(lambda vals: sorted(set(vals)))
            .reset_index(name="protein_id_list")
        )
        protein_map["protein_id"] = protein_map["protein_id_list"].str[0]
        protein_map["protein_ids"] = protein_map["protein_id_list"].apply(
            lambda vals: "|".join(vals)
        )
        protein_map["source_protein_count"] = protein_map["protein_id_list"].apply(len)
        protein_map = protein_map.drop(columns=["protein_id_list"])
        predictions = predictions.merge(protein_map, on="peptide", how="left")

    # Merge per-allele columns onto the best-allele result
    predictions = predictions.merge(per_allele_pivot, on='peptide', how='left')

    output_path = f"results/{proteome_id}_binding.csv"
    predictions.to_csv(output_path, index=False)
    print(f"[Stage 2] Retained best allele for {len(predictions)} peptides "
          f"(+ {len(alleles)} per-allele columns)")
    return predictions
