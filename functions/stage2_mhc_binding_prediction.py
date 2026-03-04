from mhcflurry import Class1PresentationPredictor

ALLELES = [
    "HLA-A*02:01",
    "HLA-A*01:01",
    "HLA-B*07:02",
    "HLA-B*08:01",
    "HLA-A*03:01",
    "HLA-A*24:02"
]

def predict_binding(peptides_df, virus_name):
    predictor = Class1PresentationPredictor.load()
    unique_peptides = peptides_df["peptide"].unique()

    predictions = predictor.predict(peptides=unique_peptides, alleles=ALLELES)

    # Merge back protein_id so we know which protein each peptide came from
    predictions = predictions.merge(peptides_df[["peptide", "protein_id"]], on="peptide", how="left")

    predictions.rename(columns={
        "affinity_nM": "affinity_nM",
        "presentation_score": "presentation_score",
        "presentation_percentile": "presentation_percentile"
    }, inplace=True)

    predictions.to_csv(f"results/{virus_name}_binding.csv", index=False)
    print(f"[Stage 2] Predicted MHC binding for {len(predictions)} peptide-allele pairs")
    return predictions