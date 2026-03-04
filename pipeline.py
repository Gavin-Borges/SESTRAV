from functions.stage1_peptide_generation import generate_peptides
from functions.stage2_mhc_binding_prediction import predict_binding
from functions.stage3_tcr_feature_extraction import extract_tcr_features
from functions.stage4_immunogenicity_scoring import score_immunogenicity,plot_immunogenicity_scores
import os

def run_pipeline(fasta_file):
    virus_name = fasta_file.replace(".fasta", "")
    fasta_path = os.path.join("data/proteomes", fasta_file)

    # Stage 1
    peptides_df = generate_peptides(fasta_path, virus_name)

    # Stage 2
    binding_df = predict_binding(peptides_df, virus_name)

    # Stage 3
    features_df = extract_tcr_features(binding_df, virus_name)

    # Stage 4
    ranked_df, model = score_immunogenicity(features_df, virus_name)

    # Plot immunogenicity scores
    plot_immunogenicity_scores(ranked_df, virus_name)

    print(f"Pipeline complete for {virus_name}\n")

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    for fasta_file in os.listdir("data/proteomes"):
        if fasta_file.endswith(".fasta"):
            run_pipeline(fasta_file)