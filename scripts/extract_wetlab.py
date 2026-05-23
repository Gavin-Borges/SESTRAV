import pandas as pd
import os
from src.iedb_data_loader import GOLD_STANDARD_EPITOPES

def main():
    print("Extracting Wet-Lab Candidates for V3...")
    
    ebv_df = pd.read_csv('results/EBV_B95_8_panel8_ranked.csv')
    hpv_df = pd.read_csv('results/HPV16_18_panel8_ranked.csv')
    
    # Extract top 50 from each
    ebv_top = ebv_df.head(50).copy()
    ebv_top['source'] = 'EBV_pipeline_top50'
    
    hpv_top = hpv_df.head(50).copy()
    hpv_top['source'] = 'HPV_pipeline_top50'
    
    # Ensure they have a label column or some structure. The ranked CSV has peptide, immunogenicity_score
    
    # Also load the gold standard positives from the dataset if they have positive labels
    dataset = pd.read_csv('data/immunogenicity_dataset_v3.csv')
    gs_df = dataset[dataset['peptide'].isin(GOLD_STANDARD_EPITOPES)].copy()
    gs_df = gs_df[gs_df['label'] == 1].copy()
    
    # Only keep unique peptides in gs_df
    gs_df = gs_df.drop_duplicates(subset=['peptide'])
    
    # We just need peptide, score if available, and source
    gs_out = pd.DataFrame({
        'peptide': gs_df['peptide'],
        'immunogenicity_score': 1.0, # Dummy high score for GS
        'source': 'Gold_Standard_Positive'
    })
    
    final_df = pd.concat([
        ebv_top[['peptide', 'immunogenicity_score', 'source']],
        hpv_top[['peptide', 'immunogenicity_score', 'source']],
        gs_out
    ], ignore_index=True)
    
    # Drop duplicates just in case a GS is in the top 50
    final_df = final_df.drop_duplicates(subset=['peptide'], keep='first')
    
    out_path = 'results/wetlab_candidates_v3.csv'
    final_df.to_csv(out_path, index=False)
    print(f"Saved {len(final_df)} candidates to {out_path}")

if __name__ == "__main__":
    main()
