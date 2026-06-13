"""
Script to generate the MHCflurry binding matrix for the v3 dataset.
"""
import os
import yaml
import pandas as pd
from src.features import BINDING_ALLELE_COLUMNS

def get_mhcflurry_predictions(peptides, alleles):
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    
    try:
        from mhcflurry import Class1PresentationPredictor
    except ImportError:
        print("MHCflurry not installed. Please install with `pip install mhcflurry`.")
        return None
    
    predictor = Class1PresentationPredictor.load()
    
    import itertools
    
    pep_list = list(peptides)
    print(f"Running predictions for {len(pep_list)} peptides × {len(alleles)} alleles...")
    
    all_pairs = list(itertools.product(pep_list, alleles))
    flat_peps = [p[0] for p in all_pairs]
    flat_alleles = [p[1] for p in all_pairs]
    
    chunk_size = 100000
    per_chunk_frames = []
    
    for i in range(0, len(flat_peps), chunk_size):
        pep_chunk = flat_peps[i:i+chunk_size]
        allele_chunk = flat_alleles[i:i+chunk_size]
        
        pred_df = predictor.predict(
            peptides=pep_chunk,
            alleles=allele_chunk,
            verbose=0
        )
        per_chunk_frames.append(pred_df)
        print(f"  Chunk {i//chunk_size + 1}: {len(pred_df)} predictions done")
        
    df_input_presentation = pd.concat(per_chunk_frames, ignore_index=True)
    
    # We want the 'presentation_score' column from predict_presentation
    # Group by peptide, then pivot_table alleles into columns
    pivot_df = df_input_presentation.pivot_table(index='peptide', columns='allele', values='presentation_score', aggfunc='mean')
    
    # Ensure all target columns exist and are named properly
    out_df = pd.DataFrame(index=pivot_df.index)
    
    for expected_col in BINDING_ALLELE_COLUMNS:
        # expected_col is like 'binding_A0201'
        allele_name = expected_col.replace('binding_', '')
        # formatting: MHCflurry sometimes uses HLA-A*02:01. The config.yaml has them as HLA-A*02:01
        formatted_allele = f"HLA-{allele_name[:1]}*{allele_name[1:3]}:{allele_name[3:]}"
        
        if formatted_allele in pivot_df.columns:
            out_df[expected_col] = pivot_df[formatted_allele]
        else:
            out_df[expected_col] = 0.0
            
    out_df.reset_index(inplace=True)
    return out_df

def main():
    print("Generating SESTRAV binding matrix V3...")
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/immunogenicity_dataset_v3.csv'))
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../config.yaml'))
    
    df = pd.read_csv(data_path)
    peptides = df['peptide'].unique()
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    alleles = config['alleles']
    print(f"Alleles from config: {alleles}")
    
    matrix_df = get_mhcflurry_predictions(peptides, alleles)
    
    if matrix_df is not None:
        out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/peptide_binding_matrix_v3.csv'))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        matrix_df.to_csv(out_path, index=False)
        print(f"Binding matrix V3 saved to {out_path}")
        print(f"Shape: {matrix_df.shape}")

if __name__ == "__main__":
    main()
