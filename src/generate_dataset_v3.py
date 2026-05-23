"""
Script to generate the version 3 immunogenicity dataset.
Includes UPDATED_ files and HPV11 from 09_Data/ directory.
"""
import os
import pandas as pd
from src.iedb_data_loader import load_and_clean_iedb

def main():
    print("Generating SESTRAV immunogenicity dataset V3...")
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../09_Data'))
    
    # Load and clean all IEDB data including HPV11
    df = load_and_clean_iedb(data_dir, include_hpv11=True)
    
    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/immunogenicity_dataset_v3.csv'))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Dataset V3 saved to {out_path}")
    print(f"Total Peptides: {len(df)}")
    print(f"Positives: {df['label'].sum()} | Negatives: {(df['label'] == 0).sum()}")

if __name__ == "__main__":
    main()
