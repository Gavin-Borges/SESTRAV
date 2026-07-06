"""
Script to generate the version 3 immunogenicity dataset.
Includes UPDATED_ files and HPV11 from data/raw/iedb_exports/ directory.
"""

import os
from src.iedb_data_loader import load_and_clean_iedb
from src.core.config import SestravConfig
from src.core.feature_store import FeatureStore


def main():
    print("Generating SESTRAV immunogenicity dataset V3...")
    config = SestravConfig.load()
    store = FeatureStore(config.output_dir)

    # Data is now stored inside the repository
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/raw/iedb_exports"))

    # We might want to pass 'include_hpv11=True' or pull it from config, assuming True for v3
    df = load_and_clean_iedb(data_dir, include_hpv11=True)

    out_path = store.save_dataset(df, "immunogenicity_dataset_v3.csv")
    print(f"Dataset V3 saved to {out_path}")
    print(f"Total Peptides: {len(df)}")
    print(f"Positives: {df['label'].sum()} | Negatives: {(df['label'] == 0).sum()}")


if __name__ == "__main__":
    main()
