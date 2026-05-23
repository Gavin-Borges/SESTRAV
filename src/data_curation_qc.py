import pandas as pd
import numpy as np
import logging
import hashlib
import json
import yaml
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IEDBDataCurator:
    """
    A strict quality-control module for IEDB exports to ensure 
    reproducibility and absolute accuracy in data parsing.
    """
    def __init__(self):
        self.required_columns = ['Epitope - Name', 'Assay - Qualitative Measure']
        
    def sanitize_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sanitize dataframe to prevent parsing errors and NaN injection."""
        logger.info(f"Initial raw data shape: {df.shape}")
        
        # Ensure required columns exist
        for col in self.required_columns:
            if col not in df.columns:
                raise ValueError(f"Security/Formatting Error: Missing required column '{col}'")
                
        # Drop rows where Epitope sequence is missing
        df = df.dropna(subset=['Epitope - Name'])
        
        # Force string types to prevent injection/type errors
        df['Epitope - Name'] = df['Epitope - Name'].astype(str).str.upper().str.strip()
        df['Assay - Qualitative Measure'] = df['Assay - Qualitative Measure'].astype(str).str.strip()
        
        return df

    def resolve_conflicts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deduplicates peptides using a majority-voting mechanism.
        Resolves the issue where IEDB has conflicting assays for the same peptide.
        """
        # Map qualitative measure to binary labels (already done in run_qc_pipeline, but keeping for standalone usage)
        if 'is_positive' not in df.columns:
            positive_keywords = ['Positive', 'Positive-High', 'Positive-Low', 'Positive-Intermediate']
            df['is_positive'] = df['Assay - Qualitative Measure'].apply(
                lambda x: 1 if any(kw in x for kw in positive_keywords) else 0
            )
        
        # Group by sequence and calculate the mean positivity
        # If mean > 0.5, it's positive. If mean < 0.5, it's negative.
        # If mean == 0.5 (perfect conflict), we conservatively discard it or label negative.
        consensus = df.groupby('Epitope - Name')['is_positive'].mean().reset_index()
        
        def assign_final_label(mean_val):
            if mean_val > 0.5:
                return 1
            elif mean_val < 0.5:
                return 0
            else:
                return np.nan # Drop ambiguous
                
        consensus['final_label'] = consensus['is_positive'].apply(assign_final_label)
        
        # Enforce QC gates for conflict ratio
        ambiguous_count = consensus['final_label'].isna().sum()
        total_unique = len(consensus)
        conflict_ratio = ambiguous_count / total_unique if total_unique > 0 else 0
        
        if conflict_ratio > 0.15:
            logger.error(f"QC Failed: Conflict ratio {conflict_ratio:.2f} exceeds threshold 0.15")
            raise RuntimeError(f"Security/QC Error: Dataset conflict ratio {conflict_ratio:.2f} is too high.")
            
        consensus = consensus.dropna(subset=['final_label'])
        consensus['final_label'] = consensus['final_label'].astype(int)
        
        # Enforce QC gates for minimum yield
        if len(consensus) < 500:
            logger.error(f"QC Failed: Peptide yield {len(consensus)} is below threshold 500")
            raise RuntimeError(f"Security/QC Error: Post-deduplication yield {len(consensus)} is insufficient.")
        
        logger.info(f"Curated consensus data shape: {consensus.shape}")
        return consensus

    def explicit_negative_mining(self, df: pd.DataFrame, binding_threshold: float = 500.0) -> pd.DataFrame:
        """
        Phase 1.3 Explicit Negative Mining:
        Extracts high-confidence negative samples (peptides that bind strongly to MHC but fail to activate T-cells).
        If 'Assay - MHC Binding Affinity' exists, we use it to filter false negatives.
        """
        # If binding affinity column doesn't exist (because we pulled only tcell assays), return as is
        if 'Assay - MHC Binding Affinity' not in df.columns:
            logger.warning("No MHC binding column found; skipping explicit negative mining.")
            return df
            
        # Filter for Negative T-cell assays where Binding Affinity < threshold (strong binder)
        # Assuming df has 'is_positive' and 'Assay - MHC Binding Affinity'
        df['Assay - MHC Binding Affinity'] = pd.to_numeric(df['Assay - MHC Binding Affinity'], errors='coerce')
        
        # Keep positive assays, OR (negative assays AND strong binding)
        strong_binders = df['Assay - MHC Binding Affinity'] <= binding_threshold
        mask = (df['is_positive'] == 1) | ((df['is_positive'] == 0) & strong_binders)
        
        mined_df = df[mask].copy()
        logger.info(f"Explicit negative mining reduced dataset from {len(df)} to {len(mined_df)} records.")
        return mined_df

    def run_qc_pipeline(self, raw_csv_path: str, output_csv_path: str):
        """Execute the full QC pipeline on a raw IEDB export."""
        logger.info(f"Running QC pipeline on {raw_csv_path}")
        try:
            # IEDB exports often have multiple header rows. We skip the first row (URL info).
            # This is a known formatting issue we are strictly guarding against.
            df = pd.read_csv(raw_csv_path, header=1, low_memory=False)
        except Exception as e:
            logger.error(f"Failed to securely parse CSV: {e}")
            raise
            
        df_clean = self.sanitize_input(df)
        
        # Add is_positive flag for explicit mining
        positive_keywords = ['Positive', 'Positive-High', 'Positive-Low', 'Positive-Intermediate']
        df_clean['is_positive'] = df_clean['Assay - Qualitative Measure'].apply(
            lambda x: 1 if any(kw in x for kw in positive_keywords) else 0
        )
        
        # Explicit Negative Mining
        df_mined = self.explicit_negative_mining(df_clean)
        
        # Resolve conflicts and deduplicate
        df_final = self.resolve_conflicts(df_mined)
        
        df_final.to_csv(output_csv_path, index=False)
        logger.info(f"Successfully wrote curated data to {output_csv_path}")

if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="SESTRAV Data Curation & QC")
    parser.add_argument("--check-dataset", type=str, help="Path to final immunogenicity_dataset.csv to enforce strict QC")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    
    if args.check_dataset:
        try:
            logger.info(f"Running strict dataset QC on {args.check_dataset}")
            df = pd.read_csv(args.check_dataset)
            
            # 1. Amino Acid Validity
            valid_chars = set("ACDEFGHIKLMNPQRSTVWY")
            invalid_mask = ~df["peptide"].apply(lambda pep: set(str(pep).upper()).issubset(valid_chars))
            if invalid_mask.any():
                raise ValueError(f"Found {invalid_mask.sum()} peptides with non-canonical amino acids.")
            
            # 2. Strict Deduplication & Conflicts
            if df.duplicated(subset=["peptide", "label"]).any():
                raise ValueError("Dataset contains identical peptide-label duplicates.")
            
            label_counts = df.groupby("peptide")["label"].nunique()
            if (label_counts > 1).any():
                raise ValueError("Dataset contains conflicting labels for the same peptide.")
                
            # 3. Class Imbalance
            pos_count = (df["label"] == 1).sum()
            neg_count = (df["label"] == 0).sum()
            if neg_count > 0:
                ratio = pos_count / neg_count
                if ratio < 0.33 or ratio > 1.0:
                    logger.warning(f"Class imbalance detected: Ratio is {ratio:.2f} (Pos:Neg)")
            
            # 4. Freeze Mode Checksum
            with open(args.config, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                
            sha256 = hashlib.sha256()
            with open(args.check_dataset, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256.update(chunk)
            checksum = sha256.hexdigest()
            logger.info(f"Dataset SHA256: {checksum}")
            
            gov = config.get("dataset_governance", {})
            if config.get("freeze_mode", False) and gov.get("require_checksum_match_in_freeze_mode", False):
                expected = gov.get("provenance", {}).get("checksum", "pending")
                if expected != "pending" and expected != checksum:
                    raise RuntimeError(f"Freeze mode violation! Dataset checksum {checksum} does not match {expected}.")
                    
            logger.info("All strict dataset QC gates passed successfully.")
            
            # Write out success file
            out_dir = Path("results/qc")
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "dataset_qc.json", "w") as f:
                json.dump({
                    "dataset": args.check_dataset,
                    "checksum": checksum,
                    "status": "PASSED",
                    "positives": int((df["label"] == 1).sum()),
                    "negatives": int((df["label"] == 0).sum())
                }, f, indent=2)
            
        except Exception as e:
            logger.error(str(e))
            sys.exit(1)
    else:
        logger.info("IEDB Data Curator Module initialized successfully.")
