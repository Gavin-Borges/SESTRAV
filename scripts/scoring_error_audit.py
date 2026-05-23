import pandas as pd
import sys
import os
from datetime import datetime
import argparse

def generate_error_audit(predictions_csv, output_path):
    try:
        df = pd.read_csv(predictions_csv)
    except Exception as e:
        print(f"Error loading {predictions_csv}: {e}")
        return

    required_cols = ['peptide', 'virus', 'label', 'score']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        print("Note: predictions_csv needs label (0/1) and score (0-1).")
        return

    # Mock allele column if missing
    if 'allele' not in df.columns:
        df['allele'] = 'Unknown'

    # Define a threshold for hard classification (e.g., Top 25% or fixed score)
    threshold = df['score'].quantile(0.75)
    df['predicted_class'] = (df['score'] >= threshold).astype(int)

    # Calculate False Positives (FP) and False Negatives (FN)
    df['is_FP'] = ((df['predicted_class'] == 1) & (df['label'] == 0)).astype(int)
    df['is_FN'] = ((df['predicted_class'] == 0) & (df['label'] == 1)).astype(int)
    
    total_fp = df['is_FP'].sum()
    total_fn = df['is_FN'].sum()

    def aggregate_errors(group_by_col):
        grouped = df.groupby(group_by_col).agg(
            total=('peptide', 'count'),
            fp=('is_FP', 'sum'),
            fn=('is_FN', 'sum')
        )
        grouped['fp_rate'] = (grouped['fp'] / grouped['total']).round(3)
        grouped['fn_rate'] = (grouped['fn'] / grouped['total']).round(3)
        return grouped.sort_values(by=['fp', 'fn'], ascending=False)

    virus_errors = aggregate_errors('virus')
    allele_errors = aggregate_errors('allele')

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# SESTRAV 2.0 Scoring Error Audit\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Overview\n")
        f.write(f"- Total Samples Evaluated: {len(df)}\n")
        f.write(f"- Classification Threshold: {threshold:.3f} (Top 25%)\n")
        f.write(f"- Total False Positives: {total_fp}\n")
        f.write(f"- Total False Negatives: {total_fn}\n\n")

        f.write("## Error Taxonomy by Virus\n")
        f.write(virus_errors.to_markdown() + "\n\n")

        f.write("## Error Taxonomy by Allele (Top 10)\n")
        f.write(allele_errors.head(10).to_markdown() + "\n\n")

        f.write("## Difficult Peptides (Top 10 High Confidence FPs)\n")
        difficult_fps = df[df['is_FP'] == 1].sort_values(by='score', ascending=False).head(10)
        f.write(difficult_fps[['peptide', 'virus', 'allele', 'score']].to_markdown() + "\n\n")

        f.write("## Missed Gold-Standards (Top 10 Lowest Confidence FNs)\n")
        difficult_fns = df[df['is_FN'] == 1].sort_values(by='score', ascending=True).head(10)
        f.write(difficult_fns[['peptide', 'virus', 'allele', 'score']].to_markdown() + "\n\n")

        f.write("## Conclusion\n")
        f.write("Use this audit to prioritize feature engineering and allele-aware modeling tracks. Peptides highlighted in the False Positive list repeatedly fool the model and require structural or physicochemical debugging.")
    
    print(f"Scoring error audit generated at {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SESTRAV Scoring Error Audit Tool")
    parser.add_argument("--predictions", type=str, required=True, help="Path to CSV containing predictions and true labels")
    parser.add_argument("--output", type=str, default="results/scoring_error_audit.md", help="Output markdown path")
    
    args = parser.parse_args()
    generate_error_audit(args.predictions, args.output)
