"""
SESTRAV Consensus Ensemble Module.
Normalizes scores and aggregates predictions from multiple models using a weighted Borda count.

Why Rank Aggregation (Borda Count) over Geometric Mean:
Geometric mean is highly sensitive to zero-cancellation (if any single model outputs
a near-zero probability, the product is zero, discarding strong signals from other models).
Borda count rank aggregation aggregates relative orderings, making it robust to
distribution scale shifts and zero-cancellation.
"""
import os
import argparse
import pandas as pd
import numpy as np

def min_max_scale(series: pd.Series) -> pd.Series:
    """Scale a series to [0, 1] range. Handle constant series gracefully."""
    s_min = series.min()
    s_max = series.max()
    if pd.isna(s_min) or s_min == s_max:
        return pd.Series(0.5, index=series.index)
    return (series - s_min) / (s_max - s_min)

def compute_borda_scores(df: pd.DataFrame, score_cols: list, weights: dict = None) -> pd.Series:
    """
    Compute Borda count score across given columns.
    For each column, rank peptides (1 = highest score/most immunogenic).
    Borda score = N - rank.
    Final score is the weighted sum of Borda scores, normalized to [0, 1].
    """
    n = len(df)
    if n == 0:
        return pd.Series(dtype=float)
        
    if weights is None:
        weights = {col: 1.0 for col in score_cols}
        
    total_weight = sum(weights[col] for col in score_cols if col in df.columns)
    if total_weight == 0:
        total_weight = 1.0
        
    borda_sum = pd.Series(0.0, index=df.index)
    
    for col in score_cols:
        if col not in df.columns:
            continue
        # Rank: higher values get rank 1 (ascending=False)
        # Use 'min' or 'average' method for ties
        ranks = df[col].rank(ascending=False, method='average')
        # Borda score: higher rank (smaller rank number) gets higher score
        borda_col = n - ranks
        weight = weights.get(col, 1.0)
        borda_sum += borda_col * weight
        
    # Scale final Borda sum to [0, 1]
    return min_max_scale(borda_sum)

def run_consensus(
    merged_csv: str,
    output_csv: str,
    score_cols: list,
    weights: dict = None,
) -> pd.DataFrame:
    """Accepts a merged CSV, aggregates scores, and appends TCR_Recognition_Propensity_Score."""
    if not os.path.isfile(merged_csv):
        raise FileNotFoundError(f"Merged scores file not found: {merged_csv}")
        
    df = pd.read_csv(merged_csv)
    print(f"[consensus] Loaded merged predictions: {len(df)} rows")
    
    # Verify cols exist
    available_cols = [c for c in score_cols if c in df.columns]
    if not available_cols:
        raise ValueError(f"None of the target score columns {score_cols} found in merged CSV. Available: {list(df.columns)}")
        
    print(f"[consensus] Aggregating columns: {available_cols}")
    
    # Fill NaNs with median/minimum before ranking so they don't break the ranks
    df_imputed = df.copy()
    for col in available_cols:
        # Fill NaN with column minimum to penalize missing predictions
        col_min = df_imputed[col].min()
        if pd.isna(col_min):
            col_min = 0.0
        df_imputed[col] = df_imputed[col].fillna(col_min)
        
    df['TCR_Recognition_Propensity_Score'] = compute_borda_scores(df_imputed, available_cols, weights)
    
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[consensus] Successfully saved consensus predictions ({len(df)} rows) to {output_csv}")
    return df

def main():
    parser = argparse.ArgumentParser(description="Compute Borda count consensus immunogenicity score")
    parser.add_argument("--merged-csv", required=True, help="Path to merged predictions CSV")
    parser.add_argument("--output-csv", required=True, help="Path to output CSV")
    parser.add_argument("--score-cols", default="sestrav_score,prime_score,predig_score", 
                        help="Comma-separated score columns to aggregate")
    parser.add_argument("--weights", default=None, 
                        help="Comma-separated weights corresponding to score-cols (e.g. 1.0,1.0,0.8)")
    args = parser.parse_args()
    
    cols = [c.strip() for c in args.score_cols.split(',')]
    
    weights = None
    if args.weights:
        w_vals = [float(w.strip()) for w in args.weights.split(',')]
        if len(w_vals) == len(cols):
            weights = dict(zip(cols, w_vals))
        else:
            print("[consensus] Warning: Length of weights does not match score-cols. Using uniform weights.")
            
    run_consensus(args.merged_csv, args.output_csv, cols, weights)

if __name__ == "__main__":
    main()
