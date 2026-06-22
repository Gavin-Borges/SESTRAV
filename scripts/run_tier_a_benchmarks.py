import os
import sys
import subprocess
import pandas as pd
import numpy as np
import argparse
import shutil

def get_data():
    # 1. Load peptides and labels
    peptides_df = pd.read_csv("results/external_validation_input.csv")
    # This has 720 rows. We only care about peptide and label
    
    # 2. Load peptide-allele pairs
    pairs_df = pd.read_csv("results/external_predig_peptide_allele_pairs.csv")
    # Columns: peptide, allele
    
    # Merge label into pairs
    df = pd.merge(pairs_df, peptides_df[['peptide', 'label']], on='peptide', how='left')
    
    # standardize columns for existing wrappers
    df = df.rename(columns={'allele': 'hla_allele'})
    
    return df, peptides_df

def run_deepimmuno(df, tmp_dir):
    print("Running DeepImmuno...")
    input_file = os.path.join(tmp_dir, "deepimmuno_in.csv")
    
    # DeepImmuno only supports 9 and 10-mers
    valid_mask = df['peptide'].str.len().isin([9, 10])
    di_df = df.loc[valid_mask, ['peptide', 'hla_allele']].copy()
    di_df['hla_allele'] = di_df['hla_allele'].str.replace(':', '')
    
    if len(di_df) == 0:
        return np.full(len(df), np.nan)
        
    di_df.to_csv(input_file, index=False, header=False)
    
    conda_exe = shutil.which("conda") or "conda"
    cmd = [
        conda_exe, "run", "-n", "di", "python",
        "deepimmuno-cnn.py",
        "--mode", "multiple",
        "--intdir", os.path.abspath(input_file),
        "--outdir", os.path.abspath(tmp_dir)
    ]
    env = os.environ.copy()
    env["TF_USE_LEGACY_KERAS"] = "1"
    subprocess.run(cmd, cwd=r"_local\tools\DeepImmuno-main", check=True, env=env)
    
    output_file = os.path.join(tmp_dir, "deepimmuno-cnn-result.txt")
    res = pd.read_csv(output_file, sep='\t')
    
    # Map back to full dataframe
    scores = np.full(len(df), np.nan)
    scores[valid_mask] = res['immunogenicity'].values
    return scores

def run_bigmhc(df, tmp_dir):
    print("Running BigMHC...")
    input_file = os.path.abspath(os.path.join(tmp_dir, "bigmhc_in.csv"))
    df[['peptide', 'hla_allele']].to_csv(input_file, index=False)
    
    cmd = [
        os.path.abspath(r"_local\venv_bigmhc\Scripts\python.exe"),
        "predict.py",
        "-i", input_file,
        "-m", "im", # BigMHC-IM for immunogenicity
        "-a", "1",  # hla is in column 1 (0-indexed)
        "-p", "0",  # peptide in col 0
        "-c", "1",  # skip header (1 row)
        "-d", "cpu"
    ]
    subprocess.run(cmd, cwd=r"_local\tools\bigmhc\src", check=True)
    
    output_file = input_file + ".prd"
    res = pd.read_csv(output_file)
    return res['BigMHC_IM'].values

def run_mixmhcpred(df, tmp_dir):
    print("Running MixMHCpred 2.2...")
    results = pd.Series(index=df.index, dtype=float)
    
    perl_exe = r"C:\Program Files\Git\usr\bin\perl.exe"
    script = os.path.abspath(r"_local\tools\MixMHCpred2.2\lib\run_MixMHCpred.pl").replace('\\', '/')
    lib_path = os.path.abspath(r"_local\tools\MixMHCpred2.2\lib").replace('\\', '/')
    
    for allele, group in df.groupby('hla_allele'):
        peptides_file = os.path.abspath(os.path.join(tmp_dir, "mix_peptides.txt"))
        out_file = os.path.abspath(os.path.join(tmp_dir, "mix_out.txt"))
        
        group['peptide'].to_csv(peptides_file, index=False, header=False)
        
        in_path = peptides_file.replace('\\', '/')
        out_path = out_file.replace('\\', '/')
        
        cmd = [
            perl_exe, script,
            "--alleles", allele,
            "--input", in_path,
            "--dir", os.path.abspath(tmp_dir).replace('\\', '/'),
            "--output", out_path,
            "--lib", lib_path
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            out_df = pd.read_csv(out_file, sep='\t', comment='#')
            score_col = 'Score_bestAllele'
            results.loc[group.index] = out_df[score_col].values
        except Exception as e:
            print(f"Failed for allele {allele}: {e}")
            results.loc[group.index] = np.nan
            
    return results.values

sys.path.insert(0, os.path.abspath('.'))
from src.evaluate_metrics import evaluate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true', help='Run on a small subset')
    args = parser.parse_args()
    
    pairs_df, peptides_df = get_data()
    
    if args.smoke:
        pairs_df = pairs_df.head(20).copy()
        
    tmp_dir = "_local/bench_tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    
    print(f"Running benchmarks on {len(pairs_df)} peptide-allele pairs...")
    
    pairs_df['deepimmuno_score'] = run_deepimmuno(pairs_df, tmp_dir)
    pairs_df['bigmhc_score'] = run_bigmhc(pairs_df, tmp_dir)
    pairs_df['mixmhcpred_score'] = run_mixmhcpred(pairs_df, tmp_dir)
    
    # Aggregate by MAX across alleles for each peptide
    print("Aggregating scores by MAX per peptide...")
    agg_df = pairs_df.groupby('peptide').agg({
        'deepimmuno_score': 'max',
        'bigmhc_score': 'max',
        'mixmhcpred_score': 'max',
        'label': 'first' # label is the same for all alleles of a peptide
    }).reset_index()
    
    # We must ensure all 720 peptides are in agg_df, even if some failed.
    agg_df = pd.merge(peptides_df[['peptide', 'label', 'rf_oof_score', 'binding_max']], agg_df.drop('label', axis=1), on='peptide', how='left')
    
    # Save the per-peptide scores
    os.makedirs("data", exist_ok=True)
    out_csv = "data/tier_a_external_benchmarks.csv"
    agg_df[['peptide', 'label', 'deepimmuno_score', 'bigmhc_score', 'mixmhcpred_score']].to_csv(out_csv, index=False)
    print(f"Saved aggregated per-peptide scores to {out_csv}")
    
    print("\n--- Benchmark Results (Tier A) ---")
    
    metrics_list = []
    
    # Evaluate sanity anchors first
    for model, col in [('SESTRAV RF', 'rf_oof_score'), ('Binding-only', 'binding_max'), ('DeepImmuno', 'deepimmuno_score'), ('BigMHC', 'bigmhc_score'), ('MixMHCpred 2.2', 'mixmhcpred_score')]:
        y_score = agg_df[col].values
        y_true = agg_df['label'].values
        valid = ~np.isnan(y_score)
        
        n_scored = valid.sum()
        coverage_pct = n_scored / len(agg_df) * 100
        
        if n_scored > 0:
            metrics = evaluate(y_true[valid], y_score[valid])
            auc = metrics['auc_roc']
            pr = metrics['auc_pr']
            issr = metrics['issr_10']
        else:
            auc = np.nan
            pr = np.nan
            issr = np.nan
            
        metrics_list.append({
            "tool": model, 
            "auc_pr": pr, 
            "auc_roc": auc, 
            "issr_10": issr, 
            "n_scored": n_scored, 
            "coverage_pct": coverage_pct
        })
        
        print(f"{model}:")
        print(f"  AUC-PR:  {pr:.4f}")
        print(f"  AUC-ROC: {auc:.4f}")
        print(f"  ISSR@10: {issr:.4f}")
        print(f"  Coverage: {n_scored}/{len(agg_df)} ({coverage_pct:.1f}%)")
        
    metrics_df = pd.DataFrame(metrics_list)
    os.makedirs("results", exist_ok=True)
    metrics_out = "results/table3_tier_a_metrics.csv"
    metrics_df.to_csv(metrics_out, index=False)
    print(f"Saved metrics to {metrics_out}")

if __name__ == "__main__":
    main()
