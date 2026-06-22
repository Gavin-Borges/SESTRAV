import os
import sys
import subprocess
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import argparse

def get_data():
    tsnadb = pd.read_csv("data/tsnadb_crossdomain_cohort.csv")
    hard_decoys = pd.read_csv("data/hard_decoys.csv")
    
    # standardize columns
    # tsnadb: peptide,hla_allele,label,...
    # hard_decoys: peptide,label,...,hla_allele,...
    
    positives = tsnadb[['peptide', 'hla_allele', 'label']].copy()
    negatives = hard_decoys[['peptide', 'hla_allele', 'label']].copy()
    
    df = pd.concat([positives, negatives], ignore_index=True)
    return df

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
    
    import shutil
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
    # BigMHC expects columns, we can provide peptide and hla
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
    # MixMHCpred requires running per allele to evaluate the exact allele
    # Actually, MixMHCpred outputs score for all provided alleles, but to be 
    # rigorous for peptide-HLA pairs, we just group by allele.
    results = pd.Series(index=df.index, dtype=float)
    
    perl_exe = r"C:\Program Files\Git\usr\bin\perl.exe"
    script = os.path.abspath(r"_local\tools\MixMHCpred2.2\lib\run_MixMHCpred.pl").replace('\\', '/')
    lib_path = os.path.abspath(r"_local\tools\MixMHCpred2.2\lib").replace('\\', '/')
    
    for allele, group in df.groupby('hla_allele'):
        peptides_file = os.path.abspath(os.path.join(tmp_dir, "mix_peptides.txt"))
        out_file = os.path.abspath(os.path.join(tmp_dir, "mix_out.txt"))
        
        group['peptide'].to_csv(peptides_file, index=False, header=False)
        
        # MixMHCpred format e.g. A0201 or HLA-A*02:01. The perl script normalizes it.
        # Use forward slashes for Perl script to avoid backslash escaping issues on Windows
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
            # The column is 'Score_best' or something similar
            # MixMHCpred 2.2 header: Peptide, Score_bestAllele, BestAllele, %Rank_bestAllele
            score_col = 'Score_bestAllele'
            
            # Map back to original indices
            # out_df should have same order as group
            results.loc[group.index] = out_df[score_col].values
        except Exception as e:
            print(f"Failed for allele {allele}: {e}")
            results.loc[group.index] = 0.0 # fallback
            
    return results.values

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true', help='Run on a small subset (20 peptides)')
    args = parser.parse_args()
    
    df = get_data()
    if args.smoke:
        # Take 10 pos and 10 neg
        df = pd.concat([df[df.label==1].head(10), df[df.label==0].head(10)], ignore_index=True)
    
    tmp_dir = "_local/bench_tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    
    print(f"Running benchmarks on {len(df)} samples...")
    
    # 1. DeepImmuno
    df['deepimmuno_score'] = run_deepimmuno(df, tmp_dir)
    
    # 2. BigMHC
    df['bigmhc_score'] = run_bigmhc(df, tmp_dir)
    
    # 3. MixMHCpred
    df['mixmhcpred_score'] = run_mixmhcpred(df, tmp_dir)
    
    # Calculate metrics
    print("\n--- Benchmark Results ---")
    
    y_true = df['label'].values
    
    metrics_list = []
    for model, col in [('DeepImmuno', 'deepimmuno_score'), ('BigMHC', 'bigmhc_score'), ('MixMHCpred 2.2', 'mixmhcpred_score')]:
        y_score = df[col].values
        valid = ~np.isnan(y_score)
        
        auc = roc_auc_score(y_true[valid], y_score[valid])
        pr = average_precision_score(y_true[valid], y_score[valid])
        
        metrics_list.append({"Model": model, "AUC-ROC": auc, "AUC-PR": pr})
        
        print(f"{model}:")
        print(f"  AUC-ROC: {auc:.4f}")
        print(f"  AUC-PR:  {pr:.4f}")
        
    metrics_df = pd.DataFrame(metrics_list)
    os.makedirs("results", exist_ok=True)
    metrics_df.to_csv("results/table3_benchmarks.csv", index=False)
    
    df.to_csv("data/external_benchmarks_results.csv", index=False)
    print("Saved aggregated metrics to results/table3_benchmarks.csv")
    print("Saved raw scores to data/external_benchmarks_results.csv")

if __name__ == "__main__":
    main()
