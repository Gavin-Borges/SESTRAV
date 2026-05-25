#!/bin/bash
#SBATCH --job-name=sestrav_pipeline
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail

echo "=== SESTRAV Pipeline Start: $(date) ==="
echo "Working directory: $(pwd)"
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"

# Initialize conda for non-interactive shell safely, then activate
CONDA_BASE=$(conda info --base 2>/dev/null || echo "")
if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    eval "$(conda shell.bash hook)"
fi
conda activate sestrav

echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Verify critical dependencies before committing to a long run
python -c "
import sys
try:
    from joblib import load
    from mhcflurry import Class1PresentationPredictor
    from Bio import SeqIO
    from src.features import TRAIN_FEATURE_COLUMNS
    import yaml

    predictor = Class1PresentationPredictor.load()

    with open('config.yaml') as f:
        cfg = yaml.safe_load(f)
    assert 'model_path' in cfg, 'config.yaml missing model_path'
    model_path = cfg['model_path']
    model = load(model_path)
    assert model.n_features_in_ in (21, 30, 50), f'Model expects unsupported feature count: {model.n_features_in_}'

    print('Pre-flight checks PASSED')
    print(f'  Model path: {model_path}')
    print(f'  Model features: {model.n_features_in_}')
    print(f'  Antigens: {cfg[\"antigens\"]}')
    print(f'  Alleles: {len(cfg[\"alleles\"])} alleles')
    print(f'  MHCflurry loaded OK')
except Exception as e:
    print(f'FATAL: Dependency verification failed: {e}', file=sys.stderr)
    sys.exit(1)
"

echo ""
echo "=== Running pipeline.py (both EBV and HPV) ==="
echo ""

mkdir -p results

python pipeline.py

echo ""
echo "=== Pipeline Complete ==="
echo ""

# Show output files
echo "Output files:"
ls -lh results/

# Print summary statistics
python -c "
import sys
try:
    import pandas as pd
    import os
    from src.naming import proteome_id_candidates

    for proteome_id in ['EBV_B95_8_panel8', 'HPV16_18_panel8']:
        ranked = None
        for candidate in proteome_id_candidates(proteome_id):
            path = f'results/{candidate}_ranked.csv'
            if os.path.isfile(path):
                ranked = path
                break
        if ranked and os.path.isfile(ranked):
            df = pd.read_csv(ranked)
            print(f'\n--- {proteome_id} ---')
            print(f'  Total peptides scored: {len(df)}')
            print(f'  Score range: [{df[\"immunogenicity_score\"].min():.4f}, {df[\"immunogenicity_score\"].max():.4f}]')
            print(f'  Top 5 peptides:')
            for _, row in df.head(5).iterrows():
                print(f'    Rank {int(row[\"rank\"]):4d}  {row[\"peptide\"]:12s}  score={row[\"immunogenicity_score\"]:.4f}  protein={row.get(\"protein_id\", \"?\")}')
except Exception as e:
    print(f'WARNING: Could not generate post-run summary stats: {e}', file=sys.stderr)
"

echo ""
echo "=== Gold-Standard Validation ==="
python -c "
import sys
try:
    from src.gold_standard import full_validation_report

    report = full_validation_report('results')
    report.to_csv('results/gold_standard_validation.csv', index=False)
    print('Validation report saved to results/gold_standard_validation.csv')
except Exception as e:
    print(f'WARNING: Gold-standard validation failed: {e}', file=sys.stderr)
"

echo ""
echo "=== SESTRAV Pipeline End: $(date) ==="
