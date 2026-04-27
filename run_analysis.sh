#!/bin/bash
#SBATCH --job-name=sestrav_analysis
#SBATCH --output=results/slurm-analysis-%j.out
#SBATCH --error=results/slurm-analysis-%j.out
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail

echo "=== SESTRAV Analysis Start: $(date) ==="
echo "Working directory: $(pwd)"
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"

eval "$(conda shell.bash hook)"
conda activate sestrav

echo "Python: $(which python)"

# Ensure shap is installed
python -c "import shap" 2>/dev/null || pip install shap --quiet

# Pre-flight: verify pipeline output exists (canonical + legacy alias fallback)
python -c "
import os
from src.naming import proteome_id_candidates, resolve_model_path

required_results = [
    ('EBV_B95_8_panel8', 'features'),
    ('EBV_B95_8_panel8', 'ranked'),
    ('HPV16_18_panel8', 'features'),
    ('HPV16_18_panel8', 'ranked'),
]
required_model_groups = [
    ['models/rf_30feature_integrated.joblib', 'models/rf_21feature_legacy.joblib'],
    ['models/xgb_30feature_integrated.joblib', 'models/xgb_21feature_legacy.joblib'],
]

missing = []
for pid, suffix in required_results:
    found = False
    for candidate in proteome_id_candidates(pid):
        if os.path.isfile(f'results/{candidate}_{suffix}.csv'):
            found = True
            break
    if not found:
        missing.append(f'results/{pid}_{suffix}.csv')

for model_group in required_model_groups:
    found = False
    for model_path in model_group:
        resolved = resolve_model_path(model_path)
        if os.path.isfile(resolved):
            found = True
            break
    if not found:
        missing.append(' or '.join(model_group))

if missing:
    print('FATAL: Missing required files:')
    for f in missing:
        print(f'  {f}')
    raise SystemExit(1)
print('Pre-flight: all required files present')
"

python run_analysis.py

echo ""
echo "=== SESTRAV Analysis End: $(date) ==="
