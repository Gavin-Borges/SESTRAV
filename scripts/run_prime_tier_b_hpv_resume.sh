#!/usr/bin/env bash
# Resume Tier B PRIME for HPV16 chunks only (EBV chunks assumed done).
set -euo pipefail
REPO="${1:-.}"
CHUNK="${2:-500}"
PRIME_HOME="$HOME/tools/sestrav_external/PRIME2.1"
export PATH="$PRIME_HOME:$HOME/tools/sestrav_external/MixMHCpred:$PATH"
STAGE="$HOME/sestrav_extval/tier_b_hpv_resume_$(date -u +%Y%m%d_%H%M)"
mkdir -p "$STAGE/in" "$STAGE/out"
cd "$REPO"
ALLELES="$(tr -d '\r\n' < results/external_tier_b_prime_alleles.txt)"
mkdir -p results/external_tool_outputs/prime_tier_b_chunks
prefix=HPV16
src=results/external_tier_b_HPV16_peptides.txt
n=0
chunk_idx=0
inp=""
while IFS= read -r pep || [[ -n "$pep" ]]; do
  pep="$(echo "$pep" | tr -d '\r')"
  [[ -z "$pep" ]] && continue
  if (( n % CHUNK == 0 )); then
    inp="$STAGE/in/${prefix}_chunk_${chunk_idx}.txt"
    outp_stage="$STAGE/out/${prefix}_chunk_${chunk_idx}_prime.txt"
    : > "$inp"
    chunk_idx=$((chunk_idx + 1))
  fi
  echo "$pep" >> "$inp"
  n=$((n + 1))
done < "$src"
for ((i=0; i<chunk_idx; i++)); do
  inp="$STAGE/in/${prefix}_chunk_${i}.txt"
  outp_stage="$STAGE/out/${prefix}_chunk_${i}_prime.txt"
  outp_repo="results/external_tool_outputs/prime_tier_b_chunks/${prefix}_chunk_${i}_prime.txt"
  echo "[hpv-resume] chunk $i: $(wc -l < "$inp") peptides"
  (cd "$PRIME_HOME" && ./PRIME -i "$inp" -o "$outp_stage" -a "$ALLELES")
  cp "$outp_stage" "$REPO/$outp_repo"
done
python3 - <<'PY'
import pandas as pd
from pathlib import Path
root = Path("results/external_tool_outputs/prime_tier_b_chunks")
parts = sorted(root.glob("*_prime.txt"))
dfs = [pd.read_csv(p, sep="\t", comment="#") for p in parts]
out = pd.concat(dfs, ignore_index=True)
pep_col = "Peptide" if "Peptide" in out.columns else out.columns[0]
out = out.drop_duplicates(subset=[pep_col], keep="first")
out_path = Path("results/external_tool_outputs/tier_b_prime21_output.txt")
out.to_csv(out_path, sep="\t", index=False)
print(f"Merged {len(out)} unique peptides -> {out_path}")
PY
