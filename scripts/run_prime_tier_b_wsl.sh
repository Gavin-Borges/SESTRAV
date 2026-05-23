#!/usr/bin/env bash
# Run PRIME 2.1 on Tier B peptide pools (chunked).
# PRIME must run from PRIME2.1 root (./PRIME), not lib/PRIME on PATH alone.
set -euo pipefail

PRIME_HOME="$HOME/tools/sestrav_external/PRIME2.1"
export PATH="$PRIME_HOME/lib:$HOME/tools/sestrav_external/MixMHCpred:$PATH"
REPO="${1:-.}"
CHUNK="${2:-500}"

if [[ ! -x "$PRIME_HOME/PRIME" ]] || [[ ! -x "$PRIME_HOME/lib/PRIME.x" ]]; then
  echo "Missing PRIME wrapper or lib/PRIME.x; run scripts/install_prime_wsl.sh" >&2
  exit 1
fi
if ! head -1 "$PRIME_HOME/PRIME" | grep -q bash; then
  echo "PRIME root must be bash wrapper (not compiled binary); run scripts/install_prime_wsl.sh" >&2
  exit 1
fi

STAGE="$HOME/sestrav_extval/tier_b_$(date -u +%Y%m%d_%H%M)"
mkdir -p "$STAGE/in" "$STAGE/out"
cd "$REPO"

ALLELES="$(tr -d '\r\n' < results/external_tier_b_prime_alleles.txt)"
mkdir -p results/external_tool_outputs/prime_tier_b_chunks

echo "[prime-tier-b] PRIME_HOME=$PRIME_HOME stage=$STAGE chunk=$CHUNK"

# Smoke from Linux home
SMOKE_IN="$STAGE/in/smoke.txt"
SMOKE_OUT="$STAGE/out/smoke_prime.txt"
head -20 results/external_prime_peptides.txt | tr -d '\r' > "$SMOKE_IN"
(
  cd "$PRIME_HOME"
  ./PRIME -i "$SMOKE_IN" -o "$SMOKE_OUT" -a "$ALLELES"
)
echo "[prime-tier-b] smoke OK"

chunk_and_run() {
  local src="$1"
  local prefix="$2"
  local n=0
  local chunk_idx=0
  local inp=""
  local outp_stage=""
  local outp_repo=""

  while IFS= read -r pep || [[ -n "$pep" ]]; do
    pep="$(echo "$pep" | tr -d '\r')"
    [[ -z "$pep" ]] && continue
    if (( n % CHUNK == 0 )); then
      inp="$STAGE/in/${prefix}_chunk_${chunk_idx}.txt"
      outp_stage="$STAGE/out/${prefix}_chunk_${chunk_idx}_prime.txt"
      outp_repo="results/external_tool_outputs/prime_tier_b_chunks/${prefix}_chunk_${chunk_idx}_prime.txt"
      : > "$inp"
      chunk_idx=$((chunk_idx + 1))
    fi
    echo "$pep" >> "$inp"
    n=$((n + 1))
  done < "$src"

  for ((i=0; i<chunk_idx; i++)); do
    inp="$STAGE/in/${prefix}_chunk_${i}.txt"
    outp_stage="$STAGE/out/${prefix}_chunk_${i}_prime.txt"
    outp_repo="results/external_tool_outputs/prime_tier_b_chunks/${prefix}_chunk_${chunk_idx}_prime.txt"
    outp_repo="results/external_tool_outputs/prime_tier_b_chunks/${prefix}_chunk_${i}_prime.txt"
    echo "[prime-tier-b] $prefix chunk $i: $(wc -l < "$inp") peptides"
    (
      cd "$PRIME_HOME"
      ./PRIME -i "$inp" -o "$outp_stage" -a "$ALLELES"
    )
    cp "$outp_stage" "$REPO/$outp_repo"
  done
}

chunk_and_run results/external_tier_b_EBV_peptides.txt EBV
chunk_and_run results/external_tier_b_HPV16_peptides.txt HPV16

cd "$REPO"
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
print(f"Merged {len(out)} -> {out_path}")
PY

echo "[prime-tier-b] done"
