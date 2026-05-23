#!/usr/bin/env bash
# Run PRIME 2.1 for Tier A from WSL without inline shell interpolation.
set -euo pipefail

REPO="${1:?repo path required}"
PRIME_HOME="$HOME/tools/sestrav_external/PRIME2.1"
export PATH="$HOME/tools/sestrav_external/MixMHCpred:$PATH"

if [[ ! -x "$PRIME_HOME/PRIME" ]]; then
  echo "Missing PRIME wrapper; run scripts/install_prime_wsl.sh" >&2
  exit 1
fi

ALLELES="$(tr -d '\r\n' < "$REPO/results/external_prime_alleles_compact.txt")"

cd "$PRIME_HOME"
./PRIME \
  -i "$REPO/results/external_prime_peptides.txt" \
  -o "$REPO/results/external_tool_outputs/prime21_output.txt" \
  -a "$ALLELES"
