#!/usr/bin/env bash
set -e
export PATH="$HOME/tools/sestrav_external/MixMHCpred:$PATH"
TMP=/tmp/mix_test_$$
mkdir -p "$TMP"
echo "GILGFVFTL" > "$TMP/peptides.txt"
cd "$HOME/tools/sestrav_external/MixMHCpred"
MixMHCpred -i "$TMP/peptides.txt" -o "$TMP/out" -a HLA-A02:01 2>&1 | tail -5
ls -la "$TMP/out" 2>/dev/null || ls -la "$TMP"
