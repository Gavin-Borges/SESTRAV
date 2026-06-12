#!/usr/bin/env bash
set -e
SESTRAV_TOOL_ROOT="${SESTRAV_TOOL_ROOT:-$HOME/tools/sestrav_external}"
export PATH="$SESTRAV_TOOL_ROOT/PRIME2.1/lib:$SESTRAV_TOOL_ROOT/MixMHCpred:$PATH"
REPO_ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

head -50 results/external_tier_b_EBV_peptides.txt > "$TMP_DIR/tb_smoke.txt"
PRIME -i "$TMP_DIR/tb_smoke.txt" -o "$TMP_DIR/tb_smoke_out.txt" -a A0201,A0101,A0301,A2402,A1101,B0702,B0801,B2705,B3501
wc -l "$TMP_DIR/tb_smoke_out.txt"
