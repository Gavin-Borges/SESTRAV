#!/usr/bin/env bash
set -e
SESTRAV_TOOL_ROOT="${SESTRAV_TOOL_ROOT:-$HOME/tools/sestrav_external}"
export PATH="$SESTRAV_TOOL_ROOT/MixMHCpred:$PATH"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
echo "GILGFVFTL" > "$TMP/peptides.txt"
cd "$SESTRAV_TOOL_ROOT/MixMHCpred"
MixMHCpred -i "$TMP/peptides.txt" -o "$TMP/out" -a HLA-A02:01 2>&1 | tail -5
ls -la "$TMP/out" 2>/dev/null || ls -la "$TMP"
