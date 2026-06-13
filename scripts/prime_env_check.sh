#!/usr/bin/env bash
set -e
SESTRAV_TOOL_ROOT="${SESTRAV_TOOL_ROOT:-$HOME/tools/sestrav_external}"
export PATH="$SESTRAV_TOOL_ROOT/PRIME2.1/lib:$SESTRAV_TOOL_ROOT/MixMHCpred:$PATH"
echo "MixMHCpred=$(command -v MixMHCpred || echo missing)"
echo "PRIME=$(command -v PRIME || echo missing)"
ls -la "$SESTRAV_TOOL_ROOT/MixMHCpred/" | head -10
file "$SESTRAV_TOOL_ROOT/PRIME2.1/lib/PRIME"
ldd "$SESTRAV_TOOL_ROOT/PRIME2.1/lib/PRIME" 2>&1 | head -15
