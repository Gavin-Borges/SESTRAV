#!/usr/bin/env bash
set -e
export PATH="$HOME/tools/sestrav_external/PRIME2.1/lib:$HOME/tools/sestrav_external/MixMHCpred:$PATH"
echo "MixMHCpred=$(command -v MixMHCpred || echo missing)"
echo "PRIME=$(command -v PRIME || echo missing)"
ls -la "$HOME/tools/sestrav_external/MixMHCpred/" | head -10
file "$HOME/tools/sestrav_external/PRIME2.1/lib/PRIME"
ldd "$HOME/tools/sestrav_external/PRIME2.1/lib/PRIME" 2>&1 | head -15
