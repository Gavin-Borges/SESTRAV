#!/usr/bin/env bash
# Install PRIME 2.1 and MixMHCpred in WSL Ubuntu for SESTRAV external validation.
set -euo pipefail

INSTALL_DIR="${HOME}/tools/sestrav_external"
PRIME_SHA_ENV="${SESTRAV_PRIME_ZIP_SHA256:-da5e65fd4b142857f42167b15495600b3a6416641c73854a8a2cbc00d67ee6a8}"
MIX_SHA_ENV="${SESTRAV_MIXMHCPRED_ZIP_SHA256:-9b3d96813368df42622ce13a96eed09b447133699d8eaf5a4793da8561981c9b}"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

verify_sha() {
  local file="$1"
  local expected="$2"
  local label="$3"
  if [[ -z "$expected" ]]; then
    echo "Missing required SHA256 for $label" >&2
    exit 1
  fi
  local actual
  actual="$(sha256sum "$file" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA256 mismatch for $label: expected $expected got $actual" >&2
    exit 1
  fi
}

if [[ ! -f PRIME2.1.zip ]]; then
  echo "Downloading PRIME 2.1..."
  curl -sL -o PRIME2.1.zip "https://github.com/GfellerLab/PRIME/archive/refs/heads/master.zip"
fi
verify_sha "PRIME2.1.zip" "$PRIME_SHA_ENV" "PRIME2.1.zip"

if [[ ! -d PRIME2.1 ]]; then
  python3 - <<'PY'
import zipfile
import os
import stat
from pathlib import Path

z = zipfile.ZipFile("PRIME2.1.zip")
root = Path(".").resolve()
for member in z.infolist():
    target = (root / member.filename).resolve()
    if root not in target.parents and target != root:
        raise RuntimeError(f"Unsafe zip member path: {member.filename}")
z.extractall(".")
os.rename("PRIME-master", "PRIME2.1")
# Ensure root wrapper is the bash script, not a compiled binary
wrapper = "PRIME2.1/PRIME"
data = z.read("PRIME-master/PRIME")
with open(wrapper, "wb") as f:
    f.write(data)
os.chmod(wrapper, os.stat(wrapper).st_mode | stat.S_IEXEC)
print("PRIME2.1 extracted")
PY
fi

if [[ ! -d MixMHCpred ]]; then
  echo "Downloading MixMHCpred..."
  curl -sL -o MixMHCpred.zip "https://github.com/GfellerLab/MixMHCpred/archive/refs/heads/master.zip"
  verify_sha "MixMHCpred.zip" "$MIX_SHA_ENV" "MixMHCpred.zip"
  python3 - <<'PY'
import zipfile, os
from pathlib import Path
z = zipfile.ZipFile("MixMHCpred.zip")
root = Path(".").resolve()
for member in z.infolist():
    target = (root / member.filename).resolve()
    if root not in target.parents and target != root:
        raise RuntimeError(f"Unsafe zip member path: {member.filename}")
z.extractall(".")
os.rename("MixMHCpred-master", "MixMHCpred")
print("MixMHCpred extracted")
PY
fi
if [[ -f MixMHCpred.zip ]]; then
  verify_sha "MixMHCpred.zip" "$MIX_SHA_ENV" "MixMHCpred.zip"
fi

if [[ -f PRIME2.1/lib/PRIME.cc ]]; then
  echo "Compiling Linux PRIME scorer (lib/PRIME.x)..."
  (cd PRIME2.1/lib && g++ -O3 PRIME.cc -o PRIME.x)
fi

# Root PRIME must remain the bash wrapper (do NOT overwrite with lib/PRIME.x).
if ! head -1 PRIME2.1/PRIME 2>/dev/null | grep -q bash; then
  echo "Restoring PRIME bash wrapper from zip..."
  python3 - <<'PY'
import zipfile, os, stat
z = zipfile.ZipFile("PRIME2.1.zip")
with open("PRIME2.1/PRIME", "wb") as f:
    f.write(z.read("PRIME-master/PRIME"))
os.chmod("PRIME2.1/PRIME", 0o755)
PY
fi

export PATH="$INSTALL_DIR/PRIME2.1:$INSTALL_DIR/MixMHCpred:$PATH"
grep -q 'sestrav_external' "$HOME/.bashrc" 2>/dev/null || \
  echo "export PATH=\"$INSTALL_DIR/PRIME2.1:$INSTALL_DIR/MixMHCpred:\$PATH\"" >> "$HOME/.bashrc"

if [[ -x PRIME2.1/PRIME ]] && [[ -x PRIME2.1/lib/PRIME.x ]]; then
  echo "PRIME installed: wrapper=$(readlink -f PRIME2.1/PRIME) scorer=$(readlink -f PRIME2.1/lib/PRIME.x)"

else
  echo "PRIME install incomplete; check g++ and PRIME2.1 paths" >&2
  exit 1
fi
