#!/usr/bin/env bash
# Install SESTRAV git hooks from the canonical sources in scripts/hooks/.
# Run once after cloning: bash scripts/hooks/install.sh
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOKS_SRC="${REPO_ROOT}/scripts/hooks"
HOOKS_DST="${REPO_ROOT}/.git/hooks"

HOOKS=(pre-commit commit-msg)

for hook in "${HOOKS[@]}"; do
    src="${HOOKS_SRC}/${hook}"
    dst="${HOOKS_DST}/${hook}"
    cp "$src" "$dst"
    chmod +x "$dst"
    echo "Installed ${hook} -> .git/hooks/${hook}"
done

echo "All SESTRAV hooks installed."
