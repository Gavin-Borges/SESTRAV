#!/usr/bin/env bash
# Activate SESTRAV git hooks from the canonical sources in scripts/hooks/.
# Run once after cloning: bash scripts/hooks/install.sh
#
# This points git's core.hooksPath at scripts/hooks/ so the tracked hooks are the
# single source of truth: edits to them take effect immediately with no copy drift,
# and the setting applies to every worktree of this clone. (git cannot auto-run
# hooks on clone by design, so this still must be run once per clone. The
# authoritative, un-bypassable secret gate for the PUBLIC repo is GitHub Secret
# Scanning + Push Protection, which local hooks cannot substitute for.)
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"

git config core.hooksPath scripts/hooks

for hook in pre-commit commit-msg pre-push; do
    chmod +x "${REPO_ROOT}/scripts/hooks/${hook}"
done

echo "SESTRAV hooks activated via core.hooksPath -> scripts/hooks/"
echo "Verify with: git config --get core.hooksPath"
