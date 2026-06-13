#!/bin/bash
# Script to generate Software Bill of Materials (SBOM) for SESTRAV
# Target: docs/sbom.json and docs/DEPENDENCY_LICENSES.md

set -e

DOCS_DIR="$(dirname "$0")/../docs"
mkdir -p "$DOCS_DIR"

echo "Generating Software Bill of Materials (SBOM)..."

# Option 1: Use syft if available
if command -v syft &> /dev/null; then
    echo "Found syft. Generating SBOM in SPDX format..."
    syft dir:. -o spdx-json="$DOCS_DIR/sbom.json"
    echo "SPDX SBOM saved to $DOCS_DIR/sbom.json"
else
    echo "syft not found. Checking for pip-licenses..."
    if pip show pip-licenses &> /dev/null; then
        echo "Generating dependency inventory via pip-licenses..."
        pip-licenses --format=json --output-file="$DOCS_DIR/sbom.json"
        pip-licenses --format=markdown --output-file="$DOCS_DIR/DEPENDENCY_LICENSES.md"
        echo "Dependency licenses saved to docs/sbom.json and docs/DEPENDENCY_LICENSES.md"
    else
        echo "Neither syft nor pip-licenses found. Attempting to install pip-licenses..."
        pip install pip-licenses
        pip-licenses --format=json --output-file="$DOCS_DIR/sbom.json"
        pip-licenses --format=markdown --output-file="$DOCS_DIR/DEPENDENCY_LICENSES.md"
        echo "Dependency licenses saved to docs/sbom.json and docs/DEPENDENCY_LICENSES.md"
    fi
fi

echo "SBOM generation run complete."
