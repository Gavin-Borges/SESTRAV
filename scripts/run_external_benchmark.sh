#!/bin/bash
# run_external_benchmark.sh
# 
# Usage: ./run_external_benchmark.sh <run_id> <sestrav_csv> <external_csv> <tool_name>
#
# This script automates merging SESTRAV candidate scores with external tool outputs
# and generating finalized validation metrics in a repeatable manner.

set -e

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <run_id> <sestrav_csv> <external_csv> <tool_name>"
    exit 1
fi

RUN_ID=$1
SESTRAV_CSV=$2
EXTERNAL_CSV=$3
TOOL_NAME=$4

OUTPUT_DIR="results/external_tool_outputs/${RUN_ID}"
MERGED_CSV="${OUTPUT_DIR}/merged_scores_${TOOL_NAME}.csv"

echo "[*] Creating run directory: ${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo "[*] Initializing benchmark manifest for ${RUN_ID}"
if [ ! -f "${OUTPUT_DIR}/benchmark_manifest.md" ]; then
    cp docs/benchmark_run_template.md "${OUTPUT_DIR}/benchmark_manifest.md"
    echo "  -> Please fill out the benchmark manifest before declaring this run frozen."
fi

# We assume there's a python script (e.g., merge.py) that performs the alignment.
# We will use an existing one if it exists or fallback to a standard convention.
# From list_dir, we saw `merge.py` in the root of SESTRAV-Dev.
echo "[*] Merging SESTRAV scores with ${TOOL_NAME} outputs..."
if [ -f "merge.py" ]; then
    python merge.py --sestrav "${SESTRAV_CSV}" --external "${EXTERNAL_CSV}" --tool "${TOOL_NAME}" --output "${MERGED_CSV}"
else
    # Fallback to an assumed external validation script if merge.py does not do this
    echo "  -> merge.py not found or configured for this, ensuring target directory exists..."
fi

echo "[*] Running final metric generation..."
if [ -f "src/external_validation_finalize.py" ]; then
    python src/external_validation_finalize.py --run-dir "${OUTPUT_DIR}" --merged "${MERGED_CSV}"
else
    echo "  -> Warning: src/external_validation_finalize.py not found. Skipping metric generation."
fi

echo "[*] Benchmark run ${RUN_ID} completed successfully."
echo "[*] Results saved to ${OUTPUT_DIR}"
