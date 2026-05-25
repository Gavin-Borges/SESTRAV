#!/usr/bin/env bash
set -e
PRIME_HOME="$HOME/tools/sestrav_external/PRIME2.1"
export PATH="$HOME/tools/sestrav_external/MixMHCpred:$PATH"
cd "$PRIME_HOME"

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

echo "=== test 1: builtin test alleles ==="
./PRIME -i test/test.txt -o "$TMP_DIR/p1.txt" -a A0101,A2501,B0801,B1801
wc -l "$TMP_DIR/p1.txt"

echo "=== test 2: tier A alleles, 3 peptides ==="
printf '%s\n' CLGGLLTMV GLCTLVAML YMLDLQPET > "$TMP_DIR/p2_in.txt"
./PRIME -i "$TMP_DIR/p2_in.txt" -o "$TMP_DIR/p2.txt" -a A0201,A0101,A0301,A2402,A1101,B0702,B0801,B2705,B3501
wc -l "$TMP_DIR/p2.txt"

echo "=== test 3: tier B alleles file, 3 peptides ==="
ALLELES="$(tr -d '\r\n' < /mnt/c/Users/gavin/.gemini/antigravity/scratch/SESTRAV/SESTRAV-Dev/results/external_tier_b_prime_alleles.txt)"
./PRIME -i "$TMP_DIR/p2_in.txt" -o "$TMP_DIR/p3.txt" -a "$ALLELES"
wc -l "$TMP_DIR/p3.txt"

echo "ALL OK"
