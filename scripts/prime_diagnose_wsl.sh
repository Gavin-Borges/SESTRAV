#!/usr/bin/env bash
set -e
PRIME_HOME="$HOME/tools/sestrav_external/PRIME2.1"
export PATH="$HOME/tools/sestrav_external/MixMHCpred:$PATH"
cd "$PRIME_HOME"

echo "=== test 1: builtin test alleles ==="
./PRIME -i test/test.txt -o /tmp/p1.txt -a A0101,A2501,B0801,B1801
wc -l /tmp/p1.txt

echo "=== test 2: tier A alleles, 3 peptides ==="
printf '%s\n' CLGGLLTMV GLCTLVAML YMLDLQPET > /tmp/p2_in.txt
./PRIME -i /tmp/p2_in.txt -o /tmp/p2.txt -a A0201,A0101,A0301,A2402,A1101,B0702,B0801,B2705,B3501
wc -l /tmp/p2.txt

echo "=== test 3: tier B alleles file, 3 peptides ==="
ALLELES="$(tr -d '\r\n' < /mnt/c/Users/gavin/.gemini/antigravity/scratch/SESTRAV/SESTRAV-Dev/results/external_tier_b_prime_alleles.txt)"
./PRIME -i /tmp/p2_in.txt -o /tmp/p3.txt -a "$ALLELES"
wc -l /tmp/p3.txt

echo "ALL OK"
