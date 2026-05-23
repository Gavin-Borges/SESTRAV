#!/usr/bin/env bash
set -e
export PATH="$HOME/tools/sestrav_external/PRIME2.1/lib:$HOME/tools/sestrav_external/MixMHCpred:$PATH"
cd /mnt/c/Users/gavin/.gemini/antigravity/scratch/SESTRAV/SESTRAV-Dev
head -50 results/external_tier_b_EBV_peptides.txt > /tmp/tb_smoke.txt
PRIME -i /tmp/tb_smoke.txt -o /tmp/tb_smoke_out.txt -a A0201,A0101,A0301,A2402,A1101,B0702,B0801,B2705,B3501
wc -l /tmp/tb_smoke_out.txt
