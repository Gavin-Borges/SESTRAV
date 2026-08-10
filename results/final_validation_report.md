# SESTRAV Final Validation Report

## Core Outputs
- Gold-standard stage validation: `results\gold_standard_validation.csv`
- Baseline comparison: `results\baseline_comparison.csv`
- H2 Tier A summary: `results\h2_tier_a_summary.csv`
- H2 Tier A markdown: `results\h2_tier_a_summary.md`

## H2 Tier A Headline
- R10 (ISSR@10 integrated / binding-only): `1.0588` (corrected 2026-08-10, D17; was `0.9494` VOID - see docs/claims_register.md)
- R25 (ISSR@25 integrated / binding-only): `1.0331` (corrected 2026-08-10, D17; was `1.0208` VOID)
- Decision (R10 >= 2 and stable denominator): **NOT SUPPORTED**

## Run Metadata
- Generated at (UTC): `2026-06-12T22:36:58.155757+00:00` (H2 Tier A Headline and the binding-matrix hash below were hand-patched 2026-08-10 per D17; the rest of this report reflects the original run)
- Freeze mode: `True`
- Input hashes:
  - Data: `b95199092c40afb156bcaca9fc97176b4ed2a187901eec1e31b5f62bb8e19e5b`
  - Binding matrix: `78aa3db8fa5f34d23eae55255cf4869b33c84fbb8416e81536e757b9c525c92c` (refreshed 2026-08-10; the prior hash `c7bb5ea1...` was the all-zeros placeholder itself - D17)
  - Model: `9a7bd2051c85e360c89a59f2bb1b4688e64a316900422e0cf45ba5967bf711f3`
