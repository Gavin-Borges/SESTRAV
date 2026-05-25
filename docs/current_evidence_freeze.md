# SESTRAV Current Evidence Freeze Pointer

Use this file as the single pointer for freeze-ready evidence status.

## Current Authoritative Freeze

- Active freeze snapshot: `docs/colloquium_evidence_freeze_v2_20260524.md`
- Latest publish-gate decision: `docs/final_publish_gate_report_20260524.md`
- Historical freeze cycle record: `docs/reproducibility_finalization_status.md` (2026-04-25 context)

## Operator Rule

- Treat the files above as authoritative only when generated with `freeze_mode: true`.
- If `results/freeze_status.json` reports `"valid": false`, do not communicate the run as freeze-ready.
- If multiple legacy/canonical output stems exist in `results/`, clean stale files and rerun before publishing claims.

## Minimum Freeze Validation Command

```bash
snakemake --snakefile pipeline.smk full_validation_report --cores 4 --forceall
```

Expected confirmation:
- `results/freeze_status.json` exists
- `results/freeze_status.json` contains `"valid": true`
