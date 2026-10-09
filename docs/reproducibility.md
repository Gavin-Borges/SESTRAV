# Reproducibility boundaries

SESTRAV has several distinct reproducibility tiers. A successful check at one
tier is not evidence for the next.

## Install and import

The repository declares its Python dependencies and optional extras. A clean
environment can install the package and import the library. The model data is
installed by the hash-verified two-step route in README's "MHCflurry model
data". On Python 3.13, MHCflurry 2.2.1's `mhcflurry-downloads fetch`, the
route's second step, fails because it imports the removed standard-library
`pipes` module (MHCflurry 2.3.0 and later do not). The first step,
`scripts/fetch_verified_mhcflurry.py`, imports no mhcflurry and runs on any
supported interpreter. This is a downloader limitation, not an import failure:
run the second step under Python 3.11 or 3.12 when live
binding prediction is needed.

The documented mode-31 RF training path does not need that fetch. It consumes
the tracked `data/immunogenicity_dataset_v5.csv` and the tracked pre-built
`models/peptide_binding_matrix_v5.csv`. Rebuilding the binding matrix does need
MHCflurry model data.

## DAG wiring from a clone

The tracked smoke fixture resolves the pipeline wiring with:

```bash
snakemake --snakefile pipeline.smk \
  --configfile tests/fixtures/dag_smoke/config.smoke.yaml \
  --cores 1 --dry-run
```

This dry-run proves that the fixture inputs and rule dependencies form a DAG.
It does not execute training, validate a model, or reproduce a scientific
number. The fixture's `rf_stub.joblib` is 22 bytes of text, not a serialized
model. Every tracked invocation of this fixture is a dry-run.

## Verification of a release bundle

The release workflow (`.github/workflows/release.yml`), run on a version tag,
builds a results bundle with `src/release_bundle.py`: a zip of the tracked
result files listed in its `CANONICAL_RESULT_FILES`, plus a manifest recording
each file's path, size and SHA-256. The manifest lets a reader verify that
downloaded bytes match the released bytes; it does not record the code
revision or the inputs that produced those results. The same workflow attests
the zip with SLSA build provenance, which binds its digest to the repository,
tag, commit and workflow run that packaged it, not to the computation that
produced the results inside it. Verification of a digest is not an independent
reproduction of the computation that produced it.
Published numbers remain governed by the claims register and their cited
result artifacts.

Pooled single-pass metrics and means over fold metrics are different
aggregations and must not be interchanged. Every v5 CV number must state its
splitter; the certified path is peptide-grouped. The historical
`docs/results_report.qmd` renders retracted ungrouped results and is retained
only as an annotated record.

## Work that needs external artifacts

A clean clone does not contain the production Stage-4 RF joblib, the neural
checkpoints required by the terminal pipeline rule, an ESM-2 embedding cache,
or MHCflurry presentation-model data. Prediction and full pipeline execution
therefore require verified external artifacts or a successful local rebuild.
`models/model_artifact_checksums.json` records a size and SHA-256 for the
Stage-4 RF joblib and other `models/` artifacts, so it can verify one of those
once obtained, but it cannot supply it. It has no entry for the terminal pipeline
rule's neural checkpoints, an ESM-2 cache or MHCflurry data.

The tracked antigen-processing cache contains documented mock-valued fields;
it is not evidence that live NetChop or TAP predictions were reproduced. The
production mode-31 path does not use those research-track fields.
