import os
import re
import sys
from pathlib import PurePosixPath

from snakemake.exceptions import WorkflowError

configfile: "config.yaml"

# ---------------------------------------------------------------------------
# Config validation prelude.
#
# Every shell: directive in this workflow and in the included
# standardize_outputs.smk interpolates config-derived values, and the workflow
# is reached by `snakemake --config key=value` as well as by a configfile, so a
# config value is untrusted input to a shell command. Those interpolations all
# carry Snakemake's `:q` quoting, which is the primary defence; this prelude is
# the second layer, bounding what a value may contain at all.
#
# An allowlist prelude is used rather than `validate(config, schema)` because:
# it adds no second file to keep in step with this one, it runs before the first
# rule is parsed so a bad value never reaches a DAG, and it can state the
# "relative path, inside the repository, no .. component" rule directly, which
# a JSON-schema `pattern` cannot express without a brittle regex.
#
# Only keys that are actually PRESENT are checked. The `config.get(key, <lit>)`
# fallbacks below are literals in this file, not caller input, so they need no
# runtime check.
# ---------------------------------------------------------------------------

# HLA-A*02:01 and the other canonical class I names. Note the default decoy
# allele is itself a `*` glob, which is why it must never reach a shell unquoted.
_ALLELE_RE = re.compile(r"[A-Za-z0-9]{1,8}-[A-Za-z0-9]{1,4}\*[0-9]{2,3}:[0-9]{2,3}[A-Za-z]?")
# Proteome ids and dataset modes: these name files and artifacts, so no spaces,
# no shell metacharacters, no leading punctuation.
_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_VERSION_RE = re.compile(r"[0-9]+(?:\.[0-9]+){1,3}(?:[-+][A-Za-z0-9.]{1,32})?")
_PATH_SEGMENT_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}")
_DRIVE_RE = re.compile(r"[A-Za-z]:")

_ALLELE_KEYS = ("decoy_allele",)
_NAME_KEYS = ("dataset_mode",)
_VERSION_KEYS = ("dataset_version", "mhcflurry_model_version")
# Integer keys that reach a shell: directive. Others are passed to script:
# directives as Python objects and never cross a shell.
_INT_KEYS = ("feature_mode", "num_decoys")
_PATH_KEYS = (
    "training_dataset",
    "binding_matrix_path",
    "model_path",
    "reference_proteome",
    "calibration_path",
    "thresholds_path",
    "antigen_processing_cache_path",
    "gnn_checkpoint",
    "structural_cache_dir",
    "output_dir",
)


def _reject(key, value, reason):
    raise WorkflowError(
        f"config[{key!r}] was rejected by the pipeline.smk validation prelude: "
        f"{reason}. Got: {value!r}"
    )


def _check_pattern(key, value, pattern, reason):
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        _reject(key, value, reason)


def _check_int(key, value):
    if isinstance(value, bool) or not isinstance(value, int):
        _reject(key, value, "must be an integer")


def _check_relative_path(key, value):
    if value == "":
        # An empty path is how this config spells "unset" (gnn_checkpoint, model_path).
        return
    if not isinstance(value, str):
        _reject(key, value, "must be a string path")
    if "\\" in value:
        _reject(key, value, "must use forward slashes, not backslashes")
    if value.startswith("/") or _DRIVE_RE.match(value) is not None:
        _reject(key, value, "must be relative to the repository root, not absolute")
    for segment in PurePosixPath(value).parts:
        if segment == "..":
            _reject(key, value, "must not escape the repository root with '..'")
        if _PATH_SEGMENT_RE.fullmatch(segment) is None:
            _reject(key, value, "path segments are restricted to [A-Za-z0-9_.-]")


def _validate_config(cfg):
    for key in _ALLELE_KEYS:
        if key in cfg:
            _check_pattern(key, cfg[key], _ALLELE_RE, "must be an allele name like HLA-A*02:01")
    for allele in cfg.get("alleles") or []:
        _check_pattern("alleles", allele, _ALLELE_RE, "must be an allele name like HLA-A*02:01")
    for antigen in cfg.get("antigens") or []:
        _check_pattern("antigens", antigen, _NAME_RE, "must be a bare proteome id")
    for proteome_id, path in (cfg.get("proteome_files") or {}).items():
        _check_pattern("proteome_files", proteome_id, _NAME_RE, "must be a bare proteome id")
        _check_relative_path(f"proteome_files[{proteome_id}]", path)
    for key in _NAME_KEYS:
        if key in cfg:
            _check_pattern(key, cfg[key], _NAME_RE, "must be a bare identifier")
    for key in _VERSION_KEYS:
        if key in cfg:
            _check_pattern(key, str(cfg[key]), _VERSION_RE, "must look like 5.0.0")
    for key in _INT_KEYS:
        if key in cfg:
            _check_int(key, cfg[key])
    for key in _PATH_KEYS:
        if key in cfg:
            _check_relative_path(key, cfg[key])


_validate_config(config)

ANTIGENS = config["antigens"]

DATASET_MODE = config.get("dataset_mode", "expansion_alpha")
DATASET_VERSION = config.get("dataset_version", "2.0.0-alpha")
TRAINING_DATASET = config.get("training_dataset", "data/immunogenicity_dataset_v4.csv")

rule Results:
    input:
        expand("results/{proteome_id}_ranked.csv", proteome_id=ANTIGENS),
        expand("results/{proteome_id}_top20_immunogenicity.png", proteome_id=ANTIGENS),
        expand("results/{proteome_id}_score_distribution.png", proteome_id=ANTIGENS),
        expand("results/{proteome_id}_standardized_outputs.csv", proteome_id=ANTIGENS),
        "models/ann/ann_model.pth",
        "models/gnn/structural_gnn_v2.pth",
        "results/final_validation_report.md"


rule generate_peptides:
    input:
        fasta = lambda wildcards: config.get("proteome_files", {}).get(
            wildcards.proteome_id,
            f"data/proteomes/{wildcards.proteome_id}.fasta"
        )
    output:
        "results/{proteome_id}_peptides.csv"
    params:
        lengths = config["peptide_lengths"]
    log:
        "logs/generate_peptides/{proteome_id}.log"
    benchmark:
        "results/benchmarks/generate_peptides/{proteome_id}.tsv"
    conda:
        "environment.yml"
    script:
        "scripts/stage1.py"


rule predict_binding:
    input:
        peptides = "results/{proteome_id}_peptides.csv"
    output:
        "results/{proteome_id}_binding.csv"
    params:
        alleles = config["alleles"]
    log:
        "logs/predict_binding/{proteome_id}.log"
    benchmark:
        "results/benchmarks/predict_binding/{proteome_id}.tsv"
    conda:
        "environment.yml"
    script:
        "scripts/stage2.py"


rule extract_features:
    input:
        binding = "results/{proteome_id}_binding.csv"
    output:
        "results/{proteome_id}_features.csv"
    log:
        "logs/extract_features/{proteome_id}.log"
    benchmark:
        "results/benchmarks/extract_features/{proteome_id}.tsv"
    conda:
        "environment.yml"
    script:
        "scripts/stage3.py"


rule score_immunogenicity:
    input:
        features = "results/{proteome_id}_features.csv"
    output:
        "results/{proteome_id}_ranked.csv",
        "results/{proteome_id}_top20_immunogenicity.png",
        "results/{proteome_id}_score_distribution.png"
    params:
        model_path = config.get("model_path", ""),
        freeze_mode = config.get("freeze_mode", False)
    log:
        "logs/score_immunogenicity/{proteome_id}.log"
    benchmark:
        "results/benchmarks/score_immunogenicity/{proteome_id}.tsv"
    conda:
        "environment.yml"
    script:
        "scripts/stage4.py"


rule generate_hard_decoys:
    input:
        fasta = config.get("reference_proteome", "data/proteomes/human_reference.fasta")
    output:
        "data/hard_decoys.csv"
    params:
        allele = config.get("decoy_allele", "HLA-A*02:01"),
        num_decoys = config.get("num_decoys", 10000)
    log:
        "logs/generate_hard_decoys.log"
    benchmark:
        "results/benchmarks/generate_hard_decoys.tsv"
    conda:
        "environment.yml"
    shell:
        # `:q` is Snakemake's shell quoting. It is platform-adaptive rather than
        # POSIX-only: snakemake.shell swaps the quote function to a cmd.exe one
        # when no bash is configured, which is the Windows default. The default
        # allele HLA-A*02:01 contains a `*`, so even the shipped value is a glob.
        "{sys.executable:q} scripts/generate_hard_decoys.py --fasta {input.fasta:q} --alleles {params.allele:q} --num-decoys {params.num_decoys:q} --output {output:q} > {log:q} 2>&1"


rule qc_dataset:
    input:
        dataset = TRAINING_DATASET
    output:
        "results/qc/dataset_qc.json"
    log:
        "logs/qc_dataset.log"
    benchmark:
        "results/benchmarks/qc_dataset.tsv"
    conda:
        "environment.yml"
    shell:
        "{sys.executable:q} src/data_curation_qc.py --check-dataset {input.dataset:q} --config config.yaml > {log:q} 2>&1"


rule train_ann:
    input:
        data = TRAINING_DATASET,
        qc = "results/qc/dataset_qc.json",
        binding_matrix = config.get("binding_matrix_path", "models/peptide_binding_matrix_v4.csv")
    output:
        "models/ann/ann_model.pth"
    params:
        feature_mode = config.get("feature_mode", 31)
    log:
        "logs/train_ann.log"
    benchmark:
        "results/benchmarks/train_ann.tsv"
    conda:
        "environment.yml"
    shell:
        "{sys.executable:q} -m src.train_ann --data {input.data:q} --model-dir models/ann --feature-mode {params.feature_mode:q} --binding-matrix {input.binding_matrix:q} > {log:q} 2>&1"


rule train_gnn:
    input:
        data = TRAINING_DATASET,
        qc = "results/qc/dataset_qc.json",
        binding_matrix = config.get("binding_matrix_path", "models/peptide_binding_matrix_v4.csv")
    output:
        # src/train_gnn.py (train_gnn_v2, architecture="v2" default) writes the checkpoint as
        # structural_gnn_v2.pth, not gnn_model.pth. This rule never passes --pooling, so it
        # takes the CLI default (pooling="mean"), which is the one pooling value that writes
        # the canonical untagged structural_gnn_v2.pth (see planned_gnn_artifact_paths() in
        # src/train_gnn.py) alongside pooling-tagged companions (gnn_scaler_mean.joblib,
        # gnn_config_mean.json, etc.) that are not declared here, matching the single-primary-
        # output convention used by the train_ann rule above.
        "models/gnn/structural_gnn_v2.pth"
    params:
        feature_mode = config.get("feature_mode", 31)
    log:
        "logs/train_gnn.log"
    benchmark:
        "results/benchmarks/train_gnn.tsv"
    conda:
        "environment.yml"
    shell:
        # --allow-overwrite is deliberate here: this rule IS the reproduction path for the
        # published models/gnn artifacts (and the OOF predictions in models/), so regenerating
        # them is the intent rather than an accident. src/train_gnn.py otherwise aborts on the
        # tracked gnn_config.json. A one-off experiment should use a scratch --model-dir instead.
        "{sys.executable:q} -m src.train_gnn --data {input.data:q} --model-dir models/gnn --allow-overwrite --feature-mode {params.feature_mode:q} --binding-matrix {input.binding_matrix:q} > {log:q} 2>&1"


rule full_validation_report:
    input:
        features = expand("results/{proteome_id}_features.csv", proteome_id=ANTIGENS),
        ranked = expand("results/{proteome_id}_ranked.csv", proteome_id=ANTIGENS),
        data = TRAINING_DATASET,
        qc = "results/qc/dataset_qc.json",
        binding_matrix = config.get("binding_matrix_path", "models/peptide_binding_matrix_v4.csv"),
        model = config.get("model_path", "models/rf_31feature_integrated.joblib")
    output:
        "results/gold_standard_validation.csv",
        "results/baseline_comparison.csv",
        "results/h2_tier_a_fold_metrics.csv",
        "results/h2_tier_a_summary.csv",
        "results/h2_tier_a_summary.md",
        "results/h2_tier_a_oof_scores.csv",
        "results/final_validation_report.md",
        "results/freeze_status.json",
        f"results/gold_standard_validation__{DATASET_MODE}__{DATASET_VERSION}.csv",
        f"results/baseline_comparison__{DATASET_MODE}__{DATASET_VERSION}.csv",
        f"results/h2_tier_a_summary__{DATASET_MODE}__{DATASET_VERSION}.csv",
    params:
        results_dir = lambda w, output: os.path.dirname(output[0]),
        model_dir = lambda w, input: os.path.dirname(input.model),
        dataset_mode = config.get("dataset_mode", "expansion_v4"),
        dataset_version = config.get("dataset_version", "4.0.0"),
        freeze_flag = "--freeze-mode" if config.get("freeze_mode", False) else ""
    log:
        "logs/full_validation_report.log"
    benchmark:
        "results/benchmarks/full_validation_report.tsv"
    conda:
        "environment.yml"
    shell:
        # --allow-overwrite is deliberate here: this rule declares the 10 published
        # results/ artifacts as its own Snakemake outputs, so it IS the reproduction
        # path for them - a re-run (e.g. `snakemake -f`) is the intent, not an
        # accident. src/final_validation_report.py otherwise aborts on the tracked
        # h2_tier_a_summary.md. A one-off experiment should use a scratch --results-dir.
        # {params.freeze_flag:q} is safe when the flag is empty: Snakemake's
        # QuotedFormatter skips quoting an empty string, so an unset flag
        # interpolates to nothing rather than to an empty '' argument.
        "{sys.executable:q} -m src.final_validation_report "
        "--results-dir {params.results_dir:q} "
        "--model-dir {params.model_dir:q} "
        "--data {input.data:q} "
        "--binding-matrix {input.binding_matrix:q} "
        "--model-path {input.model:q} "
        "--dataset-mode {params.dataset_mode:q} "
        "--dataset-version {params.dataset_version:q} "
        "--allow-overwrite "
        "{params.freeze_flag:q} "
        "> {log:q} 2>&1"


rule extract_verify_data:
    input:
        config = "src/verify/targets.json"
    output:
        csvs = [
            "results/verify/sars_cov_2_verify.csv",
            "results/verify/influenza_a_verify.csv",
            "results/verify/hcv_verify.csv"
        ]
    params:
        mock_flag = "--mock" if config.get("mock_ingestion", False) else ""
    log:
        "logs/extract_verify_data.log"
    benchmark:
        "results/benchmarks/extract_verify_data.tsv"
    conda:
        "environment.yml"
    shell:
        "{sys.executable:q} src/verify/iedb_multi_virus_extractor.py {input.config:q} {params.mock_flag:q} > {log:q} 2>&1"


rule evaluate_verify_gnn:
    input:
        targets = "src/verify/targets.json",
        data = [
            "results/verify/sars_cov_2_verify.csv",
            "results/verify/influenza_a_verify.csv",
            "results/verify/hcv_verify.csv"
        ]
    output:
        report = "results/verify/validation_report.json"
    params:
        checkpoint = config.get("gnn_checkpoint", ""),
        mock_flag = "--mock" if config.get("mock_evaluation", False) else ""
    log:
        "logs/evaluate_verify_gnn.log"
    benchmark:
        "results/benchmarks/evaluate_verify_gnn.tsv"
    # This was a `run:` block that built the command as an f-string and handed the
    # result to shell(). An f-string is interpolated by Python BEFORE Snakemake's
    # formatter ever sees it, so `:q` cannot apply and no quoting was possible:
    # params.checkpoint came straight from config into a shell command. A plain
    # `shell:` directive is used rather than a subprocess.run([...]) argv list
    # because every other rule in this workflow is a `shell:` directive and the
    # redirection to {log} is part of the contract, which subprocess.run would
    # have to re-implement with file handles. The two conditional appends are not
    # needed: an empty params value interpolates to nothing under `:q`, so the
    # argv the evaluator receives is identical to what the f-string produced.
    shell:
        "{sys.executable:q} src/verify/sestrav_evaluator.py {input.targets:q} "
        "{params.checkpoint:q} {params.mock_flag:q} > {log:q} 2>&1"


include: "standardize_outputs.smk"
