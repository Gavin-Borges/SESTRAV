import os

configfile: "config.yaml"

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
        "models/gnn/gnn_model.pth",
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
    conda:
        "environment.yml"
    shell:
        "python scripts/generate_hard_decoys.py --fasta {input.fasta} --allele {params.allele} --num_decoys {params.num_decoys} --output {output} > {log} 2>&1"


rule qc_dataset:
    input:
        dataset = TRAINING_DATASET
    output:
        "results/qc/dataset_qc.json"
    log:
        "logs/qc_dataset.log"
    conda:
        "environment.yml"
    shell:
        "python src/data_curation_qc.py --check-dataset {input.dataset} --config config.yaml > {log} 2>&1"


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
    conda:
        "environment.yml"
    shell:
        "python -m src.train_ann --data {input.data} --model-dir models/ann --feature-mode {params.feature_mode} --binding-matrix {input.binding_matrix} > {log} 2>&1"


rule train_gnn:
    input:
        data = TRAINING_DATASET,
        qc = "results/qc/dataset_qc.json",
        binding_matrix = config.get("binding_matrix_path", "models/peptide_binding_matrix_v4.csv")
    output:
        "models/gnn/gnn_model.pth"
    params:
        feature_mode = config.get("feature_mode", 31)
    log:
        "logs/train_gnn.log"
    conda:
        "environment.yml"
    shell:
        "python -m src.train_gnn --data {input.data} --model-dir models/gnn --feature-mode {params.feature_mode} --binding-matrix {input.binding_matrix} > {log} 2>&1"


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
    conda:
        "environment.yml"
    shell:
        "python -m src.final_validation_report "
        "--results-dir {params.results_dir} "
        "--model-dir {params.model_dir} "
        "--data {input.data} "
        "--binding-matrix {input.binding_matrix} "
        "--model-path {input.model} "
        "--dataset-mode {params.dataset_mode} "
        "--dataset-version {params.dataset_version} "
        "{params.freeze_flag} "
        "> {log} 2>&1"


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
    conda:
        "environment.yml"
    shell:
        "python src/verify/iedb_multi_virus_extractor.py {input.config} {params.mock_flag} > {log} 2>&1"


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
    run:
        cmd = f"python src/verify/sestrav_evaluator.py {input.targets}"
        if params.checkpoint:
            cmd += f" {params.checkpoint}"
        if params.mock_flag:
            cmd += f" {params.mock_flag}"
        cmd += f" > {log[0]} 2>&1"
        shell(cmd)


include: "standardize_outputs.smk"
