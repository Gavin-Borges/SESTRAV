configfile: "config.yaml"

ANTIGENS = config["antigens"]

DATASET_MODE = config.get("dataset_mode", "modeB_updated")
DATASET_VERSION = config.get("dataset_version", "IEDB-unknown")

rule Results:
    input:
        expand("results/{proteome_id}_ranked.csv", proteome_id=ANTIGENS),
        expand("results/{proteome_id}_top20_immunogenicity.png", proteome_id=ANTIGENS),
        expand("results/{proteome_id}_score_distribution.png", proteome_id=ANTIGENS),
        "models/ann/ann_model.pth",
        "models/gnn/gnn_model.pth",
        "results/final_validation_report.md"


rule generate_peptides:
    input:
        lambda wildcards: config.get("proteome_files", {}).get(
            wildcards.proteome_id,
            f"data/proteomes/{wildcards.proteome_id}.fasta"
        )
    output:
        "results/{proteome_id}_peptides.csv"
    params:
        lengths = config["peptide_lengths"]
    script:
        "scripts/stage1.py"


rule predict_binding:
    input:
        "results/{proteome_id}_peptides.csv"
    output:
        "results/{proteome_id}_binding.csv"
    params:
        alleles = config["alleles"]
    script:
        "scripts/stage2.py"


rule extract_features:
    input:
        "results/{proteome_id}_binding.csv"
    output:
        "results/{proteome_id}_features.csv"
    script:
        "scripts/stage3.py"


rule score_immunogenicity:
    input:
        "results/{proteome_id}_features.csv"
    output:
        "results/{proteome_id}_ranked.csv",
        "results/{proteome_id}_top20_immunogenicity.png",
        "results/{proteome_id}_score_distribution.png"
    params:
        model_path = config.get("model_path", ""),
        freeze_mode = config.get("freeze_mode", False)
    script:
        "scripts/stage4.py"


rule qc_dataset:
    input:
        "immunogenicity_dataset.csv"
    output:
        "results/qc/dataset_qc.json"
    shell:
        "python src/data_curation_qc.py --check-dataset {input[0]} --config config.yaml"


rule train_ann:
    input:
        "immunogenicity_dataset.csv",
        "results/qc/dataset_qc.json",
        config.get("binding_matrix_path", "models/peptide_binding_matrix.csv")
    output:
        "models/ann/ann_model.pth"
    params:
        feature_mode = config.get("feature_mode", 21),
        binding_matrix = config.get("binding_matrix_path", "models/peptide_binding_matrix.csv")
    shell:
        "python -m src.train_ann --data {input[0]} --model-dir models/ann --feature-mode {params.feature_mode} --binding-matrix {params.binding_matrix}"


rule train_gnn:
    input:
        "immunogenicity_dataset.csv",
        "results/qc/dataset_qc.json",
        config.get("binding_matrix_path", "models/peptide_binding_matrix.csv")
    output:
        "models/gnn/gnn_model.pth"
    params:
        feature_mode = config.get("feature_mode", 21),
        binding_matrix = config.get("binding_matrix_path", "models/peptide_binding_matrix.csv")
    shell:
        "python -m src.train_gnn --data {input[0]} --model-dir models/gnn --feature-mode {params.feature_mode} --binding-matrix {params.binding_matrix}"


rule full_validation_report:
    input:
        expand("results/{proteome_id}_features.csv", proteome_id=ANTIGENS),
        expand("results/{proteome_id}_ranked.csv", proteome_id=ANTIGENS),
        "immunogenicity_dataset.csv",
        "results/qc/dataset_qc.json",
        config.get("binding_matrix_path", "models/peptide_binding_matrix.csv"),
        config.get("model_path", "models/rf_30feature_integrated.joblib")
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
        results_dir = config.get("output_dir", "results"),
        model_dir = "models",
        data_path = "immunogenicity_dataset.csv",
        binding_matrix = config.get("binding_matrix_path", "models/peptide_binding_matrix.csv"),
        model_path = config.get("model_path", "models/rf_30feature_integrated.joblib"),
        dataset_mode = config.get("dataset_mode", "modeB_updated"),
        dataset_version = config.get("dataset_version", "IEDB-unknown"),
        freeze_flag = "--freeze-mode" if config.get("freeze_mode", False) else ""
    shell:
        "python -m src.final_validation_report "
        "--results-dir {params.results_dir} "
        "--model-dir {params.model_dir} "
        "--data {params.data_path} "
        "--binding-matrix {params.binding_matrix} "
        "--model-path {params.model_path} "
        "--dataset-mode {params.dataset_mode} "
        "--dataset-version {params.dataset_version} "
        "{params.freeze_flag}"
