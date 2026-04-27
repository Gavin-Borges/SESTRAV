configfile: "config.yaml"

ANTIGENS = config["antigens"]


rule Results:
    input:
        expand("results/{proteome_id}_ranked.csv", proteome_id=ANTIGENS),
        expand("results/{proteome_id}_top20_immunogenicity.png", proteome_id=ANTIGENS),
        expand("results/{proteome_id}_score_distribution.png", proteome_id=ANTIGENS)


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
        model_path = config.get("model_path", "")
    script:
        "scripts/stage4.py"


rule full_validation_report:
    input:
        expand("results/{proteome_id}_features.csv", proteome_id=ANTIGENS),
        expand("results/{proteome_id}_ranked.csv", proteome_id=ANTIGENS),
        "immunogenicity_dataset.csv",
        config.get("binding_matrix_path", "models/peptide_binding_matrix.csv"),
        config.get("model_path", "models/rf_30feature_integrated.joblib")
    output:
        "results/gold_standard_validation.csv",
        "results/baseline_comparison.csv",
        "results/h2_tier_a_fold_metrics.csv",
        "results/h2_tier_a_summary.csv",
        "results/h2_tier_a_summary.md",
        "results/final_validation_report.md"
    params:
        results_dir = config.get("output_dir", "results"),
        model_dir = "models",
        data_path = "immunogenicity_dataset.csv",
        binding_matrix = config.get("binding_matrix_path", "models/peptide_binding_matrix.csv"),
        model_path = config.get("model_path", "models/rf_30feature_integrated.joblib"),
        dataset_mode = config.get("dataset_mode", "modeB_updated"),
        dataset_version = config.get("dataset_version", "IEDB-unknown")
    shell:
        "python -m src.final_validation_report "
        "--results-dir {params.results_dir} "
        "--model-dir {params.model_dir} "
        "--data {params.data_path} "
        "--binding-matrix {params.binding_matrix} "
        "--model-path {params.model_path} "
        "--dataset-mode {params.dataset_mode} "
        "--dataset-version {params.dataset_version}"
