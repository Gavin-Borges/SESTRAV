configfile: "config.yaml"

ANTIGENS = config["antigens"]

rule Results:
    input:
        expand("results/{virus}_ranked.csv", virus=ANTIGENS),
        expand("results/{virus}_top20_immunogenicity.png", virus=ANTIGENS)


rule generate_peptides:
    input:
        "data/proteomes/{virus}.fasta"
    output:
        "results/{virus}_peptides.csv"
    script:
        "scripts/stage1.py"


rule predict_binding:
    input:
        "results/{virus}_peptides.csv"
    output:
        "results/{virus}_binding.csv"
    script:
        "scripts/stage2.py"


rule extract_features:
    input:
        "results/{virus}_binding.csv"
    output:
        "results/{virus}_features.csv"
    script:
        "scripts/stage3.py"


rule score_immunogenicity:
    input:
        "results/{virus}_features.csv"
    output:
        "results/{virus}_ranked.csv",
        "results/{virus}_top20_immunogenicity.png"
    script:
        "scripts/stage4.py"