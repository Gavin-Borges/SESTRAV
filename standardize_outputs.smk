# standardize_outputs.smk
# Rules to run PRIME, PredIG, and standardize output schemas

rule run_prime:
    input:
        binding = "results/{proteome_id}_binding.csv"
    output:
        prime_out = "results/{proteome_id}_prime_output.txt"
    conda:
        "environments/prime.yaml"
    params:
        alleles = lambda wildcards: ",".join(config["alleles"])
    shell:
        "python scripts/run_prime_wrapper.py --binding-csv {input.binding} --output {output.prime_out} --alleles '{params.alleles}'"

rule run_predig:
    input:
        binding = "results/{proteome_id}_binding.csv"
    output:
        predig_out = "results/{proteome_id}_predig_output.csv"
    conda:
        "environments/predig.yaml"
    params:
        alleles = lambda wildcards: ",".join(config["alleles"])
    shell:
        "python scripts/run_predig_wrapper.py --binding-csv {input.binding} --output {output.predig_out} --alleles '{params.alleles}'"

rule standardize_predictor_outputs:
    input:
        binding = "results/{proteome_id}_binding.csv",
        prime = "results/{proteome_id}_prime_output.txt",
        predig = "results/{proteome_id}_predig_output.csv",
        sestrav = "results/{proteome_id}_ranked.csv"
    output:
        std_out = "results/{proteome_id}_standardized_outputs.csv"
    shell:
        "python scripts/standardize_outputs.py "
        "--binding {input.binding} "
        "--prime {input.prime} "
        "--predig {input.predig} "
        "--sestrav {input.sestrav} "
        "--output {output.std_out}"
