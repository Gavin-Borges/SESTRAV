# standardize_outputs.smk
# Rules to run PRIME, PredIG, and standardize output schemas

import sys

rule run_prime:
    input:
        binding = "results/{proteome_id}_binding.csv"
    output:
        prime_out = "results/{proteome_id}_prime_output.txt"
    params:
        alleles = lambda wildcards: ",".join(config["alleles"])
    log:
        "logs/run_prime/{proteome_id}.log"
    benchmark:
        "results/benchmarks/run_prime/{proteome_id}.tsv"
    conda:
        "environments/prime.yaml"
    shell:
        # The hand-written '...' around {params.alleles} is replaced by `:q`. The
        # manual single quotes were POSIX-only and offered no protection: a config
        # allele containing a single quote closed them and escaped the argument.
        "{sys.executable:q} scripts/run_prime_wrapper.py --binding-csv {input.binding:q} --output {output.prime_out:q} --alleles {params.alleles:q} > {log:q} 2>&1"

rule run_predig:
    input:
        binding = "results/{proteome_id}_binding.csv",
        # The FASTA this antigen's peptides were cut from, taken from generate_peptides'
        # own input function rather than restating its lookup, so the two rules cannot
        # resolve one proteome to different files. pipeline.smk defines that rule before
        # its include of this file, which is the only way this file is ever loaded.
        fasta = rules.generate_peptides.input.fasta
    output:
        predig_out = "results/{proteome_id}_predig_output.csv"
    params:
        alleles = lambda wildcards: ",".join(config["alleles"])
    log:
        "logs/run_predig/{proteome_id}.log"
    benchmark:
        "results/benchmarks/run_predig/{proteome_id}.tsv"
    conda:
        "environments/predig.yaml"
    shell:
        # --proteome-id is load-bearing, not cosmetic: the wrapper hands this binding CSV
        # to build_recombinant_table as peptide metadata, and stage 2 writes no
        # proteome_id column, so without it no recombinant input can be built and PredIG
        # never runs. Quoted with :q like every other interpolation in this file.
        # --proteome-fasta gives every antigen its real parent proteins; without it
        # only the two panels PROTEOME_FASTA names found any, and PredIG scored the
        # rest on synthetic glycine context.
        "{sys.executable:q} scripts/run_predig_wrapper.py --binding-csv {input.binding:q} --output {output.predig_out:q} --alleles {params.alleles:q} --proteome-id {wildcards.proteome_id:q} --proteome-fasta {input.fasta:q} > {log:q} 2>&1"

rule standardize_predictor_outputs:
    input:
        binding = "results/{proteome_id}_binding.csv",
        prime = "results/{proteome_id}_prime_output.txt",
        predig = "results/{proteome_id}_predig_output.csv",
        sestrav = "results/{proteome_id}_ranked.csv"
    output:
        std_out = "results/{proteome_id}_standardized_outputs.csv"
    log:
        "logs/standardize_predictor_outputs/{proteome_id}.log"
    benchmark:
        "results/benchmarks/standardize_predictor_outputs/{proteome_id}.tsv"
    conda:
        "environment.yml"
    shell:
        "{sys.executable:q} scripts/standardize_outputs.py "
        "--binding {input.binding:q} "
        "--prime {input.prime:q} "
        "--predig {input.predig:q} "
        "--sestrav {input.sestrav:q} "
        "--output {output.std_out:q} "
        "> {log:q} 2>&1"
