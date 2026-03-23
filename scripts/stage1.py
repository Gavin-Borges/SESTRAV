from functions.stage1_peptide_generation import generate_peptides

fasta = snakemake.input[0]
virus = snakemake.wildcards.virus

generate_peptides(fasta, virus)