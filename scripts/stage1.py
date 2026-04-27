from functions.stage1_peptide_generation import generate_peptides

fasta = snakemake.input[0]
proteome_id = snakemake.wildcards.proteome_id
lengths = snakemake.params.get("lengths", [8, 9, 10, 11])

generate_peptides(fasta, proteome_id, peptide_lengths=lengths)
