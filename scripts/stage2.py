import pandas as pd
from functions.stage2_mhc_binding_prediction import predict_binding

proteome_id = snakemake.wildcards.proteome_id
peptides = pd.read_csv(snakemake.input[0])
alleles = snakemake.params.get("alleles", None)

predict_binding(peptides, proteome_id, alleles=alleles)
