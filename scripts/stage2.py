import pandas as pd
from functions.stage2_mhc_binding_prediction import predict_binding

virus = snakemake.wildcards.virus
peptides = pd.read_csv(snakemake.input[0])

predict_binding(peptides, virus)